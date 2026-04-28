#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import List

from run_vlm_eval import (
    GEMINI_BATCH_TERMINAL_STATES,
    collect_gemini_batch_records,
    gemini_job_state_name,
    get_gemini_api_key,
    load_dataset,
    load_existing_records,
    load_local_env_files,
    parse_batch_ids,
    poll_gemini_batch_until_terminal,
    write_summary,
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[2]
    default_data_root = project_root / "data"
    default_dataset = default_data_root / "train_60k_v13_2.jsonl"

    parser = argparse.ArgumentParser(
        description="Lightweight Gemini batch recovery without prepare_items image preprocessing."
    )
    parser.add_argument(
        "--batch-id",
        default="",
        help="Comma-separated Gemini batch names. Example: batches/a,batches/b",
    )
    parser.add_argument(
        "--batch-file",
        default=None,
        help=(
            "Optional text file containing Gemini batch names. Supports both plain submitted list "
            "files (one `batches/...` per line) and log lines containing `name=batches/...`."
        ),
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-2.5-flash",
        help="Model name used when writing recovered records.",
    )
    parser.add_argument(
        "--gemini-api-key-env",
        default="GEMINI_API_KEY",
        help="Environment variable name to read Gemini API key from, e.g. GEMINI_API_KEY_2.",
    )
    parser.add_argument("--dataset", default=str(default_dataset), help="Path to source JSONL dataset.")
    parser.add_argument(
        "--base-dir",
        default=str(default_data_root),
        help="Base directory used to resolve relative paths in the dataset.",
    )
    parser.add_argument("--output", required=True, help="Recovered JSONL output path.")
    parser.add_argument(
        "--wait-for-batch",
        action="store_true",
        help="Wait for non-terminal Gemini batch jobs before collecting results.",
    )
    parser.add_argument(
        "--batch-poll-seconds",
        type=float,
        default=30.0,
        help="Polling interval in seconds when waiting for batch completion.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output instead of resuming.")
    return parser.parse_args()


def load_batch_names(batch_id_arg: str, batch_file: str) -> List[str]:
    batch_names = parse_batch_ids(batch_id_arg) if batch_id_arg else []
    if batch_file:
        text = Path(batch_file).expanduser().resolve().read_text(encoding="utf-8")
        # Works for both:
        # 1) logs: "... name=batches/xxxx state=..."
        # 2) submitted list: "batches/xxxx"
        extracted = re.findall(r"batches/[A-Za-z0-9_-]+", text)
        batch_names.extend(extracted)

    deduped: List[str] = []
    seen = set()
    for name in batch_names:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    load_local_env_files(script_dir)

    dataset_path = Path(args.dataset).expanduser().resolve()
    base_dir = Path(args.base_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_path.with_suffix(".summary.json")

    batch_names = load_batch_names(args.batch_id, args.batch_file)
    if not batch_names:
        raise ValueError("No Gemini batch names were provided. Use --batch-id or --batch-file.")

    if args.overwrite and output_path.exists():
        output_path.unlink()

    existing_records = load_existing_records(output_path)
    completed_task_ids = set(record.get("task_id") for record in existing_records if record.get("task_id"))

    # Recovery only needs a task_id -> item mapping for build_record().
    items = load_dataset(dataset_path, base_dir=base_dir)
    prepared_by_task_id = {
        str(item["task_id"]): {"task_id": str(item["task_id"]), "item": item}
        for item in items
    }

    from google import genai

    client = genai.Client(api_key=get_gemini_api_key(args.gemini_api_key_env))

    print(f"Dataset: {dataset_path}")
    print(f"Base dir: {base_dir}")
    print(f"Output: {output_path}")
    print(f"Gemini API key env: {args.gemini_api_key_env}")
    print(f"Loaded dataset items: {len(prepared_by_task_id)}")
    print(f"Existing recovered records: {len(existing_records)}")
    print(f"Batch count: {len(batch_names)}")

    terminal_batches = []
    for batch_name in batch_names:
        batch = client.batches.get(name=batch_name)
        state_name = gemini_job_state_name(batch)
        print(f"Loaded Gemini batch: name={batch_name} state={state_name}")
        if args.wait_for_batch and state_name not in GEMINI_BATCH_TERMINAL_STATES:
            batch = poll_gemini_batch_until_terminal(client, batch_name, args.batch_poll_seconds)
            state_name = gemini_job_state_name(batch)
        if state_name not in GEMINI_BATCH_TERMINAL_STATES:
            print(f"Skipping non-terminal batch: name={batch_name} state={state_name}")
            continue
        terminal_batches.append(batch)

    new_records = []
    for index, batch in enumerate(terminal_batches, start=1):
        batch_name = str(getattr(batch, "name", f"batch_{index}"))
        print(f"Collecting Gemini batch {index}/{len(terminal_batches)}: {batch_name}")
        batch_output_raw_path = output_path.with_suffix(f".gemini_batch_{index}.raw.json")
        batch_error_raw_path = output_path.with_suffix(f".gemini_batch_{index}.errors.json")
        new_records.extend(
            collect_gemini_batch_records(
                batch=batch,
                prepared_by_task_id=prepared_by_task_id,
                provider="gemini",
                model=args.gemini_model,
                completed_task_ids=completed_task_ids,
                output_raw_path=batch_output_raw_path,
                error_raw_path=batch_error_raw_path,
            )
        )

    if not new_records:
        print("No new completed records were collected from the Gemini batch job(s).")
        if existing_records:
            write_summary(existing_records, summary_path)
            print(f"Saved summary to: {summary_path}")
        return

    with output_path.open("a", encoding="utf-8") as f:
        for record in new_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    all_records = existing_records + new_records
    write_summary(all_records, summary_path)
    print(f"Collected {len(new_records)} new records from Gemini batch job(s)")
    print(f"Saved JSONL to: {output_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
