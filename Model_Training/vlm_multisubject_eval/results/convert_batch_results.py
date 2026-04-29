#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from run_vlm_eval import build_record, extract_json_object, normalize_result


PROJECT_ROOT = EVAL_DIR.parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_DATASET = DATA_ROOT / "train_60k_v13_2.jsonl"
DEFAULT_OUTPUT = EVAL_DIR / "results" / "2_5_merged_sorted.jsonl"
DEFAULT_ERROR_OUTPUT = DEFAULT_OUTPUT.with_suffix(".errors.jsonl")

SOURCE_SPECS = [
    {
        "dir": SCRIPT_DIR / "2_5_8001_22027_batches",
        "start_index": 8001,
        "end_index": 22027,
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "output": SCRIPT_DIR / "2_5_batch_results_8001_22027.jsonl",
    },
    {
        "dir": SCRIPT_DIR / "2_5_22290_28000_batches",
        "start_index": 22290,
        "end_index": 28000,
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "output": SCRIPT_DIR / "2_5_batch_results_22290_28000.jsonl",
    },
    {
        "dir": SCRIPT_DIR / "2_5_28001_31870_batches",
        "start_index": 28001,
        "end_index": 31870,
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "output": SCRIPT_DIR / "2_5_batch_results_28001_31870.jsonl",
    },
    {
        "dir": SCRIPT_DIR / "2_5_32192_34978_batches",
        "start_index": 32192,
        "end_index": 34978,
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "output": SCRIPT_DIR / "2_5_batch_results_32192_34978.jsonl",
    },
    {
        "dir": SCRIPT_DIR / "2_5_38493_40000_batches",
        "start_index": 38493,
        "end_index": 40000,
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "output": SCRIPT_DIR / "2_5_batch_results_38493_40000.jsonl",
    },
    {
        "dir": SCRIPT_DIR / "2_5_40001_43935_batches",
        "start_index": 40001,
        "end_index": 43935,
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "output": SCRIPT_DIR / "2_5_batch_results_40001_43935.jsonl",
    },
    {
        "dir": SCRIPT_DIR / "2_5_44190_50000_batches",
        "start_index": 44190,
        "end_index": 50000,
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "output": SCRIPT_DIR / "2_5_batch_results_44190_50000.jsonl",
    },
    {
        "dir": SCRIPT_DIR / "2_5_50001_52013_batches",
        "start_index": 50001,
        "end_index": 52013,
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "output": SCRIPT_DIR / "2_5_batch_results_50001_52013.jsonl",
    },
    {
        "dir": SCRIPT_DIR / "2_5_52336_55354_batches",
        "start_index": 52336,
        "end_index": 55354,
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "output": SCRIPT_DIR / "2_5_batch_results_52336_55354.jsonl",
    },
    {
        "dir": SCRIPT_DIR / "2_5_55359_60000_batches",
        "start_index": 55359,
        "end_index": 60000,
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "output": SCRIPT_DIR / "2_5_batch_results_55359_60000.jsonl",
    },
]


def batch_file_sort_key(path: Path) -> int:
    match = re.search(r"gemini_batch_(\d+)\.raw\.json$", path.name)
    if not match:
        raise ValueError(f"Unexpected batch raw filename: {path}")
    return int(match.group(1))


def iter_batch_files(directory: Path) -> List[Path]:
    files = sorted(directory.glob("*.raw.json"), key=batch_file_sort_key)
    if not files:
        raise FileNotFoundError(f"No raw batch files found under {directory}")
    return files


def extract_response_text(response_payload: Dict[str, object]) -> Optional[str]:
    candidates = response_payload.get("candidates")
    if not isinstance(candidates, list):
        return None

    chunks: List[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text)
    if not chunks:
        return None
    return "".join(chunks)


def build_item_from_train_record(record: Dict[str, object], line_number: int) -> Dict[str, object]:
    people_names = list(record.get("people_names") or [])
    object_names = list(record.get("object_names") or [])
    prompt = record.get("prompt_en") or record.get("prompt_zh") or ""
    return {
        "task_id": str(record.get("id") or line_number),
        "subject_count": record.get("total_entities"),
        "prompt": prompt,
        "metadata": {
            "ratio_type": record.get("ratio_type", "unknown"),
            "model_A_name": "A",
            "model_B_name": "B",
            "level": record.get("level"),
            "class_tag": record.get("class_tag"),
            "seed_id": record.get("seed_id"),
            "n_humans": record.get("n_humans"),
            "n_objects": record.get("n_objects"),
            "people_names": people_names,
            "object_names": object_names,
        },
    }


def load_train_items(dataset_file: Path) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    with dataset_file.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object at line {line_number}, got {type(record)}")
            item = build_item_from_train_record(record, line_number)
            items.append(item)
    return items


def flatten_directory_responses(directory: Path) -> List[Dict[str, object]]:
    flattened: List[Dict[str, object]] = []
    for batch_index, batch_file in enumerate(iter_batch_files(directory), start=1):
        payload = json.loads(batch_file.read_text(encoding="utf-8"))
        responses = ((payload.get("dest") or {}).get("inlined_responses") or [])
        if not isinstance(responses, list):
            raise ValueError(f"Invalid inlined_responses in {batch_file}")
        print(f"Loading {batch_file.name}: {len(responses)} responses")
        for response_index, response in enumerate(responses, start=1):
            flattened.append(
                {
                    "batch_file": str(batch_file),
                    "batch_index": batch_index,
                    "response_index": response_index,
                    "payload": response,
                }
            )
    return flattened


def convert_source(
    source_spec: Dict[str, object],
    dataset_items: List[Dict[str, object]],
) -> (List[Dict[str, object]], List[Dict[str, object]]):
    source_dir = Path(source_spec["dir"])
    start_index = int(source_spec["start_index"])
    end_index = int(source_spec["end_index"])
    provider = str(source_spec["provider"])
    fallback_model = str(source_spec["model"])

    flattened_responses = flatten_directory_responses(source_dir)
    expected_count = end_index - start_index + 1
    if len(flattened_responses) != expected_count:
        raise ValueError(
            f"{source_dir} contains {len(flattened_responses)} responses, expected {expected_count}"
        )

    item_slice = dataset_items[start_index - 1:end_index]
    if len(item_slice) != expected_count:
        raise ValueError(
            f"Dataset slice {start_index}-{end_index} has {len(item_slice)} items, expected {expected_count}"
        )

    converted_records: List[Dict[str, object]] = []
    error_records: List[Dict[str, object]] = []

    for item, response_info in zip(item_slice, flattened_responses):
        payload = response_info["payload"]
        if not isinstance(payload, dict):
            error_records.append(
                {
                    "task_id": item.get("task_id"),
                    "batch_file": response_info["batch_file"],
                    "response_index": response_info["response_index"],
                    "error": "Response payload is not a JSON object",
                }
            )
            continue

        response_body = payload.get("response")
        if not isinstance(response_body, dict):
            error_records.append(
                {
                    "task_id": item.get("task_id"),
                    "batch_file": response_info["batch_file"],
                    "response_index": response_info["response_index"],
                    "error": "Missing response body",
                }
            )
            continue

        raw_response_text = extract_response_text(response_body)
        if not raw_response_text:
            error_records.append(
                {
                    "task_id": item.get("task_id"),
                    "batch_file": response_info["batch_file"],
                    "response_index": response_info["response_index"],
                    "error": "Response does not contain text",
                    "response": response_body,
                }
            )
            continue

        try:
            parsed = extract_json_object(raw_response_text)
            normalized = normalize_result(parsed)
            model = str(response_body.get("model_version") or fallback_model)
            converted_records.append(
                build_record(
                    item=item,
                    normalized_result=normalized,
                    provider=provider,
                    model=model,
                    raw_response_text=raw_response_text,
                )
            )
        except Exception as exc:
            error_records.append(
                {
                    "task_id": item.get("task_id"),
                    "batch_file": response_info["batch_file"],
                    "response_index": response_info["response_index"],
                    "error": str(exc),
                    "raw_response_text": raw_response_text,
                }
            )

    return converted_records, error_records


def write_jsonl(records: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl_records(jsonl_path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{jsonl_path}:{line_number} is not a JSON object")
            records.append(payload)
    return records


def merge_and_sort_records(jsonl_paths: List[Path]) -> List[Dict[str, object]]:
    merged_by_task_id: Dict[str, Dict[str, object]] = {}
    for jsonl_path in jsonl_paths:
        print(f"Merging {jsonl_path} ...")
        for record in load_jsonl_records(jsonl_path):
            task_id = str(record.get("task_id"))
            if not task_id or task_id == "None":
                continue
            merged_by_task_id[task_id] = record
    return sorted(merged_by_task_id.values(), key=lambda record: int(str(record["task_id"])))


def discover_existing_merge_inputs(output_file: Path, converted_jsonl_paths: List[Path]) -> List[Path]:
    converted_names = {path.name for path in converted_jsonl_paths}
    candidates = sorted(SCRIPT_DIR.glob("*.jsonl"))
    selected: List[Path] = []
    for path in candidates:
        if path.name == output_file.name:
            continue
        if "error" in path.name.lower():
            continue
        if path.name in converted_names:
            continue
        if (
            path.name.startswith("2_5")
            or path.name.startswith("gemini_2_5_flash_")
        ):
            selected.append(path)
    return selected


def convert_batch_results(
    output_file: Path = DEFAULT_OUTPUT,
    error_output_file: Path = DEFAULT_ERROR_OUTPUT,
    dataset_file: Path = DEFAULT_DATASET,
) -> None:
    print(f"Loading dataset from {dataset_file} ...")
    dataset_items = load_train_items(dataset_file)
    print(f"Loaded {len(dataset_items)} dataset items")

    all_records: List[Dict[str, object]] = []
    all_errors: List[Dict[str, object]] = []
    converted_jsonl_paths: List[Path] = []

    for source_spec in SOURCE_SPECS:
        print(
            f"Converting {source_spec['dir']} for dataset slice "
            f"{source_spec['start_index']}-{source_spec['end_index']} ..."
        )
        records, errors = convert_source(source_spec, dataset_items)
        all_records.extend(records)
        all_errors.extend(errors)
        converted_output_path = Path(source_spec["output"])
        write_jsonl(records, converted_output_path)
        converted_jsonl_paths.append(converted_output_path)
        print(f"  saved converted jsonl to {converted_output_path}")
        print(f"  converted={len(records)} errors={len(errors)}")

    existing_jsonl_paths = discover_existing_merge_inputs(output_file, converted_jsonl_paths)
    merge_inputs = existing_jsonl_paths + converted_jsonl_paths
    merged_records = merge_and_sort_records(merge_inputs)
    write_jsonl(merged_records, output_file)

    if all_errors:
        write_jsonl(all_errors, error_output_file)

    print(f"Converted {len(all_records)} new batch results")
    print(f"Merged {len(merge_inputs)} jsonl files into {output_file}")
    print(f"Final merged record count: {len(merged_records)}")
    if all_errors:
        print(f"Saved {len(all_errors)} conversion errors to {error_output_file}")


if __name__ == "__main__":
    convert_batch_results()
