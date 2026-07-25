import argparse
import json
from collections import Counter
from pathlib import Path


PROMPT_REQUIRED_FIELDS = {
    "id": int,
    "seed_id": int,
    "level": int,
    "class_tag": str,
    "ratio_type": str,
    "n_humans": int,
    "n_objects": int,
    "total_entities": int,
    "people_names": list,
    "object_names": list,
    "prompt_en": str,
    "prompt_zh": str,
    "token_len_est": int,
}

LABEL_REQUIRED_FIELDS = {
    "task_id": str,
    "provider": str,
    "model": str,
    "a_existence": (int, float),
    "a_appearance": (int, float),
    "a_interaction": (int, float),
    "b_existence": (int, float),
    "b_appearance": (int, float),
    "b_interaction": (int, float),
    "winner": str,
    "reason": str,
    "metadata": dict,
    "subject_count": int,
    "prompt": str,
}

LABEL_METADATA_REQUIRED_FIELDS = {
    "ratio_type": str,
    "level": int,
    "class_tag": str,
    "seed_id": int,
    "n_humans": int,
    "n_objects": int,
    "people_names": list,
    "object_names": list,
}


def load_jsonl(path):
    records = []
    errors = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                errors.append(f"{path.name}:{line_no} empty line")
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name}:{line_no} invalid json: {exc}")
                continue
            if not isinstance(obj, dict):
                errors.append(f"{path.name}:{line_no} json record is not an object")
                continue
            records.append((line_no, obj))
    return records, errors


def check_required_fields(record, required_fields, prefix):
    errors = []
    for key, expected_type in required_fields.items():
        if key not in record:
            errors.append(f"{prefix} missing required field `{key}`")
            continue
        if not isinstance(record[key], expected_type):
            errors.append(
                f"{prefix} field `{key}` has wrong type: "
                f"expected {expected_type}, got {type(record[key]).__name__}"
            )
    return errors


def validate_prompt_file(path):
    records, errors = load_jsonl(path)
    prompts_by_id = {}
    duplicate_ids = Counter()
    level_counter = Counter()

    for line_no, record in records:
        prefix = f"{path.name}:{line_no}"
        errors.extend(check_required_fields(record, PROMPT_REQUIRED_FIELDS, prefix))
        prompt_id = record.get("id")
        if not isinstance(prompt_id, int):
            continue

        if prompt_id in prompts_by_id:
            duplicate_ids[prompt_id] += 1
        else:
            prompts_by_id[prompt_id] = record

        level = record.get("level")
        if isinstance(level, int):
            level_counter[level] += 1
            if level not in {2, 4, 6, 8}:
                errors.append(f"{prefix} unexpected level `{level}`")

        people = record.get("people_names", [])
        objects = record.get("object_names", [])
        n_humans = record.get("n_humans")
        n_objects = record.get("n_objects")
        total = record.get("total_entities")

        if isinstance(people, list) and any(not isinstance(x, str) or not x for x in people):
            errors.append(f"{prefix} `people_names` must be a list of non-empty strings")
        if isinstance(objects, list) and any(not isinstance(x, str) or not x for x in objects):
            errors.append(f"{prefix} `object_names` must be a list of non-empty strings")
        if isinstance(n_humans, int) and isinstance(people, list) and n_humans != len(people):
            errors.append(f"{prefix} `n_humans` != len(people_names)")
        if isinstance(n_objects, int) and isinstance(objects, list) and n_objects != len(objects):
            errors.append(f"{prefix} `n_objects` != len(object_names)")
        if (
            isinstance(total, int)
            and isinstance(n_humans, int)
            and isinstance(n_objects, int)
            and total != n_humans + n_objects
        ):
            errors.append(f"{prefix} `total_entities` != n_humans + n_objects")
        if isinstance(total, int) and isinstance(people, list) and isinstance(objects, list):
            if total != len(people) + len(objects):
                errors.append(f"{prefix} `total_entities` != len(people_names) + len(object_names)")
        if not record.get("prompt_en"):
            errors.append(f"{prefix} empty `prompt_en`")

    if duplicate_ids:
        dup_preview = ", ".join(str(x) for x in sorted(duplicate_ids)[:10])
        errors.append(f"{path.name} duplicate prompt ids found: {dup_preview}")

    return {
        "records": records,
        "prompts_by_id": prompts_by_id,
        "errors": errors,
        "level_counter": level_counter,
    }


def validate_label_file(path):
    records, errors = load_jsonl(path)
    labels_by_id = {}
    duplicate_ids = Counter()
    winner_counter = Counter()

    for line_no, record in records:
        prefix = f"{path.name}:{line_no}"
        errors.extend(check_required_fields(record, LABEL_REQUIRED_FIELDS, prefix))
        metadata = record.get("metadata")
        if isinstance(metadata, dict):
            errors.extend(check_required_fields(metadata, LABEL_METADATA_REQUIRED_FIELDS, f"{prefix}.metadata"))

        task_id = record.get("task_id")
        if not isinstance(task_id, str):
            continue
        if task_id in labels_by_id:
            duplicate_ids[task_id] += 1
        else:
            labels_by_id[task_id] = record

        winner = record.get("winner")
        if isinstance(winner, str):
            winner_counter[winner] += 1
            if winner not in {"A", "B"}:
                errors.append(f"{prefix} invalid `winner`: {winner}")

        for score_key in [
            "a_existence",
            "a_appearance",
            "a_interaction",
            "b_existence",
            "b_appearance",
            "b_interaction",
        ]:
            score = record.get(score_key)
            if isinstance(score, (int, float)) and not (0 <= score <= 1):
                errors.append(f"{prefix} `{score_key}` out of range [0, 1]: {score}")

        metadata = record.get("metadata", {})
        people = metadata.get("people_names", [])
        objects = metadata.get("object_names", [])
        n_humans = metadata.get("n_humans")
        n_objects = metadata.get("n_objects")
        subject_count = record.get("subject_count")

        if isinstance(n_humans, int) and isinstance(people, list) and n_humans != len(people):
            errors.append(f"{prefix} metadata `n_humans` != len(people_names)")
        if isinstance(n_objects, int) and isinstance(objects, list) and n_objects != len(objects):
            errors.append(f"{prefix} metadata `n_objects` != len(object_names)")
        if isinstance(subject_count, int) and isinstance(people, list) and isinstance(objects, list):
            if subject_count != len(people) + len(objects):
                errors.append(f"{prefix} `subject_count` != len(people_names) + len(object_names)")
        if not record.get("prompt"):
            errors.append(f"{prefix} empty `prompt`")

    if duplicate_ids:
        dup_preview = ", ".join(sorted(duplicate_ids)[:10])
        errors.append(f"{path.name} duplicate task_ids found: {dup_preview}")

    return {
        "records": records,
        "labels_by_id": labels_by_id,
        "errors": errors,
        "winner_counter": winner_counter,
    }


def cross_check(prompt_info, label_infos):
    warnings = []
    errors = []
    prompts_by_id = prompt_info["prompts_by_id"]
    prompt_ids = set(str(x) for x in prompts_by_id.keys())

    label_id_sets = []
    for label_name, info in label_infos.items():
        label_ids = set(info["labels_by_id"].keys())
        label_id_sets.append(label_ids)

        only_in_label = sorted(label_ids - prompt_ids)
        if only_in_label:
            warnings.append(
                f"{label_name}: {len(only_in_label)} task_ids exist in labels but not in prompt file"
            )

        for task_id, label in info["labels_by_id"].items():
            prompt = prompts_by_id.get(int(task_id)) if task_id.isdigit() else None
            if prompt is None:
                continue
            metadata = label["metadata"]
            if label.get("prompt") != prompt.get("prompt_en"):
                errors.append(f"{label_name}: task_id={task_id} label prompt != prompt_en")
            if label.get("subject_count") != prompt.get("total_entities"):
                errors.append(f"{label_name}: task_id={task_id} subject_count mismatch")
            for key in ["seed_id", "level", "class_tag", "ratio_type", "n_humans", "n_objects"]:
                if metadata.get(key) != prompt.get(key):
                    errors.append(f"{label_name}: task_id={task_id} metadata `{key}` mismatch")
            if metadata.get("people_names") != prompt.get("people_names"):
                errors.append(f"{label_name}: task_id={task_id} people_names mismatch")
            if metadata.get("object_names") != prompt.get("object_names"):
                errors.append(f"{label_name}: task_id={task_id} object_names mismatch")

    if len(label_id_sets) >= 2:
        common_ids = set.intersection(*label_id_sets)
        union_ids = set.union(*label_id_sets)
        warnings.append(f"label intersection size: {len(common_ids)}")
        warnings.append(f"label union size: {len(union_ids)}")
        for label_name, ids in zip(label_infos.keys(), label_id_sets):
            warnings.append(f"{label_name}: {len(ids)} unique task_ids")
        conflicting_winners = 0
        for task_id in common_ids:
            winners = {
                label_infos[label_name]["labels_by_id"][task_id]["winner"]
                for label_name in label_infos
            }
            if len(winners) > 1:
                conflicting_winners += 1
        warnings.append(f"label winner conflicts on common ids: {conflicting_winners}")

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description="Validate Model_Training data_v2 prompt and label jsonl files.")
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data_v2" / "prompt" / "train_60k_v13_2.jsonl"),
        help="Path to prompt jsonl",
    )
    parser.add_argument(
        "--label_paths",
        nargs="+",
        default=[
            str(Path(__file__).resolve().parent.parent / "data_v2" / "60k_LLM_Result" / "2_5_merged_sorted.jsonl"),
            str(Path(__file__).resolve().parent.parent / "data_v2" / "60k_LLM_Result" / "3_1_merged_sorted.jsonl"),
        ],
        help="Paths to label jsonl files",
    )
    args = parser.parse_args()

    prompt_path = Path(args.prompt_path)
    label_paths = [Path(x) for x in args.label_paths]

    prompt_info = validate_prompt_file(prompt_path)
    label_infos = {path.name: validate_label_file(path) for path in label_paths}

    all_errors = list(prompt_info["errors"])
    for info in label_infos.values():
        all_errors.extend(info["errors"])

    cross_errors, warnings = cross_check(prompt_info, label_infos)
    all_errors.extend(cross_errors)

    print("=== V2 Data Check Summary ===")
    print(f"Prompt file: {prompt_path}")
    print(f"Prompt records: {len(prompt_info['records'])}")
    print(f"Prompt unique ids: {len(prompt_info['prompts_by_id'])}")
    print(f"Prompt level distribution: {dict(sorted(prompt_info['level_counter'].items()))}")
    print()

    for name, info in label_infos.items():
        print(f"Label file: {name}")
        print(f"  Records: {len(info['records'])}")
        print(f"  Unique task_ids: {len(info['labels_by_id'])}")
        print(f"  Winner distribution: {dict(sorted(info['winner_counter'].items()))}")
        print()

    if warnings:
        print("=== Warnings ===")
        for warning in warnings:
            print(f"- {warning}")
        print()

    if all_errors:
        print("=== Errors ===")
        for error in all_errors[:200]:
            print(f"- {error}")
        if len(all_errors) > 200:
            print(f"... and {len(all_errors) - 200} more errors")
        raise SystemExit(1)

    print("All checks passed.")


if __name__ == "__main__":
    main()
