import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_subject_refs(people_names, object_names, refs_prefix):
    subject_refs = []
    for name in people_names:
        subject_refs.append({"id": name, "image_path": f"{refs_prefix}/{name}.jpg"})
    for name in object_names:
        subject_refs.append({"id": name, "image_path": f"{refs_prefix}/{name}.jpg"})
    return subject_refs


def resolve_image_path(image_root, task_id, preferred_ext, fallback_exts):
    candidates = [preferred_ext] + [ext for ext in fallback_exts if ext != preferred_ext]
    for ext in candidates:
        path = image_root / f"{task_id}{ext}"
        if path.exists():
            return path, ext
    return None, None


def aggregate_labels(labels):
    count = len(labels)
    avg_a = {
        "existence": sum(float(x["a_existence"]) for x in labels) / count,
        "appearance": sum(float(x["a_appearance"]) for x in labels) / count,
        "interaction": sum(float(x["a_interaction"]) for x in labels) / count,
    }
    avg_b = {
        "existence": sum(float(x["b_existence"]) for x in labels) / count,
        "appearance": sum(float(x["b_appearance"]) for x in labels) / count,
        "interaction": sum(float(x["b_interaction"]) for x in labels) / count,
    }
    winners = {x.get("winner") for x in labels}
    winners.discard(None)
    preference = next(iter(winners)) if len(winners) == 1 else None

    return {
        "preference": preference,
        "category_scores_A": avg_a,
        "category_scores_B": avg_b,
        "label_sources": [
            {
                "provider": x.get("provider"),
                "model": x.get("model"),
                "winner": x.get("winner"),
            }
            for x in labels
        ],
    }


def stratified_group_split(records, train_ratio, val_ratio, seed):
    rng = random.Random(seed)
    seed_groups = defaultdict(list)
    seed_buckets = {}

    for item in records:
        seed_id = item["metadata"].get("seed_id", item["task_id"])
        seed_groups[seed_id].append(item)
        seed_buckets.setdefault(
            seed_id,
            (
                item["metadata"].get("ratio_type", "unknown"),
                item["metadata"].get("class_tag", "unknown"),
            ),
        )

    bucket_to_seed_ids = defaultdict(list)
    for seed_id, bucket in seed_buckets.items():
        bucket_to_seed_ids[bucket].append(seed_id)

    train_records, val_records, test_records = [], [], []

    for seed_ids in bucket_to_seed_ids.values():
        rng.shuffle(seed_ids)
        n = len(seed_ids)
        train_n = int(n * train_ratio)
        val_n = int(n * val_ratio)

        train_ids = seed_ids[:train_n]
        val_ids = seed_ids[train_n : train_n + val_n]
        test_ids = seed_ids[train_n + val_n :]

        for sid in train_ids:
            train_records.extend(seed_groups[sid])
        for sid in val_ids:
            val_records.extend(seed_groups[sid])
        for sid in test_ids:
            test_records.extend(seed_groups[sid])

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)
    return train_records, val_records, test_records


def main(args):
    prompt_records = load_jsonl(args.prompt_path)
    image_a_root = Path(args.image_a_root)
    image_b_root = Path(args.image_b_root)
    image_ext_fallbacks = [".png", ".jpg", ".jpeg", ".webp"]

    labels_by_task = defaultdict(list)
    for label_path in args.labels_paths:
        for row in load_jsonl(label_path):
            labels_by_task[str(row["task_id"])].append(row)

    output_records = []
    missing_labels = 0
    missing_preference = 0
    missing_images = 0
    fallback_hits_a = 0
    fallback_hits_b = 0

    for row in prompt_records:
        task_id = str(row["id"])
        labels = labels_by_task.get(task_id, [])
        if not labels:
            missing_labels += 1
            continue

        aggregate = aggregate_labels(labels)
        if aggregate["preference"] is None:
            missing_preference += 1
            continue

        image_a_path, ext_a = resolve_image_path(image_a_root, task_id, args.image_a_ext, image_ext_fallbacks)
        image_b_path, ext_b = resolve_image_path(image_b_root, task_id, args.image_b_ext, image_ext_fallbacks)
        if image_a_path is None or image_b_path is None:
            missing_images += 1
            continue
        if ext_a != args.image_a_ext:
            fallback_hits_a += 1
        if ext_b != args.image_b_ext:
            fallback_hits_b += 1

        item = {
            "task_id": task_id,
            "prompt": row.get("prompt_en") or row.get("prompt") or "",
            "prompt_zh": row.get("prompt_zh", ""),
            "subject_count": int(row.get("total_entities", 0)),
            "subject_refs": build_subject_refs(
                row.get("people_names", []),
                row.get("object_names", []),
                args.refs_prefix,
            ),
            "image_A_path": str(image_a_path),
            "image_B_path": str(image_b_path),
            "preference": aggregate["preference"],
            "category_scores_A": aggregate["category_scores_A"],
            "category_scores_B": aggregate["category_scores_B"],
            "metadata": {
                "source": "V2 synthetic + VLM consensus",
                "seed_id": row.get("seed_id"),
                "level": row.get("level"),
                "class_tag": row.get("class_tag"),
                "ratio_type": row.get("ratio_type"),
                "n_humans": row.get("n_humans"),
                "n_objects": row.get("n_objects"),
                "people_names": row.get("people_names", []),
                "object_names": row.get("object_names", []),
                "token_len_est": row.get("token_len_est"),
                "label_sources": aggregate["label_sources"],
                "num_label_votes": len(labels),
            },
        }
        output_records.append(item)

    train_records, val_records, test_records = stratified_group_split(
        output_records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train_v2.json"
    val_path = output_dir / "val_v2.json"
    test_path = output_dir / "test_v2.json"

    train_path.write_text(json.dumps(train_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    val_path.write_text(json.dumps(val_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    test_path.write_text(json.dumps(test_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Loaded prompts: {len(prompt_records)}")
    print(f"Built usable records: {len(output_records)}")
    print(f"Dropped for missing labels: {missing_labels}")
    print(f"Dropped for unresolved preference ties: {missing_preference}")
    print(f"Dropped for missing generated images: {missing_images}")
    print(f"Image A fallback extension hits: {fallback_hits_a}")
    print(f"Image B fallback extension hits: {fallback_hits_b}")
    print(f"Train split: {len(train_records)} -> {train_path}")
    print(f"Val split: {len(val_records)} -> {val_path}")
    print(f"Test split: {len(test_records)} -> {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build PrismBench V2 train/val/test manifests.")
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent / "data_v2" / "prompt" / "train_60k_v13_2.jsonl"
        ),
        help="Path to the v2 prompt jsonl",
    )
    parser.add_argument(
        "--labels_paths",
        nargs="+",
        default=[
            str(Path(__file__).resolve().parent.parent / "data_v2" / "60k_LLM_Result" / "2_5_merged_sorted.jsonl"),
            str(Path(__file__).resolve().parent.parent / "data_v2" / "60k_LLM_Result" / "3_1_merged_sorted.jsonl"),
        ],
        help="One or more label jsonl files to aggregate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data_v2"),
        help="Directory where train_v2.json / val_v2.json / test_v2.json will be written",
    )
    parser.add_argument(
        "--refs_prefix",
        type=str,
        default="./data_v2/refs",
        help="Relative prefix stored inside subject_refs image paths",
    )
    parser.add_argument(
        "--image_a_root",
        type=str,
        default="/root/data/A",
        help="Root directory for model A generated images",
    )
    parser.add_argument(
        "--image_b_root",
        type=str,
        default="/root/data/B",
        help="Root directory for model B generated images",
    )
    parser.add_argument("--image_a_ext", type=str, default=".png", help="Extension for model A generated images")
    parser.add_argument("--image_b_ext", type=str, default=".jpg", help="Extension for model B generated images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for grouped splitting")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio by seed_id group")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation split ratio by seed_id group")
    args = parser.parse_args()
    main(args)
