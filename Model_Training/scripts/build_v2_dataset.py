import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from tqdm.auto import tqdm


def log(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [build_v2_dataset] {message}", flush=True)


def load_jsonl(path, desc):
    records = []
    log(f"Loading JSONL: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=desc, unit=" lines", mininterval=2.0):
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    log(f"Loaded {len(records)} records from {path}")
    return records


def build_subject_refs(people_names, object_names, refs_root, refs_prefix, fallback_exts):
    subject_refs = []
    missing_refs = []
    fallback_hits = 0

    for name in people_names + object_names:
        resolved = None
        for ext in fallback_exts:
            candidate = refs_root / f"{name}{ext}"
            if candidate.exists():
                resolved = ext
                break
        if resolved is None:
            missing_refs.append(name)
            continue
        if resolved != ".jpg":
            fallback_hits += 1
        subject_refs.append({"id": name, "image_path": f"{refs_prefix}/{name}{resolved}"})

    return subject_refs, missing_refs, fallback_hits


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

    log(
        f"Preparing grouped split across {len(seed_groups)} seed groups and "
        f"{len(bucket_to_seed_ids)} stratification buckets"
    )
    train_records, val_records, test_records = [], [], []

    for seed_ids in tqdm(
        bucket_to_seed_ids.values(),
        total=len(bucket_to_seed_ids),
        desc="Splitting buckets",
        unit=" bucket",
        mininterval=1.0,
    ):
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
    log("Starting V2 dataset build")
    log(f"Prompt path: {args.prompt_path}")
    log(f"Label paths: {', '.join(args.labels_paths)}")
    log(f"Image A root: {args.image_a_root}")
    log(f"Image B root: {args.image_b_root}")
    log(f"Refs root: {args.refs_root}")
    prompt_records = load_jsonl(args.prompt_path, desc="Loading prompts")
    image_a_root = Path(args.image_a_root)
    image_b_root = Path(args.image_b_root)
    refs_root = Path(args.refs_root)
    image_ext_fallbacks = [".png", ".jpg", ".jpeg", ".webp"]
    ref_ext_fallbacks = [".jpg", ".png", ".jpeg", ".webp"]

    labels_by_task = defaultdict(list)
    for label_path in args.labels_paths:
        label_rows = load_jsonl(label_path, desc=f"Loading {Path(label_path).name}")
        for row in label_rows:
            labels_by_task[str(row["task_id"])].append(row)
        log(f"Indexed labels from {label_path}; current unique task ids: {len(labels_by_task)}")

    output_records = []
    missing_labels = 0
    missing_preference = 0
    missing_images = 0
    missing_refs = 0
    fallback_hits_a = 0
    fallback_hits_b = 0
    fallback_hits_refs = 0

    progress = tqdm(prompt_records, desc="Building usable samples", unit=" sample", mininterval=1.0)
    for index, row in enumerate(progress, start=1):
        task_id = str(row["id"])
        labels = labels_by_task.get(task_id, [])
        if not labels:
            missing_labels += 1
        else:
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

            subject_refs, row_missing_refs, row_ref_fallbacks = build_subject_refs(
                row.get("people_names", []),
                row.get("object_names", []),
                refs_root,
                args.refs_prefix,
                ref_ext_fallbacks,
            )
            if row_missing_refs:
                missing_refs += 1
                continue
            fallback_hits_refs += row_ref_fallbacks

            item = {
                "task_id": task_id,
                "prompt": row.get("prompt_en") or row.get("prompt") or "",
                "prompt_zh": row.get("prompt_zh", ""),
                "subject_count": int(row.get("total_entities", 0)),
                "subject_refs": subject_refs,
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

        if index % 1000 == 0 or index == len(prompt_records):
            progress.set_postfix(
                kept=len(output_records),
                no_label=missing_labels,
                pref_drop=missing_preference,
                img_drop=missing_images,
                ref_drop=missing_refs,
            )

    log(f"Finished sample filtering. Usable records: {len(output_records)}")
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

    log(f"Writing train split to {train_path}")
    train_path.write_text(json.dumps(train_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"Writing val split to {val_path}")
    val_path.write_text(json.dumps(val_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"Writing test split to {test_path}")
    test_path.write_text(json.dumps(test_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log(f"Loaded prompts: {len(prompt_records)}")
    log(f"Built usable records: {len(output_records)}")
    log(f"Dropped for missing labels: {missing_labels}")
    log(f"Dropped for unresolved preference ties: {missing_preference}")
    log(f"Dropped for missing generated images: {missing_images}")
    log(f"Dropped for missing reference images: {missing_refs}")
    log(f"Image A fallback extension hits: {fallback_hits_a}")
    log(f"Image B fallback extension hits: {fallback_hits_b}")
    log(f"Reference fallback extension hits: {fallback_hits_refs}")
    log(f"Train split: {len(train_records)} -> {train_path}")
    log(f"Val split: {len(val_records)} -> {val_path}")
    log(f"Test split: {len(test_records)} -> {test_path}")
    log("Dataset build completed successfully")


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
        "--refs_root",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data_v2" / "refs"),
        help="Directory containing reference images",
    )
    parser.add_argument(
        "--image_a_root",
        type=str,
        default="/workspace/data/A",
        help="Root directory for model A generated images",
    )
    parser.add_argument(
        "--image_b_root",
        type=str,
        default="/workspace/data/B",
        help="Root directory for model B generated images",
    )
    parser.add_argument("--image_a_ext", type=str, default=".png", help="Extension for model A generated images")
    parser.add_argument("--image_b_ext", type=str, default=".jpg", help="Extension for model B generated images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for grouped splitting")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio by seed_id group")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation split ratio by seed_id group")
    args = parser.parse_args()
    main(args)
