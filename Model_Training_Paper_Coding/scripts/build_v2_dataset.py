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


def load_json(path):
    log(f"Loading JSON: {path}")
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    log(f"Loaded {len(records)} records from {path}")
    return records


def write_json(path, records, desc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Writing {desc} to {path}")
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_file_index(root, allowed_exts, desc):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root}")

    ext_priority = {ext.lower(): idx for idx, ext in enumerate(allowed_exts)}
    index = {}
    total_files = 0

    log(f"Scanning directory for {desc}: {root}")
    for path in tqdm(root.iterdir(), desc=desc, unit=" file", mininterval=1.0):
        if not path.is_file():
            continue
        total_files += 1
        ext = path.suffix.lower()
        if ext not in ext_priority:
            continue
        stem = path.stem
        existing = index.get(stem)
        if existing is None or ext_priority[ext] < ext_priority[existing[1].lower()]:
            index[stem] = (path, path.suffix)

    log(
        f"Indexed {len(index)} usable names from {root} "
        f"({total_files} filesystem entries scanned)"
    )
    return index


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


def build_subject_refs_from_index(people_names, object_names, refs_index, refs_prefix, preferred_ext):
    subject_refs = []
    missing_refs = []
    fallback_hits = 0

    for name in people_names + object_names:
        resolved = refs_index.get(name)
        if resolved is None:
            missing_refs.append(name)
            continue
        path, ext = resolved
        if ext.lower() != preferred_ext.lower():
            fallback_hits += 1
        subject_refs.append({"id": name, "image_path": f"{refs_prefix}/{path.name}"})

    return subject_refs, missing_refs, fallback_hits


def resolve_image_path_from_index(image_index, task_id):
    resolved = image_index.get(str(task_id))
    if resolved is None:
        return None, None
    path, ext = resolved
    return path, ext


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


def prepare_prefiltered_candidates(args):
    log("Preparing prefiltered candidates from prompt/label/ref data")
    log(f"Prompt path: {args.prompt_path}")
    log(f"Label paths: {', '.join(args.labels_paths)}")
    log(f"Refs root: {args.refs_root}")

    prompt_records = load_jsonl(args.prompt_path, desc="Loading prompts")
    refs_root = Path(args.refs_root)
    ref_ext_fallbacks = [".jpg", ".png", ".jpeg", ".webp"]
    refs_index = build_file_index(refs_root, ref_ext_fallbacks, desc="Indexing refs root")

    labels_by_task = defaultdict(list)
    for label_path in args.labels_paths:
        label_rows = load_jsonl(label_path, desc=f"Loading {Path(label_path).name}")
        for row in label_rows:
            labels_by_task[str(row["task_id"])].append(row)
        log(f"Indexed labels from {label_path}; current unique task ids: {len(labels_by_task)}")

    candidate_records = []
    missing_labels = 0
    missing_preference = 0
    missing_refs = 0
    fallback_hits_refs = 0

    progress = tqdm(prompt_records, desc="Preparing candidates", unit=" sample", mininterval=1.0)
    for index, row in enumerate(progress, start=1):
        task_id = str(row["id"])
        labels = labels_by_task.get(task_id, [])
        if not labels:
            missing_labels += 1
            continue

        aggregate = aggregate_labels(labels)
        if aggregate["preference"] is None:
            missing_preference += 1
            continue

        subject_refs, row_missing_refs, row_ref_fallbacks = build_subject_refs_from_index(
            row.get("people_names", []),
            row.get("object_names", []),
            refs_index,
            args.refs_prefix,
            preferred_ext=".jpg",
        )
        if row_missing_refs:
            missing_refs += 1
            continue
        fallback_hits_refs += row_ref_fallbacks

        candidate_records.append(
            {
                "task_id": task_id,
                "prompt": row.get("prompt_en") or row.get("prompt") or "",
                "prompt_zh": row.get("prompt_zh", ""),
                "subject_count": int(row.get("total_entities", 0)),
                "subject_refs": subject_refs,
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
        )

        if index % 1000 == 0 or index == len(prompt_records):
            progress.set_postfix(
                kept=len(candidate_records),
                no_label=missing_labels,
                pref_drop=missing_preference,
                ref_drop=missing_refs,
            )

    stats = {
        "loaded_prompts": len(prompt_records),
        "candidate_records": len(candidate_records),
        "missing_labels": missing_labels,
        "missing_preference": missing_preference,
        "missing_refs": missing_refs,
        "fallback_hits_refs": fallback_hits_refs,
    }
    log(
        "Prefilter stage summary | "
        f"loaded_prompts={stats['loaded_prompts']} candidates={stats['candidate_records']} "
        f"missing_labels={stats['missing_labels']} pref_drop={stats['missing_preference']} "
        f"ref_drop={stats['missing_refs']} ref_fallback_hits={stats['fallback_hits_refs']}"
    )
    return candidate_records, stats


def finalize_candidates_with_images(candidate_records, args):
    log("Finalizing candidate records with generated image checks and dataset split")
    log(f"Image A root: {args.image_a_root}")
    log(f"Image B root: {args.image_b_root}")

    image_a_root = Path(args.image_a_root)
    image_b_root = Path(args.image_b_root)
    image_ext_fallbacks = [".png", ".jpg", ".jpeg", ".webp"]
    image_a_ext_order = [args.image_a_ext] + [ext for ext in image_ext_fallbacks if ext != args.image_a_ext]
    image_b_ext_order = [args.image_b_ext] + [ext for ext in image_ext_fallbacks if ext != args.image_b_ext]

    image_a_index = build_file_index(image_a_root, image_a_ext_order, desc="Indexing image A root")
    image_b_index = build_file_index(image_b_root, image_b_ext_order, desc="Indexing image B root")

    output_records = []
    missing_images = 0
    fallback_hits_a = 0
    fallback_hits_b = 0

    progress = tqdm(candidate_records, desc="Finalizing usable samples", unit=" sample", mininterval=1.0)
    for index, item in enumerate(progress, start=1):
        task_id = str(item["task_id"])
        image_a_path, ext_a = resolve_image_path_from_index(image_a_index, task_id)
        image_b_path, ext_b = resolve_image_path_from_index(image_b_index, task_id)
        if image_a_path is None or image_b_path is None:
            missing_images += 1
            continue
        if ext_a.lower() != args.image_a_ext.lower():
            fallback_hits_a += 1
        if ext_b.lower() != args.image_b_ext.lower():
            fallback_hits_b += 1

        finalized_item = dict(item)
        finalized_item["image_A_path"] = str(image_a_path)
        finalized_item["image_B_path"] = str(image_b_path)
        output_records.append(finalized_item)

        if index % 1000 == 0 or index == len(candidate_records):
            progress.set_postfix(
                kept=len(output_records),
                img_drop=missing_images,
                a_fallback=fallback_hits_a,
                b_fallback=fallback_hits_b,
            )

    log(f"Finished image validation. Usable records after image check: {len(output_records)}")
    train_records, val_records, test_records = stratified_group_split(
        output_records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    stats = {
        "candidate_records": len(candidate_records),
        "usable_records": len(output_records),
        "missing_images": missing_images,
        "fallback_hits_a": fallback_hits_a,
        "fallback_hits_b": fallback_hits_b,
        "train_records": len(train_records),
        "val_records": len(val_records),
        "test_records": len(test_records),
    }
    log(
        "Finalize stage summary | "
        f"candidates={stats['candidate_records']} usable={stats['usable_records']} "
        f"missing_images={stats['missing_images']} image_a_fallback_hits={stats['fallback_hits_a']} "
        f"image_b_fallback_hits={stats['fallback_hits_b']}"
    )
    return train_records, val_records, test_records, stats


def main(args):
    log("Starting V2 dataset build")
    if args.prepare_candidates_only and not args.prefilter_cache_path:
        raise ValueError("--prepare_candidates_only requires --prefilter_cache_path")

    prefilter_stats = None
    if args.prefilter_cache_path and args.reuse_prefilter_cache and Path(args.prefilter_cache_path).exists():
        log(f"Reusing existing prefilter cache: {args.prefilter_cache_path}")
        candidate_records = load_json(args.prefilter_cache_path)
    else:
        candidate_records, prefilter_stats = prepare_prefiltered_candidates(args)
        if args.prefilter_cache_path:
            write_json(args.prefilter_cache_path, candidate_records, desc="prefilter candidate cache")

    if args.prepare_candidates_only:
        log("Prepare-only mode enabled; skipping generated image checks and split writing")
        return

    train_records, val_records, test_records, finalize_stats = finalize_candidates_with_images(candidate_records, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train_v2.json"
    val_path = output_dir / "val_v2.json"
    test_path = output_dir / "test_v2.json"

    write_json(train_path, train_records, desc="train split")
    write_json(val_path, val_records, desc="val split")
    write_json(test_path, test_records, desc="test split")

    if prefilter_stats is not None:
        log(f"Loaded prompts: {prefilter_stats['loaded_prompts']}")
        log(f"Prefilter candidates: {prefilter_stats['candidate_records']}")
        log(f"Dropped for missing labels: {prefilter_stats['missing_labels']}")
        log(f"Dropped for unresolved preference ties: {prefilter_stats['missing_preference']}")
        log(f"Dropped for missing reference images: {prefilter_stats['missing_refs']}")
        log(f"Reference fallback extension hits: {prefilter_stats['fallback_hits_refs']}")
    else:
        log(f"Prefilter candidates loaded from cache: {len(candidate_records)}")

    log(f"Built usable records: {finalize_stats['usable_records']}")
    log(f"Dropped for missing generated images: {finalize_stats['missing_images']}")
    log(f"Image A fallback extension hits: {finalize_stats['fallback_hits_a']}")
    log(f"Image B fallback extension hits: {finalize_stats['fallback_hits_b']}")
    log(f"Train split: {finalize_stats['train_records']} -> {train_path}")
    log(f"Val split: {finalize_stats['val_records']} -> {val_path}")
    log(f"Test split: {finalize_stats['test_records']} -> {test_path}")
    log("Dataset build completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build MIB V2 train/val/test manifests.")
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
    parser.add_argument(
        "--prefilter_cache_path",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data_v2" / "v2_prefilter_candidates.json"),
        help="Reusable cache file for prompt/label/ref-prefiltered candidate records",
    )
    parser.add_argument(
        "--reuse_prefilter_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing prefilter cache when available",
    )
    parser.add_argument(
        "--prepare_candidates_only",
        action="store_true",
        help="Only build and save the reusable prefilter candidate cache; skip image checks and final split",
    )
    args = parser.parse_args()
    main(args)
