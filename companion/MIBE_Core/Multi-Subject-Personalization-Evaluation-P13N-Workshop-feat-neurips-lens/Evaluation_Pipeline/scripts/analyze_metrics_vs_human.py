import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


CATEGORY_NAMES = ("existence", "appearance", "interaction")

MODEL_NAME_MAP = {
    "mosaic_v10": "mosaic",
    "nano_banana_v10": "nano_banana",
    "flux2_klein_9b_kv": "flux",
    "gpt_image_1_5": "gpt-image-1.5",
    "seedream45": "seedream4.5",
    "seedream4.5": "seedream4.5",
    "seedream_4_5": "seedream4.5",
    "glm": "glm",
    "flux": "flux",
    "mosaic": "mosaic",
    "nano_banana": "nano_banana",
}


def normalize_model_name(name: str) -> str:
    key = (name or "").strip()
    return MODEL_NAME_MAP.get(key, key)


def canonical_pair_key(dataset: str, base_id: str, model_a: str, model_b: str) -> Tuple[str, str, Tuple[str, str]]:
    return dataset, str(base_id), tuple(sorted((normalize_model_name(model_a), normalize_model_name(model_b))))


def parse_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_csv(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def infer_dataset_from_pair_id(pair_id: str) -> str:
    return "v10" if pair_id.startswith("v10_") else "v13"


def infer_base_id_from_pair_id(pair_id: str) -> str:
    return pair_id.rsplit("_", 1)[-1]


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def threshold_label(score: float, threshold: float = 0.5) -> int:
    return 1 if score >= threshold else 0


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "support": tp + tn + fp + fn,
    }


def parse_size_and_mode(metrics_model_name: str) -> Tuple[str, str]:
    match = re.search(r"qwen35_(08b|2b|4b)_(layer_only|lora_layer)", metrics_model_name)
    if not match:
        return "unknown", "unknown"
    return match.group(1), match.group(2)


def build_human_reference(v10_csv: Path, v13_csv: Path) -> Dict:
    human_rows = []
    for row in parse_csv(v10_csv):
        row["_dataset"] = "v10"
        row["_base_id"] = str(row["combo_id"])
        human_rows.append(row)
    for row in parse_csv(v13_csv):
        row["_dataset"] = "v13"
        row["_base_id"] = str(row["base_id"])
        human_rows.append(row)

    grouped: Dict[Tuple[str, str, Tuple[str, str]], List[Dict]] = defaultdict(list)
    for row in human_rows:
        key = canonical_pair_key(
            row["_dataset"],
            row["_base_id"],
            row["model_a"],
            row["model_b"],
        )
        grouped[key].append(row)

    kept_pairs = {}
    dropped_preference_inconsistent = 0
    category_ambiguous_counter = Counter()

    for key, rows in grouped.items():
        winner_models = set()
        category_votes = {
            model_name: {category: [] for category in CATEGORY_NAMES}
            for model_name in key[2]
        }

        for row in rows:
            model_a = normalize_model_name(row["model_a"])
            model_b = normalize_model_name(row["model_b"])
            preference = row["preference"].strip().upper()
            if preference == "A":
                winner_models.add(model_a)
            elif preference == "B":
                winner_models.add(model_b)

            for model_name, prefix in ((model_a, "a"), (model_b, "b")):
                for category in CATEGORY_NAMES:
                    value = row[f"{prefix}_{category}"]
                    category_votes[model_name][category].append(int(value))

        if len(winner_models) != 1:
            dropped_preference_inconsistent += 1
            continue

        human_winner_model = next(iter(winner_models))
        resolved_categories = {}
        for model_name, votes_by_category in category_votes.items():
            resolved_categories[model_name] = {}
            for category, votes in votes_by_category.items():
                positives = sum(votes)
                negatives = len(votes) - positives
                if positives == negatives:
                    resolved_categories[model_name][category] = None
                    category_ambiguous_counter[category] += 1
                else:
                    resolved_categories[model_name][category] = 1 if positives > negatives else 0

        kept_pairs[key] = {
            "human_winner_model": human_winner_model,
            "rows": rows,
            "categories": resolved_categories,
            "dataset": key[0],
            "base_id": key[1],
        }

    return {
        "pairs": kept_pairs,
        "stats": {
            "total_raw_groups": len(grouped),
            "kept_pairs": len(kept_pairs),
            "dropped_preference_inconsistent": dropped_preference_inconsistent,
            "category_ambiguous": dict(category_ambiguous_counter),
        },
    }


def evaluate_metrics_file(metrics_path: Path, human_ref: Dict) -> Dict:
    records = parse_jsonl(metrics_path)
    image_index: Dict[Tuple[str, str, str], Dict] = {}
    metrics_pair_keys_seen = set()
    for record in records:
        dataset = infer_dataset_from_pair_id(record["pair_id"])
        base_id = infer_base_id_from_pair_id(record["pair_id"])
        model_name = normalize_model_name(record["gen_image_model_name"])
        image_index[(dataset, base_id, model_name)] = record
        metrics_pair_keys_seen.add(record["pair_id"])

    pairwise_matches = []
    pairwise_matches_by_dataset = defaultdict(list)
    correct_side_confidence = []
    matched_pairs_by_dataset = Counter()

    category_true = {category: [] for category in CATEGORY_NAMES}
    category_pred = {category: [] for category in CATEGORY_NAMES}
    category_by_dataset = {
        dataset: {category: {"y_true": [], "y_pred": []} for category in CATEGORY_NAMES}
        for dataset in ("v10", "v13")
    }

    evaluated_pairs = 0
    missing_human_pairs = 0  # kept for backward compatibility of output schema
    missing_metrics_pairs = 0

    for pair_key, human_pair in human_ref["pairs"].items():
        dataset, base_id, model_pair = pair_key
        model_a, model_b = model_pair
        record_a = image_index.get((dataset, base_id, model_a))
        record_b = image_index.get((dataset, base_id, model_b))
        if record_a is None or record_b is None:
            missing_metrics_pairs += 1
            continue

        evaluated_pairs += 1
        matched_pairs_by_dataset[dataset] += 1
        predicted_winner = record_a if float(record_a["preference_raw_score"]) >= float(record_b["preference_raw_score"]) else record_b
        human_winner_model = human_pair["human_winner_model"]
        is_correct = predicted_winner["gen_image_model_name"] == human_winner_model
        pairwise_matches.append(1 if is_correct else 0)
        pairwise_matches_by_dataset[dataset].append(1 if is_correct else 0)

        correct_side_record = (
            record_a if record_a["gen_image_model_name"] == human_winner_model else record_b
        )
        other_record = record_b if correct_side_record is record_a else record_a
        correct_side_confidence.append(
            sigmoid(float(correct_side_record["preference_raw_score"]) - float(other_record["preference_raw_score"]))
        )

        for record in (record_a, record_b):
            model_name = normalize_model_name(record["gen_image_model_name"])
            human_categories = human_pair["categories"].get(model_name, {})
            for category in CATEGORY_NAMES:
                human_label = human_categories.get(category)
                if human_label is None:
                    continue
                category_true[category].append(human_label)
                pred_label = threshold_label(float(record[f"{category}_score"]))
                category_pred[category].append(pred_label)
                category_by_dataset[dataset][category]["y_true"].append(human_label)
                category_by_dataset[dataset][category]["y_pred"].append(pred_label)

    metrics_model_name = records[0]["metrics_model_name"] if records else metrics_path.stem
    overall_pairwise_acc = mean(pairwise_matches) if pairwise_matches else float("nan")
    overall_pairwise_conf = mean(correct_side_confidence) if correct_side_confidence else float("nan")

    category_metrics = {
        category: binary_metrics(category_true[category], category_pred[category])
        for category in CATEGORY_NAMES
    }
    macro_f1 = mean([category_metrics[c]["f1"] for c in CATEGORY_NAMES])
    macro_precision = mean([category_metrics[c]["precision"] for c in CATEGORY_NAMES])
    macro_accuracy = mean([category_metrics[c]["accuracy"] for c in CATEGORY_NAMES])

    return {
        "metrics_model_name": metrics_model_name,
        "size": parse_size_and_mode(metrics_model_name)[0],
        "mode": parse_size_and_mode(metrics_model_name)[1],
        "rows": len(records),
        "evaluated_pairs": evaluated_pairs,
        "missing_human_pairs": missing_human_pairs,
        "missing_metrics_pairs": missing_metrics_pairs,
        "pairwise_accuracy": overall_pairwise_acc,
        "pairwise_accuracy_v10": mean(pairwise_matches_by_dataset["v10"]) if pairwise_matches_by_dataset["v10"] else float("nan"),
        "pairwise_accuracy_v13": mean(pairwise_matches_by_dataset["v13"]) if pairwise_matches_by_dataset["v13"] else float("nan"),
        "correct_side_win_prob_mean": overall_pairwise_conf,
        "matched_pairs_by_dataset": dict(matched_pairs_by_dataset),
        "macro_precision": macro_precision,
        "macro_f1": macro_f1,
        "macro_accuracy": macro_accuracy,
        "category_metrics": category_metrics,
        "category_by_dataset": {
            dataset: {
                category: binary_metrics(values["y_true"], values["y_pred"])
                for category, values in per_category.items()
            }
            for dataset, per_category in category_by_dataset.items()
        },
    }


def infer_side(record_id: str) -> str:
    parts = record_id.split("::")
    return parts[1] if len(parts) >= 2 else "?"


def summarize_results(results: List[Dict], human_stats: Dict) -> str:
    lines = [
        "=== Human Label Filtering ===",
        (
            f"raw_groups={human_stats['total_raw_groups']} | "
            f"kept_pairs={human_stats['kept_pairs']} | "
            f"dropped_preference_inconsistent={human_stats['dropped_preference_inconsistent']}"
        ),
        "category_ambiguous_drops: " + ", ".join(
            f"{k}={v}" for k, v in sorted(human_stats["category_ambiguous"].items())
        ),
        "",
        "=== Metrics vs Human ===",
    ]

    ordered = sorted(results, key=lambda x: (-x["pairwise_accuracy"], -x["macro_f1"], x["metrics_model_name"]))
    for result in ordered:
        match_info = ", ".join(f"{k}={v}" for k, v in sorted(result["matched_pairs_by_dataset"].items()))
        lines.extend(
            [
                (
                    f"{result['metrics_model_name']}: "
                    f"pairs={result['evaluated_pairs']}, "
                    f"matched_by_dataset=[{match_info}], "
                    f"pair_acc={result['pairwise_accuracy']:.4f}, "
                    f"v10_acc={result['pairwise_accuracy_v10']:.4f}, "
                    f"v13_acc={result['pairwise_accuracy_v13']:.4f}, "
                    f"macro_precision={result['macro_precision']:.4f}, "
                    f"macro_f1={result['macro_f1']:.4f}, "
                    f"macro_acc={result['macro_accuracy']:.4f}, "
                    f"mean_correct_win_prob={result['correct_side_win_prob_mean']:.4f}"
                ),
                (
                    "  categories: "
                    + " | ".join(
                        f"{category}: P={result['category_metrics'][category]['precision']:.4f}, "
                        f"R={result['category_metrics'][category]['recall']:.4f}, "
                        f"F1={result['category_metrics'][category]['f1']:.4f}, "
                        f"Acc={result['category_metrics'][category]['accuracy']:.4f}"
                        for category in CATEGORY_NAMES
                    )
                ),
            ]
        )

    lines.append("")
    lines.append("=== Scale / Mode Comparison ===")

    by_size = defaultdict(dict)
    for result in results:
        by_size[result["size"]][result["mode"]] = result

    for size, entries in sorted(by_size.items()):
        if "layer_only" in entries and "lora_layer" in entries:
            layer = entries["layer_only"]
            lora = entries["lora_layer"]
            lines.append(
                (
                    f"{size}: lora-layer vs layer-only | "
                    f"pair_acc_delta={lora['pairwise_accuracy'] - layer['pairwise_accuracy']:+.4f}, "
                    f"macro_f1_delta={lora['macro_f1'] - layer['macro_f1']:+.4f}"
                )
            )

    best_by_mode = defaultdict(list)
    for result in results:
        best_by_mode[result["mode"]].append(result)
    for mode, entries in sorted(best_by_mode.items()):
        size_order = {"08b": 0, "2b": 1, "4b": 2}
        entries = sorted(entries, key=lambda x: size_order.get(x["size"], 999))
        if len(entries) >= 2:
            for earlier, later in zip(entries, entries[1:]):
                lines.append(
                    (
                        f"{mode}: {earlier['size']} -> {later['size']} | "
                        f"pair_acc_delta={later['pairwise_accuracy'] - earlier['pairwise_accuracy']:+.4f}, "
                        f"macro_f1_delta={later['macro_f1'] - earlier['macro_f1']:+.4f}"
                    )
                )

    if all((result["matched_pairs_by_dataset"].get("v13", 0) == 0) for result in results):
        lines.append("")
        lines.append("=== Important Note ===")
        lines.append(
            "No v13 human-aligned pairs were matched. The current exported v13 metrics pairs do not use the same model pairing schema as the human annotation CSV "
            "(human CSV pairs are flux vs gpt-image-1.5 and glm vs seedream4.5, while current metrics export uses glm vs flux and gpt-image-1.5 vs seedream4.5)."
        )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze metrics JSONL files against human annotations.")
    parser.add_argument(
        "--v10_csv",
        type=str,
        required=True,
        help="Path to round10 human annotation CSV.",
    )
    parser.add_argument(
        "--v13_csv",
        type=str,
        required=True,
        help="Path to v13 human annotation CSV.",
    )
    parser.add_argument(
        "metrics_jsonl_files",
        nargs="+",
        help="One or more exported metrics JSONL files to evaluate.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save machine-readable evaluation results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    human_ref = build_human_reference(Path(args.v10_csv).resolve(), Path(args.v13_csv).resolve())
    results = [evaluate_metrics_file(Path(path).resolve(), human_ref) for path in args.metrics_jsonl_files]
    print(summarize_results(results, human_ref["stats"]))

    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "human_stats": human_ref["stats"],
                    "results": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print()
        print(f"Saved evaluation JSON to: {output_path}")


if __name__ == "__main__":
    main()
