import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else float("nan")


def pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def infer_side(record_id: str) -> str:
    parts = record_id.split("::")
    return parts[1] if len(parts) >= 2 else "?"


def build_pair_view(records: List[Dict]) -> Dict[str, Dict[str, Dict]]:
    pairs: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    for record in records:
        side = infer_side(record["id"])
        pairs[record["pair_id"]][side] = record
    return pairs


def summarize_file(records: List[Dict], name: str) -> str:
    rows = len(records)
    pair_view = build_pair_view(records)
    pair_count = len(pair_view)

    dataset_counter = Counter(r.get("dataset", "unknown") for r in records)
    generator_counter = Counter(r.get("gen_image_model_name", "unknown") for r in records)
    pref_counter = Counter(r.get("predicted_preference", "unknown") for r in records)

    raw_scores = [float(r["preference_raw_score"]) for r in records]
    win_probs = [float(r["pairwise_win_prob"]) for r in records]
    existence_scores = [float(r["existence_score"]) for r in records]
    appearance_scores = [float(r["appearance_score"]) for r in records]
    interaction_scores = [float(r["interaction_score"]) for r in records]

    generator_win_counter = Counter()
    generator_margin_sum = Counter()
    valid_pairs = 0
    for pair_id, sides in pair_view.items():
        if "A" not in sides or "B" not in sides:
            continue
        valid_pairs += 1
        record_a = sides["A"]
        record_b = sides["B"]
        winner = record_a if float(record_a["pairwise_win_prob"]) >= 0.5 else record_b
        loser = record_b if winner is record_a else record_a
        generator_win_counter[winner["gen_image_model_name"]] += 1
        generator_margin_sum[winner["gen_image_model_name"]] += (
            float(winner["preference_raw_score"]) - float(loser["preference_raw_score"])
        )

    lines = [
        f"=== {name} ===",
        f"rows={rows} | pairs={pair_count} | valid_pairs={valid_pairs}",
        "datasets: " + ", ".join(f"{k}={v}" for k, v in sorted(dataset_counter.items())),
        "generators: " + ", ".join(f"{k}={v}" for k, v in sorted(generator_counter.items())),
        "predicted_preference: " + ", ".join(f"{k}={v}" for k, v in sorted(pref_counter.items())),
        (
            "score_stats: "
            f"raw_mean={mean(raw_scores):.4f}, raw_min={min(raw_scores):.4f}, raw_max={max(raw_scores):.4f}, "
            f"win_prob_mean={mean(win_probs):.4f}"
        ),
        (
            "category_means: "
            f"existence={mean(existence_scores):.4f}, "
            f"appearance={mean(appearance_scores):.4f}, "
            f"interaction={mean(interaction_scores):.4f}"
        ),
    ]

    if valid_pairs:
        win_parts = []
        for gen_name, wins in sorted(generator_win_counter.items()):
            avg_margin = generator_margin_sum[gen_name] / wins if wins else 0.0
            win_parts.append(
                f"{gen_name}: wins={wins} ({pct(wins, valid_pairs):.1f}%), avg_margin={avg_margin:.4f}"
            )
        lines.append("pair_winners: " + " | ".join(win_parts))
    return "\n".join(lines)


def extract_pair_winners(records: List[Dict]) -> Dict[str, Dict]:
    winners = {}
    for pair_id, sides in build_pair_view(records).items():
        if "A" not in sides or "B" not in sides:
            continue
        record_a = sides["A"]
        record_b = sides["B"]
        winner = record_a if float(record_a["pairwise_win_prob"]) >= 0.5 else record_b
        winners[pair_id] = {
            "winner_side": infer_side(winner["id"]),
            "winner_model": winner["gen_image_model_name"],
            "winner_prob": float(winner["pairwise_win_prob"]),
        }
    return winners


def compare_files(named_records: List[Tuple[str, List[Dict]]]) -> str:
    if len(named_records) < 2:
        return ""

    lines = ["=== Cross-File Comparison ==="]

    pair_winners_by_file = {
        name: extract_pair_winners(records)
        for name, records in named_records
    }

    names = [name for name, _ in named_records]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_i = names[i]
            name_j = names[j]
            winners_i = pair_winners_by_file[name_i]
            winners_j = pair_winners_by_file[name_j]
            common_pairs = sorted(set(winners_i) & set(winners_j))
            if not common_pairs:
                continue
            agree = 0
            same_winner_model = 0
            prob_deltas = []
            for pair_id in common_pairs:
                wi = winners_i[pair_id]
                wj = winners_j[pair_id]
                if wi["winner_side"] == wj["winner_side"]:
                    agree += 1
                if wi["winner_model"] == wj["winner_model"]:
                    same_winner_model += 1
                prob_deltas.append(abs(wi["winner_prob"] - wj["winner_prob"]))
            lines.append(
                (
                    f"{name_i} vs {name_j}: "
                    f"common_pairs={len(common_pairs)}, "
                    f"winner_side_agreement={pct(agree, len(common_pairs)):.2f}%, "
                    f"winner_model_agreement={pct(same_winner_model, len(common_pairs)):.2f}%, "
                    f"avg_conf_delta={mean(prob_deltas):.4f}"
                )
            )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze exported LENS metrics JSONL files.")
    parser.add_argument(
        "jsonl_files",
        nargs="+",
        help="Paths to one or more metrics JSONL files.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to dump a machine-readable summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    named_records: List[Tuple[str, List[Dict]]] = []
    summary_json = {}

    for file_name in args.jsonl_files:
        path = Path(file_name).resolve()
        records = load_jsonl(path)
        name = path.stem
        named_records.append((name, records))
        print(summarize_file(records, name))
        print()

        summary_json[name] = {
            "rows": len(records),
            "pairs": len(build_pair_view(records)),
        }

    comparison = compare_files(named_records)
    if comparison:
        print(comparison)

    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary_json, f, indent=2, ensure_ascii=False)
        print()
        print(f"Saved summary JSON to: {output_path}")


if __name__ == "__main__":
    main()
