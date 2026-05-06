import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PAPER_DATA_ROOT = ROOT / "paper_data"

SILVER_25_PATH = REPO_ROOT / "Model_Training" / "data_v2" / "60k_LLM_Result" / "2_5_merged_sorted.jsonl"
SILVER_31_PATH = REPO_ROOT / "Model_Training" / "data_v2" / "60k_LLM_Result" / "3_1_merged_sorted.jsonl"

V10_HUMAN_PATH = ROOT / "data" / "human_annotations" / "round10_annotations_latest.csv"
V13_HUMAN_PATH = ROOT / "data" / "human_annotations" / "export_round1260_20260503.csv"

METRICS_JSONL_DIR = ROOT / "outputs" / "jsonl"
METRICS_SUMMARY_PATH = ROOT / "outputs" / "summaries" / "metrics_vs_human_summary.json"

BASELINE_ANALYSIS_DIR = (
    REPO_ROOT
    / "multisubject_generation_eval"
    / "data"
    / "v13_2_1.26k_evl"
    / "eval_results"
    / "analysis"
)
BASELINE_EVAL_RESULTS_DIR = (
    REPO_ROOT
    / "multisubject_generation_eval"
    / "data"
    / "v13_2_1.26k_evl"
    / "eval_results"
)

CATEGORY_NAMES = ("existence", "appearance", "interaction")
LABEL_KEYS = (
    "a_existence",
    "a_appearance",
    "a_interaction",
    "b_existence",
    "b_appearance",
    "b_interaction",
)

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


def read_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_csv_rows(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalize_model_name(name: str) -> str:
    key = (name or "").strip()
    return MODEL_NAME_MAP.get(key, key)


def canonical_pair_key(dataset: str, base_id: str, model_a: str, model_b: str) -> Tuple[str, str, Tuple[str, str]]:
    return dataset, str(base_id), tuple(sorted((normalize_model_name(model_a), normalize_model_name(model_b))))


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_group_agreement(rows: List[Dict], group_key: str) -> List[Dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)

    summary_rows = []
    for value, items in sorted(groups.items(), key=lambda item: item[0]):
        pref_agree = sum(1 for item in items if item["preference_agree"])
        diag_all_agree = sum(1 for item in items if item["all_diag_agree"])
        out = {
            group_key: value,
            "matched_tasks": len(items),
            "preference_agreement_rate": safe_ratio(pref_agree, len(items)),
            "all_diagnostic_agreement_rate": safe_ratio(diag_all_agree, len(items)),
        }
        for label_key in LABEL_KEYS:
            out[f"{label_key}_agreement_rate"] = safe_ratio(
                sum(1 for item in items if item["label_agreement"][label_key]),
                len(items),
            )
        summary_rows.append(out)
    return summary_rows


def build_section_4_1() -> None:
    out_dir = PAPER_DATA_ROOT / "section_4_1_mib_silver"
    records_25 = {record["task_id"]: record for record in read_jsonl(SILVER_25_PATH)}
    records_31 = {record["task_id"]: record for record in read_jsonl(SILVER_31_PATH)}

    task_ids_25 = set(records_25)
    task_ids_31 = set(records_31)
    common_ids = sorted(task_ids_25 & task_ids_31, key=lambda x: int(x))

    joined_rows = []
    for task_id in common_ids:
        rec_25 = records_25[task_id]
        rec_31 = records_31[task_id]
        meta_25 = rec_25.get("metadata", {})
        label_agreement = {key: rec_25.get(key) == rec_31.get(key) for key in LABEL_KEYS}
        joined_rows.append(
            {
                "task_id": task_id,
                "subject_count": rec_25.get("subject_count"),
                "level": meta_25.get("level"),
                "class_tag": meta_25.get("class_tag"),
                "ratio_type": meta_25.get("ratio_type"),
                "winner_25": rec_25.get("winner"),
                "winner_31": rec_31.get("winner"),
                "preference_agree": rec_25.get("winner") == rec_31.get("winner"),
                "label_agreement": label_agreement,
                "all_diag_agree": all(label_agreement.values()),
            }
        )

    pref_agree = sum(1 for row in joined_rows if row["preference_agree"])
    diag_all_agree = sum(1 for row in joined_rows if row["all_diag_agree"])
    summary = {
        "source_files": [str(SILVER_25_PATH), str(SILVER_31_PATH)],
        "record_counts": {
            "gemini_2_5": len(records_25),
            "gemini_3_1": len(records_31),
            "matched_tasks": len(common_ids),
            "only_in_2_5": len(task_ids_25 - task_ids_31),
            "only_in_3_1": len(task_ids_31 - task_ids_25),
        },
        "overall_agreement": {
            "preference_agreement_rate": safe_ratio(pref_agree, len(common_ids)),
            "all_diagnostic_agreement_rate": safe_ratio(diag_all_agree, len(common_ids)),
            **{
                f"{label_key}_agreement_rate": safe_ratio(
                    sum(1 for row in joined_rows if row["label_agreement"][label_key]),
                    len(common_ids),
                )
                for label_key in LABEL_KEYS
            },
        },
    }

    write_jsonl(out_dir / "silver_joined_llm_annotations.jsonl", joined_rows)
    write_json(out_dir / "silver_agreement_summary.json", summary)
    for group_key in ("subject_count", "level", "class_tag", "ratio_type"):
        rows = summarize_group_agreement(joined_rows, group_key)
        fieldnames = [group_key, "matched_tasks", "preference_agreement_rate", "all_diagnostic_agreement_rate"] + [
            f"{label_key}_agreement_rate" for label_key in LABEL_KEYS
        ]
        write_csv(out_dir / f"silver_agreement_by_{group_key}.csv", rows, fieldnames)
        write_jsonl(out_dir / f"silver_agreement_by_{group_key}.jsonl", rows)


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def build_section_4_2() -> None:
    out_dir = PAPER_DATA_ROOT / "section_4_2_mib_gold"
    human_rows = []
    for row in read_csv_rows(V10_HUMAN_PATH):
        row["_dataset"] = "v10"
        row["_base_id"] = str(row["combo_id"])
        human_rows.append(row)
    for row in read_csv_rows(V13_HUMAN_PATH):
        row["_dataset"] = "v13"
        row["_base_id"] = str(row["base_id"])
        human_rows.append(row)

    grouped = defaultdict(list)
    for row in human_rows:
        key = canonical_pair_key(row["_dataset"], row["_base_id"], row["model_a"], row["model_b"])
        grouped[key].append(row)

    group_rows = []
    dataset_summary = Counter()
    kept_pairs_by_dataset = Counter()
    dropped_pairs_by_dataset = Counter()
    category_ambiguous = Counter()
    for key, rows in grouped.items():
        dataset, base_id, model_pair = key
        first = rows[0]
        winner_models = set()
        prompt_ilogical_true = sum(1 for row in rows if parse_bool(row.get("prompt_ilogical", "False")))
        category_votes = {
            model_name: {category: [] for category in CATEGORY_NAMES}
            for model_name in model_pair
        }
        for row in rows:
            model_a = normalize_model_name(row["model_a"])
            model_b = normalize_model_name(row["model_b"])
            pref = row["preference"].strip().upper()
            if pref == "A":
                winner_models.add(model_a)
            elif pref == "B":
                winner_models.add(model_b)
            for model_name, prefix in ((model_a, "a"), (model_b, "b")):
                for category in CATEGORY_NAMES:
                    category_votes[model_name][category].append(int(row[f"{prefix}_{category}"]))

        preference_consistent = len(winner_models) == 1
        if preference_consistent:
            kept_pairs_by_dataset[dataset] += 1
        else:
            dropped_pairs_by_dataset[dataset] += 1

        categories = {}
        for model_name, per_category in category_votes.items():
            categories[model_name] = {}
            for category, votes in per_category.items():
                positives = sum(votes)
                negatives = len(votes) - positives
                majority = None if positives == negatives else (1 if positives > negatives else 0)
                if majority is None:
                    category_ambiguous[category] += 1
                categories[model_name][category] = {
                    "mean": positives / len(votes) if votes else 0.0,
                    "majority": majority,
                    "votes": votes,
                }

        group_rows.append(
            {
                "dataset": dataset,
                "base_id": base_id,
                "pair_models": list(model_pair),
                "annotation_count": len(rows),
                "level": int(first["level"]),
                "class_tag": first["class_tag"],
                "ratio_type": first["ratio_type"],
                "prompt_ilogical_true_count": prompt_ilogical_true,
                "prompt_ilogical_any": prompt_ilogical_true > 0,
                "preference_consistent": preference_consistent,
                "human_winner_model": next(iter(winner_models)) if preference_consistent else None,
                "categories": categories,
            }
        )
        dataset_summary[dataset] += 1

    summary = {
        "source_files": [str(V10_HUMAN_PATH), str(V13_HUMAN_PATH)],
        "group_counts": {
            "total_groups": len(group_rows),
            "by_dataset": dict(dataset_summary),
            "kept_pairs_by_dataset": dict(kept_pairs_by_dataset),
            "dropped_preference_inconsistent_by_dataset": dict(dropped_pairs_by_dataset),
        },
        "category_ambiguous_counts": dict(category_ambiguous),
    }

    write_jsonl(out_dir / "gold_human_annotation_groups.jsonl", group_rows)
    write_json(out_dir / "gold_human_annotation_summary.json", summary)

    for group_key in ("dataset", "level", "class_tag", "ratio_type"):
        grouped_rows = defaultdict(list)
        for row in group_rows:
            grouped_rows[str(row[group_key])].append(row)
        csv_rows = []
        for value, items in sorted(grouped_rows.items(), key=lambda item: item[0]):
            csv_rows.append(
                {
                    group_key: value,
                    "groups": len(items),
                    "preference_consistent_groups": sum(1 for item in items if item["preference_consistent"]),
                    "preference_consistency_rate": safe_ratio(
                        sum(1 for item in items if item["preference_consistent"]),
                        len(items),
                    ),
                    "prompt_ilogical_any_rate": safe_ratio(
                        sum(1 for item in items if item["prompt_ilogical_any"]),
                        len(items),
                    ),
                }
            )
        write_csv(
            out_dir / f"gold_summary_by_{group_key}.csv",
            csv_rows,
            [group_key, "groups", "preference_consistent_groups", "preference_consistency_rate", "prompt_ilogical_any_rate"],
        )
        write_jsonl(out_dir / f"gold_summary_by_{group_key}.jsonl", csv_rows)


def copy_and_convert_csv(src: Path, dst_dir: Path, stem: str) -> None:
    rows = read_csv_rows(src)
    shutil.copy2(src, dst_dir / f"{stem}.csv")
    write_jsonl(dst_dir / f"{stem}.jsonl", rows)


def build_section_5_1() -> None:
    out_dir = PAPER_DATA_ROOT / "section_5_1_existing_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    file_map = {
        "baseline_human_alignment_v10_v13": BASELINE_ANALYSIS_DIR / "human_vs_metric_pairwise_accuracy_v10_v13.csv",
        "baseline_human_alignment_overall": BASELINE_ANALYSIS_DIR / "human_vs_metric_pairwise_accuracy.csv",
        "baseline_metric_means_overall": BASELINE_ANALYSIS_DIR / "metric_means_overall.csv",
        "baseline_metric_means_by_subject_count": BASELINE_ANALYSIS_DIR / "metric_means_by_subject_count.csv",
        "baseline_metric_means_by_occlusion_interaction": BASELINE_ANALYSIS_DIR / "metric_means_by_occlusion_interaction.csv",
    }
    for stem, src in file_map.items():
        copy_and_convert_csv(src, out_dir, stem)

    raw_metric_files = sorted(path.name for path in BASELINE_EVAL_RESULTS_DIR.glob("*.json"))
    write_json(
        out_dir / "baseline_sources.json",
        {
            "analysis_files": {stem: str(src) for stem, src in file_map.items()},
            "raw_metric_result_files": raw_metric_files,
        },
    )


def flatten_metrics_summary(results: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    overall_rows = []
    category_rows = []
    category_dataset_rows = []
    for result in results:
        overall_rows.append(
            {
                "metrics_model_name": result["metrics_model_name"],
                "size": result["size"],
                "mode": result["mode"],
                "rows": result["rows"],
                "evaluated_pairs": result["evaluated_pairs"],
                "pairwise_accuracy": result["pairwise_accuracy"],
                "pairwise_accuracy_v10": result["pairwise_accuracy_v10"],
                "pairwise_accuracy_v13": result["pairwise_accuracy_v13"],
                "macro_precision": result["macro_precision"],
                "macro_f1": result["macro_f1"],
                "macro_accuracy": result["macro_accuracy"],
                "correct_side_win_prob_mean": result["correct_side_win_prob_mean"],
            }
        )
        for category, metrics in result["category_metrics"].items():
            category_rows.append(
                {
                    "metrics_model_name": result["metrics_model_name"],
                    "size": result["size"],
                    "mode": result["mode"],
                    "category": category,
                    **metrics,
                }
            )
        for dataset_name, per_category in result["category_by_dataset"].items():
            for category, metrics in per_category.items():
                category_dataset_rows.append(
                    {
                        "metrics_model_name": result["metrics_model_name"],
                        "size": result["size"],
                        "mode": result["mode"],
                        "dataset": dataset_name,
                        "category": category,
                        **metrics,
                    }
                )
    return overall_rows, category_rows, category_dataset_rows


def build_section_5_2_and_5_3() -> None:
    summary = json.loads(METRICS_SUMMARY_PATH.read_text(encoding="utf-8"))
    results = summary["results"]
    overall_rows, category_rows, category_dataset_rows = flatten_metrics_summary(results)

    out_52 = PAPER_DATA_ROOT / "section_5_2_mie_alignment"
    out_53 = PAPER_DATA_ROOT / "section_5_3_breakdown"
    source_jsonl_dir = out_52 / "source_jsonl"
    source_jsonl_dir.mkdir(parents=True, exist_ok=True)

    for src in sorted(METRICS_JSONL_DIR.glob("*.jsonl")):
        shutil.copy2(src, source_jsonl_dir / src.name)

    write_json(out_52 / "mie_vs_human_summary.json", summary)
    write_csv(
        out_52 / "mie_overall_metrics.csv",
        overall_rows,
        [
            "metrics_model_name",
            "size",
            "mode",
            "rows",
            "evaluated_pairs",
            "pairwise_accuracy",
            "pairwise_accuracy_v10",
            "pairwise_accuracy_v13",
            "macro_precision",
            "macro_f1",
            "macro_accuracy",
            "correct_side_win_prob_mean",
        ],
    )
    write_jsonl(out_52 / "mie_overall_metrics.jsonl", overall_rows)
    write_csv(
        out_52 / "mie_category_metrics.csv",
        category_rows,
        ["metrics_model_name", "size", "mode", "category", "tp", "tn", "fp", "fn", "precision", "recall", "f1", "accuracy", "support"],
    )
    write_jsonl(out_52 / "mie_category_metrics.jsonl", category_rows)
    write_csv(
        out_52 / "mie_category_by_dataset.csv",
        category_dataset_rows,
        ["metrics_model_name", "size", "mode", "dataset", "category", "tp", "tn", "fp", "fn", "precision", "recall", "f1", "accuracy", "support"],
    )
    write_jsonl(out_52 / "mie_category_by_dataset.jsonl", category_dataset_rows)
    write_json(
        out_52 / "mie_sources.json",
        {
            "summary_json": str(METRICS_SUMMARY_PATH),
            "source_jsonl_files": [str(path) for path in sorted(METRICS_JSONL_DIR.glob("*.jsonl"))],
            "note": "Current export contains the 0.8B and 2B evaluator variants only.",
        },
    )

    seen_unseen_rows = []
    for row in overall_rows:
        seen_unseen_rows.append(
            {
                "metrics_model_name": row["metrics_model_name"],
                "size": row["size"],
                "mode": row["mode"],
                "seen_pairwise_accuracy_v10": row["pairwise_accuracy_v10"],
                "unseen_pairwise_accuracy_v13": row["pairwise_accuracy_v13"],
                "unseen_minus_seen": row["pairwise_accuracy_v13"] - row["pairwise_accuracy_v10"],
            }
        )

    by_size = defaultdict(dict)
    for row in overall_rows:
        by_size[row["size"]][row["mode"]] = row
    lora_vs_layer_rows = []
    for size, entries in sorted(by_size.items()):
        if "layer_only" in entries and "lora_layer" in entries:
            layer = entries["layer_only"]
            lora = entries["lora_layer"]
            lora_vs_layer_rows.append(
                {
                    "size": size,
                    "pairwise_accuracy_delta_lora_minus_layer": lora["pairwise_accuracy"] - layer["pairwise_accuracy"],
                    "pairwise_accuracy_v10_delta": lora["pairwise_accuracy_v10"] - layer["pairwise_accuracy_v10"],
                    "pairwise_accuracy_v13_delta": lora["pairwise_accuracy_v13"] - layer["pairwise_accuracy_v13"],
                    "macro_f1_delta_lora_minus_layer": lora["macro_f1"] - layer["macro_f1"],
                }
            )

    by_mode = defaultdict(list)
    for row in overall_rows:
        by_mode[row["mode"]].append(row)
    scaling_rows = []
    size_order = {"08b": 0, "2b": 1, "4b": 2}
    for mode, rows in sorted(by_mode.items()):
        rows = sorted(rows, key=lambda item: size_order.get(item["size"], 999))
        for earlier, later in zip(rows, rows[1:]):
            scaling_rows.append(
                {
                    "mode": mode,
                    "from_size": earlier["size"],
                    "to_size": later["size"],
                    "pairwise_accuracy_delta": later["pairwise_accuracy"] - earlier["pairwise_accuracy"],
                    "pairwise_accuracy_v10_delta": later["pairwise_accuracy_v10"] - earlier["pairwise_accuracy_v10"],
                    "pairwise_accuracy_v13_delta": later["pairwise_accuracy_v13"] - earlier["pairwise_accuracy_v13"],
                    "macro_f1_delta": later["macro_f1"] - earlier["macro_f1"],
                }
            )

    write_csv(
        out_53 / "mie_seen_unseen_table.csv",
        seen_unseen_rows,
        ["metrics_model_name", "size", "mode", "seen_pairwise_accuracy_v10", "unseen_pairwise_accuracy_v13", "unseen_minus_seen"],
    )
    write_jsonl(out_53 / "mie_seen_unseen_table.jsonl", seen_unseen_rows)
    write_csv(
        out_53 / "mie_lora_vs_layer_table.csv",
        lora_vs_layer_rows,
        [
            "size",
            "pairwise_accuracy_delta_lora_minus_layer",
            "pairwise_accuracy_v10_delta",
            "pairwise_accuracy_v13_delta",
            "macro_f1_delta_lora_minus_layer",
        ],
    )
    write_jsonl(out_53 / "mie_lora_vs_layer_table.jsonl", lora_vs_layer_rows)
    write_csv(
        out_53 / "mie_scaling_table.csv",
        scaling_rows,
        ["mode", "from_size", "to_size", "pairwise_accuracy_delta", "pairwise_accuracy_v10_delta", "pairwise_accuracy_v13_delta", "macro_f1_delta"],
    )
    write_jsonl(out_53 / "mie_scaling_table.jsonl", scaling_rows)
    write_csv(
        out_53 / "mie_category_by_dataset.csv",
        category_dataset_rows,
        ["metrics_model_name", "size", "mode", "dataset", "category", "tp", "tn", "fp", "fn", "precision", "recall", "f1", "accuracy", "support"],
    )
    write_jsonl(out_53 / "mie_category_by_dataset.jsonl", category_dataset_rows)
    write_json(
        out_53 / "mie_breakdown_manifest.json",
        {
            "summary_json": str(out_52 / "mie_vs_human_summary.json"),
            "note": "Breakdown tables are derived from the current 0.8B and 2B evaluator summary.",
        },
    )


def write_readme() -> None:
    readme_path = PAPER_DATA_ROOT / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# paper_data",
                "",
                "This folder contains section-oriented data exports for the paper results sections.",
                "",
                "Generated sections:",
                "",
                "- `section_4_1_mib_silver`",
                "- `section_4_2_mib_gold`",
                "- `section_5_1_existing_metrics`",
                "- `section_5_2_mie_alignment`",
                "- `section_5_3_breakdown`",
                "",
                "Regenerate everything with:",
                "",
                "```bash",
                "python3 build_paper_data.py",
                "```",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    PAPER_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    build_section_4_1()
    build_section_4_2()
    build_section_5_1()
    build_section_5_2_and_5_3()
    write_readme()
    print(f"Generated paper data under: {PAPER_DATA_ROOT}")


if __name__ == "__main__":
    main()
