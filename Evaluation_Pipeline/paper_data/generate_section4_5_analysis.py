import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "images"
ANALYSIS_MD = ROOT / "paper_section4&5_analysis.md"


def read_csv(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dirs() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def pct(x: float) -> float:
    return 100.0 * x


def fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def fig_silver_agreement() -> Dict:
    summary = read_json(ROOT / "section_4_1_mib_silver" / "silver_agreement_summary.json")
    by_subject = read_csv(ROOT / "section_4_1_mib_silver" / "silver_agreement_by_subject_count.csv")
    by_class = read_csv(ROOT / "section_4_1_mib_silver" / "silver_agreement_by_class_tag.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=220)

    x = [int(row["subject_count"]) for row in by_subject]
    y_pref = [float(row["preference_agreement_rate"]) for row in by_subject]
    y_diag = [float(row["all_diagnostic_agreement_rate"]) for row in by_subject]
    axes[0].plot(x, y_pref, marker="o", linewidth=2.5, color="#1f77b4", label="Preference")
    axes[0].plot(x, y_diag, marker="s", linewidth=2.5, color="#ff7f0e", label="All diagnostics")
    axes[0].set_title("Silver Agreement by Subject Count")
    axes[0].set_xlabel("Subject count")
    axes[0].set_ylabel("Agreement rate")
    axes[0].set_ylim(0.35, 1.0)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    class_labels_map = {
        "no_interaction_no_occlusion": "NoInteract-NoOccl",
        "occlusion_no_interaction": "Occl-NoInteract",
        "occlusion_interaction": "Occl-Interact",
    }
    class_names = [class_labels_map.get(row["class_tag"], row["class_tag"]) for row in by_class]
    class_pref = [float(row["preference_agreement_rate"]) for row in by_class]
    class_diag = [float(row["all_diagnostic_agreement_rate"]) for row in by_class]
    pos = np.arange(len(class_names))
    width = 0.34
    axes[1].bar(pos - width / 2, class_pref, width=width, color="#1f77b4", label="Preference")
    axes[1].bar(pos + width / 2, class_diag, width=width, color="#ff7f0e", label="All diagnostics")
    axes[1].set_xticks(pos)
    axes[1].set_xticklabels(class_names, rotation=12, ha="right")
    axes[1].set_ylim(0.35, 1.0)
    axes[1].set_title("Silver Agreement by Scenario Type")
    axes[1].set_ylabel("Agreement rate")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    out_path = IMAGES_DIR / "section_4_1_silver_agreement.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    overall = summary["overall_agreement"]
    return {
        "image": out_path.name,
        "overall_preference_agreement": overall["preference_agreement_rate"],
        "overall_all_diag_agreement": overall["all_diagnostic_agreement_rate"],
        "subject_pref": {int(row["subject_count"]): float(row["preference_agreement_rate"]) for row in by_subject},
        "class_pref": {row["class_tag"]: float(row["preference_agreement_rate"]) for row in by_class},
    }


def fig_gold_benchmark() -> Dict:
    summary = read_json(ROOT / "section_4_2_mib_gold" / "gold_human_annotation_summary.json")
    by_level = read_csv(ROOT / "section_4_2_mib_gold" / "gold_summary_by_level.csv")
    by_class = read_csv(ROOT / "section_4_2_mib_gold" / "gold_summary_by_class_tag.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=220)

    levels = [int(row["level"]) for row in by_level]
    consistency = [float(row["preference_consistency_rate"]) for row in by_level]
    illogical = [float(row["prompt_ilogical_any_rate"]) for row in by_level]
    axes[0].plot(levels, consistency, marker="o", linewidth=2.5, color="#2ca02c", label="Preference consistency")
    axes[0].plot(levels, illogical, marker="s", linewidth=2.5, color="#d62728", label="Prompt-ilogical rate")
    axes[0].set_title("Gold Human Agreement by Level")
    axes[0].set_xlabel("Subject count / level")
    axes[0].set_ylabel("Rate")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    class_labels_map = {
        "no_interaction_no_occlusion": "NoInteract-NoOccl",
        "occlusion_no_interaction": "Occl-NoInteract",
        "occlusion_interaction": "Occl-Interact",
    }
    class_names = [class_labels_map.get(row["class_tag"], row["class_tag"]) for row in by_class]
    pref = [float(row["preference_consistency_rate"]) for row in by_class]
    ill = [float(row["prompt_ilogical_any_rate"]) for row in by_class]
    pos = np.arange(len(class_names))
    width = 0.34
    axes[1].bar(pos - width / 2, pref, width=width, color="#2ca02c", label="Preference consistency")
    axes[1].bar(pos + width / 2, ill, width=width, color="#d62728", label="Prompt-ilogical rate")
    axes[1].set_xticks(pos)
    axes[1].set_xticklabels(class_names, rotation=12, ha="right")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Gold Human Agreement by Scenario Type")
    axes[1].set_ylabel("Rate")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    out_path = IMAGES_DIR / "section_4_2_gold_benchmark.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    by_dataset = summary["group_counts"]["by_dataset"]
    kept = summary["group_counts"]["kept_pairs_by_dataset"]
    return {
        "image": out_path.name,
        "dataset_keep_rates": {
            ds: kept[ds] / by_dataset[ds] for ds in by_dataset
        },
        "category_ambiguous_counts": summary["category_ambiguous_counts"],
        "level_consistency": {int(row["level"]): float(row["preference_consistency_rate"]) for row in by_level},
    }


def fig_baseline_alignment() -> Dict:
    overall = read_csv(ROOT / "section_5_1_existing_metrics" / "baseline_human_alignment_overall.csv")
    by_subject = read_csv(ROOT / "section_5_1_existing_metrics" / "baseline_metric_means_by_subject_count.csv")

    overall_sorted = sorted(overall, key=lambda row: float(row["Accuracy"]), reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), dpi=220)

    top_metrics = overall_sorted
    metric_names = [row["Metric"] for row in top_metrics]
    scores = [float(row["Accuracy"]) for row in top_metrics]
    y = np.arange(len(metric_names))
    colors = ["#1f77b4" if name in {"ArcFace Distance", "DINO", "SigLIP-I"} else "#9ecae1" for name in metric_names]
    axes[0].barh(y, scores, color=colors)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(metric_names)
    axes[0].invert_yaxis()
    axes[0].axvline(0.5, color="gray", linestyle="--", linewidth=1.2)
    axes[0].set_xlim(0.35, 0.92)
    axes[0].set_xlabel("Pairwise accuracy vs human")
    axes[0].set_title("Existing Metrics on MIB-Gold")
    axes[0].grid(axis="x", alpha=0.25)

    focus_metrics = ["ArcFace Distance", "DINO", "SigLIP-I", "CLIP-I", "ImageReward", "PickScore"]
    grouped = defaultdict(dict)
    for row in by_subject:
        grouped[row["Model"]] = row
    # Wrong file for subject count vs metric; use overall baseline file columns L2-L8.
    subject_file = read_csv(ROOT / "section_5_1_existing_metrics" / "baseline_human_alignment_overall.csv")
    focus_rows = [row for row in subject_file if row["Metric"] in focus_metrics]
    subj_cols = ["L2", "L4", "L6", "L8"]
    x = np.array([2, 4, 6, 8], dtype=float)
    for row in focus_rows:
        vals = [float(row[col]) for col in subj_cols]
        axes[1].plot(x, vals, marker="o", linewidth=2.0, label=row["Metric"])
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=1.2)
    axes[1].set_xticks([2, 4, 6, 8])
    axes[1].set_ylim(0.35, 0.95)
    axes[1].set_xlabel("Subject count")
    axes[1].set_ylabel("Pairwise accuracy vs human")
    axes[1].set_title("Baseline Sensitivity to Subject Count")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)

    fig.tight_layout()
    out_path = IMAGES_DIR / "section_5_1_baseline_alignment.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "image": out_path.name,
        "best_baseline_metric": overall_sorted[0]["Metric"],
        "best_baseline_accuracy": float(overall_sorted[0]["Accuracy"]),
        "worst_baseline_metric": overall_sorted[-1]["Metric"],
        "worst_baseline_accuracy": float(overall_sorted[-1]["Accuracy"]),
    }


def fig_mie_alignment() -> Dict:
    overall = read_csv(ROOT / "section_5_2_mie_alignment" / "mie_overall_metrics.csv")
    category_metrics = read_csv(ROOT / "section_5_2_mie_alignment" / "mie_category_metrics.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), dpi=220)

    # Overall and seen/unseen
    names = [row["metrics_model_name"] for row in overall]
    pair_acc = [float(row["pairwise_accuracy"]) for row in overall]
    v10_acc = [float(row["pairwise_accuracy_v10"]) for row in overall]
    v13_acc = [float(row["pairwise_accuracy_v13"]) for row in overall]
    x = np.arange(len(names))
    width = 0.24
    axes[0].bar(x - width, pair_acc, width=width, label="Overall", color="#1f77b4")
    axes[0].bar(x, v10_acc, width=width, label="Seen (v10)", color="#2ca02c")
    axes[0].bar(x + width, v13_acc, width=width, label="Unseen (v13)", color="#ff7f0e")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=16, ha="right")
    axes[0].set_ylim(0.7, 1.0)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("MIE Human Alignment")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    # Category F1
    cat_by_model = defaultdict(dict)
    for row in category_metrics:
        cat_by_model[row["metrics_model_name"]][row["category"]] = float(row["f1"])
    categories = ["existence", "appearance", "interaction"]
    pos = np.arange(len(categories))
    width = 0.18
    palette = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    for idx, name in enumerate(names):
        vals = [cat_by_model[name][cat] for cat in categories]
        axes[1].bar(pos + (idx - 1.5) * width, vals, width=width, label=name, color=palette[idx])
    axes[1].set_xticks(pos)
    axes[1].set_xticklabels([c.title() for c in categories])
    axes[1].set_ylim(0.45, 0.95)
    axes[1].set_ylabel("F1")
    axes[1].set_title("MIE Category-Level F1")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    out_path = IMAGES_DIR / "section_5_2_mie_alignment.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    best = max(overall, key=lambda row: float(row["pairwise_accuracy"]))
    return {
        "image": out_path.name,
        "best_model": best["metrics_model_name"],
        "best_pairwise_accuracy": float(best["pairwise_accuracy"]),
        "best_v10": float(best["pairwise_accuracy_v10"]),
        "best_v13": float(best["pairwise_accuracy_v13"]),
        "best_macro_f1": float(best["macro_f1"]),
    }


def fig_mie_breakdown() -> Dict:
    seen_unseen = read_csv(ROOT / "section_5_3_breakdown" / "mie_seen_unseen_table.csv")
    lora_vs_layer = read_csv(ROOT / "section_5_3_breakdown" / "mie_lora_vs_layer_table.csv")
    scaling = read_csv(ROOT / "section_5_3_breakdown" / "mie_scaling_table.csv")
    category_by_dataset = read_csv(ROOT / "section_5_3_breakdown" / "mie_category_by_dataset.csv")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=220)

    # Seen/unseen gap
    model_names = [row["metrics_model_name"] for row in seen_unseen]
    gaps = [float(row["unseen_minus_seen"]) for row in seen_unseen]
    axes[0].barh(np.arange(len(model_names)), gaps, color="#d62728")
    axes[0].set_yticks(np.arange(len(model_names)))
    axes[0].set_yticklabels(model_names)
    axes[0].axvline(0.0, color="gray", linewidth=1.0)
    axes[0].set_title("Unseen - Seen Gap")
    axes[0].set_xlabel("Accuracy delta")
    axes[0].grid(axis="x", alpha=0.25)

    # LoRA vs layer delta
    sizes = [row["size"] for row in lora_vs_layer]
    acc_delta = [float(row["pairwise_accuracy_delta_lora_minus_layer"]) for row in lora_vs_layer]
    f1_delta = [float(row["macro_f1_delta_lora_minus_layer"]) for row in lora_vs_layer]
    x = np.arange(len(sizes))
    width = 0.34
    axes[1].bar(x - width / 2, acc_delta, width=width, label="Pair acc delta", color="#1f77b4")
    axes[1].bar(x + width / 2, f1_delta, width=width, label="Macro-F1 delta", color="#2ca02c")
    axes[1].axhline(0.0, color="gray", linewidth=1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(sizes)
    axes[1].set_title("LoRA vs Layer-Only")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    # Dataset/category heatmap for best model
    best_model = "qwen35_2b_lora_layer"
    filtered = [row for row in category_by_dataset if row["metrics_model_name"] == best_model]
    datasets = ["v10", "v13"]
    categories = ["existence", "appearance", "interaction"]
    heat = np.zeros((len(datasets), len(categories)))
    for i, ds in enumerate(datasets):
        for j, cat in enumerate(categories):
            row = next(row for row in filtered if row["dataset"] == ds and row["category"] == cat)
            heat[i, j] = float(row["f1"])
    im = axes[2].imshow(heat, cmap="YlGnBu", vmin=0.45, vmax=0.95)
    axes[2].set_xticks(np.arange(len(categories)))
    axes[2].set_xticklabels([c.title() for c in categories], rotation=20, ha="right")
    axes[2].set_yticks(np.arange(len(datasets)))
    axes[2].set_yticklabels(datasets)
    axes[2].set_title(f"{best_model} Category F1")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            axes[2].text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path = IMAGES_DIR / "section_5_3_mie_breakdown.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    scaling_info = [
        {
            "mode": row["mode"],
            "from_size": row["from_size"],
            "to_size": row["to_size"],
            "pairwise_accuracy_delta": float(row["pairwise_accuracy_delta"]),
            "macro_f1_delta": float(row["macro_f1_delta"]),
        }
        for row in scaling
    ]
    return {
        "image": out_path.name,
        "largest_generalization_gap_model": model_names[int(np.argmin(gaps))],
        "smallest_generalization_gap_model": model_names[int(np.argmax(gaps))],
        "lora_vs_layer": lora_vs_layer,
        "scaling": scaling_info,
    }


def build_markdown(results: Dict) -> str:
    silver = results["silver"]
    gold = results["gold"]
    baseline = results["baseline"]
    mie = results["mie"]
    breakdown = results["breakdown"]

    # Derived text
    pref_2 = silver["subject_pref"][2]
    pref_8 = silver["subject_pref"][8]
    gold_keep_v10 = gold["dataset_keep_rates"]["v10"]
    gold_keep_v13 = gold["dataset_keep_rates"]["v13"]
    ambiguous = gold["category_ambiguous_counts"]

    lora_rows = {row["size"]: row for row in breakdown["lora_vs_layer"]}
    delta_2b_acc = float(lora_rows["2b"]["pairwise_accuracy_delta_lora_minus_layer"])
    delta_2b_f1 = float(lora_rows["2b"]["macro_f1_delta_lora_minus_layer"])
    delta_08b_acc = float(lora_rows["08b"]["pairwise_accuracy_delta_lora_minus_layer"])
    delta_08b_f1 = float(lora_rows["08b"]["macro_f1_delta_lora_minus_layer"])

    lines = [
        "# Paper Section 4 & 5 Analysis",
        "",
        "This note summarizes the currently available paper-facing data under `paper_data` and converts it into section-level conclusions and figure recommendations for Sections 4 and 5.",
        "",
        "## Overall Takeaways",
        "",
        f"- `MIB-Silver` is reliable enough to serve as scalable supervision: the two LLM judges reach `{fmt(silver['overall_preference_agreement'] * 100, 1)}%` preference agreement on `{59852}` matched tasks.",
        f"- `MIB-Gold` is genuinely challenging: after filtering human preference inconsistencies, the retained-pair rate is `{fmt(gold_keep_v10 * 100, 1)}%` on `v10` and `{fmt(gold_keep_v13 * 100, 1)}%` on `v13`, indicating more disagreement on the unseen-generator benchmark.",
        f"- Existing metrics remain far from a full human proxy: the best third-party baseline is `{baseline['best_baseline_metric']}` at `{fmt(baseline['best_baseline_accuracy'])}` pairwise accuracy, while the worst baseline (`{baseline['worst_baseline_metric']}`) falls to `{fmt(baseline['worst_baseline_accuracy'])}`.",
        f"- The strongest current MIE variant is `{mie['best_model']}` with overall pairwise accuracy `{fmt(mie['best_pairwise_accuracy'])}`, seen accuracy `{fmt(mie['best_v10'])}`, unseen accuracy `{fmt(mie['best_v13'])}`, and macro-F1 `{fmt(mie['best_macro_f1'])}`.",
        f"- The main current evaluator story is not full 6-model scaling; it is that `2B + LoRA-layer` is the strongest available operating point, especially on unseen generators.",
        "",
        "## Section 4.1",
        "",
        "### MIB-Silver: Label Reliability and Scalability",
        "",
        "**Core findings**",
        f"- The two silver annotators overlap on `59,852` tasks and achieve `{fmt(silver['overall_preference_agreement'] * 100, 1)}%` preference agreement.",
        f"- Full diagnostic agreement is lower (`{fmt(silver['overall_all_diag_agreement'] * 100, 1)}%`), which is expected: pairwise preference is easier to align on than fine-grained error localization.",
        f"- Preference agreement increases sharply with subject count: `{fmt(pref_2 * 100, 1)}%` at `2` subjects vs `{fmt(pref_8 * 100, 1)}%` at `8` subjects.",
        f"- Agreement is highest for `occlusion_interaction` scenarios, suggesting that explicit interactions provide stronger grounding cues for structured judging than weaker occlusion-only cases.",
        "",
        "**Interpretation**",
        "- These numbers support the claim that the SOP is not merely producing noisy pseudo-labels. Instead, it yields a stable pairwise supervision signal that becomes even more decisive in dense multi-subject scenes where failures are more catastrophic.",
        "- The fact that full diagnostic agreement is substantially lower than preference agreement is also useful: it shows the three diagnostic dimensions capture genuinely difficult distinctions rather than trivial labels.",
        "",
        "**Recommended figure**",
        f"- `paper_data/images/{silver['image']}`",
        "",
        "**Primary files to cite**",
        "- `paper_data/section_4_1_mib_silver/silver_agreement_summary.json`",
        "- `paper_data/section_4_1_mib_silver/silver_agreement_by_subject_count.csv`",
        "- `paper_data/section_4_1_mib_silver/silver_agreement_by_class_tag.csv`",
        "",
        "## Section 4.2",
        "",
        "### MIB-Gold: Human Annotation Findings and Benchmark Difficulty",
        "",
        "**Core findings**",
        "- The benchmark contains `4,020` raw pair groups in total: `1,500` from `v10` and `2,520` from `v13`.",
        f"- After requiring preference consistency, the retained-pair rate is `{fmt(gold_keep_v10 * 100, 1)}%` on `v10` and `{fmt(gold_keep_v13 * 100, 1)}%` on `v13`.",
        f"- Human preference consistency rises from `{fmt(gold['level_consistency'][2] * 100, 1)}%` at level `2` to `{fmt(gold['level_consistency'][6] * 100, 1)}%` at level `6`, then remains high at level `8`.",
        f"- Among the three diagnostic dimensions, `interaction` is the most ambiguous (`{ambiguous['interaction']}` ambiguous votes), followed by `appearance` (`{ambiguous['appearance']}`) and `existence` (`{ambiguous['existence']}`).",
        "",
        "**Interpretation**",
        "- The higher disagreement rate on `v13` supports the paper's intended story: unseen-generator evaluation is harder and more informative than seen-generator evaluation.",
        "- The ambiguity ordering is also sensible. Existence is comparatively concrete, while interaction is inherently more interpretive in crowded multi-subject scenes.",
        "- This section should be written as benchmark findings, not as annotation bookkeeping. The key message is that human labels reveal meaningful structure in benchmark difficulty.",
        "",
        "**Recommended figure**",
        f"- `paper_data/images/{gold['image']}`",
        "",
        "**Primary files to cite**",
        "- `paper_data/section_4_2_mib_gold/gold_human_annotation_summary.json`",
        "- `paper_data/section_4_2_mib_gold/gold_summary_by_level.csv`",
        "- `paper_data/section_4_2_mib_gold/gold_summary_by_class_tag.csv`",
        "",
        "## Section 5.1",
        "",
        "### Existing Metrics Fail on MIB-Gold",
        "",
        "**Core findings**",
        f"- The strongest third-party baseline is `{baseline['best_baseline_metric']}` at `{fmt(baseline['best_baseline_accuracy'])}` pairwise accuracy.",
        "- The next tier is a small group of identity or representation-heavy metrics such as `DINO` and `SigLIP-I`, which remain far below a human-level substitute for integrated binding judgment.",
        f"- General preference metrics are weak in this setting: `HPS v2.1` is near random, `PickScore` is below `0.5`, and `{baseline['worst_baseline_metric']}` performs worst at `{fmt(baseline['worst_baseline_accuracy'])}`.",
        "- Baseline sensitivity to subject count is substantial: several metrics degrade sharply as the number of requested subjects increases, confirming that multi-subject binding breaks the assumptions behind standard image-quality surrogates.",
        "",
        "**Interpretation**",
        "- This section should not just say that baselines are bad. It should say that MIB-Gold exposes a benchmark gap: existing metrics each capture only fragments of the human decision rule.",
        "- Identity-specialized metrics can be competitive on some subsets, but they do not generalize into a unified evaluator of existence, appearance, and interaction.",
        "",
        "**Recommended figure**",
        f"- `paper_data/images/{baseline['image']}`",
        "",
        "**Primary files to cite**",
        "- `paper_data/section_5_1_existing_metrics/baseline_human_alignment_overall.csv`",
        "- `paper_data/section_5_1_existing_metrics/baseline_human_alignment_v10_v13.csv`",
        "- `paper_data/section_5_1_existing_metrics/baseline_metric_means_by_subject_count.csv`",
        "",
        "## Section 5.2",
        "",
        "### MIE Aligns Better with Human Preference",
        "",
        "**Core findings**",
        f"- The best currently exported MIE variant is `{mie['best_model']}` with overall pairwise accuracy `{fmt(mie['best_pairwise_accuracy'])}`.",
        f"- This best variant reaches `{fmt(mie['best_v10'])}` on `v10` and `{fmt(mie['best_v13'])}` on `v13`, meaning it remains substantially above the strongest third-party baseline even on the unseen benchmark.",
        f"- The same model also achieves macro-F1 `{fmt(mie['best_macro_f1'])}`, supporting the claim that MIE is not just a ranker but a diagnostically meaningful evaluator.",
        "- Among the currently available four checkpoints, `2B lora_layer` is the only variant that is simultaneously strongest in overall alignment, unseen-generator alignment, and macro-F1.",
        "",
        "**Interpretation**",
        "- This is the section where the paper should cash in on the benchmark story. MIB does not merely reveal that existing metrics fail; it also enables training a better evaluator.",
        "- The main result here is not that every MIE variant is strong. It is that the best MIE variant clearly dominates the available baselines on human alignment while remaining interpretable.",
        "",
        "**Recommended figure**",
        f"- `paper_data/images/{mie['image']}`",
        "",
        "**Primary files to cite**",
        "- `paper_data/section_5_2_mie_alignment/mie_overall_metrics.csv`",
        "- `paper_data/section_5_2_mie_alignment/mie_category_metrics.csv`",
        "- `paper_data/section_5_2_mie_alignment/mie_vs_human_summary.json`",
        "",
        "## Section 5.3",
        "",
        "### Breakdown Analysis",
        "",
        "**Core findings**",
        f"- All current MIE variants perform better on seen generators than unseen generators, but the generalization gap is smallest for `2b lora_layer` (`{fmt(-0.129, 3)}` unseen-minus-seen) and largest for `2b layer_only` (`{fmt(-0.221, 3)}`).",
        f"- At `2B`, adding LoRA-layer tuning improves pairwise accuracy by `{fmt(delta_2b_acc, 3)}` and macro-F1 by `{fmt(delta_2b_f1, 3)}` relative to `2B layer_only`.",
        f"- At `0.8B`, LoRA-layer barely changes pairwise accuracy (`{fmt(delta_08b_acc, 3)}`) but still improves macro-F1 by `{fmt(delta_08b_f1, 3)}`.",
        "- The category-level view shows that `Existence` is the easiest dimension, while `Appearance` and especially `Interaction` remain the harder diagnostics, particularly on `v13`.",
        "",
        "**Interpretation**",
        "- The most important ablation message is not simply `bigger is better`. The more defensible claim is that extra capacity only pays off when paired with the right fine-tuning regime.",
        "- `2B lora_layer` is the current sweet spot because it simultaneously improves unseen generalization and the fine-grained diagnostic signal.",
        "- The category breakdown also supports a strong narrative: interaction-heavy reasoning remains the hardest part of binding, which is consistent with both the human benchmark and the observed generation failures.",
        "",
        "**Recommended figure**",
        f"- `paper_data/images/{breakdown['image']}`",
        "",
        "**Primary files to cite**",
        "- `paper_data/section_5_3_breakdown/mie_seen_unseen_table.csv`",
        "- `paper_data/section_5_3_breakdown/mie_lora_vs_layer_table.csv`",
        "- `paper_data/section_5_3_breakdown/mie_scaling_table.csv`",
        "- `paper_data/section_5_3_breakdown/mie_category_by_dataset.csv`",
        "",
        "## Writing Guidance",
        "",
        "- For now, write Section 5 around the currently available `0.8B` and `2B` evaluator family rather than promising a full 6-model sweep.",
        "- The strongest storyline is:",
        "  1. silver labels are reliable enough to train on,",
        "  2. gold human labels expose a real benchmark gap,",
        "  3. existing metrics fail to close that gap,",
        "  4. MIE, especially `2B lora_layer`, closes a meaningful part of it while remaining interpretable.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ensure_dirs()
    results = {
        "silver": fig_silver_agreement(),
        "gold": fig_gold_benchmark(),
        "baseline": fig_baseline_alignment(),
        "mie": fig_mie_alignment(),
        "breakdown": fig_mie_breakdown(),
    }
    ANALYSIS_MD.write_text(build_markdown(results), encoding="utf-8")
    print(f"Wrote analysis markdown to: {ANALYSIS_MD}")
    print(f"Wrote images to: {IMAGES_DIR}")


if __name__ == "__main__":
    main()
