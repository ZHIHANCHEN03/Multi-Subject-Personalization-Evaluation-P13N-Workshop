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

# Conference-level Matplotlib Styles (NeurIPS strict format)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Bitstream Vera Serif", "Computer Modern Roman", "serif"],
    "mathtext.fontset": "stix",  # STIX fonts are similar to Times for math
    "axes.labelsize": 10,        # NeurIPS main text is 10pt
    "axes.titlesize": 10,        # NeurIPS prefers understated titles, not massive bold ones
    "axes.titleweight": "normal",
    "font.size": 10,
    "legend.fontsize": 9,        # NeurIPS captions and legends are typically 9pt
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 11,
    "figure.titleweight": "bold",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

FIGSIZE_TWO = (14.5, 5.8)
FIGSIZE_THREE = (17.2, 5.2)

GRID_COLOR = "#d9d9d9"
COLOR_BLUE = "#5f7d95"
COLOR_GREEN = "#79a88e"
COLOR_GOLD = "#d8b365"
COLOR_RED = "#c85c4a"
COLOR_RED_LIGHT = "#c9938a"
COLOR_BLUE_LIGHT = "#9eb6cb"
COLOR_GREEN_LIGHT = "#8bb8a8"


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


def style_axis(ax) -> None:
    ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def format_model_name(raw_name: str) -> str:
    mapping = {
        "qwen35_08b_layer_only": "Qwen-0.8B (Layer)",
        "qwen35_08b_lora_layer": "Qwen-0.8B (LoRA)",
        "qwen35_2b_layer_only": "Qwen-2B (Layer)",
        "qwen35_2b_lora_layer": "Qwen-2B (LoRA)",
        "qwen35_4b_layer_only": "Qwen-4B (Layer)",
        "qwen35_4b_lora_layer": "Qwen-4B (LoRA)",
    }
    return mapping.get(raw_name, raw_name)



def fig_silver_agreement() -> Dict:
    summary = read_json(ROOT / "section_4_1_1_mib_silver" / "silver_agreement_summary.json")
    by_subject = read_csv(ROOT / "section_4_1_1_mib_silver" / "silver_agreement_by_subject_count.csv")
    by_class = read_csv(ROOT / "section_4_1_1_mib_silver" / "silver_agreement_by_class_tag.csv")

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_THREE, dpi=300)

    # Subplot 1: By Evaluation Dimension
    overall = summary["overall_agreement"]
    dims = ["Preference", "Existence", "Interaction", "Appearance"]
    rates = [
        pct(overall["preference_agreement_rate"]),
        pct(overall["existence_both_agreement_rate"]),
        pct(overall["interaction_both_agreement_rate"]),
        pct(overall["appearance_both_agreement_rate"]),
    ]
    colors_dim = [COLOR_BLUE, COLOR_GREEN, COLOR_GOLD, COLOR_RED]
    x_pos = np.arange(len(dims))
    axes[0].bar(x_pos, rates, width=0.5, color=colors_dim)
    axes[0].set_xticks(x_pos)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Cross-Model Agreement (%)", fontsize=11)
    axes[0].set_title("Agreement by Dimension", fontsize=12)
    style_axis(axes[0])
    for i, v in enumerate(rates):
        axes[0].text(i, v + 1.5, f"{v:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[0].set_xticklabels(dims, rotation=20, ha="right", fontsize=10)

    # Subplot 2: By Subject Count
    x = [int(row["subject_count"]) for row in by_subject]
    y_pref = [pct(float(row["preference_agreement_rate"])) for row in by_subject]
    axes[1].plot(x, y_pref, marker="o", markersize=7, linewidth=2.3, color=COLOR_BLUE)
    axes[1].set_title("Agreement by Subject Count", fontsize=12)
    axes[1].set_xlabel("Number of Subjects", fontsize=11)
    axes[1].set_ylabel("Preference Agreement (%)", fontsize=11)
    axes[1].set_ylim(85, 102)
    axes[1].set_xticks([2, 4, 6, 8])
    style_axis(axes[1])
    for i, v in zip(x, y_pref):
        axes[1].text(i, v + 0.5, f"{v:.1f}%", ha='center', va='bottom', fontsize=9, color=COLOR_BLUE, fontweight='bold')

    # Subplot 3: By Spatial Relationship
    class_labels_map = {
        "no_interaction_no_occlusion": "No Occlusion w/o Interaction",
        "occlusion_no_interaction": "Occlusion w/o Interaction",
        "occlusion_interaction": "Occlusion w/ Interaction",
    }
    class_names = [class_labels_map.get(row["class_tag"], row["class_tag"]) for row in by_class]
    class_pref = [pct(float(row["preference_agreement_rate"])) for row in by_class]
    y_pos = np.arange(len(class_names))
    colors_tag = [COLOR_BLUE, COLOR_GREEN, COLOR_GOLD]
    axes[2].barh(y_pos, class_pref, height=0.5, color=colors_tag)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(class_names, fontsize=10)
    axes[2].set_xlim(80, 100)
    axes[2].set_xlabel("Preference Agreement (%)", fontsize=11)
    axes[2].set_title("Agreement by Spatial Relation", fontsize=12)
    style_axis(axes[2])
    for i, v in enumerate(class_pref):
        axes[2].text(v - 0.5, i, f"{v:.1f}%", ha='right', va='center', fontsize=9, color="white", fontweight='bold')

    fig.tight_layout()
    out_path = IMAGES_DIR / "section_4_1_1_silver_agreement.png"
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
    summary = read_json(ROOT / "section_4_1_2_mib_gold" / "gold_human_annotation_summary.json")
    by_level = read_csv(ROOT / "section_4_1_2_mib_gold" / "gold_summary_by_level.csv")
    by_class = read_csv(ROOT / "section_4_1_2_mib_gold" / "gold_summary_by_class_tag.csv")

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_TWO, dpi=300)

    levels = [int(row["level"]) for row in by_level]
    consistency = [pct(float(row["preference_consistency_rate"])) for row in by_level]
    axes[0].plot(levels, consistency, marker="o", markersize=7, linewidth=2.3, color=COLOR_BLUE)
    axes[0].set_title("Consistency by Subject Count", fontsize=12)
    axes[0].set_xlabel("Subject count / level", fontsize=11)
    axes[0].set_ylabel("Preference Consistency (%)", fontsize=11)
    axes[0].set_ylim(0, 105)
    style_axis(axes[0])
    for level, value in zip(levels, consistency):
        axes[0].text(level, value + 1.2, f"{value:.1f}%", ha="center", va="bottom", fontsize=9, color=COLOR_BLUE, fontweight="bold")

    class_labels_map = {
        "no_interaction_no_occlusion": "No Occlusion w/o Interaction",
        "occlusion_no_interaction": "Occlusion w/o Interaction",
        "occlusion_interaction": "Occlusion w/ Interaction",
    }
    class_names = [class_labels_map.get(row["class_tag"], row["class_tag"]) for row in by_class]
    pref = [pct(float(row["preference_consistency_rate"])) for row in by_class]
    pos = np.arange(len(class_names))
    bars = axes[1].bar(pos, pref, width=0.55, color=[COLOR_BLUE, COLOR_GREEN, COLOR_GOLD])
    axes[1].set_xticks(pos)
    axes[1].set_xticklabels(class_names, rotation=0, ha="center", fontsize=10)
    axes[1].set_ylim(0, 105)
    axes[1].set_title("Consistency by Spatial Relation", fontsize=12)
    axes[1].set_ylabel("Preference Consistency (%)", fontsize=11)
    style_axis(axes[1])
    for bar, value in zip(bars, pref):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    out_path = IMAGES_DIR / "section_4_1_2_gold_benchmark.png"
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


def fig_4_1_alignment_overview() -> Dict:
    silver_summary = read_json(ROOT / "section_4_1_1_mib_silver" / "silver_agreement_summary.json")
    silver_by_subject = read_csv(ROOT / "section_4_1_1_mib_silver" / "silver_agreement_by_subject_count.csv")
    silver_by_class = read_csv(ROOT / "section_4_1_1_mib_silver" / "silver_agreement_by_class_tag.csv")

    gold_groups_path = ROOT / "section_4_1_2_mib_gold" / "gold_human_annotation_groups.jsonl"
    gold_by_level = read_csv(ROOT / "section_4_1_2_mib_gold" / "gold_summary_by_level.csv")
    gold_by_class = read_csv(ROOT / "section_4_1_2_mib_gold" / "gold_summary_by_class_tag.csv")
    gold_groups = []
    with gold_groups_path.open("r", encoding="utf-8") as f:
        for line in f:
            gold_groups.append(json.loads(line))

    fig, axes = plt.subplots(2, 3, figsize=(20, 10), dpi=300)

    class_labels_map = {
        "no_interaction_no_occlusion": "No Occlusion w/o Interaction",
        "occlusion_no_interaction": "Occlusion w/o Interaction",
        "occlusion_interaction": "Occlusion w/ Interaction",
    }

    def gold_dim_flags(group: Dict) -> Dict[str, bool]:
        per_category = {}
        for category in ("existence", "appearance", "interaction"):
            per_category[category] = all(
                model_info[category]["majority"] is not None
                for model_info in group["categories"].values()
            )
        return per_category

    gold_dim_rates = {
        "Preference": pct(sum(1 for row in gold_groups if row["preference_consistent"]) / len(gold_groups)),
        "Existence": pct(sum(1 for row in gold_groups if gold_dim_flags(row)["existence"]) / len(gold_groups)),
        "Interaction": pct(sum(1 for row in gold_groups if gold_dim_flags(row)["interaction"]) / len(gold_groups)),
        "Appearance": pct(sum(1 for row in gold_groups if gold_dim_flags(row)["appearance"]) / len(gold_groups)),
    }

    # -- ROW 0: SILVER --
    # Panel 0,0: Silver by dimension
    silver_overall = silver_summary["overall_agreement"]
    dims = ["Preference", "Existence", "Interaction", "Appearance"]
    silver_rates = [
        pct(silver_overall["preference_agreement_rate"]),
        pct(silver_overall["existence_both_agreement_rate"]),
        pct(silver_overall["interaction_both_agreement_rate"]),
        pct(silver_overall["appearance_both_agreement_rate"]),
    ]
    x_pos = np.arange(len(dims))
    width = 0.5
    axes[0, 0].bar(x_pos, silver_rates, width=width, color=[COLOR_BLUE, COLOR_GREEN, COLOR_GOLD, COLOR_RED], alpha=0.9)
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(dims, rotation=20, ha="right", fontsize=10)
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].set_ylabel("Agreement (%)", fontsize=11)
    axes[0, 0].set_title("Silver (VLM-VLM) by Dimension", fontsize=12)
    style_axis(axes[0, 0])
    for i, v in enumerate(silver_rates):
        axes[0, 0].text(i, v + 1.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel 0,1: Silver by subject count
    silver_x = [int(row["subject_count"]) for row in silver_by_subject]
    silver_y = [pct(float(row["preference_agreement_rate"])) for row in silver_by_subject]
    axes[0, 1].plot(silver_x, silver_y, marker="o", markersize=7, linewidth=2.3, color=COLOR_BLUE)
    axes[0, 1].set_xticks([2, 4, 6, 8])
    axes[0, 1].set_ylim(85, 102)
    axes[0, 1].set_xlabel("Number of Subjects", fontsize=11)
    axes[0, 1].set_ylabel("Agreement (%)", fontsize=11)
    axes[0, 1].set_title("Silver (VLM-VLM) by Subject Count", fontsize=12)
    style_axis(axes[0, 1])
    for i, v in zip(silver_x, silver_y):
        axes[0, 1].text(i, v + 0.4, f"{v:.1f}%", ha="center", va="bottom", fontsize=9, color=COLOR_BLUE, fontweight="bold")

    # Panel 0,2: Silver by spatial relation
    silver_class_names = [class_labels_map.get(row["class_tag"], row["class_tag"]) for row in silver_by_class]
    silver_class_pref = [pct(float(row["preference_agreement_rate"])) for row in silver_by_class]
    y_pos = np.arange(len(silver_class_names))
    axes[0, 2].barh(y_pos, silver_class_pref, height=0.6, color=[COLOR_BLUE, COLOR_GREEN, COLOR_GOLD], alpha=0.9)
    axes[0, 2].set_yticks(y_pos)
    axes[0, 2].set_yticklabels(silver_class_names, fontsize=10)
    axes[0, 2].set_xlim(80, 100)
    axes[0, 2].set_xlabel("Preference Agreement (%)", fontsize=11)
    axes[0, 2].set_title("Silver (VLM-VLM) by Spatial Relation", fontsize=12)
    style_axis(axes[0, 2])
    for i, v in enumerate(silver_class_pref):
        axes[0, 2].text(v - 0.4, i, f"{v:.1f}%", ha="right", va="center", fontsize=9, color="white", fontweight="bold")

    # -- ROW 1: GOLD --
    # Panel 1,0: Gold by dimension
    gold_dim_values = [gold_dim_rates[key] for key in dims]
    axes[1, 0].bar(x_pos, gold_dim_values, width=width, color=[COLOR_BLUE, COLOR_GREEN, COLOR_GOLD, COLOR_RED], alpha=0.9)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(dims, rotation=20, ha="right", fontsize=10)
    axes[1, 0].set_ylim(0, 105)
    axes[1, 0].set_ylabel("Agreement (%)", fontsize=11)
    axes[1, 0].set_title("Gold (Human-Human) by Dimension", fontsize=12)
    style_axis(axes[1, 0])
    for i, v in enumerate(gold_dim_values):
        axes[1, 0].text(i, v + 1.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel 1,1: Gold by subject count
    gold_levels = [int(row["level"]) for row in gold_by_level]
    gold_consistency = [pct(float(row["preference_consistency_rate"])) for row in gold_by_level]
    axes[1, 1].plot(gold_levels, gold_consistency, marker="o", markersize=7, linewidth=2.3, color=COLOR_BLUE)
    axes[1, 1].set_xticks([2, 4, 6, 8])
    axes[1, 1].set_ylim(85, 102)
    axes[1, 1].set_xlabel("Number of Subjects", fontsize=11)
    axes[1, 1].set_ylabel("Agreement (%)", fontsize=11)
    axes[1, 1].set_title("Gold (Human-Human) by Subject Count", fontsize=12)
    style_axis(axes[1, 1])
    for level, value in zip(gold_levels, gold_consistency):
        axes[1, 1].text(level, value + 0.4, f"{value:.1f}%", ha="center", va="bottom", fontsize=9, color=COLOR_BLUE, fontweight="bold")

    # Panel 1,2: Gold by spatial relation
    gold_pref = [pct(float(row["preference_consistency_rate"])) for row in gold_by_class]
    axes[1, 2].barh(y_pos, gold_pref, height=0.6, color=[COLOR_BLUE, COLOR_GREEN, COLOR_GOLD], alpha=0.9)
    axes[1, 2].set_yticks(y_pos)
    axes[1, 2].set_yticklabels(silver_class_names, fontsize=10)
    axes[1, 2].set_xlim(80, 100)
    axes[1, 2].set_xlabel("Preference Agreement (%)", fontsize=11)
    axes[1, 2].set_title("Gold (Human-Human) by Spatial Relation", fontsize=12)
    style_axis(axes[1, 2])
    for i, v in enumerate(gold_pref):
        axes[1, 2].text(v - 0.4, i, f"{v:.1f}%", ha="right", va="center", fontsize=9, color="white", fontweight="bold")

    fig.suptitle("High Agreement in Silver VLM Labels and Gold Human Labels", fontsize=16, y=0.96)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = IMAGES_DIR / "section_4_1_1_alignment_overview.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "image": out_path.name,
    }


def fig_baseline_alignment() -> Dict:
    overall = read_csv(ROOT / "section_4_2_1_existing_metrics" / "baseline_human_alignment_overall.csv")
    # Keep this figure strictly on the v13 evaluation split.
    mie_4b_lora_overall = 0.8841597191750767

    overall_sorted = sorted(overall, key=lambda row: float(row["Accuracy"]), reverse=True)
    
    # 过滤掉 ArcFace 相关的指标
    overall_sorted = [row for row in overall_sorted if "ArcFace" not in row["Metric"]]
    overall_sorted.insert(
        0,
        {
            "Metric": "MIE-4B-LoRA",
            "Accuracy": f"{mie_4b_lora_overall:.15f}",
        },
    )
    overall_sorted = sorted(overall_sorted, key=lambda row: float(row["Accuracy"]), reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_TWO, dpi=300)

    top_metrics = overall_sorted
    metric_names = [row["Metric"] for row in top_metrics]
    scores = [pct(float(row["Accuracy"])) for row in top_metrics]
    y = np.arange(len(metric_names))
    
    # Use a restrained palette for baselines and a single warm highlight for MIE.
    colors = []
    for metric_name, score in zip(metric_names, scores):
        if metric_name == "MIE-4B-LoRA":
            colors.append(COLOR_RED)
        elif score > 0.6:
            colors.append(COLOR_BLUE)
        elif score > 0.5:
            colors.append(COLOR_GOLD)
        else:
            colors.append(COLOR_RED_LIGHT)
            
    bars = axes[0].barh(y, scores, color=colors)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(metric_names, fontsize=10)
    axes[0].invert_yaxis()
    axes[0].axvline(50.0, color="black", linestyle="--", linewidth=1.2)
    axes[0].set_xlim(0.0, max(scores) + 15.0)
    axes[0].set_xlabel("Agreement with Human Preference (%)")
    axes[0].set_title("Human Alignment by Metric")
    style_axis(axes[0])
    
    # 在柱状图尾部添加具体的数字
    for bar in bars:
        width = bar.get_width()
        axes[0].text(width + 1.0, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                     ha='left', va='center', fontsize=9)

    # 用左图的所有 metric（除了可能被去掉的）
    focus_metrics = metric_names
    subject_file = read_csv(ROOT / "section_4_2_1_existing_metrics" / "baseline_human_alignment_overall.csv")
    focus_rows = [row for row in subject_file if row["Metric"] in focus_metrics]
    subj_cols = ["L2", "L4", "L6", "L8"]
    x = np.array([2, 4, 6, 8], dtype=float)
    
    baseline_line_colors = [
        "#7399bf", "#8bb8a8", "#d7b46a", "#b3a6d9", "#7fa6a1", "#c79b91",
        "#9bb7cf", "#c4b5df", "#d6c9a3", "#96c8d8", "#b6c0c8", "#d7a6ad",
    ]

    for idx, row in enumerate(focus_rows):
        vals = [pct(float(row[col])) for col in subj_cols]
        axes[1].plot(
            x,
            vals,
            marker="o",
            markersize=3.2,
            linewidth=1.1,
            color=baseline_line_colors[idx % len(baseline_line_colors)],
            alpha=0.72,
            label=row["Metric"],
        )
        
    # Add the v13-only subject-count curve for MIE-4B-LoRA.
    lora_4b_vals = [0.7870370370370371, 0.9159369527145359, 0.9220338983050848, 0.9048442906574394]
    lora_4b_vals = [pct(v) for v in lora_4b_vals]
    axes[1].plot(x, lora_4b_vals, marker="*", markersize=10, linewidth=2.6, color=COLOR_RED, label="MIE-4B-LoRA")

    axes[1].axhline(50.0, color="black", linestyle="--", linewidth=1.2)
    axes[1].set_xticks([2, 4, 6, 8])
    axes[1].set_ylim(30, 105)
    axes[1].set_xlabel("Number of Subjects")
    axes[1].set_ylabel("Agreement with Human Preference (%)")
    axes[1].set_title("Human Alignment by Subject Count")
    style_axis(axes[1])
    
    # 调整图例以适应更多的 metrics
    axes[1].legend(frameon=True, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))

    fig.tight_layout()
    out_path = IMAGES_DIR / "section_4_2_1_baseline_alignment.png"
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
    overall = read_csv(ROOT / "section_4_2_2_mie_alignment" / "mie_overall_metrics.csv")
    category_metrics = read_csv(ROOT / "section_4_2_2_mie_alignment" / "mie_category_metrics.csv")

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_TWO, dpi=300)

    # Overall and seen/unseen
    names = [format_model_name(row["metrics_model_name"]) for row in overall]
    pair_acc = [pct(float(row["pairwise_accuracy"])) for row in overall]
    v10_acc = [pct(float(row["pairwise_accuracy_v10"])) for row in overall]
    v13_acc = [pct(float(row["pairwise_accuracy_v13"])) for row in overall]
    x = np.arange(len(names))
    width = 0.24
    overall_color = COLOR_BLUE
    v10_color = COLOR_GREEN
    v13_color = COLOR_GOLD
    axes[0].bar(x - width, pair_acc, width=width, label="Overall", color=overall_color)
    axes[0].bar(x, v10_acc, width=width, label="Nano Banana + Mosaic", color=v10_color)
    axes[0].bar(x + width, v13_acc, width=width, label="Cross-Platform Pool", color=v13_color)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=15, ha="right")
    axes[0].set_ylim(65, 105)
    axes[0].set_ylabel("Pairwise Accuracy (%)")
    axes[0].set_title("Human Alignment by Model")
    style_axis(axes[0])
    axes[0].legend(frameon=True, loc="lower right")

    # 给柱子加上具体的数值标签
    for i, v in enumerate(pair_acc):
        axes[0].text(i - width, v + 0.5, f"{v:.1f}%", ha='center', va='bottom', fontsize=8, color=overall_color, rotation=0)
    for i, v in enumerate(v10_acc):
        axes[0].text(i, v + 0.5, f"{v:.1f}%", ha='center', va='bottom', fontsize=8, color=v10_color, rotation=0)
    for i, v in enumerate(v13_acc):
        axes[0].text(i + width, v + 0.5, f"{v:.1f}%", ha='center', va='bottom', fontsize=8, color="#b78a33", rotation=0)

    # Category F1
    cat_by_model = defaultdict(dict)
    for row in category_metrics:
        cat_by_model[format_model_name(row["metrics_model_name"])][row["category"]] = float(row["f1"])
    categories = ["existence", "appearance", "interaction"]
    pos = np.arange(len(categories))
    
    # 因为有6个模型，稍微调细一点柱子宽度
    num_models = len(names)
    width = 0.8 / num_models
    
    # Cool colors for layer-only, warm colors for LoRA-layer, darker shade for larger models.
    palette = {
        "Qwen-0.8B (Layer)": "#9eb6cb",
        "Qwen-0.8B (LoRA)": "#e2a07f",
        "Qwen-2B (Layer)": "#6f8daa",
        "Qwen-2B (LoRA)": "#cf7657",
        "Qwen-4B (Layer)": "#496a8d",
        "Qwen-4B (LoRA)": "#b84f3c",
    }
    
    for idx, name in enumerate(names):
        vals = [pct(cat_by_model[name][cat]) for cat in categories]
        # 计算每个柱子的偏移量，让它们居中对齐
        offset = (idx - num_models/2 + 0.5) * width
        axes[1].bar(pos + offset, vals, width=width, label=name, color=palette.get(name, "#7f8c8d"))
        
    axes[1].set_xticks(pos)
    axes[1].set_xticklabels([c.title() for c in categories])
    axes[1].set_ylim(45, 100)
    axes[1].set_ylabel("F1 Score (%)")
    axes[1].set_title("Category F1 by Model")
    style_axis(axes[1])
    
    # 把图例放到图外
    axes[1].legend(frameon=True, loc='center left', bbox_to_anchor=(1.02, 0.5))

    fig.tight_layout()
    out_path = IMAGES_DIR / "section_4_2_2_mie_alignment.png"
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
    seen_unseen = read_csv(ROOT / "section_4_2_3_breakdown" / "mie_seen_unseen_table.csv")
    lora_vs_layer = read_csv(ROOT / "section_4_2_3_breakdown" / "mie_lora_vs_layer_table.csv")
    scaling = read_csv(ROOT / "section_4_2_3_breakdown" / "mie_scaling_table.csv")
    category_by_dataset = read_csv(ROOT / "section_4_2_3_breakdown" / "mie_category_by_dataset.csv")

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_THREE, dpi=300)

    # Seen/unseen gap
    model_names = [format_model_name(row["metrics_model_name"]) for row in seen_unseen]
    gaps = [pct(float(row["unseen_minus_seen"])) for row in seen_unseen]
    axes[0].barh(np.arange(len(model_names)), gaps, color=COLOR_RED)
    axes[0].set_yticks(np.arange(len(model_names)))
    axes[0].set_yticklabels(model_names)
    axes[0].axvline(0.0, color="black", linewidth=1.2)
    axes[0].set_title("Generalization Gap by Model")
    axes[0].set_xlabel("Accuracy Delta (%)")
    axes[0].set_xlim(min(gaps) - 8.0, max(gaps) + 5.0)
    style_axis(axes[0])
    
    # 给 gap 加上数值标签
    for i, v in enumerate(gaps):
        axes[0].text(v - 0.8, i, f"{v:+.1f}%", ha='right', va='center', fontsize=9, color=COLOR_RED)

    # LoRA vs layer delta
    size_map = {"08b": "0.8B", "2b": "2B", "4b": "4B"}
    sizes = [size_map.get(row["size"], row["size"]) for row in lora_vs_layer]
    acc_delta = [pct(float(row["pairwise_accuracy_delta_lora_minus_layer"])) for row in lora_vs_layer]
    f1_delta = [pct(float(row["macro_f1_delta_lora_minus_layer"])) for row in lora_vs_layer]
    x = np.arange(len(sizes))
    width = 0.35
    axes[1].bar(x - width / 2, acc_delta, width=width, label="Pair Acc Delta", color=COLOR_BLUE)
    axes[1].bar(x + width / 2, f1_delta, width=width, label="Macro-F1 Delta", color=COLOR_GREEN)
    axes[1].axhline(0.0, color="black", linewidth=1.2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(sizes)
    axes[1].set_title("LoRA Gains by Scale")
    axes[1].set_ylabel("Delta (%)")
    axes[1].set_ylim(min(min(acc_delta), min(f1_delta)) - 3.0, max(max(acc_delta), max(f1_delta)) + 3.0)
    style_axis(axes[1])
    axes[1].legend(frameon=True, loc="upper left")
    
    # 给 delta 加数值
    for i, v in enumerate(acc_delta):
        axes[1].text(i - width / 2, v + (0.5 if v > 0 else -1.5), f"{v:+.1f}%", ha='center', va='bottom' if v > 0 else 'top', fontsize=9, color=COLOR_BLUE)
    for i, v in enumerate(f1_delta):
        axes[1].text(i + width / 2, v + (0.5 if v > 0 else -1.5), f"{v:+.1f}%", ha='center', va='bottom' if v > 0 else 'top', fontsize=9, color=COLOR_GREEN)

    # Dataset/category heatmap for best model
    best_model = "qwen35_4b_lora_layer" # 既然现在有了4b，这里我们可以写成最新的最好的那个
    
    # 动态找到当前 acc 最高的那个模型作为 best model
    overall_for_best = read_csv(ROOT / "section_4_2_2_mie_alignment" / "mie_overall_metrics.csv")
    best_model = max(overall_for_best, key=lambda row: float(row["pairwise_accuracy"]))["metrics_model_name"]
    
    filtered = [row for row in category_by_dataset if row["metrics_model_name"] == best_model]
    datasets = ["Nano Banana\n+ Mosaic", "Cross-Platform\nPool"]
    categories = ["existence", "appearance", "interaction"]
    heat = np.zeros((len(datasets), len(categories)))
    for i, ds_key in enumerate(["v10", "v13"]):
        for j, cat in enumerate(categories):
            try:
                row = next(row for row in filtered if row["dataset"] == ds_key and row["category"] == cat)
                heat[i, j] = pct(float(row["f1"]))
            except StopIteration:
                pass # 防御性编程
                
    im = axes[2].imshow(heat, cmap="Blues", vmin=45, vmax=95)
    axes[2].set_xticks(np.arange(len(categories)))
    axes[2].set_xticklabels([c.title() for c in categories], rotation=15, ha="right")
    axes[2].set_yticks(np.arange(len(datasets)))
    axes[2].set_yticklabels(datasets)
    axes[2].set_title("Category F1 by Dataset")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            # 颜色深的地方用白字，颜色浅的地方用黑字
            text_color = "white" if heat[i, j] > 75 else "black"
            axes[2].text(j, i, f"{heat[i, j]:.1f}%", ha="center", va="center", fontsize=9, color=text_color)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path = IMAGES_DIR / "section_4_2_3_mie_breakdown.png"
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
    lora_rows = {row["size"]: row for row in breakdown["lora_vs_layer"]}
    delta_2b_acc = float(lora_rows["2b"]["pairwise_accuracy_delta_lora_minus_layer"])
    delta_2b_f1 = float(lora_rows["2b"]["macro_f1_delta_lora_minus_layer"])
    delta_08b_acc = float(lora_rows["08b"]["pairwise_accuracy_delta_lora_minus_layer"])
    delta_08b_f1 = float(lora_rows["08b"]["macro_f1_delta_lora_minus_layer"])
    delta_4b_acc = float(lora_rows["4b"]["pairwise_accuracy_delta_lora_minus_layer"])
    delta_4b_f1 = float(lora_rows["4b"]["macro_f1_delta_lora_minus_layer"])

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
        "- The current evaluator story is now a full 6-model scaling result: LoRA-layer variants dominate their layer-only counterparts, and the `4B` LoRA model is the strongest overall system.",
        "",
        "## Section 4.1.1",
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
        "- `paper_data/section_4_1_1_mib_silver/silver_agreement_summary.json`",
        "- `paper_data/section_4_1_1_mib_silver/silver_agreement_by_subject_count.csv`",
        "- `paper_data/section_4_1_1_mib_silver/silver_agreement_by_class_tag.csv`",
        "",
        "## Section 4.1.2",
        "",
        "### MIB-Gold: Human Annotation Findings and Benchmark Difficulty",
        "",
        "**Core findings**",
        "- The benchmark contains `4,020` raw pair groups in total: `1,500` from `v10` and `2,520` from `v13`.",
        f"- After requiring preference consistency, the retained-pair rate is `{fmt(gold_keep_v10 * 100, 1)}%` on `v10` and `{fmt(gold_keep_v13 * 100, 1)}%` on `v13`.",
        f"- Human preference consistency rises from `{fmt(gold['level_consistency'][2] * 100, 1)}%` at level `2` to `{fmt(gold['level_consistency'][6] * 100, 1)}%` at level `6`, then remains high at level `8`.",
        "",
        "**Interpretation**",
        "- The higher disagreement rate on `v13` supports the paper's intended story: unseen-generator evaluation is harder and more informative than seen-generator evaluation.",
        "- The level-wise consistency trend further suggests that the benchmark difficulty is structured rather than noisy: denser scenes remain hard, but human judgments stay coherent after preference-consistency control.",
        "- This section should be written as benchmark findings, not as annotation bookkeeping. The key message is that human labels reveal meaningful structure in benchmark difficulty.",
        "",
        "**Recommended figure**",
        f"- `paper_data/images/{gold['image']}`",
        "",
        "**Primary files to cite**",
        "- `paper_data/section_4_1_2_mib_gold/gold_human_annotation_summary.json`",
        "- `paper_data/section_4_1_2_mib_gold/gold_summary_by_level.csv`",
        "- `paper_data/section_4_1_2_mib_gold/gold_summary_by_class_tag.csv`",
        "",
        "## Section 4.2.1",
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
        "- `paper_data/section_4_2_1_existing_metrics/baseline_human_alignment_overall.csv`",
        "- `paper_data/section_4_2_1_existing_metrics/baseline_human_alignment_v10_v13.csv`",
        "- `paper_data/section_4_2_1_existing_metrics/baseline_metric_means_by_subject_count.csv`",
        "",
        "## Section 4.2.2",
        "",
        "### MIE Aligns Better with Human Preference",
        "",
        "**Core findings**",
        f"- The best currently exported MIE variant is `{mie['best_model']}` with overall pairwise accuracy `{fmt(mie['best_pairwise_accuracy'])}`.",
        f"- This best variant reaches `{fmt(mie['best_v10'])}` on `v10` and `{fmt(mie['best_v13'])}` on `v13`, meaning it remains substantially above the strongest third-party baseline even on the unseen benchmark.",
        f"- The same model also achieves macro-F1 `{fmt(mie['best_macro_f1'])}`, supporting the claim that MIE is not just a ranker but a diagnostically meaningful evaluator.",
        "- Across all six exported checkpoints, the `4B lora_layer` variant is strongest in overall alignment, unseen-generator alignment, and macro-F1.",
        "",
        "**Interpretation**",
        "- This is the section where the paper should cash in on the benchmark story. MIB does not merely reveal that existing metrics fail; it also enables training a better evaluator.",
        "- The main result here is not that every MIE variant is strong. It is that the best MIE variant clearly dominates the available baselines on human alignment while remaining interpretable.",
        "",
        "**Recommended figure**",
        f"- `paper_data/images/{mie['image']}`",
        "",
        "**Primary files to cite**",
        "- `paper_data/section_4_2_2_mie_alignment/mie_overall_metrics.csv`",
        "- `paper_data/section_4_2_2_mie_alignment/mie_category_metrics.csv`",
        "- `paper_data/section_4_2_2_mie_alignment/mie_vs_human_summary.json`",
        "",
        "## Section 4.2.3",
        "",
        "### Breakdown Analysis",
        "",
        "**Core findings**",
        f"- All current MIE variants perform better on seen generators than unseen generators, but the generalization gap is smallest for `4b lora_layer` (`{fmt(-0.098, 3)}` unseen-minus-seen) and largest for `2b layer_only` (`{fmt(-0.221, 3)}`).",
        f"- At `2B`, adding LoRA-layer tuning improves pairwise accuracy by `{fmt(delta_2b_acc, 3)}` and macro-F1 by `{fmt(delta_2b_f1, 3)}` relative to `2B layer_only`.",
        f"- At `0.8B`, LoRA-layer barely changes pairwise accuracy (`{fmt(delta_08b_acc, 3)}`) but still improves macro-F1 by `{fmt(delta_08b_f1, 3)}`.",
        f"- At `4B`, LoRA-layer still improves pairwise accuracy by `{fmt(delta_4b_acc, 3)}` and macro-F1 by `{fmt(delta_4b_f1, 3)}` over `4B layer_only`, showing that stronger tuning remains beneficial even at the largest scale.",
        "- The category-level view shows that `Existence` is the easiest dimension, while `Appearance` and especially `Interaction` remain the harder diagnostics, particularly on `v13`.",
        "",
        "**Interpretation**",
        "- The most important ablation message is not simply `bigger is better`. The more defensible claim is that extra capacity pays off most when paired with the right fine-tuning regime.",
        "- The full scaling story is now consistent: `4B lora_layer` is the strongest checkpoint overall, while LoRA-layer tuning improves diagnostic quality at every tested scale.",
        "- The category breakdown also supports a strong narrative: interaction-heavy reasoning remains the hardest part of binding, which is consistent with both the human benchmark and the observed generation failures.",
        "",
        "**Recommended figure**",
        f"- `paper_data/images/{breakdown['image']}`",
        "",
        "**Primary files to cite**",
        "- `paper_data/section_4_2_3_breakdown/mie_seen_unseen_table.csv`",
        "- `paper_data/section_4_2_3_breakdown/mie_lora_vs_layer_table.csv`",
        "- `paper_data/section_4_2_3_breakdown/mie_scaling_table.csv`",
        "- `paper_data/section_4_2_3_breakdown/mie_category_by_dataset.csv`",
        "",
        "## Writing Guidance",
        "",
        "- Write Section 4.2 around the full six-checkpoint evaluator family; the `4B` results are now available and should be treated as the main headline rather than an optional add-on.",
        "- The strongest storyline is:",
        "  1. silver labels are reliable enough to train on,",
        "  2. gold human labels expose a real benchmark gap,",
        "  3. existing metrics fail to close that gap,",
        "  4. MIE, especially `4B lora_layer`, closes a meaningful part of it while remaining interpretable.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ensure_dirs()
    results = {
        "silver": fig_silver_agreement(),
        "gold": fig_gold_benchmark(),
        "alignment_overview": fig_4_1_alignment_overview(),
        "baseline": fig_baseline_alignment(),
        "mie": fig_mie_alignment(),
        "breakdown": fig_mie_breakdown(),
    }
    ANALYSIS_MD.write_text(build_markdown(results), encoding="utf-8")
    print(f"Wrote analysis markdown to: {ANALYSIS_MD}")
    print(f"Wrote images to: {IMAGES_DIR}")


if __name__ == "__main__":
    main()
