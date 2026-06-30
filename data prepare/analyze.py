"""Analysis & presentation for the DPO pair selection.

Reads the outputs of data_prepare.py:
    selected_pairs_{label}.jsonl
    sampling_stats_{label}.json

Produces, per size label:
    figures/dashboard_{label}.png   multi-panel visual summary
    figures/funnel_{label}.png      selection funnel
    figures/grid_{label}.png        (N x primary_dim) selected vs pool heatmaps
    report_{label}.md               paper-ready tables

Run:  .venv/bin/python analyze.py
"""

import argparse
import json
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIMS = ["existence", "appearance", "interaction"]
DIM_SHORT = {"existence": "exist", "appearance": "appear", "interaction": "interact"}
SUBJECT_COUNTS = [2, 4, 6, 8]
COLORS = {"existence": "#4C72B0", "appearance": "#DD8452", "interaction": "#55A868"}


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- plots

def plot_funnel(stats, label, out):
    steps = [
        ("Paired\n(both sources)", stats["total_paired"]),
        ("Winner agree", stats["total_paired"] - stats["drop_disagree"]),
        ("Strong signal", stats["total_paired"] - stats["drop_disagree"] - stats["drop_weak"]),
        ("Kept (Layer 0)", stats["kept_after_layer0"]),
        (f"Selected ({label})", stats["total_selected"]),
    ]
    names = [s[0] for s in steps]
    vals = [s[1] for s in steps]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(range(len(vals)), vals, color="#4C72B0")
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("number of pairs")
    ax.set_title(f"Selection funnel — {label}")
    for b, v in zip(bars, vals):
        ax.text(v + max(vals) * 0.01, b.get_y() + b.get_height() / 2,
                f"{v:,}", va="center", fontsize=9)
    ax.set_xlim(0, max(vals) * 1.12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_grid(stats, label, out):
    cells = stats["cells"]
    sel = np.array([[cells[f"N{n}_{d}"]["selected"] for d in DIMS] for n in SUBJECT_COUNTS])
    pool = np.array([[cells[f"N{n}_{d}"]["after_filter"] for d in DIMS] for n in SUBJECT_COUNTS])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, mat, title, cmap in [
        (axes[0], sel, "Selected per cell", "Blues"),
        (axes[1], pool, "Available pool (after Layer-2 filter)", "Greens"),
    ]:
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(DIMS)))
        ax.set_xticklabels([DIM_SHORT[d] for d in DIMS])
        ax.set_yticks(range(len(SUBJECT_COUNTS)))
        ax.set_yticklabels([f"N={n}" for n in SUBJECT_COUNTS])
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                ax.text(j, i, f"{v:,}", ha="center", va="center",
                        color="white" if v > mat.max() * 0.5 else "black", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"(N x primary dimension) grid — {label}", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dashboard(rows, stats, label, out):
    n_counts = Counter(r["N"] for r in rows)
    dim_counts = Counter(r["primary_dim"] for r in rows)
    winner_counts = Counter(r["winner"] for r in rows)
    s_vals = [r["S"] for r in rows]
    norm_vals = [r["delta_norm"] for r in rows]

    # stacked N x dim
    nd = defaultdict(lambda: defaultdict(int))
    for r in rows:
        nd[r["N"]][r["primary_dim"]] += 1

    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))

    # (0,0) N bucket distribution
    ax = axs[0, 0]
    vals = [n_counts[n] for n in SUBJECT_COUNTS]
    ax.bar([f"N={n}" for n in SUBJECT_COUNTS], vals, color="#4C72B0")
    tot = sum(vals)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v}\n{v/tot*100:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_title("Layer 1: subject-count (N) buckets")
    ax.set_ylabel("pairs")
    ax.set_ylim(0, max(vals) * 1.2)

    # (0,1) primary dimension distribution
    ax = axs[0, 1]
    vals = [dim_counts[d] for d in DIMS]
    ax.bar([DIM_SHORT[d] for d in DIMS], vals, color=[COLORS[d] for d in DIMS])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v}\n{v/tot*100:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_title("Layer 3: primary dimension")
    ax.set_ylabel("pairs")
    ax.set_ylim(0, max(vals) * 1.2)

    # (0,2) winner A/B
    ax = axs[0, 2]
    wv = [winner_counts.get("A", 0), winner_counts.get("B", 0)]
    ax.bar(["A wins", "B wins"], wv, color=["#C44E52", "#8172B3"])
    for i, v in enumerate(wv):
        ax.text(i, v, f"{v}\n{v/tot*100:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_title("Winner balance")
    ax.set_ylabel("pairs")
    ax.set_ylim(0, max(wv) * 1.2)

    # (1,0) stacked N x dim
    ax = axs[1, 0]
    bottom = np.zeros(len(SUBJECT_COUNTS))
    x = np.arange(len(SUBJECT_COUNTS))
    for d in DIMS:
        vals = np.array([nd[n][d] for n in SUBJECT_COUNTS])
        ax.bar(x, vals, bottom=bottom, label=DIM_SHORT[d], color=COLORS[d])
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in SUBJECT_COUNTS])
    ax.set_title("N x primary dimension (stacked)")
    ax.set_ylabel("pairs")
    ax.legend(fontsize=8)

    # (1,1) significance S histogram
    ax = axs[1, 1]
    ax.hist(s_vals, bins=20, color="#937860", edgecolor="white")
    ax.axvline(float(np.median(s_vals)), color="black", ls="--", lw=1,
               label=f"median={np.median(s_vals):.3f}")
    ax.set_title("Significance  S = max(delta) - mean(delta)")
    ax.set_xlabel("S")
    ax.set_ylabel("pairs")
    ax.legend(fontsize=8)

    # (1,2) delta_norm histogram
    ax = axs[1, 2]
    ax.hist(norm_vals, bins=20, color="#55A868", edgecolor="white")
    ax.axvline(float(np.median(norm_vals)), color="black", ls="--", lw=1,
               label=f"median={np.median(norm_vals):.3f}")
    ax.set_title("Signal magnitude  ||delta||_1")
    ax.set_xlabel("||delta||_1")
    ax.set_ylabel("pairs")
    ax.legend(fontsize=8)

    fig.suptitle(f"DPO selection summary — {label}  (n={len(rows):,})", fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- report

def write_report(rows, stats, label, out):
    n_counts = Counter(r["N"] for r in rows)
    dim_counts = Counter(r["primary_dim"] for r in rows)
    winner_counts = Counter(r["winner"] for r in rows)
    tot = len(rows)
    cells = stats["cells"]

    nd = defaultdict(lambda: defaultdict(int))
    for r in rows:
        nd[r["N"]][r["primary_dim"]] += 1

    L = []
    L.append(f"# DPO selection report — {label}\n")

    L.append("## Selection funnel\n")
    L.append("| stage | pairs | kept % |")
    L.append("|---|---:|---:|")
    base = stats["total_paired"]
    funnel = [
        ("paired (both sources)", base),
        ("winner agree", base - stats["drop_disagree"]),
        ("strong signal", base - stats["drop_disagree"] - stats["drop_weak"]),
        ("kept after Layer 0", stats["kept_after_layer0"]),
        (f"selected ({label})", stats["total_selected"]),
    ]
    for name, v in funnel:
        L.append(f"| {name} | {v:,} | {v/base*100:.1f}% |")
    L.append("")
    L.append(f"- dropped — winner disagree: **{stats['drop_disagree']:,}**, "
             f"weak signal (‖δ‖≈0): **{stats['drop_weak']:,}**, "
             f"missing N: **{stats['drop_no_N']:,}**\n")

    L.append("## Layer 1 — subject-count (N) buckets\n")
    L.append("| N | selected | share |")
    L.append("|---:|---:|---:|")
    for n in SUBJECT_COUNTS:
        L.append(f"| {n} | {n_counts[n]} | {n_counts[n]/tot*100:.1f}% |")
    L.append("")

    L.append("## Layer 3 — primary dimension\n")
    L.append("| dimension | selected | share |")
    L.append("|---|---:|---:|")
    for d in DIMS:
        L.append(f"| {d} | {dim_counts[d]} | {dim_counts[d]/tot*100:.1f}% |")
    L.append("")

    L.append("## (N x primary dimension) grid — selected / pool after filter\n")
    L.append("| N | " + " | ".join(DIMS) + " | row total |")
    L.append("|---:|" + "|".join(["---:"] * (len(DIMS) + 1)) + "|")
    for n in SUBJECT_COUNTS:
        cellsrow = []
        rowtot = 0
        for d in DIMS:
            sel = cells[f"N{n}_{d}"]["selected"]
            pool = cells[f"N{n}_{d}"]["after_filter"]
            rowtot += sel
            mark = "" if sel * 1.0 >= stats["per_cell_target"] else " ⚠"
            cellsrow.append(f"{sel} / {pool}{mark}")
        L.append(f"| {n} | " + " | ".join(cellsrow) + f" | {rowtot} |")
    L.append("")
    L.append(f"- per-cell target = **{stats['per_cell_target']}**; "
             "⚠ marks cells that fell back to taking the full pool (scarce stratum).\n")

    L.append("## Winner balance\n")
    L.append(f"- A wins: **{winner_counts.get('A',0)}** ({winner_counts.get('A',0)/tot*100:.1f}%)  ")
    L.append(f"- B wins: **{winner_counts.get('B',0)}** ({winner_counts.get('B',0)/tot*100:.1f}%)\n")

    s_vals = [r["S"] for r in rows]
    norm_vals = [r["delta_norm"] for r in rows]
    L.append("## Signal statistics (selected set)\n")
    L.append("| metric | mean | median | min | max |")
    L.append("|---|---:|---:|---:|---:|")
    L.append(f"| significance S | {np.mean(s_vals):.3f} | {np.median(s_vals):.3f} | "
             f"{np.min(s_vals):.3f} | {np.max(s_vals):.3f} |")
    L.append(f"| ‖δ‖₁ | {np.mean(norm_vals):.3f} | {np.median(norm_vals):.3f} | "
             f"{np.min(norm_vals):.3f} | {np.max(norm_vals):.3f} |")
    L.append("")

    L.append("## Figures\n")
    L.append(f"- ![funnel](figures/funnel_{label}.png)")
    L.append(f"- ![grid](figures/grid_{label}.png)")
    L.append(f"- ![dashboard](figures/dashboard_{label}.png)\n")

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="2k,5k")
    ap.add_argument("--pairs_tmpl", default="selected_pairs_{label}.jsonl")
    ap.add_argument("--stats_tmpl", default="sampling_stats_{label}.json")
    ap.add_argument("--figdir", default="figures")
    args = ap.parse_args()

    os.makedirs(args.figdir, exist_ok=True)
    for label in [x.strip() for x in args.labels.split(",") if x.strip()]:
        rows = load_jsonl(args.pairs_tmpl.format(label=label))
        stats = load_json(args.stats_tmpl.format(label=label))
        plot_funnel(stats, label, f"{args.figdir}/funnel_{label}.png")
        plot_grid(stats, label, f"{args.figdir}/grid_{label}.png")
        plot_dashboard(rows, stats, label, f"{args.figdir}/dashboard_{label}.png")
        write_report(rows, stats, label, f"report_{label}.md")
        print(f"[{label}] n={len(rows):,} -> figures/*_{label}.png, report_{label}.md")


if __name__ == "__main__":
    main()
