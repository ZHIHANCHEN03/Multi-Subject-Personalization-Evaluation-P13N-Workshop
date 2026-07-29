"""Build the selective-correction (gating) figure for the paper's Sec. 4.5.

Recomputes the numbers from the committed FLUX.2 records -- no hard-coded
values -- and writes a single-column vector figure for main.tex.

    python3 plot_gating.py                    # -> ../paper/figures/gating.pdf (+ .png preview)

Story the figure has to carry: MIDC is a *selective* fixer. It triggers on the
harder minority of rows and cuts SCR sharply there; on the majority it declines
to act and lands on top of one-shot.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FLUX2 = os.path.join(HERE, "results_flux2")
OUT_DIR = os.path.join(HERE, os.pardir, "paper", "figures")

SEEDS = (0, 1, 2)
METHODS = [("oneshot", "one-shot"), ("bon", "best-of-8"), ("ours", "MIDC (ours)")]

# Categorical slots 1-3 of the validated reference palette. Validated with
# scripts/validate_palette.js --pairs all --mode light: all checks pass (worst
# CVD dE 9.2, worst normal-vision dE 24.0). The aqua contrast WARN is
# discharged by the relief rule -- every bar carries a visible value label.
COLORS = {"oneshot": "#2a78d6", "bon": "#eb6834", "ours": "#1baf7a"}
# Orange and aqua sit at nearly the same lightness, so they collapse onto each
# other in a grayscale printout. One sparse, hairline hatch on best-of-8 breaks
# that tie without turning the figure into texture noise.
HATCH = {"oneshot": "", "bon": "//", "ours": ""}

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"


def read_records(path):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def gating_stats(entity):
    """Pool over (task, seed) rows; split by whether MIDC accepted a step."""
    rows = defaultdict(dict)
    for key, _ in METHODS:
        for s in SEEDS:
            for r in read_records(os.path.join(FLUX2, f"flux2_{entity}_{key}_s{s}", "records.jsonl")):
                rows[(r["task_id"], s)][key] = r

    triggered = {k: v for k, v in rows.items() if v.get("ours", {}).get("accepted_steps", 0) > 0}
    noop = {k: v for k, v in rows.items() if k not in triggered}
    total = len(rows)

    out = {}
    for name, subset in (("triggered", triggered), ("no-op", noop)):
        out[name] = {
            "n": len(subset),
            "pct": 100 * len(subset) / total if total else float("nan"),
            **{key: statistics.mean([v[key]["scr"] for v in subset.values() if key in v])
               for key, _ in METHODS},
        }
    return out


def build(stats_by_entity, out_pdf, out_png):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 7,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,   # embed TrueType -- AAAI requires Type 1/TrueType
        "ps.fonttype": 42,
        "hatch.linewidth": 0.35,
    })

    # Single AAAI column is 3.3in. The paper sits exactly on the 7-page content
    # limit, so this figure has to be no taller than the table it replaces.
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.78), sharey=True)
    subsets = ["triggered", "no-op"]
    x = np.arange(len(subsets))
    width = 0.24

    for ax, entity in zip(axes, (6, 8)):
        st = stats_by_entity[entity]
        for i, (key, label) in enumerate(METHODS):
            vals = [st[s][key] for s in subsets]
            bars = ax.bar(
                x + (i - 1) * width, vals, width * 0.92,
                label=label, color=COLORS[key], hatch=HATCH[key],
                edgecolor="white", linewidth=0.6, zorder=3,
            )
            for rect, v in zip(bars, vals):
                ax.text(rect.get_x() + rect.get_width() / 2, v + 0.010, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=4.4, color=INK, zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels([f"triggered\n{st[s]['n']} rows ({st[s]['pct']:.0f}%)" if s == "triggered"
                            else f"no-op\n{st[s]['n']} rows ({st[s]['pct']:.0f}%)"
                            for s in subsets], fontsize=6, color=INK)
        ax.set_title(f"{entity} entities", fontsize=7, color=INK, pad=3)
        ax.set_ylim(0, 0.82)
        ax.tick_params(axis="both", length=2, width=0.6, colors=INK_MUTED, labelsize=6)
        ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)

        # Call out the gap that carries the story: on the triggered rows MIDC
        # drops well below the base generator. Drawn in the gutter between the
        # two groups so it cannot collide with a bar or a value label.
        top, bot = st["triggered"]["oneshot"], st["triggered"]["ours"]
        gx = width + 0.20
        ax.plot([0.0, gx], [top, top], color=GRID, linewidth=0.5,
                linestyle=(0, (2, 2)), zorder=1)
        ax.annotate("", xy=(gx, bot), xytext=(gx, top), zorder=4,
                    arrowprops=dict(arrowstyle="<|-|>", color=INK_MUTED, linewidth=0.5,
                                    mutation_scale=4, shrinkA=0, shrinkB=0))
        ax.text(gx, top + 0.018, f"$-${top - bot:.3f}", fontsize=5.4,
                color=INK_MUTED, ha="center", va="bottom", zorder=4)

    axes[0].set_ylabel("SCR  (lower is better)", fontsize=6.5, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.005),
                     ncol=3, frameon=False, fontsize=6.3, handlelength=1.3,
                     handleheight=0.9, columnspacing=1.1, handletextpad=0.4)
    for t in leg.get_texts():
        t.set_color(INK)

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.subplots_adjust(wspace=0.10)
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_png, dpi=400, bbox_inches="tight", pad_inches=0.01)
    print(f"wrote {out_pdf}\nwrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=os.path.normpath(os.path.join(OUT_DIR, "gating.pdf")))
    ap.add_argument("--png", default=os.path.normpath(os.path.join(OUT_DIR, "gating.png")))
    args = ap.parse_args()

    stats = {e: gating_stats(e) for e in (6, 8)}
    for e, st in stats.items():
        for s, v in st.items():
            print(f"{e}-entity {s:9} n={v['n']:3} ({v['pct']:.0f}%)  "
                  + "  ".join(f"{lab} {v[k]:.3f}" for k, lab in METHODS))
    build(stats, args.pdf, args.png)


if __name__ == "__main__":
    main()
