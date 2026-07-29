"""Build the MIDC pipeline figure (paper Figure 2).

Replaces a plain box-and-arrow TikZ diagram. The problem with that version was
that it *named* the mechanism without showing it: a reader saw boxes labelled
"Calibrated Routing" and "Dual-Signal Diagnosis" but could not see what routing
decides or what the second signal adds. Here the two decisions are drawn as the
small bar charts they actually are, with real numbers from one task
(`hard_058139`), so the figure carries the method rather than restating the text.

Numbers come from the committed record for that task, not from prose, so they
cannot drift out of sync with the results.

    python3 plot_pipeline.py     # -> ../paper/figures/pipeline.pdf (+ .png)
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, os.pardir, "paper", "figures")
RECORD = os.path.join(HERE, "results_r2", "merged", "ours_v2_s0", "records.jsonl")
TASK = "hard_058139"

FACETS = ("existence", "appearance", "interaction")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
ACCENT = "#2a78d6"     # the routed facet / the selected subject
NEUTRAL = "#b9c6d6"    # everything not selected
GOOD = "#1baf7a"


def load_case() -> dict:
    with open(RECORD) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            r = json.loads(line)
            if r.get("task_id") == TASK:
                acc = next(s for s in r["step_log"] if s.get("accepted"))
                # The record does not store the pre-correction per-subject sims,
                # and r["dino_sims"] is the *accepted* candidate's -- whose argmin
                # need not be the diagnosed subject. Use the first (rejected)
                # candidate's sims instead: that array is recorded, and its argmin
                # is exactly weak_subject, so the panel shows the real decision.
                first = r["step_log"][0]
                return {
                    "deficits": acc["deficits_before"],
                    "routed": acc["routed_dim"],
                    "weak": acc["weak_subject"],
                    "action": acc["action"],
                    "sims_before": first["candidate_sims"],
                    "sims_after": r["dino_sims"],
                    "init_total": r["init_total"],
                    "final_total": r["final_total"],
                    "scr": r["scr"],
                }
    raise SystemExit(f"{TASK} not found in {RECORD}")


def box(ax, x, y, w, h, text, fc="#f2f4f7", fs=4.6, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                linewidth=0.5, edgecolor=GRID, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, zorder=3, linespacing=1.25, fontweight=weight)


def arrow(ax, p0, p1, rad=0.0, color=MUTED, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=4.5,
                                 linewidth=0.55, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=4,
                                 shrinkA=1.5, shrinkB=1.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=os.path.normpath(os.path.join(OUT_DIR, "pipeline.pdf")))
    ap.add_argument("--png", default=os.path.normpath(os.path.join(OUT_DIR, "pipeline.png")))
    args = ap.parse_args()
    c = load_case()

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(3.35, 1.62))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---------------- top row: the loop ----------------
    yb, hb = 0.775, 0.185
    box(ax, 0.005, yb, 0.135, hb, "prompt\n$+$ refs", fc="#ffffff")
    box(ax, 0.165, yb, 0.125, hb, "generator\n$G$")
    box(ax, 0.315, yb, 0.145, hb, "decomposed\nverifier $V$")
    box(ax, 0.485, yb, 0.165, hb, "action\nportfolio")
    box(ax, 0.675, yb, 0.15, hb, "guarded\naccept?")
    box(ax, 0.85, yb, 0.145, hb, "corrected\nimage", fc="#ffffff")
    for a, b in ((0.140, 0.165), (0.290, 0.315), (0.650, 0.675), (0.825, 0.850)):
        arrow(ax, (a, yb + hb / 2), (b, yb + hb / 2))
    # reject path folds back to the generator
    arrow(ax, (0.75, yb), (0.2275, yb), rad=-0.30, ls=(0, (2.2, 1.6)))
    ax.text(0.49, yb - 0.135, "reject: roll back, resample", fontsize=3.9,
            color=MUTED, ha="center", style="italic")

    # ---------------- the two decisions, drawn ----------------
    # (a) calibrated routing over facet deficits
    axr = fig.add_axes([0.075, 0.145, 0.335, 0.40])
    vals = [c["deficits"][f] for f in FACETS]
    colors = [ACCENT if f == c["routed"] else NEUTRAL for f in FACETS]
    axr.bar(range(3), vals, 0.62, color=colors, edgecolor="white", linewidth=0.4, zorder=3)
    axr.axhline(0, color=GRID, linewidth=0.5, zorder=1)
    axr.set_xticks(range(3))
    axr.set_xticklabels(["exist.", "appear.", "interact."], fontsize=4.0, color=INK)
    axr.tick_params(axis="both", length=1.5, width=0.4, labelsize=3.8, colors=MUTED)
    for s in ("top", "right"):
        axr.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axr.spines[s].set_color(GRID); axr.spines[s].set_linewidth(0.5)
    axr.set_title("calibrated routing:  $\\arg\\max_d\\ \\delta_d$",
                  fontsize=4.3, color=INK, pad=2)
    axr.set_ylabel("deficit $\\delta_d$", fontsize=4.0, color=MUTED, labelpad=1)
    hi = FACETS.index(c["routed"])
    axr.set_ylim(min(vals) * 1.25, max(vals) * 1.9)
    axr.annotate(f"routed here", xy=(hi, vals[hi]), xytext=(hi - 1.05, max(vals) * 1.45),
                 fontsize=3.8, color=ACCENT, ha="center",
                 arrowprops=dict(arrowstyle="-|>", color=ACCENT,
                                 linewidth=0.45, mutation_scale=3.5))

    # (b) dual-signal diagnosis over per-subject identity
    axs = fig.add_axes([0.605, 0.145, 0.345, 0.40])
    sims = c["sims_before"]
    w = c["weak"]
    assert sims.index(min(sims)) == w, "argmin must be the diagnosed subject"
    axs.bar(range(len(sims)), sims, 0.62,
            color=[ACCENT if i == w else NEUTRAL for i in range(len(sims))],
            edgecolor="white", linewidth=0.4, zorder=3)
    axs.set_xticks(range(len(sims)))
    axs.set_xticklabels([f"$s_{i+1}$" for i in range(len(sims))], fontsize=4.0, color=INK)
    axs.set_ylim(-1.2, 1.25)
    axs.axhline(0, color=GRID, linewidth=0.5, zorder=1)
    axs.tick_params(axis="both", length=1.5, width=0.4, labelsize=3.8, colors=MUTED)
    for s in ("top", "right"):
        axs.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axs.spines[s].set_color(GRID); axs.spines[s].set_linewidth(0.5)
    axs.set_title("dual signal:  $\\arg\\min_i\\ \\mathrm{sim}_i$",
                  fontsize=4.3, color=INK, pad=2)
    axs.set_ylabel("identity sim.", fontsize=4.0, color=MUTED, labelpad=1)
    axs.annotate("collapsed\nsubject\n(not detected)", xy=(w, sims[w]),
                 xytext=(w - 1.0, -0.75), fontsize=3.8, color=ACCENT, ha="center",
                 arrowprops=dict(arrowstyle="-|>", color=ACCENT, linewidth=0.45,
                                 mutation_scale=3.5))

    # ---------------- the resulting step, with real numbers ----------------
    after = ", ".join(f"{v:.2f}" for v in c["sims_after"])
    ax.text(0.5, 0.005,
            f"route {c['routed']}  $\\rightarrow$  act {c['action'].replace('_', '+')}  "
            f"$\\rightarrow$  accepted: verifier total {c['init_total']:.2f} to "
            f"{c['final_total']:.2f}, all four subjects recovered ({after}), SCR "
            f"{c['scr']:.2f}",
            fontsize=3.9, color=INK, ha="center", va="bottom")

    os.makedirs(os.path.dirname(args.pdf), exist_ok=True)
    fig.savefig(args.pdf, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(args.png, dpi=400, bbox_inches="tight", pad_inches=0.01)
    print(f"wrote {args.pdf}\nwrote {args.png}")
    print(f"case: routed={c['routed']} weak={c['weak']} action={c['action']} "
          f"total {c['init_total']:.3f}->{c['final_total']:.3f} scr={c['scr']}")


if __name__ == "__main__":
    main()
