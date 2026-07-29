#!/usr/bin/env python3
"""OmniGen2 qualitative panels (v2), annotated per subject.

Builds both OmniGen2 comparison figures with the same 4-panel grammar
(one-shot / UMO / best-of-8 / MIDC, per-subject DINOv2 chips that SCR
thresholds at 0.5, so the reader counts collapsed subjects directly):

  figures/teaser.pdf          <- hard_054547 (seed 0), Figure 1
  figures/umo_qualitative.pdf <- hard_046947 (seed 0), experiments section

Both tasks' panels were verified against the manifest prompt and the committed
records (chips agree with what is visible in each image). hard_054547 has
-1.0 sims (subject not found by the detector), rendered as "miss".

Geometry note: authored at 7in wide but rendered in the paper at
\\columnwidth (~3.3in), i.e. ~47%. All font sizes below are therefore ~2.1x the
intended final size. Verify with the true-size preview the script writes.

Usage:  python3 plot_umo_qualitative_v2.py
Writes: ../paper/figures/{teaser,umo_qualitative}.pdf (+ .png, + _preview.png)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.image as mpimg
from PIL import Image

THRESH = 0.5          # SCR threshold on per-subject DINOv2 similarity
COL_WIDTH_IN = 3.3    # AAAI single column, for the true-size preview

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "paper", "figures")
IMG = os.path.join(HERE, "fig_assets")

METHODS = [
    ("one_shot", "one-shot"),
    ("umo", "UMO (retrained)"),
    ("best_of_n", "best-of-8"),
    ("ours_v2", "MIDC (ours)"),
]

FIGURES = [
    ("teaser", "hard_054547", 0,
     r"$s_1$ man denim jacket   $s_2$ deer   "
     r"$s_3$ donut   $s_4$ spaghetti"),
    ("umo_qualitative", "hard_046947", 0,
     r"$s_1$ elderly black woman   $s_2$ man bomber jacket   "
     r"$s_3$ man black suit   $s_4$ rabbit"),
]

INK, GREY = "#1a1a1a", "#8a8a8a"
FAIL, PASS, HILITE = "#c0392b", "#2e7d32", "#1f4e79"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load_records(task, seed):
    out = {}
    for key, _ in METHODS:
        path = os.path.join(HERE, "results_r2", "merged", f"{key}_s{seed}", "records.jsonl")
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                if r["task_id"] == task:
                    out[key] = r
                    break
    missing = [k for k, _ in METHODS if k not in out]
    if missing:
        raise SystemExit(f"no record for {missing} on {task} seed {seed}")
    return out


def build(out_name, task, seed, legend):
    rec = load_records(task, seed)
    fig_w, fig_h = 7.0, 3.15
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    # vertical layout (figure fraction), top -> bottom
    Y_NAME, Y_SCR = 0.955, 0.884
    IMG_Y, IMG_H = 0.335, 0.520
    CHIP_Y, CHIP_H = 0.212, 0.078
    Y_SLBL, Y_VERD, Y_LEGEND = 0.176, 0.098, 0.028

    pan_w, gap = 0.212, 0.026
    px0 = 0.5 - (4 * pan_w + 3 * gap) / 2

    for j, (key, name) in enumerate(METHODS):
        r = rec[key]
        sims, scr = r["dino_sims"], r["scr"]
        n_bad = sum(s < THRESH for s in sims)
        ours = key == "ours_v2"
        x = px0 + j * (pan_w + gap)

        if ours:  # highlight band behind our column, drawn under everything
            fig.patches.append(FancyBboxPatch(
                (x - 0.013, Y_VERD - 0.038), pan_w + 0.026, (Y_NAME + 0.038) - (Y_VERD - 0.038),
                boxstyle="round,pad=0.003,rounding_size=0.008",
                transform=fig.transFigure, facecolor="#eef3f9",
                edgecolor=HILITE, linewidth=1.1, zorder=-5))

        fig.text(x + pan_w / 2, Y_NAME, name, ha="center", va="top",
                 fontsize=17.0, color=HILITE if ours else INK,
                 fontweight="bold" if ours else "normal")
        fig.text(x + pan_w / 2, Y_SCR, f"SCR {scr:.2f}", ha="center", va="top",
                 fontsize=15.5, color=PASS if n_bad == 0 else FAIL,
                 fontweight="bold" if n_bad == 0 else "normal")

        ax = fig.add_axes([x, IMG_Y, pan_w, IMG_H])
        ax.set_zorder(5)
        p = os.path.join(IMG, f"{task}__{key}.png")
        if os.path.exists(p):
            ax.imshow(mpimg.imread(p))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(HILITE if ours else GREY)
            sp.set_linewidth(1.2 if ours else 0.6)

        # per-subject chips -- what SCR is actually thresholded from
        cw, cgap = pan_w / 4.42, 0.0055
        cx0 = x + (pan_w - (4 * cw + 3 * cgap)) / 2
        for i, sim in enumerate(sims):
            bad = sim < THRESH
            cx = cx0 + i * (cw + cgap)
            fig.patches.append(Rectangle(
                (cx, CHIP_Y), cw, CHIP_H, transform=fig.transFigure,
                facecolor=(FAIL if bad else PASS), alpha=0.14,
                edgecolor=(FAIL if bad else PASS), linewidth=0.8, zorder=2))
            label = "miss" if sim < 0 else f"{sim:.2f}"
            fig.text(cx + cw / 2, CHIP_Y + CHIP_H / 2, label,
                     ha="center", va="center", fontsize=12.5, zorder=3,
                     color=(FAIL if bad else PASS),
                     fontweight="normal" if bad else "bold")
            fig.text(cx + cw / 2, Y_SLBL, f"$s_{i+1}$", ha="center", va="top",
                     fontsize=10.5, color=GREY, zorder=3)

        fig.text(x + pan_w / 2, Y_VERD,
                 "4/4 retained" if n_bad == 0 else f"{n_bad}/4 collapsed",
                 ha="center", va="center", fontsize=13.0,
                 color=(PASS if n_bad == 0 else FAIL),
                 fontweight="bold" if n_bad == 0 else "normal")

    fig.text(0.5, Y_LEGEND, legend, ha="center", va="center",
             fontsize=11.0, color=GREY)

    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{out_name}.{ext}"), dpi=300,
                    facecolor="white", bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)

    # true-size preview: exactly how it looks at \columnwidth, 300 dpi print
    png = os.path.join(OUT, f"{out_name}.png")
    im = Image.open(png)
    w = int(COL_WIDTH_IN * 300)
    im.resize((w, int(im.size[1] * w / im.size[0])), Image.LANCZOS).save(
        os.path.join(OUT, f"{out_name}_preview.png"))

    print("wrote", os.path.abspath(os.path.join(OUT, f"{out_name}.pdf")))
    print(f"  final size in paper: {COL_WIDTH_IN}in x "
          f"{COL_WIDTH_IN * fig_h / fig_w:.2f}in")
    for key, name in METHODS:
        r = rec[key]
        print(f"  {name:18s} SCR {r['scr']:.2f}  DINO {r['dino_mean']:.3f}  "
              f"sims {[round(s, 3) for s in r['dino_sims']]}")


def main():
    for out_name, task, seed, legend in FIGURES:
        build(out_name, task, seed, legend)


if __name__ == "__main__":
    main()
