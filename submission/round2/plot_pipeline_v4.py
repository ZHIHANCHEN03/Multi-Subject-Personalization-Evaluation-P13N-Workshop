#!/usr/bin/env python3
"""MIDC architecture figure (paper Figure 2), v4.

Drawn in the grammar of a conventional architecture diagram: pastel rounded
blocks on a white ground, black arrows, a grey container marking the part that
repeats, and a labelled feedback path. Colour encodes role and is reused across
the figure -- pink for the generator, orange for the two scorers, blue for the
two decisions, purple for the action, olive for the gate, green for the output.

Earlier versions either restated the text in generic boxes (v1/v2) or replaced
the architecture with a trajectory (v3). The trajectory is better placed in the
caption, which carries every number including the step the gate rejected; the
figure's job is to show the shape of the loop, which the prose cannot.

    python3 plot_pipeline_v4.py   # -> ../paper/figures/pipeline.pdf (+ .png)
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, os.pardir, "paper", "figures")
COL_IN = 3.35

INK = "#000000"
CONTAINER = "#efefef"
PINK, ORANGE, BLUE = "#f8cecc", "#ffe6cc", "#dae8fc"
PURPLE, OLIVE, GREEN = "#e1d5e7", "#e8f0c8", "#d5e8d4"

FS = 6.1     # block labels
FS_S = 5.5   # small annotations


def blk(ax, cx, cy, w, h, text, fc, fs=FS, lw=0.8):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.018",
        linewidth=lw, edgecolor=INK, facecolor=fc, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color=INK, zorder=4, linespacing=1.22)


def arw(ax, p0, p1, rad=0.0, lw=1.0, ls="-", color=INK, ms=6.5):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
        color=color, linestyle=ls, connectionstyle=f"arc3,rad={rad}",
        zorder=5, shrinkA=0.5, shrinkB=0.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=os.path.normpath(os.path.join(OUT_DIR, "pipeline.pdf")))
    ap.add_argument("--png", default=os.path.normpath(os.path.join(OUT_DIR, "pipeline.png")))
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "stix",
    })
    fig = plt.figure(figsize=(COL_IN, 2.24))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_facecolor("white")

    xl, xr = 0.315, 0.685      # the two parallel branches
    xc = (xl + xr) / 2
    hb = 0.098                 # standard block height

    # ---- grey container: everything that repeats K times --------------------
    ax.add_patch(FancyBboxPatch(
        (0.135, 0.207), 0.720, 0.575,
        boxstyle="round,pad=0.0,rounding_size=0.022",
        linewidth=0.9, edgecolor="#9a9a9a", facecolor=CONTAINER, zorder=1))
    ax.text(0.872, 0.495, "$\\times K$", ha="left", va="center",
            fontsize=6.6, color=INK, zorder=6)

    # ---- inputs -------------------------------------------------------------
    ax.text(0.315, 0.038, "prompt", ha="center", va="center", fontsize=FS, color=INK)
    ax.text(0.560, 0.038, "refs $\\{r_i\\}$", ha="center", va="center", fontsize=FS, color=INK)
    blk(ax, 0.437, 0.140, 0.250, hb, "generator $G$", PINK)
    arw(ax, (0.315, 0.062), (0.360, 0.140 - hb / 2))
    arw(ax, (0.560, 0.062), (0.515, 0.140 - hb / 2))


    # ---- scorers ------------------------------------------------------------
    y1 = 0.300
    blk(ax, xl, y1, 0.300, hb, "decomposed\nverifier $V$", ORANGE)
    blk(ax, xr, y1, 0.300, hb, "DINOv2 $+$\nGrounding-DINO", ORANGE)
    arw(ax, (0.378, 0.140 + hb / 2), (xl, y1 - hb / 2), rad=-0.10)
    arw(ax, (0.496, 0.140 + hb / 2), (xr, y1 - hb / 2), rad=0.10)

    # ---- decisions ----------------------------------------------------------
    y2 = 0.455
    blk(ax, xl, y2, 0.300, hb, "calibrated routing\n$d^\\star=\\arg\\max_d\\ \\delta_d$", BLUE)
    blk(ax, xr, y2, 0.300, hb, "dual-signal diag.\n$i^\\star=\\arg\\min_i\\ \\mathrm{sim}_i$", BLUE)
    arw(ax, (xl, y1 + hb / 2), (xl, y2 - hb / 2))
    arw(ax, (xr, y1 + hb / 2), (xr, y2 - hb / 2))

    # ---- action -------------------------------------------------------------
    y3 = 0.600
    blk(ax, xc, y3, 0.560, hb,
        "action portfolio on $(d^\\star,i^\\star)$\n"
        "reorder refs $\\;|\\;$ layout hint $\\;|\\;$ rewrite", PURPLE)
    arw(ax, (xl, y2 + hb / 2), (xc - 0.130, y3 - hb / 2), rad=-0.12)
    arw(ax, (xr, y2 + hb / 2), (xc + 0.130, y3 - hb / 2), rad=0.12)

    # ---- gate ---------------------------------------------------------------
    y4 = 0.730
    blk(ax, xc, y4, 0.440, 0.080,
        "guarded accept: $V_{\\mathrm{total}}\\uparrow$ and $\\mathrm{sim}_{i^\\star}\\uparrow$", OLIVE)
    arw(ax, (xc, y3 + hb / 2), (xc, y4 - 0.040))

    # ---- output -------------------------------------------------------------
    blk(ax, xc, 0.880, 0.330, 0.082, "corrected image", GREEN)
    arw(ax, (xc, y4 + 0.040), (xc, 0.880 - 0.041))
    ax.text(xc + 0.022, 0.812, "accept", ha="left", va="center",
            fontsize=FS_S, color=INK, zorder=6)

    # ---- reject path: back to the generator ---------------------------------
    XR = 0.045                       # the return rail, left of the container
    ax.plot([xc - 0.220, XR], [y4, y4], color=INK, lw=0.9, zorder=5,
            solid_capstyle="round")
    ax.plot([XR, XR], [y4, 0.645], color=INK, lw=0.9, zorder=5,
            solid_capstyle="round")
    ax.plot([XR, XR], [0.245, 0.140], color=INK, lw=0.9, zorder=5,
            solid_capstyle="round")
    arw(ax, (XR, 0.140), (0.437 - 0.125, 0.140), lw=0.9)
    ax.text(XR, 0.445, "reject: roll back, resample", ha="center", va="center",
            fontsize=FS_S, color=INK, rotation=90, zorder=6)

    os.makedirs(os.path.dirname(args.pdf), exist_ok=True)
    fig.savefig(args.pdf, facecolor="white", pad_inches=0.012)
    fig.savefig(args.png, dpi=400, facecolor="white", pad_inches=0.012)
    im = Image.open(args.png); w = int(COL_IN * 300)
    im.resize((w, int(im.size[1] * w / im.size[0])), Image.LANCZOS).save(
        os.path.join(OUT_DIR, "pipeline_preview.png"))
    print(f"wrote {args.pdf}")


if __name__ == "__main__":
    main()
