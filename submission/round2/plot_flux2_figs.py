#!/usr/bin/env python3
"""FLUX.2 comparison figure (paper Figure 4 grid).

NOTE: the Figure 1 teaser is no longer built here -- as of 2026-07-28 it is an
OmniGen2 one-shot/UMO/MIDC panel (hard_043091) built by plot_teaser_omnigen.py,
because the old teaser duplicated this grid's top row (hard_048829). Do not
re-enable build_teaser here: it would overwrite the new teaser.pdf.

Annotation grammar matches Figure 3: a per-subject strip under each image, one
cell per subject, red when that subject's DINOv2 similarity falls below the 0.5
threshold SCR is defined by. SCR is then the fraction of red cells, so the
comparison is countable rather than asserted.

Layout is computed in inches and each panel's axes is given exactly the source
image's aspect ratio, so the image fills its axes with no letterboxing and the
strip, the header and the picture share the same left and right edges. (An
earlier version laid out in figure fractions; square axes letterboxed the
cropped images and the strip ended up wider than the picture above it -- the
misalignment was visible.) The figure is saved without `bbox_inches="tight"`,
so the authored aspect ratio is exactly what lands on the page.

The teaser prints the similarity inside each cell; the grid does not, because
at three panels a cell is ~9pt across at \\columnwidth and the digits would be
unreadable. The grid's numbers are quoted in its caption.

All values are read from the committed records; no number is typed by hand.

    python3 plot_flux2_figs.py   # -> ../paper/figures/{teaser,results_grid}.pdf
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.pardir, "paper", "figures")
IMG = os.path.join(HERE, "fig_assets")
THRESH = 0.5

# printed width of each figure (the \includegraphics width in main.tex)
TEASER_W_IN = 0.92 * 3.3
GRID_W_IN = 0.90 * 3.3

# (task, seed): each row chosen so a *single* seed is monotonic across methods,
# so the panels of a row differ only in method.
GRID = [("hard_048829", 1), ("hard_047461", 2), ("hard_057505", 1)]

INK, GREY = "#1a1a1a", "#8a8a8a"
FAIL, PASS, HILITE = "#c0392b", "#2e7d32", "#1f4e79"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "stix",
})


def rec(task, seed, method):
    path = os.path.join(HERE, "results_flux2", f"flux2_8_{method}_s{seed}",
                        "records.jsonl")
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r["task_id"] == task:
                return r
    raise SystemExit(f"{task} not in {path}")


def aspect(task, method="oneshot"):
    im = Image.open(os.path.join(IMG, f"crop_{method}_{task}.png"))
    return im.size[0] / im.size[1]


class Canvas:
    """Inch-based layout on a fixed-size figure; nothing is auto-cropped."""

    def __init__(self, w_in, h_in, printed_w_in):
        self.W, self.H = w_in, h_in
        self.fig = plt.figure(figsize=(w_in, h_in))
        self.fig.patch.set_facecolor("white")
        self.k = printed_w_in / w_in           # authored -> printed scale
        self.printed_h = h_in * self.k

    def pt(self, printed_pt):
        """Authored point size that prints at `printed_pt`."""
        return printed_pt / self.k

    def fx(self, x): return x / self.W
    def fy(self, y): return y / self.H

    def text(self, x, y, s, printed_pt, **kw):
        self.fig.text(self.fx(x), self.fy(y), s, fontsize=self.pt(printed_pt), **kw)

    def rect(self, x, y, w, h, **kw):
        self.fig.patches.append(Rectangle(
            (self.fx(x), self.fy(y)), self.fx(w), self.fy(h),
            transform=self.fig.transFigure, **kw))

    def round_rect(self, x, y, w, h, **kw):
        self.fig.patches.append(FancyBboxPatch(
            (self.fx(x), self.fy(y)), self.fx(w), self.fy(h),
            boxstyle="round,pad=0,rounding_size=0.008",
            transform=self.fig.transFigure, **kw))

    def axes(self, x, y, w, h):
        return self.fig.add_axes([self.fx(x), self.fy(y), self.fx(w), self.fy(h)])

    def save(self, name):
        for ext in ("pdf", "png"):
            self.fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300,
                             facecolor="white")
        im = Image.open(os.path.join(OUT, f"{name}.png"))
        w = int(self.k * self.W * 300)
        im.resize((w, int(im.size[1] * w / im.size[0])), Image.LANCZOS).save(
            os.path.join(OUT, f"{name}_preview.png"))
        plt.close(self.fig)


def draw_panel(c, x, y, pw, ph, task, seed, method, name, *,
               ours=False, show_name=True, cells=False, note=False,
               strip_h=0.16, note_gap=0.05, strip_gap=0.055,
               dy_name=0.57, dy_scr=0.21, dy_note=0.17,
               pt_name=8.4, pt_scr=7.8, pt_cell=5.6, pt_note=6.2):
    """One method panel: header, image (axes aspect == image aspect), strip."""
    r = rec(task, seed, method)
    sims, scr = r["dino_sims"], r["scr"]
    nbad = sum(s < THRESH for s in sims)
    strip_y = y - strip_gap - strip_h

    # every panel gets the same frame, so the columns are structurally
    # identical and only the colour marks ours; an earlier version boxed only
    # ours and the asymmetry read as misalignment.
    pad = 0.075
    bot = (strip_y - note_gap - dy_note - 0.14) if note else (strip_y - 0.075)
    top = y + ph + (dy_name + 0.20 if show_name else dy_scr + 0.19)
    c.round_rect(x - pad, bot, pw + 2 * pad, top - bot,
                 facecolor="#eef3f9" if ours else "#ffffff",
                 edgecolor=HILITE if ours else "#cfcfcf",
                 linewidth=0.9 if ours else 0.6, zorder=-5)

    if show_name:
        c.text(x + pw / 2, y + ph + dy_name, name, pt_name, ha="center", va="center",
               color=HILITE if ours else INK,
               fontweight="bold" if ours else "normal")
    c.text(x + pw / 2, y + ph + dy_scr, f"SCR {scr:.3f}".rstrip("0").rstrip("."),
           pt_scr, ha="center", va="center",
           color=PASS if nbad <= 1 else FAIL,
           fontweight="bold" if ours else "normal")

    ax = c.axes(x, y, pw, ph); ax.set_zorder(5)
    ax.imshow(mpimg.imread(os.path.join(IMG, f"crop_{method}_{task}.png")))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(HILITE if ours else GREY)
        s.set_linewidth(1.0 if ours else 0.5)

    n = len(sims)
    gp = pw * 0.011
    cw = (pw - gp * (n - 1)) / n
    for i, s in enumerate(sims):
        bad = s < THRESH
        cx = x + i * (cw + gp)
        c.rect(cx, strip_y, cw, strip_h,
               facecolor=(FAIL if bad else PASS), alpha=0.20,
               edgecolor=(FAIL if bad else PASS), linewidth=0.6, zorder=3)
        if cells:
            c.text(cx + cw / 2, strip_y + strip_h / 2, f"{s:.2f}".lstrip("0"),
                   pt_cell, ha="center", va="center", zorder=4,
                   color=(FAIL if bad else PASS),
                   fontweight="normal" if bad else "bold")
    if note:
        c.text(x + pw / 2, strip_y - note_gap - dy_note, f"{nbad} of 8 collapsed",
               pt_note, ha="center", va="center",
               color=(PASS if nbad <= 1 else FAIL),
               fontweight="bold" if nbad <= 1 else "normal")
    return nbad, scr, sims


def build_teaser():
    task, seed = TEASER
    pw = 3.98
    ph = pw / aspect(task)
    gap, mar = 0.34, 0.19
    strip_h, strip_gap, note_gap = 0.30, 0.07, 0.05
    top_pad, bot_pad = 0.79, 0.62
    W = 2 * pw + gap + 2 * mar
    H = ph + top_pad + strip_gap + strip_h + note_gap + bot_pad
    c = Canvas(W, H, TEASER_W_IN)

    y = bot_pad + note_gap + strip_h + strip_gap
    out = []
    LEGEND_Y = 0.145
    for j, (m, name, ours) in enumerate([("oneshot", "one-shot", False),
                                         ("ours", "MIDC (ours)", True)]):
        out.append(draw_panel(c, mar + j * (pw + gap), y, pw, ph, task, seed, m,
                              name, ours=ours, cells=True, note=True,
                              strip_h=strip_h, strip_gap=strip_gap,
                              note_gap=note_gap, pt_cell=5.8,
                              dy_name=0.57, dy_scr=0.21, dy_note=0.17))
    c.text(W / 2, LEGEND_Y,
           "red $=$ identity similarity below $0.5$ (collapsed)   ·   green $=$ retained",
           5.8, ha="center", va="center", color=GREY, style="italic")
    c.save("teaser")
    return out, c.printed_h


def build_grid():
    pw = 2.82
    gap, mar_l, mar_r = 0.14, 0.42, 0.19
    strip_h, strip_gap = 0.16, 0.055
    top_pad, row_gap, bot_pad = 0.82, 0.46, 0.42
    hs = [pw / aspect(t) for t, _ in GRID]
    W = 3 * pw + 2 * gap + mar_l + mar_r
    strip_row = strip_gap + strip_h
    H = sum(hs) + top_pad + len(hs) * strip_row + (len(hs) - 1) * row_gap + bot_pad
    c = Canvas(W, H, GRID_W_IN)

    # row bottoms, from the bottom row up: each row = strip + image + gap
    y = bot_pad + strip_row
    bottoms = {len(hs) - 1: y}
    for ri in range(len(hs) - 2, -1, -1):
        y = y + hs[ri + 1] + row_gap + strip_row
        bottoms[ri] = y
    ys = [bottoms[ri] for ri in range(len(hs))]
    methods = [("oneshot", "one-shot"), ("bon", "best-of-8"), ("ours", "MIDC (ours)")]
    out = []
    for ri, (task, seed) in enumerate(GRID):
        for j, (m, name) in enumerate(methods):
            out.append(draw_panel(c, mar_l + j * (pw + gap), ys[ri], pw, hs[ri],
                                  task, seed, m, name, ours=(m == "ours"),
                                  show_name=(ri == 0), cells=False, note=False,
                                  strip_h=strip_h, strip_gap=strip_gap,
                                  pt_name=8.0, pt_scr=7.2,
                                  dy_name=0.589, dy_scr=0.214))
        c.text(mar_l - 0.15, ys[ri] + hs[ri] / 2, task.replace("hard_", "task "),
               6.0, ha="center", va="center", color=GREY, rotation=90)
    c.text(W / 2, 0.16,
           "red $=$ identity similarity below $0.5$ (collapsed)   ·   green $=$ retained",
           5.8, ha="center", va="center", color=GREY, style="italic")
    c.save("results_grid")
    return out, c.printed_h


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for label, fn, wid in (("results_grid", build_grid, GRID_W_IN),):
        res, hh = fn()
        print(f"{label}: prints at {wid:.2f} x {hh:.2f} in")
        for nbad, scr, sims in res:
            print(f"   SCR {scr:.3f}  collapsed {nbad}/8  "
                  f"sims {[round(s, 2) for s in sims]}")
