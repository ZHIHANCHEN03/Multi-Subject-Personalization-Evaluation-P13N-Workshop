"""Build the MIDC-vs-UMO qualitative figure (paper Sec. 4.2).

The headline comparison of the paper is against the retrained SOTA (UMO), but
the qualitative grid in Sec. 4.3 shows FLUX.2 vs one-shot/best-of-8 only. This
builds the missing panel: one OmniGen2 hard 4-entity task with the reference
subjects shown, so a reader can verify the identity claims rather than take the
metric on faith.

Task selection was deliberately constrained rather than eyeballed: among the 250
hard_4 tasks we kept only those where MIDC beats all three baselines on both SCR
and DINO *and* the row is a no-op (`accepted_steps == 0`, i.e. no layout or
reference-set manipulation was applied, so the output cannot be a
spatial-separation artifact). `hard_050399` is the clearest of that pool.

Inputs are not in the repo (generated images live on the compute server); pass
--images pointing at a directory containing `<task>__<method>.png`.

    python3 plot_umo_qualitative.py --images /path/to/pngs --refs ../../meta/refs
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# hard_041671: "adult man beard pets the brown horse.
#               man sportswear folds the folding umbrella."
# Chosen over the earlier candidate because all four methods land at comparable
# sharpness here (Laplacian variance 931/711/1033/988), so any visible difference
# is structural rather than a by-product of UMO's softer runner config.
TASK = "hard_041671"
SUBJECTS = ["adult_man_beard", "man_sportswear", "brown_horse", "folding_umbrella"]
# (records key, display label) -- SCR is read from the records, never hard-coded.
PANELS = [("one_shot", "one-shot"), ("umo", "UMO (retrained)"),
          ("best_of_n", "best-of-8"), ("ours_v2", "MIDC (ours)")]
RECORDS = "results_r2/merged/{method}_s0/records.jsonl"

INK = "#0b0b0b"
INK_MUTED = "#52514e"


def square(im, size):
    """Center-crop to a square then resize, so panels share one aspect ratio."""
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return im.resize((size, size), Image.LANCZOS)


def load_scores(round2_dir: str) -> dict[str, tuple[float, float]]:
    """Read this task's SCR and DINO straight from the committed records."""
    out = {}
    for key, _ in PANELS:
        path = os.path.join(round2_dir, RECORDS.format(method=key))
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                r = json.loads(line)
                if r.get("task_id") == TASK:
                    out[key] = (r["scr"], r["dino_mean"])
                    break
        if key not in out:
            raise SystemExit(f"{TASK} not found in {path}")
    return out


def build(images, refs, out_pdf, out_png, scores):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(3.35, 1.50))
    # Top strip: the four reference subjects. Bottom row: the four methods.
    gs = fig.add_gridspec(2, 4, height_ratios=[0.58, 1.0], hspace=0.42, wspace=0.04,
                          left=0.005, right=0.995, top=0.87, bottom=0.005)

    for j, name in enumerate(SUBJECTS):
        ax = fig.add_subplot(gs[0, j])
        p = os.path.join(refs, f"{name}.jpg")
        ax.imshow(square(Image.open(p).convert("RGB"), 256))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#d8d7d2"); s.set_linewidth(0.5)
        ax.set_title(name.replace("_", " "), fontsize=3.6, color=INK_MUTED, pad=1.4)

    for j, (key, label) in enumerate(PANELS):
        ax = fig.add_subplot(gs[1, j])
        p = os.path.join(images, f"{TASK}__{key}.png")
        ax.imshow(square(Image.open(p).convert("RGB"), 420))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#d8d7d2"); s.set_linewidth(0.5)
        weight = "bold" if key == "ours_v2" else "normal"
        scr, dino = scores[key]
        # Two lines: one four-panel row cannot fit label and both metrics side by
        # side at this width without the titles running into each other.
        ax.set_title(f"{label}\nSCR {scr:.2f} · DINO {dino:.2f}", fontsize=4.4,
                     color=INK, pad=1.4, fontweight=weight, linespacing=1.35)

    fig.text(0.5, 0.955, "reference subjects", ha="center", va="center",
             fontsize=4.4, color=INK_MUTED)
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_png, dpi=400, bbox_inches="tight", pad_inches=0.01)
    print(f"wrote {out_pdf}\nwrote {out_png}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="dir with <task>__<method>.png")
    ap.add_argument("--refs", default=os.path.normpath(
        os.path.join(here, os.pardir, os.pardir, "meta", "refs")))
    ap.add_argument("--pdf", default=os.path.normpath(os.path.join(
        here, os.pardir, "paper", "figures", "umo_qualitative.pdf")))
    ap.add_argument("--png", default=os.path.normpath(os.path.join(
        here, os.pardir, "paper", "figures", "umo_qualitative.png")))
    ap.add_argument("--round2", default=os.path.dirname(os.path.abspath(__file__)),
                    help="dir containing results_r2/ (for reading SCR/DINO)")
    a = ap.parse_args()
    build(a.images, a.refs, a.pdf, a.png, load_scores(a.round2))


if __name__ == "__main__":
    main()
