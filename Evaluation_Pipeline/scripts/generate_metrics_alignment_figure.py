import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
INPUT_JSON = ROOT / "metrics_vs_human_summary.json"
OUTPUT_DIR = ROOT / "images"
OUTPUT_PATH = OUTPUT_DIR / "metrics_vs_human_alignment.png"

W, H = 2200, 1380
BG = "#F7F5F1"
TEXT = "#1F1A17"
MUTED = "#5D5751"
LINE = "#CEC7BF"
WHITE = "#FFFFFF"
BLACK = "#15120F"

ORANGE = "#F28C65"
YELLOW = "#F2C14E"
GREEN = "#A4D0A4"
BLUE = "#5A9CF0"
BLUE_DARK = "#2C63B5"
RED = "#E96A62"
LORA = "#6C5CE7"
LAYER = "#8AA1B6"


def load_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
        )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_TITLE = load_font(42, True)
FONT_SUB = load_font(21, False)
FONT_H2 = load_font(27, True)
FONT_BODY = load_font(20, False)
FONT_SMALL = load_font(17, False)
FONT_BOLD = load_font(20, True)
FONT_TINY = load_font(16, False)


def rounded(draw, box, fill, outline=LINE, radius=22, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text(draw, xy, value, font, fill=TEXT):
    draw.text(xy, value, font=font, fill=fill)


def center_text(draw, box, value, font, fill=TEXT):
    x0, y0, x1, y1 = box
    bbox = draw.multiline_textbbox((0, 0), value, font=font, spacing=4, align="center")
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.multiline_text((x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2), value, font=font, fill=fill, spacing=4, align="center")


def format_pct(x: float) -> str:
    return f"{100*x:.1f}%"


def model_label(result):
    size = result["size"].replace("08b", "0.8B").upper()
    mode = "LoRA" if result["mode"] == "lora_layer" else "Layer"
    return f"Qwen3.5-{size}\n{mode}"


def model_color(result):
    return LORA if result["mode"] == "lora_layer" else LAYER


def draw_bar_panel(draw, box, title, subtitle, items, key, color_fn):
    x0, y0, x1, y1 = box
    rounded(draw, box, WHITE)
    text(draw, (x0 + 28, y0 + 20), title, FONT_H2)
    text(draw, (x0 + 28, y0 + 58), subtitle, FONT_SMALL, MUTED)

    bar_x0 = x0 + 220
    bar_x1 = x1 - 70
    row_h = 78
    start_y = y0 + 108
    draw.line((bar_x0, start_y - 12, bar_x0, y1 - 35), fill=LINE, width=2)

    for idx, item in enumerate(items):
        y = start_y + idx * row_h
        label = model_label(item)
        center_text(draw, (x0 + 24, y - 8, x0 + 188, y + 46), label, FONT_BODY)
        value = item[key]
        width = max(2, (bar_x1 - bar_x0) * value)
        fill = color_fn(item)
        rounded(draw, (bar_x0, y, bar_x0 + width, y + 34), fill, outline=fill, radius=12, width=1)
        draw.text((bar_x0 + width + 12, y + 2), format_pct(value), font=FONT_BOLD, fill=TEXT)


def draw_summary_card(draw, box, title, body, accent):
    rounded(draw, box, WHITE)
    x0, y0, x1, y1 = box
    draw.rounded_rectangle((x0 + 16, y0 + 16, x1 - 16, y0 + 54), radius=14, fill=accent)
    center_text(draw, (x0 + 20, y0 + 17, x1 - 20, y0 + 52), title, FONT_BOLD, BLACK)
    center_text(draw, (x0 + 20, y0 + 66, x1 - 20, y1 - 16), body, FONT_BODY, TEXT)


def draw_heatmap(draw, box, title, results):
    x0, y0, _, _ = box
    rounded(draw, box, WHITE)
    text(draw, (x0 + 28, y0 + 20), title, FONT_H2)
    text(draw, (x0 + 28, y0 + 56), "Cell value = F1 against majority-vote human diagnostic labels", FONT_SMALL, MUTED)

    categories = ["existence", "appearance", "interaction"]
    col_x = [x0 + 300, x0 + 560, x0 + 820]
    row_y0 = y0 + 120
    row_h = 78

    for cidx, category in enumerate(categories):
        center_text(draw, (col_x[cidx], y0 + 88, col_x[cidx] + 180, y0 + 118), category.title(), FONT_BOLD)

    for ridx, result in enumerate(results):
        y = row_y0 + ridx * row_h
        center_text(draw, (x0 + 24, y, x0 + 260, y + 48), model_label(result), FONT_BODY)
        for cidx, category in enumerate(categories):
            f1 = result["category_metrics"][category]["f1"]
            shade = int(255 - 95 * f1)
            fill = (230 - int(85 * f1), 240 - int(70 * f1), shade)
            rounded(draw, (col_x[cidx], y, col_x[cidx] + 180, y + 46), fill, outline=LINE, radius=12, width=1)
            center_text(draw, (col_x[cidx], y, col_x[cidx] + 180, y + 46), f"{f1:.3f}", FONT_BOLD)


def draw_key_numbers_panel(draw, box, results):
    x0, y0, x1, y1 = box
    rounded(draw, box, WHITE)
    text(draw, (x0 + 28, y0 + 22), "Key Numbers", FONT_H2)
    text(draw, (x0 + 28, y0 + 58), "Pairwise agreement with majority-vote human preference", FONT_SMALL, MUTED)

    table_x0 = x0 + 26
    table_x1 = x1 - 26
    header_y = y0 + 104
    row_y = header_y + 44
    row_h = 74

    col_model = table_x0 + 8
    col_overall = x0 + 420
    col_split = x0 + 560

    text(draw, (col_model, header_y), "Model", FONT_BOLD, MUTED)
    text(draw, (col_overall, header_y), "Overall", FONT_BOLD, MUTED)
    text(draw, (col_split, header_y), "Seen / Unseen", FONT_BOLD, MUTED)
    draw.line((table_x0, header_y + 30, table_x1, header_y + 30), fill=LINE, width=2)

    for idx, result in enumerate(results):
        y = row_y + idx * row_h
        if idx > 0:
            draw.line((table_x0, y - 14, table_x1, y - 14), fill=LINE, width=1)
        label = model_label(result).replace("\n", " / ")
        draw.text((col_model, y), label, font=FONT_BODY, fill=TEXT)
        draw.text((col_overall, y), format_pct(result["pairwise_accuracy"]), font=FONT_BOLD, fill=model_color(result))
        split_text = f"V10 {format_pct(result['pairwise_accuracy_v10'])}   |   V13 {format_pct(result['pairwise_accuracy_v13'])}"
        draw.text((col_split, y + 2), split_text, font=FONT_TINY, fill=MUTED)

    footer = "LoRA leads overall; V13 remains the harder unseen-generator split."
    text(draw, (x0 + 28, y1 - 42), footer, FONT_SMALL, MUTED)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with INPUT_JSON.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    results = payload["results"]
    results = sorted(results, key=lambda x: (-x["pairwise_accuracy"], -x["macro_f1"]))

    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)

    title = "Human Alignment of Learned Metrics on PrismBench"
    subtitle = "Models are evaluated against majority-vote human labels after dropping preference-inconsistent pairs."
    text(draw, (W / 2 - 480, 34), title, FONT_TITLE)
    text(draw, (W / 2 - 505, 88), subtitle, FONT_SUB, MUTED)

    human_stats = payload["human_stats"]
    draw_summary_card(
        draw,
        (70, 150, 480, 320),
        "Filtered Human Benchmark",
        (
            f"Raw groups: {human_stats['total_raw_groups']}\n"
            f"Kept pairs: {human_stats['kept_pairs']}\n"
            f"Dropped pref-inconsistent: {human_stats['dropped_preference_inconsistent']}"
        ),
        ORANGE,
    )
    best = results[0]
    draw_summary_card(
        draw,
        (520, 150, 930, 320),
        "Best Overall Model",
        (
            f"{best['metrics_model_name']}\n"
            f"Pairwise accuracy: {format_pct(best['pairwise_accuracy'])}\n"
            f"Macro-F1: {best['macro_f1']:.3f}"
        ),
        GREEN,
    )

    lora_models = [r for r in results if r["mode"] == "lora_layer"]
    layer_models = [r for r in results if r["mode"] == "layer_only"]
    best_lora = max(lora_models, key=lambda x: x["pairwise_accuracy"])
    best_layer = max(layer_models, key=lambda x: x["pairwise_accuracy"])
    draw_summary_card(
        draw,
        (970, 150, 1380, 320),
        "LoRA vs Layer",
        (
            f"Best LoRA: {format_pct(best_lora['pairwise_accuracy'])}\n"
            f"Best Layer: {format_pct(best_layer['pairwise_accuracy'])}\n"
            f"Delta: {format_pct(best_lora['pairwise_accuracy'] - best_layer['pairwise_accuracy'])}"
        ),
        YELLOW,
    )

    draw_summary_card(
        draw,
        (1420, 150, 2130, 320),
        "Takeaway",
        "LoRA models align better with humans.\n2B LoRA is the strongest among the analyzed 0.8B / 2B metrics.\nV13 is substantially harder than V10.",
        BLUE,
    )

    draw_bar_panel(
        draw,
        (70, 360, 1060, 770),
        "Overall Pairwise Alignment",
        "Higher is better. Computed from human pairwise preference after consistency filtering.",
        results,
        "pairwise_accuracy",
        model_color,
    )

    draw_bar_panel(
        draw,
        (1140, 360, 2130, 770),
        "Seen vs Unseen Generalization",
        "V10 corresponds to seen-generator benchmark; V13 corresponds to unseen-generator benchmark.",
        sorted(results, key=lambda x: (-x["pairwise_accuracy_v13"], -x["pairwise_accuracy_v10"])),
        "pairwise_accuracy_v13",
        model_color,
    )

    draw_heatmap(draw, (70, 820, 1270, 1310), "Category-Level Human Alignment", results)

    draw_key_numbers_panel(draw, (1320, 820, 2130, 1310), results)

    image.save(OUTPUT_PATH, quality=96)
    print(f"Saved figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
