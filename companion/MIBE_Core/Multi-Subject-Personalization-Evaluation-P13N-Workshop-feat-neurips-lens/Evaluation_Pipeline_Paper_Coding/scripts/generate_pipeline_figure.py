from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "images"
OUTPUT_PATH = IMAGES_DIR / "mib_mie_pipeline.png"

CANVAS_W = 2400
CANVAS_H = 1500

BG = "#F6F4EF"
CARD = "#FFFDF8"
TEXT = "#1D1A17"
MUTED = "#5F5A55"
OUTLINE = "#2C2824"
SOFT = "#CFC7BD"
WHITE = "#FFFFFF"

BLUE = "#4A90E2"
BLUE_SOFT = "#DCEBFA"
BLUE_DARK = "#275EA6"
RED = "#EF6A62"
RED_SOFT = "#FBE2DE"
RED_DARK = "#B63E38"
PURPLE = "#8B79B8"
PURPLE_SOFT = "#E9E2F6"
ORANGE = "#DA8B54"
ORANGE_SOFT = "#F6E1D4"
GREEN_SOFT = "#E8F2E8"
YELLOW = "#E3B34C"
YELLOW_SOFT = "#F9EECC"
BLACK = "#111111"


def load_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
        )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_TITLE = load_font(50, bold=True)
FONT_SUB = load_font(23, bold=False)
FONT_SECTION = load_font(27, bold=True)
FONT_STEP = load_font(25, bold=True)
FONT_BODY = load_font(20, bold=False)
FONT_BODY_BOLD = load_font(20, bold=True)
FONT_SMALL = load_font(17, bold=False)
FONT_TAG = load_font(18, bold=True)
FONT_NUM = load_font(30, bold=True)


def round_box(draw, box, fill, outline=OUTLINE, radius=28, width=3):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def shadowed_box(base, box, fill, outline=OUTLINE, radius=28, shadow_offset=(8, 10)):
    shadow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    sx0, sy0, sx1, sy1 = box
    dx, dy = shadow_offset
    shadow_draw.rounded_rectangle((sx0 + dx, sy0 + dy, sx1 + dx, sy1 + dy), radius=radius, fill=(0, 0, 0, 55))
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))
    base.alpha_composite(shadow)
    draw = ImageDraw.Draw(base)
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=3)


def wrapped_text(text, width):
    lines = []
    for part in text.split("\n"):
        lines.extend(textwrap.wrap(part, width=width) or [""])
    return "\n".join(lines)


def center_multiline(draw, box, text, font, fill=TEXT, spacing=6):
    x0, y0, x1, y1 = box
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = x0 + (x1 - x0 - tw) / 2
    y = y0 + (y1 - y0 - th) / 2
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=spacing, align="center")


def left_multiline(draw, xy, text, font, fill=TEXT, spacing=6):
    draw.multiline_text(xy, text, font=font, fill=fill, spacing=spacing)


def numbered_badge(draw, cx, cy, label, fill=BLACK):
    r = 28
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill)
    bbox = draw.textbbox((0, 0), label, font=FONT_NUM)
    draw.text((cx - (bbox[2] - bbox[0]) / 2, cy - (bbox[3] - bbox[1]) / 2 - 1), label, font=FONT_NUM, fill=WHITE)


def arrow(draw, p1, p2, color=OUTLINE, width=5, head=16):
    draw.line([p1, p2], fill=color, width=width)
    x1, y1 = p1
    x2, y2 = p2
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 > x1 else -1
        draw.polygon(
            [(x2, y2), (x2 - direction * head, y2 - head / 1.3), (x2 - direction * head, y2 + head / 1.3)],
            fill=color,
        )
    else:
        direction = 1 if y2 > y1 else -1
        draw.polygon(
            [(x2, y2), (x2 - head / 1.3, y2 - direction * head), (x2 + head / 1.3, y2 - direction * head)],
            fill=color,
        )


def bent_arrow(draw, start, via, end, color=OUTLINE, width=5):
    draw.line([start, via, end], fill=color, width=width)
    arrow(draw, via, end, color=color, width=width)


def step_card(base, box, color, step_no, title, body):
    shadowed_box(base, box, CARD)
    draw = ImageDraw.Draw(base)
    x0, y0, x1, _ = box
    draw.rounded_rectangle((x0, y0, x1, y0 + 16), radius=28, fill=color)
    draw.rectangle((x0, y0 + 14, x1, y0 + 28), fill=color)
    numbered_badge(draw, x0 + 52, y0 + 54, str(step_no))
    draw.text((x0 + 100, y0 + 28), title, font=FONT_STEP, fill=TEXT)
    body = wrapped_text(body, 37)
    left_multiline(draw, (x0 + 36, y0 + 92), body, FONT_BODY, MUTED, spacing=7)


def panel_header(draw, box, color, title):
    x0, y0, x1, y1 = box
    draw.rectangle((x0, y0, x1, y1), fill=color)
    bbox = draw.textbbox((0, 0), title, font=FONT_SECTION)
    draw.text((x0 + (x1 - x0 - (bbox[2] - bbox[0])) / 2, y0 + 8), title, font=FONT_SECTION, fill=BLACK)


def summary_panel(base, box, color, title, lines):
    shadowed_box(base, box, CARD, radius=22, shadow_offset=(6, 8))
    draw = ImageDraw.Draw(base)
    x0, y0, x1, y1 = box
    panel_header(draw, (x0 + 12, y0 + 12, x1 - 12, y0 + 54), color, title)
    y = y0 + 72
    for left, right in lines:
        draw.text((x0 + 30, y), left, font=FONT_BODY, fill=TEXT)
        rb = draw.textbbox((0, 0), right, font=FONT_BODY)
        draw.text((x1 - 30 - (rb[2] - rb[0]), y), right, font=FONT_BODY, fill=TEXT)
        y += 34
    draw.line((x0 + 24, y1 - 20, x1 - 24, y1 - 20), fill=SOFT, width=2)


def mini_people(draw, x, y, colors):
    offsets = [0, 46, 92, 138]
    for dx, color in zip(offsets, colors):
        cx = x + dx
        draw.ellipse((cx, y, cx + 34, y + 34), outline=color, width=5)
        draw.arc((cx - 14, y + 28, cx + 48, y + 92), start=200, end=340, fill=color, width=5)


def mini_chat_stack(base, x, y, w, h):
    draw = ImageDraw.Draw(base)
    for offset in [26, 13, 0]:
        shadowed_box(base, (x + offset, y + offset, x + w + offset, y + h + offset), "#ECE8DE", outline=SOFT, radius=32, shadow_offset=(0, 0))
    shadowed_box(base, (x, y, x + w, y + h), "#EFEBDD", outline=OUTLINE, radius=32, shadow_offset=(0, 0))
    draw = ImageDraw.Draw(base)
    bubble_colors = [BLUE, YELLOW, BLUE, RED]
    for row_y in [y + 44, y + 168]:
        draw.ellipse((x + 28, row_y, x + 74, row_y + 46), outline=OUTLINE, width=3)
        draw.line((x + 51, row_y + 46, x + 51, row_y + 64), fill=OUTLINE, width=3)
        draw.arc((x + 34, row_y + 58, x + 68, row_y + 82), 200, 340, fill=OUTLINE, width=3)
        round_box(draw, (x + 88, row_y, x + 360, row_y + 44), WHITE, OUTLINE, radius=18, width=3)
        draw.line((x + 112, row_y + 15, x + 314, row_y + 15), fill=OUTLINE, width=3)
        draw.line((x + 112, row_y + 30, x + 284, row_y + 30), fill=OUTLINE, width=3)
        bx = x + 42
        for color in bubble_colors:
            round_box(draw, (bx, row_y + 78, bx + 72, row_y + 128), WHITE, color, radius=14, width=3)
            draw.line((bx + 16, row_y + 97, bx + 50, row_y + 97), fill=color, width=3)
            draw.line((bx + 16, row_y + 112, bx + 42, row_y + 112), fill=color, width=3)
            bx += 94


def mini_icon_pool(draw, x, y):
    colors = [RED, BLUE, ORANGE, BLUE, RED, ORANGE]
    positions = [(0, 0), (52, 12), (104, 2), (12, 54), (66, 64), (118, 50)]
    for (dx, dy), color in zip(positions, colors):
        round_box(draw, (x + dx, y + dy, x + dx + 42, y + dy + 42), WHITE, color, radius=10, width=3)
        draw.line((x + dx + 12, y + dy + 18, x + dx + 30, y + dy + 18), fill=color, width=3)
        draw.line((x + dx + 12, y + dy + 29, x + dx + 24, y + dy + 29), fill=color, width=3)


def tag(draw, x, y, text, fill):
    bbox = draw.textbbox((0, 0), text, font=FONT_TAG)
    w = bbox[2] - bbox[0] + 40
    h = 40
    round_box(draw, (x, y, x + w, y + h), fill=fill, outline=SOFT, radius=14, width=2)
    draw.text((x + 20, y + 8), text, font=FONT_TAG, fill=TEXT)
    return w


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(image)

    title = "MIB + MIE: Scalable Metric Training and Rigorous Meta-Evaluation"
    subtitle = "A publication-style overview of how the benchmark is constructed, filtered, trained, and evaluated"
    tb = draw.textbbox((0, 0), title, font=FONT_TITLE)
    draw.text(((CANVAS_W - (tb[2] - tb[0])) / 2, 34), title, font=FONT_TITLE, fill=TEXT)
    sb = draw.textbbox((0, 0), subtitle, font=FONT_SUB)
    draw.text(((CANVAS_W - (sb[2] - sb[0])) / 2, 98), subtitle, font=FONT_SUB, fill=MUTED)

    # Top storytelling row
    numbered_badge(draw, 530, 243, "1")
    draw.text((580, 195), "Build a controlled multi-subject reference pool", font=FONT_STEP, fill=TEXT)
    left_multiline(draw, (580, 236), "80 reference images\n30 humans + 50 objects\nstandardized at 512 x 512 px", FONT_BODY, MUTED)
    mini_people(draw, 150, 202, [RED, ORANGE, BLUE, YELLOW])
    shadowed_box(image, (280, 170, 500, 315), ORANGE_SOFT, outline=ORANGE, radius=26)
    draw = ImageDraw.Draw(image)
    center_multiline(draw, (300, 190, 480, 295), "Reference\nPool", FONT_SECTION, TEXT)

    numbered_badge(draw, 1040, 243, "2")
    draw.text((1090, 188), "Construct prompts hierarchically", font=FONT_STEP, fill=TEXT)
    left_multiline(draw, (1090, 230), "15K seeds -> 60K prompts\n36 buckets: relation x ratio x complexity\nN in {2, 4, 6, 8}", FONT_BODY, MUTED)
    shadowed_box(image, (880, 166, 1010, 318), PURPLE_SOFT, outline=PURPLE, radius=26)
    draw = ImageDraw.Draw(image)
    for i in range(4):
        round_box(draw, (904, 190 + i * 28, 986, 212 + i * 28), WHITE, PURPLE, radius=10, width=2)
        draw.line((922, 201 + i * 28, 966, 201 + i * 28), fill=PURPLE, width=3)

    numbered_badge(draw, 1605, 243, "3")
    draw.text((1652, 180), "Generate paired images and collect ratings", font=FONT_STEP, fill=TEXT)
    left_multiline(draw, (1652, 222), "Silver path: Nano Banana vs Mosaic + dual VLM labels\nGold path: six generators + double-blind human annotation", FONT_BODY, MUTED)
    mini_chat_stack(image, 1860, 154, 360, 254)
    draw = ImageDraw.Draw(image)
    mini_icon_pool(draw, 2190, 200)

    arrow(draw, (508, 244), (548, 244))
    arrow(draw, (1018, 244), (1058, 244))
    arrow(draw, (1538, 244), (1580, 244))

    # Two main columns
    left_x, right_x = 110, 1250
    card_w = 1040
    step_h = 145
    gap = 28
    start_y = 470

    draw.text((left_x + 20, start_y - 42), "Silver Path (scale)", font=FONT_SECTION, fill=BLUE_DARK)
    draw.text((right_x + 20, start_y - 42), "Gold Path (rigor)", font=FONT_SECTION, fill=RED_DARK)

    left_steps = [
        (3, "Silver Set Generation", "Two generators: Nano Banana + Mosaic\n60K image pairs, paired as Image A vs Image B", BLUE_SOFT, BLUE),
        (4, "Dual VLM Annotation", "Gemini 2.5 Flash + Gemini 3.1 Flash Lite\nOutputs: Existence, Appearance, Interaction, Pairwise preference", BLUE_SOFT, BLUE),
        (5, "Agreement-based Filtering", "57,055 usable pairs retained\ntrain 51K / val 2.8K / test 2.9K", BLUE_SOFT, BLUE_DARK),
        (6, "MIE Training on Silver Set", "Backbone: Qwen3.5-VL (0.8B / 2B / 4B)\nModes: LoRA + layer-only\nDual-head loss: Ranking + Diagnostic BCE", BLUE, BLUE_DARK),
    ]
    right_steps = [
        (3, "Gold Set Generation", "Six generators: Nano Banana, Mosaic, Flux, GLM,\nGPT-Image-1.5, Seedream 4.5\n4,000 image pairs", RED_SOFT, RED),
        (4, "Double-blind Human Annotation", "Annotators are blind to generator identity\nThey rate the same diagnostic dimensions and pairwise preference", RED_SOFT, RED),
        (5, "Gold Evaluation Set", "Seen generators: 1,500 pairs (Nano Banana, Mosaic)\nUnseen generators: 2,500 pairs (Flux, GLM, GPT-Image-1.5, Seedream 4.5)", RED_SOFT, RED_DARK),
        (7, "MIE Evaluation on Gold Set", "Compare CLIP-I, CLIP-T, DINOv2, ArcFace,\nImageReward, SCR, PSNR and more\nMeasures: Pairwise accuracy, Spearman, Kendall-T", RED, RED_DARK),
    ]

    left_boxes = []
    right_boxes = []
    y = start_y
    for step in left_steps:
        box = (left_x, y, left_x + card_w, y + step_h)
        left_boxes.append(box)
        step_card(image, box, step[4], step[0], step[1], step[2])
        y += step_h + gap

    y = start_y
    for step in right_steps:
        box = (right_x, y, right_x + card_w, y + step_h)
        right_boxes.append(box)
        step_card(image, box, step[4], step[0], step[1], step[2])
        y += step_h + gap

    draw = ImageDraw.Draw(image)
    for boxes in (left_boxes, right_boxes):
        for i in range(len(boxes) - 1):
            b0 = boxes[i]
            b1 = boxes[i + 1]
            arrow(draw, ((b0[0] + b0[2]) / 2, b0[3]), ((b1[0] + b1[2]) / 2, b1[1]))

    mid_y = left_boxes[-1][1] + 74
    draw.line((left_boxes[-1][2], mid_y, right_boxes[-1][0] - 56, mid_y), fill=OUTLINE, width=4)
    arrow(draw, (right_boxes[-1][0] - 56, mid_y), (right_boxes[-1][0], mid_y))
    draw.text((1160, mid_y - 34), "evaluate on", font=FONT_SMALL, fill=MUTED)

    # Bottom summary panels
    panel_y = 1160
    panel_h = 250
    panel_gap = 38
    panel_w = (CANVAS_W - 2 * 110 - 2 * panel_gap) // 3

    summary_panel(
        image,
        (110, panel_y, 110 + panel_w, panel_y + panel_h),
        "#F58C6B",
        "Reference Pool",
        [
            ("Reference subjects", "80"),
            ("Human identities", "30"),
            ("Object identities", "50"),
            ("Image size", "512 x 512"),
            ("Use", "personalization anchors"),
        ],
    )
    summary_panel(
        image,
        (110 + panel_w + panel_gap, panel_y, 110 + 2 * panel_w + panel_gap, panel_y + panel_h),
        "#F1BE4B",
        "MIB Construction",
        [
            ("Seeds", "15,000"),
            ("Prompts", "60,000"),
            ("Buckets", "36"),
            ("Subject count", "{2, 4, 6, 8}"),
            ("Relations", "3 types"),
        ],
    )
    summary_panel(
        image,
        (110 + 2 * (panel_w + panel_gap), panel_y, 110 + 3 * panel_w + 2 * panel_gap, panel_y + panel_h),
        "#A9D4B9",
        "Evaluation Outputs",
        [
            ("Silver labels", "dual VLM"),
            ("Gold labels", "double-blind human"),
            ("Metric outputs", "pref + 3 diagnostics"),
            ("Main measures", "accuracy / Spearman / Kendall-T"),
            ("Downstream use", "benchmarking + evaluator dev"),
        ],
    )

    draw = ImageDraw.Draw(image)
    draw.text((116, 1442), "Community Uses:", font=FONT_BODY_BOLD, fill=MUTED)
    x = 300
    for text, fill in [
        ("Metric Training", GREEN_SOFT),
        ("Generator Benchmarking", "#EFEFEA"),
        ("RLHF / DPO Preference Data", "#EFEFEA"),
        ("Evaluator Development", "#EFEFEA"),
        ("Generative Alignment", "#EFEFEA"),
    ]:
        width = tag(draw, x, 1422, text, fill)
        x += width + 18

    image.convert("RGB").save(OUTPUT_PATH, quality=96)
    print(f"Saved optimized figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
