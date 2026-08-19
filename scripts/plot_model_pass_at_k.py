#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError as exc:
    raise SystemExit("Pillow is required. Install it with: pip install -e '.[analysis]'") from exc


WIDTH, HEIGHT = 1800, 1080
BG, INK, MUTED, GRID = "#F7F8FA", "#17212B", "#66717E", "#D9DEE5"
MODEL_COLORS = ("#1F6F8B", "#D1495B", "#2A9D8F", "#E9A23B")


def font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


TITLE, SUBTITLE, LABEL, SMALL, VALUE = font(46, True), font(24), font(23), font(19), font(21, True)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot model-level Pass@k charts.")
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def canvas(title: str, subtitle: str):
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.text((80, 55), title, fill=INK, font=TITLE)
    draw.text((80, 120), subtitle, fill=MUTED, font=SUBTITLE)
    return image, draw


def axes(draw, left, top, right, bottom):
    for tick in range(0, 101, 20):
        y = bottom - (bottom - top) * tick / 100
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((left - 72, y - 11), f"{tick}%", fill=MUTED, font=SMALL)


def legend(draw, models, x=150, y=980):
    for index, model in enumerate(models):
        offset = x + index * 520
        draw.rounded_rectangle((offset, y, offset + 25, y + 25), radius=4, fill=MODEL_COLORS[index])
        draw.text((offset + 37, y - 2), model["label"], fill=INK, font=LABEL)


def overall_chart(analysis, output_dir: Path):
    image, draw = canvas("Model-level Pass@k", "Three rollouts are aggregated per model; rounds are not shown as separate models")
    left, top, right, bottom = 155, 225, 1710, 900
    axes(draw, left, top, right, bottom)
    models, ks = analysis["models"], analysis["ks"]
    slot = (right - left) / len(ks)
    bar_width = min(125, slot * 0.62 / len(models))
    for ki, k in enumerate(ks):
        center = left + slot * (ki + 0.5)
        start = center - bar_width * len(models) / 2
        for mi, model in enumerate(models):
            rate = 100 * model["pass_at_k"][f"pass@{k}"]
            x0, x1 = start + mi * bar_width + 7, start + (mi + 1) * bar_width - 7
            y0 = bottom - (bottom - top) * rate / 100
            draw.rounded_rectangle((x0, y0, x1, bottom), radius=6, fill=MODEL_COLORS[mi])
            draw.text(((x0 + x1) / 2, y0 - 28), f"{rate:.1f}", fill=INK, font=VALUE, anchor="mm")
        draw.text((center, bottom + 35), f"Pass@{k}", fill=INK, font=LABEL, anchor="ma")
    legend(draw, models)
    image.save(output_dir / "pass_at_k_overall.png", optimize=True)


def dataset_chart(analysis, output_dir: Path):
    image, draw = canvas("Pass@k by benchmark", "Each panel compares the same two models across cumulative rollout budgets")
    models, ks = analysis["models"], analysis["ks"]
    panels = ["Overall", "MBPP+", "HumanEval+", "SWE-smith"]
    positions = [(90, 215, 855, 555), (945, 215, 1710, 555), (90, 650, 855, 990), (945, 650, 1710, 990)]
    for panel, (left, top, right, bottom) in zip(panels, positions):
        draw.text((left, top - 45), panel, fill=INK, font=LABEL)
        for tick in (0, 25, 50, 75, 100):
            y = bottom - (bottom - top) * tick / 100
            draw.line((left, y, right, y), fill=GRID, width=1)
            draw.text((left - 55, y - 10), str(tick), fill=MUTED, font=SMALL)
        x_positions = [left + (right - left) * (index + 1) / (len(ks) + 1) for index in range(len(ks))]
        for mi, model in enumerate(models):
            metrics = model["pass_at_k"] if panel == "Overall" else model["by_dataset"][panel]["pass_at_k"]
            points = [(x, bottom - (bottom - top) * (100 * metrics[f"pass@{k}"]) / 100) for x, k in zip(x_positions, ks)]
            draw.line(points, fill=MODEL_COLORS[mi], width=7)
            for (x, y), k in zip(points, ks):
                draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=MODEL_COLORS[mi])
                label_y = y - 27 if mi == 0 else min(bottom - 10, y + 22)
                draw.text((x, label_y), f"{100 * metrics[f'pass@{k}']:.1f}", fill=INK, font=SMALL, anchor="mm")
        for x, k in zip(x_positions, ks):
            draw.text((x, bottom + 16), f"P@{k}", fill=INK, font=SMALL, anchor="ma")
    legend(draw, models, x=515, y=1025)
    image.save(output_dir / "pass_at_k_by_dataset.png", optimize=True)


def main():
    args = parse_args()
    analysis = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    overall_chart(analysis, output_dir)
    dataset_chart(analysis, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "charts": 2}))


if __name__ == "__main__":
    main()
