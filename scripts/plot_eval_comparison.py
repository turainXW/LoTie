#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError as exc:
    raise SystemExit("Pillow is required. Install it with: pip install -e '.[analysis]'") from exc


WIDTH = 1800
HEIGHT = 1050
BG = "#F7F8FA"
INK = "#17212B"
MUTED = "#66717E"
GRID = "#D9DEE5"
COLORS = ("#1F6F8B", "#D1495B", "#2A9D8F", "#E9A23B", "#6C5CE7", "#627D98")
FAILURE_COLORS = {
    "tests_failed": "#D1495B",
    "unresolved_patch": "#E9A23B",
    "no_effective_edit": "#627D98",
    "no_patch_max_steps": "#8D6E63",
    "protocol_parse_error": "#6C5CE7",
    "compile_failed": "#B23A48",
    "context_overflow": "#355070",
    "infrastructure": "#111111",
}


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


TITLE = font(46, bold=True)
SUBTITLE = font(24)
LABEL = font(23)
SMALL = font(19)
VALUE = font(22, bold=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render evaluation comparison PNGs from evaluation_analysis.json.")
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def series_from_analysis(analysis: dict) -> list[dict]:
    series = []
    for run in analysis["runs"]:
        rounds = run.get("rounds", {})
        if len(rounds) <= 1:
            payload = next(iter(rounds.values()), run)
            series.append({"label": run["label"], **payload})
        else:
            for round_index, payload in sorted(rounds.items(), key=lambda item: int(item[0])):
                series.append({"label": f"{run['label']} R{round_index}", **payload})
    return series


def new_canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.text((80, 55), title, fill=INK, font=TITLE)
    draw.text((80, 120), subtitle, fill=MUTED, font=SUBTITLE)
    return image, draw


def save(image: Image.Image, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image.save(output_dir / name, format="PNG", optimize=True)


def performance_chart(series: list[dict], output_dir: Path) -> None:
    image, draw = new_canvas(
        "Code Agent pass rate by benchmark",
        "Resolved by verifier; each sampling round is shown separately",
    )
    left, top, right, bottom = 150, 220, 1720, 900
    categories = ["Overall", "MBPP+", "HumanEval+", "SWE-smith"]
    for tick in range(0, 101, 20):
        y = bottom - (bottom - top) * tick / 100
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((70, y - 12), f"{tick}%", fill=MUTED, font=SMALL)
    group_width = (right - left) / len(categories)
    bar_width = min(72, group_width * 0.72 / max(1, len(series)))
    for category_index, category in enumerate(categories):
        center = left + group_width * (category_index + 0.5)
        total_width = bar_width * len(series)
        start = center - total_width / 2
        for series_index, item in enumerate(series):
            if category == "Overall":
                rate = float(item["pass_rate"])
            else:
                rate = float(item.get("by_dataset", {}).get(category, {}).get("pass_rate", 0.0))
            x0 = start + series_index * bar_width + 4
            x1 = start + (series_index + 1) * bar_width - 4
            y0 = bottom - (bottom - top) * rate / 100
            draw.rounded_rectangle((x0, y0, x1, bottom), radius=5, fill=COLORS[series_index % len(COLORS)])
            text = f"{rate:.1f}"
            box = draw.textbbox((0, 0), text, font=SMALL)
            draw.text(((x0 + x1 - (box[2] - box[0])) / 2, y0 - 27), text, fill=INK, font=SMALL)
        box = draw.textbbox((0, 0), category, font=LABEL)
        draw.text((center - (box[2] - box[0]) / 2, bottom + 25), category, fill=INK, font=LABEL)
    legend_x, legend_y = 150, 955
    for index, item in enumerate(series):
        x = legend_x + index * 385
        draw.rounded_rectangle((x, legend_y, x + 24, legend_y + 24), radius=4, fill=COLORS[index % len(COLORS)])
        draw.text((x + 34, legend_y - 2), item["label"], fill=INK, font=SMALL)
    save(image, output_dir, "pass_rate_by_dataset.png")


def failure_chart(series: list[dict], output_dir: Path) -> None:
    image, draw = new_canvas(
        "Failure composition",
        "Mutually exclusive failure categories as a share of all evaluated samples",
    )
    left, top, right = 320, 245, 1690
    bar_height, gap = 82, 68
    categories = sorted(
        {
            category
            for item in series
            for category in item.get("canonical_outcomes", {})
            if category != "resolved"
        }
    )
    for index, item in enumerate(series):
        y0 = top + index * (bar_height + gap)
        y1 = y0 + bar_height
        draw.text((60, y0 + 24), item["label"], fill=INK, font=LABEL)
        x = left
        total = max(1, int(item["total"]))
        for category in categories:
            count = int(item.get("canonical_outcomes", {}).get(category, 0))
            if not count:
                continue
            width = (right - left) * count / total
            draw.rectangle((x, y0, x + width, y1), fill=FAILURE_COLORS.get(category, "#999999"))
            if width >= 45:
                draw.text((x + 8, y0 + 26), str(count), fill="white", font=SMALL)
            x += width
        failure_count = total - int(item.get("resolved", 0))
        draw.text((right + 12, y0 + 24), f"{failure_count}/{total}", fill=MUTED, font=VALUE)
    legend_y = 890
    x = 80
    for category in categories:
        color = FAILURE_COLORS.get(category, "#999999")
        draw.rounded_rectangle((x, legend_y, x + 22, legend_y + 22), radius=3, fill=color)
        draw.text((x + 31, legend_y - 2), category.replace("_", " "), fill=INK, font=SMALL)
        x += 250
        if x > 1550:
            x = 80
            legend_y += 48
    save(image, output_dir, "failure_composition.png")


def coverage_chart(analysis: dict, output_dir: Path) -> None:
    image, draw = new_canvas(
        "Task coverage across samples",
        "Coverage means a task was solved at least once; multi-round runs are best-of-N",
    )
    left, top, right, bottom = 210, 250, 1660, 860
    runs = analysis["runs"]
    for tick in range(0, 101, 20):
        y = bottom - (bottom - top) * tick / 100
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((125, y - 12), f"{tick}%", fill=MUTED, font=SMALL)
    slot = (right - left) / max(1, len(runs))
    for index, run in enumerate(runs):
        coverage = run["coverage"]
        rate = float(coverage["rate"])
        center = left + slot * (index + 0.5)
        x0, x1 = center - 105, center + 105
        y0 = bottom - (bottom - top) * rate / 100
        draw.rounded_rectangle((x0, y0, x1, bottom), radius=8, fill=COLORS[index % len(COLORS)])
        value = f"{coverage['covered_tasks']}/{coverage['total_tasks']}\n{rate:.1f}%"
        draw.multiline_text((center, y0 - 70), value, fill=INK, font=VALUE, anchor="mm", align="center", spacing=4)
        label = run["label"]
        draw.multiline_text((center, bottom + 35), label, fill=INK, font=LABEL, anchor="ma", align="center")
    save(image, output_dir, "task_coverage.png")


def main() -> None:
    args = parse_args()
    analysis = json.loads(Path(args.analysis).expanduser().read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir).expanduser().resolve()
    series = series_from_analysis(analysis)
    performance_chart(series, output_dir)
    failure_chart(series, output_dir)
    coverage_chart(analysis, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "charts": 3}))


if __name__ == "__main__":
    main()
