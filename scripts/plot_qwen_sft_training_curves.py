#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1600
HEIGHT = 1080
BACKGROUND = "#f7f8fa"
PANEL = "#ffffff"
GRID = "#dfe3e8"
TEXT = "#20262e"
MUTED = "#68717d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot live Lottie Qwen LoRA SFT metrics from JSONL logs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--planned-updates", type=int, default=458)
    parser.add_argument("--smooth-window", type=int, default=20)
    return parser.parse_args()


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size, index=1 if bold and path.endswith(".ttc") else 0)
        except OSError:
            continue
    return ImageFont.load_default()


def moving_average(values: list[float], window: int) -> list[float]:
    result: list[float] = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
        result.append(running / min(index + 1, window))
    return result


def finite_extent(values: list[float], *, include_zero: bool = False) -> tuple[float, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return 0.0, 1.0
    low = min(finite)
    high = max(finite)
    if include_zero:
        low = min(low, 0.0)
    if low == high:
        padding = max(abs(low) * 0.1, 1.0)
    else:
        padding = (high - low) * 0.1
    lower = 0.0 if include_zero and low >= 0.0 else low - padding
    return lower, high + padding


def draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    title: str,
    x_values: list[float],
    series: list[tuple[str, list[float], str, int]],
    y_formatter: Callable[[float], str],
    y_include_zero: bool = False,
    epoch_boundaries: list[int] | None = None,
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=8, fill=PANEL, outline="#e3e6ea", width=1)
    title_font = load_font(22, bold=True)
    label_font = load_font(16)
    legend_font = load_font(15)
    draw.text((left + 22, top + 18), title, font=title_font, fill=TEXT)

    plot_left = left + 82
    plot_top = top + 58
    plot_right = right - 24
    plot_bottom = bottom - 50
    all_values = [value for _, values, _, _ in series for value in values]
    y_min, y_max = finite_extent(all_values, include_zero=y_include_zero)
    x_min = min(x_values) if x_values else 0.0
    x_max = max(x_values) if x_values else 1.0
    if x_min == x_max:
        x_max = x_min + 1.0

    def map_x(value: float) -> float:
        return plot_left + (value - x_min) / (x_max - x_min) * (plot_right - plot_left)

    def map_y(value: float) -> float:
        return plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

    for tick in range(6):
        ratio = tick / 5
        y = plot_bottom - ratio * (plot_bottom - plot_top)
        value = y_min + ratio * (y_max - y_min)
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=1)
        label = y_formatter(value)
        label_box = draw.textbbox((0, 0), label, font=label_font)
        draw.text((plot_left - 12 - (label_box[2] - label_box[0]), y - 8), label, font=label_font, fill=MUTED)
    for tick in range(6):
        ratio = tick / 5
        x = plot_left + ratio * (plot_right - plot_left)
        value = x_min + ratio * (x_max - x_min)
        draw.line((x, plot_top, x, plot_bottom), fill="#eef0f3", width=1)
        label = str(int(round(value)))
        label_box = draw.textbbox((0, 0), label, font=label_font)
        draw.text((x - (label_box[2] - label_box[0]) / 2, plot_bottom + 10), label, font=label_font, fill=MUTED)

    for boundary_index, boundary in enumerate(epoch_boundaries or [], start=1):
        if x_min < boundary < x_max:
            x = map_x(boundary)
            draw.line((x, plot_top, x, plot_bottom), fill="#9aa2ac", width=2)
            label = f"Epoch {boundary_index} -> {boundary_index + 1}"
            draw.text((x + 5, plot_top + 5), label, font=legend_font, fill=MUTED)

    for label, values, color, width in series:
        points = [(map_x(x), map_y(value)) for x, value in zip(x_values, values, strict=True) if math.isfinite(value)]
        if len(points) >= 2:
            draw.line(points, fill=color, width=width, joint="curve")
        elif points:
            x, y = points[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)

    legend_x = left + 40 + draw.textlength(title, font=title_font)
    legend_y = top + 22
    for label, _, color, width in series:
        draw.line((legend_x, legend_y + 8, legend_x + 24, legend_y + 8), fill=color, width=max(width, 2))
        draw.text((legend_x + 31, legend_y), label, font=legend_font, fill=MUTED)
        legend_x += 42 + draw.textlength(label, font=legend_font)


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("training log is empty")

    steps = [float(row["micro_step"]) for row in rows]
    losses = [float(row["loss"]) for row in rows]
    smooth_losses = moving_average(losses, args.smooth_window)
    learning_rates = [float(row["learning_rate"]) for row in rows]
    throughput = [
        float(row.get("tokens_per_sec_current_run", row.get("tokens_per_sec_cumulative", 0.0)))
        for row in rows
    ]
    allocated = [float(row["peak_allocated_gib"]) for row in rows]
    reserved = [float(row["peak_reserved_gib"]) for row in rows]
    epoch_boundaries: list[int] = []
    for previous, current in zip(rows, rows[1:]):
        if int(current["epoch"]) != int(previous["epoch"]):
            epoch_boundaries.append(int(current["micro_step"]))

    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    heading = load_font(32, bold=True)
    subtitle = load_font(18)
    draw.text((54, 30), "Lottie Qwen3-4B LoRA SFT - Live Curves", font=heading, fill=TEXT)
    latest = rows[-1]
    status = (
        f"epoch {latest['epoch']} | update {latest['update_step']}/{args.planned_updates} | "
        f"micro-step {latest['micro_step']} | latest loss {latest['loss']:.4f}"
    )
    draw.text((55, 76), status, font=subtitle, fill=MUTED)

    gap = 22
    panel_width = (WIDTH - 108 - gap) // 2
    panel_height = 430
    boxes = [
        (54, 116, 54 + panel_width, 116 + panel_height),
        (54 + panel_width + gap, 116, WIDTH - 54, 116 + panel_height),
        (54, 116 + panel_height + gap, 54 + panel_width, HEIGHT - 54),
        (54 + panel_width + gap, 116 + panel_height + gap, WIDTH - 54, HEIGHT - 54),
    ]
    draw_panel(
        draw,
        boxes[0],
        title="Training loss",
        x_values=steps,
        series=[
            ("raw", losses, "#9dc7ba", 2),
            (f"MA({args.smooth_window})", smooth_losses, "#087f5b", 4),
        ],
        y_formatter=lambda value: f"{value:.2f}",
        y_include_zero=True,
        epoch_boundaries=epoch_boundaries,
    )
    draw_panel(
        draw,
        boxes[1],
        title="Learning rate",
        x_values=steps,
        series=[("lr", learning_rates, "#d9485f", 4)],
        y_formatter=lambda value: f"{value * 1e5:.2f}e-5",
        y_include_zero=True,
        epoch_boundaries=epoch_boundaries,
    )
    draw_panel(
        draw,
        boxes[2],
        title="Cumulative throughput",
        x_values=steps,
        series=[("tokens/s", throughput, "#2166ac", 4)],
        y_formatter=lambda value: f"{value:.0f}",
        epoch_boundaries=epoch_boundaries,
    )
    draw_panel(
        draw,
        boxes[3],
        title="Peak GPU memory",
        x_values=steps,
        series=[
            ("allocated GiB", allocated, "#e08214", 4),
            ("reserved GiB", reserved, "#7b3294", 4),
        ],
        y_formatter=lambda value: f"{value:.0f}",
        y_include_zero=True,
        epoch_boundaries=epoch_boundaries,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)
    print(output)


if __name__ == "__main__":
    main()
