#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lottie-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / "remote_summaries"
OUTPUT_DIR = HERE / "charts"

MODELS = [
    ("qwen35_4b", "Qwen3.5 4B", "#147D92"),
    ("qwen3_4b_baseline", "Qwen3 4B Base", "#7A8694"),
    ("epoch_001", "Epoch 1", "#E29A2D"),
    ("epoch_003", "Epoch 3", "#2F8F62"),
    ("epoch_005", "Epoch 5", "#B34D72"),
]
KS = (1, 2, 3)
INK = "#17212B"
MUTED = "#677483"
GRID = "#D8DEE6"
BG = "#F7F8FA"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": "#FFFFFF",
            "savefig.facecolor": BG,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelcolor": MUTED,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def load_rows() -> list[dict[str, Any]]:
    rows = []
    expected_harness = None
    for key, label, color in MODELS:
        path = SOURCE_DIR / key / "combined_pass_at_k_summary.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["task_count"] != 90 or row["valid_samples"] != 270:
            raise RuntimeError(f"Incomplete result for {key}: {row['task_count']} tasks, {row['valid_samples']} samples")
        if expected_harness is None:
            expected_harness = row["harness"]
        elif row["harness"] != expected_harness:
            raise RuntimeError(f"Harness mismatch for {key}")
        row.update({"key": key, "label": label, "color": color})
        rows.append(row)
    return rows


def finish_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=220, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def padded_limits(values: list[float], minimum_span: float = 0.18, pad: float = 0.04) -> tuple[float, float]:
    low = max(0.0, min(values) - pad)
    high = min(1.0, max(values) + pad)
    if high - low < minimum_span:
        midpoint = (high + low) / 2
        low = max(0.0, midpoint - minimum_span / 2)
        high = min(1.0, midpoint + minimum_span / 2)
    return low, high


def overall_bars(rows: list[dict[str, Any]], *, full_scale: bool) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 7.2), constrained_layout=True)
    x = list(range(len(KS)))
    width = 0.15
    offsets = [(index - (len(rows) - 1) / 2) * width for index in range(len(rows))]
    all_values = []
    for offset, row in zip(offsets, rows):
        values = [row["aggregate"][f"pass@{k}"] for k in KS]
        all_values.extend(values)
        bars = ax.bar(
            [position + offset for position in x],
            values,
            width=width * 0.9,
            color=row["color"],
            label=row["label"],
            zorder=3,
        )
        ax.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=4, fontsize=9, color=INK)

    ax.set_xticks(x, [f"Pass@{k}" for k in KS], fontsize=12, color=INK)
    if full_scale:
        ax.set_ylim(0, 1)
        title = "Harness V3 overall Pass@k, full scale"
        subtitle = "Absolute 0–100% axis retained for conservative visual comparison"
        stem = "overall_pass_at_k_full_scale"
    else:
        ax.set_ylim(*padded_limits(all_values, minimum_span=0.36, pad=0.035))
        title = "Harness V3 overall Pass@k"
        subtitle = "Zoomed axis for checkpoint comparison; 90 tasks and 270 valid samples per model"
        stem = "overall_pass_at_k"
    ax.set_title(title, loc="left", pad=28, color=INK)
    ax.text(0, 1.025, subtitle, transform=ax.transAxes, color=MUTED, fontsize=10.5)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.10), handlelength=1.2, columnspacing=1.5)
    finish_axis(ax)
    save_figure(fig, stem)


def benchmark_panels(rows: list[dict[str, Any]]) -> None:
    panels = [
        ("Overall", None),
        ("HumanEval+", "humanevalplus"),
        ("MBPP+", "mbppplus"),
        ("SWE-smith", "swesmith_py"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.13, hspace=0.46, wspace=0.16)
    fig.suptitle("Harness V3 Pass@k by benchmark", x=0.07, y=0.975, ha="left", fontsize=19, fontweight="bold", color=INK)
    fig.text(
        0.055,
        0.925,
        "Each panel uses a local y-axis so differences remain legible; exact values are in metrics.csv",
        color=MUTED,
        fontsize=10.5,
    )
    handles = []
    for ax, (title, dataset_key) in zip(axes.flat, panels):
        panel_values = []
        for row in rows:
            metrics = row["aggregate"] if dataset_key is None else row["by_dataset"][dataset_key]
            values = [metrics[f"pass@{k}"] for k in KS]
            panel_values.extend(values)
            line = ax.plot(KS, values, marker="o", markersize=6, linewidth=2.4, color=row["color"], label=row["label"])[0]
            if len(handles) < len(rows):
                handles.append(line)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xticks(KS, [f"P@{k}" for k in KS])
        ax.set_xlim(0.75, 3.25)
        ax.set_ylim(*padded_limits(panel_values, minimum_span=0.25, pad=0.045))
        finish_axis(ax)
    fig.legend(handles, [row["label"] for row in rows], ncol=5, loc="lower center", bbox_to_anchor=(0.5, 0.025))
    save_figure(fig, "pass_at_k_by_benchmark")


def epoch_trend(rows: list[dict[str, Any]]) -> None:
    stages = [rows[1], rows[2], rows[3], rows[4]]
    stage_labels = ["Base", "Epoch 1", "Epoch 3", "Epoch 5"]
    pass_colors = {1: "#376996", 2: "#D17A22", 3: "#2F8F62"}
    panels: list[tuple[str, Callable[[dict[str, Any], int], float]]] = [
        ("Overall", lambda row, k: row["aggregate"][f"pass@{k}"]),
        ("SWE-smith", lambda row, k: row["by_dataset"]["swesmith_py"][f"pass@{k}"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.2))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.18, wspace=0.17)
    fig.suptitle("Qwen3 4B fine-tuning progression", x=0.07, y=0.97, ha="left", fontsize=19, fontweight="bold", color=INK)
    fig.text(0.07, 0.91, "Epoch 3 is the strongest fine-tuned checkpoint; Epoch 5 regresses slightly", color=MUTED, fontsize=10.5)
    handles = []
    for ax, (title, getter) in zip(axes, panels):
        panel_values = []
        for k in KS:
            values = [getter(stage, k) for stage in stages]
            panel_values.extend(values)
            line = ax.plot(stage_labels, values, marker="o", markersize=7, linewidth=2.7, color=pass_colors[k], label=f"Pass@{k}")[0]
            if len(handles) < 3:
                handles.append(line)
            if k in (1, 3):
                vertical_offset = 8 if k == 3 else -14
                for index, value in enumerate(values):
                    ax.annotate(
                        f"{value:.1%}",
                        (index, value),
                        xytext=(0, vertical_offset),
                        textcoords="offset points",
                        ha="center",
                        va="bottom" if k == 3 else "top",
                        fontsize=8.5,
                        color=INK,
                    )
        ax.set_title(title, loc="left", color=INK)
        ax.set_ylim(*padded_limits(panel_values, minimum_span=0.27, pad=0.045))
        finish_axis(ax)
    fig.legend(handles, [f"Pass@{k}" for k in KS], ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.035))
    save_figure(fig, "finetuning_epoch_trend")


def failure_buckets(statuses: dict[str, int]) -> dict[str, int]:
    buckets = {name: 0 for name in ("resolved", "tests failed", "no edit", "max steps", "parse error", "context overflow", "other")}
    for status, count in statuses.items():
        if status == "resolved":
            bucket = "resolved"
        elif "tests_failed" in status:
            bucket = "tests failed"
        elif "no_edit" in status:
            bucket = "no edit"
        elif "max_steps" in status:
            bucket = "max steps"
        elif "parse_error" in status:
            bucket = "parse error"
        elif "context_overflow" in status:
            bucket = "context overflow"
        else:
            bucket = "other"
        buckets[bucket] += count
    return buckets


def outcome_distribution(rows: list[dict[str, Any]]) -> None:
    categories = ["resolved", "tests failed", "no edit", "max steps", "parse error", "context overflow", "other"]
    colors = ["#2F8F62", "#D17A22", "#B34D72", "#735CDD", "#C94747", "#3997A8", "#AAB1BA"]
    fig, ax = plt.subplots(figsize=(12.8, 6.8), constrained_layout=True)
    left = [0.0] * len(rows)
    y = list(range(len(rows)))
    for category, color in zip(categories, colors):
        values = [failure_buckets(row["verifier_status"])[category] / 270 for row in rows]
        bars = ax.barh(y, values, left=left, height=0.62, color=color, label=category, zorder=3)
        for bar, value in zip(bars, values):
            if value >= 0.055:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{round(value * 270):d}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if category != "other" else INK,
                    fontweight="bold",
                )
        left = [current + value for current, value in zip(left, values)]
    ax.set_yticks(y, [row["label"] for row in rows], color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title("Verifier outcome distribution", loc="left", pad=28, color=INK)
    ax.text(0, 1.025, "Counts across 270 valid samples per model; categories are mutually exclusive", transform=ax.transAxes, color=MUTED)
    ax.legend(ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), columnspacing=1.2)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    save_figure(fig, "verifier_outcome_distribution")


def write_data(rows: list[dict[str, Any]]) -> None:
    analysis = {
        "format": "lottie_v3_eval_chart_data_v1",
        "task_count_per_model": 90,
        "rollouts_per_task": 3,
        "valid_samples_per_model": 270,
        "total_valid_samples": 1350,
        "harness": rows[0]["harness"],
        "models": rows,
    }
    (HERE / "chart_data.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (HERE / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "benchmark", "pass@1", "pass@2", "pass@3", "resolved_rollouts", "valid_samples"])
        for row in rows:
            datasets = [("overall", row["aggregate"]), *sorted(row["by_dataset"].items())]
            for benchmark, metrics in datasets:
                writer.writerow(
                    [
                        row["label"],
                        benchmark,
                        f"{metrics['pass@1']:.10f}",
                        f"{metrics['pass@2']:.10f}",
                        f"{metrics['pass@3']:.10f}",
                        row["resolved_rollouts"] if benchmark == "overall" else "",
                        row["valid_samples"],
                    ]
                )


def write_report(rows: list[dict[str, Any]]) -> None:
    baseline, epoch3, epoch5 = rows[1], rows[3], rows[4]
    lines = [
        "# Harness V3 全量 Pass@3 评测图表",
        "",
        "- 口径：5 个模型，每个模型 90 题 x 3 rollout = 270 条，共 1350 条有效评测样本。",
        "- Harness：`lottie_code_agent_harness_v3`，策略 `append_nonmod3_action_reset_v1`。",
        "- 采样：temperature=0.5，top_p=0.95，thinking disabled。",
        "- 步数：函数题 20 步，SWE-smith 60 步。",
        "",
        "| 模型 | Overall P@1 | P@2 | P@3 | SWE-smith P@1 | P@2 | P@3 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        overall = row["aggregate"]
        smith = row["by_dataset"]["swesmith_py"]
        lines.append(
            f"| {row['label']} | {overall['pass@1']:.2%} | {overall['pass@2']:.2%} | {overall['pass@3']:.2%} "
            f"| {smith['pass@1']:.2%} | {smith['pass@2']:.2%} | {smith['pass@3']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- Epoch 3 是本轮最优微调 checkpoint：Overall P@1 相比 baseline 提升 {(epoch3['aggregate']['pass@1'] - baseline['aggregate']['pass@1']):.2%}，SWE-smith P@3 提升 {(epoch3['by_dataset']['swesmith_py']['pass@3'] - baseline['by_dataset']['swesmith_py']['pass@3']):.2%}。",
            f"- Epoch 5 相比 Epoch 3 略有回落：Overall P@1 变化 {(epoch5['aggregate']['pass@1'] - epoch3['aggregate']['pass@1']):.2%}，SWE-smith P@1 变化 {(epoch5['by_dataset']['swesmith_py']['pass@1'] - epoch3['by_dataset']['swesmith_py']['pass@1']):.2%}。",
            "- Qwen3.5 4B 仍是最强参照，但 Epoch 3 已明显缩小 Qwen3 4B baseline 在仓库级任务上的差距。",
            "",
            "## 图表口径",
            "",
            "- `overall_pass_at_k.png` 使用收紧纵轴，便于观察 checkpoint 差异。",
            "- `overall_pass_at_k_full_scale.png` 使用 0–100% 纵轴，避免视觉放大造成误判。",
            "- 每张图同时提供 PNG 和 PDF；PDF 可直接用于论文或演示文稿。",
        ]
    )
    (HERE / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    overall_bars(rows, full_scale=False)
    overall_bars(rows, full_scale=True)
    benchmark_panels(rows)
    epoch_trend(rows)
    outcome_distribution(rows)
    write_data(rows)
    write_report(rows)
    print(json.dumps({"charts": 5, "formats": ["png", "pdf"], "models": 5, "valid_samples": 1350}))


if __name__ == "__main__":
    main()
