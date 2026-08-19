#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lottie-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, PercentFormatter


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
EVAL_DATA_PATH = HERE / "chart_data.json"
TOKEN_DATA_PATH = PROJECT / "outputs/training_packages/lottie_train1200_cleaned_complete_v1/metadata/qwen3_4b_token_lengths.json"
TRAIN_CONFIG_PATH = HERE.parent / "qwen_sft_mask_v2_20260815/experiment_config_e2_v1.json"
TRAIN_LOG_DIR = HERE.parent / "qwen_sft_mask_v2_20260815/curves_e1_e5"
OUTPUT_DIR = HERE / "charts/detailed"
ANALYSIS_PATH = HERE / "detailed_analysis_data.json"

LOG_PATHS = [
    TRAIN_LOG_DIR / "epoch1_2.jsonl",
    TRAIN_LOG_DIR / "epoch3.jsonl",
    TRAIN_LOG_DIR / "epoch4_5_live.jsonl",
]

INK = "#17212B"
MUTED = "#687585"
GRID = "#D8DEE6"
BG = "#F6F8FA"
WHITE = "#FFFFFF"
BLUE = "#2C6EAA"
TEAL = "#147D92"
GREEN = "#2F8F62"
AMBER = "#D88A20"
MAGENTA = "#B34D72"
RED = "#C94747"
VIOLET = "#735CDD"

MODEL_ORDER = ["qwen35_4b", "qwen3_4b_baseline", "epoch_001", "epoch_003", "epoch_005"]
FT_ORDER = ["qwen3_4b_baseline", "epoch_001", "epoch_003", "epoch_005"]
MODEL_LABELS = {
    "qwen35_4b": "Qwen3.5 4B",
    "qwen3_4b_baseline": "Qwen3 4B Base",
    "epoch_001": "Epoch 1",
    "epoch_003": "Epoch 3",
    "epoch_005": "Epoch 5",
}
MODEL_COLORS = {
    "qwen35_4b": TEAL,
    "qwen3_4b_baseline": "#7A8694",
    "epoch_001": AMBER,
    "epoch_003": GREEN,
    "epoch_005": MAGENTA,
}
DATASET_LABELS = {
    "humanevalplus": "HumanEval+",
    "mbppplus": "MBPP+",
    "swesmith_py": "SWE-smith",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": WHITE,
            "savefig.facecolor": BG,
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 15,
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=220, bbox_inches="tight", pad_inches=0.16)
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def clean_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


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


def task_consistency(metrics: dict[str, float], task_count: int = 90) -> list[int]:
    p1, p2, p3 = (metrics[f"pass@{k}"] for k in (1, 2, 3))
    successful_tasks = task_count * p3
    one = 3 * task_count * (p3 - p2)
    total_successes = 3 * task_count * p1
    three = total_successes + one - 2 * successful_tasks
    two = successful_tasks - one - three
    zero = task_count - successful_tasks
    counts = [round(value) for value in (zero, one, two, three)]
    if sum(counts) != task_count:
        counts[0] += task_count - sum(counts)
    return counts


def task_contribution_variance(counts: list[int], task_count: int = 90) -> dict[str, tuple[float, float, float]]:
    successes = np.repeat(np.arange(4), counts)
    contributions = {
        "pass@1": successes / 3,
        "pass@2": np.where(successes == 0, 0.0, np.where(successes == 1, 2 / 3, 1.0)),
        "pass@3": (successes > 0).astype(float),
    }
    result = {}
    for key, values in contributions.items():
        mean = float(np.mean(values))
        se = float(np.std(values, ddof=1) / math.sqrt(task_count))
        result[key] = (mean, max(0.0, mean - 1.96 * se), min(1.0, mean + 1.96 * se))
    return result


def load_training_logs() -> dict[int, list[dict[str, Any]]]:
    epochs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for path in LOG_PATHS:
        for row in load_jsonl(path):
            epochs[int(row["epoch"])].append(row)
    for epoch in epochs:
        epochs[epoch].sort(key=lambda row: int(row["micro_step"]))
    return dict(sorted(epochs.items()))


def summarize_epoch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    losses = np.array([float(row["loss"]) for row in rows])
    supervised = np.array([int(row["supervised_tokens"]) for row in rows])
    real_tokens = np.array([int(row["real_tokens"]) for row in rows])
    gradients = np.array([float(row["gradient_norm"]) for row in rows if row.get("gradient_norm") is not None])
    learning_rates = np.array([float(row["learning_rate"]) for row in rows])
    summary: dict[str, Any] = {
        "microbatches": len(rows),
        "examples": int(sum(int(row["examples"]) for row in rows)),
        "context_tokens": int(real_tokens.sum()),
        "supervised_tokens": int(supervised.sum()),
        "supervised_share": float(supervised.sum() / real_tokens.sum()),
        "weighted_loss": float(np.average(losses, weights=supervised)),
        "median_loss": float(np.median(losses)),
        "learning_rate_max": float(learning_rates.max()),
        "learning_rate_end": float(learning_rates[-1]),
        "peak_allocated_gib": float(max(float(row.get("peak_allocated_gib", 0)) for row in rows)),
        "peak_reserved_gib": float(max(float(row.get("peak_reserved_gib", 0)) for row in rows)),
    }
    if gradients.size:
        summary["gradient_norm_median"] = float(np.median(gradients))
        summary["gradient_norm_p95"] = float(np.quantile(gradients, 0.95))
    return summary


def rolling_mean(values: np.ndarray, window: int = 25) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_task_consistency(models: dict[str, dict[str, Any]], consistency: dict[str, list[int]]) -> None:
    fig, ax = plt.subplots(figsize=(12.4, 6.9), constrained_layout=True)
    categories = ["0 / 3", "1 / 3", "2 / 3", "3 / 3"]
    colors = ["#DDE2E8", "#E6B657", "#68A8B5", GREEN]
    y = np.arange(len(MODEL_ORDER))
    left = np.zeros(len(MODEL_ORDER))
    for index, (category, color) in enumerate(zip(categories, colors)):
        values = np.array([consistency[key][index] for key in MODEL_ORDER])
        bars = ax.barh(y, values, left=left, height=0.62, color=color, label=category, zorder=3)
        for bar, value in zip(bars, values):
            if value >= 5:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, str(value), ha="center", va="center", color=INK, fontweight="bold", fontsize=9)
        left += values
    ax.set_yticks(y, [MODEL_LABELS[key] for key in MODEL_ORDER], color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 90)
    ax.set_xlabel("Tasks")
    ax.set_title("Task consistency across three rollouts", loc="left", pad=28, color=INK)
    ax.text(0, 1.025, "How many of each model's three attempts passed on the same task", transform=ax.transAxes, color=MUTED)
    ax.legend(title="Successful rollouts", ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    clean_axis(ax, "x")
    save_figure(fig, "task_consistency_distribution")


def metric_vector(model: dict[str, Any]) -> tuple[list[str], list[float]]:
    labels = ["Overall P@1", "Overall P@3"]
    values = [model["aggregate"]["pass@1"], model["aggregate"]["pass@3"]]
    for dataset in ("humanevalplus", "mbppplus", "swesmith_py"):
        label = DATASET_LABELS[dataset]
        labels.extend([f"{label} P@1", f"{label} P@3"])
        values.extend([model["by_dataset"][dataset]["pass@1"], model["by_dataset"][dataset]["pass@3"]])
    return labels, values


def draw_heatmap(ax: plt.Axes, values: np.ndarray, rows: list[str], columns: list[str], title: str) -> None:
    limit = max(0.10, float(np.max(np.abs(values))))
    cmap = LinearSegmentedColormap.from_list("delta", ["#B94C5C", "#F8F9FA", "#2D8B68"])
    image = ax.imshow(values, cmap=cmap, norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
    ax.set_xticks(np.arange(len(columns)), columns, color=INK)
    ax.set_yticks(np.arange(len(rows)), rows, color=INK)
    ax.set_title(title, loc="left", color=INK, pad=14)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            ax.text(j, i, f"{value * 100:+.1f}", ha="center", va="center", color=INK, fontsize=9, fontweight="bold")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def plot_delta_heatmap(models: dict[str, dict[str, Any]]) -> None:
    metric_labels, baseline_values = metric_vector(models["qwen3_4b_baseline"])
    _, q35_values = metric_vector(models["qwen35_4b"])
    checkpoint_keys = ["epoch_001", "epoch_003", "epoch_005"]
    checkpoint_values = [metric_vector(models[key])[1] for key in checkpoint_keys]
    vs_base = np.array(checkpoint_values).T - np.array(baseline_values)[:, None]
    vs_q35 = np.array(checkpoint_values).T - np.array(q35_values)[:, None]

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 8.2))
    fig.subplots_adjust(left=0.12, right=0.985, top=0.83, bottom=0.10, wspace=0.36)
    fig.suptitle("Checkpoint deltas by capability", x=0.08, y=0.965, ha="left", fontsize=19, fontweight="bold", color=INK)
    fig.text(0.08, 0.915, "Cell values are percentage-point differences; green is better", color=MUTED)
    columns = ["E1", "E3", "E5"]
    draw_heatmap(axes[0], vs_base, metric_labels, columns, "Fine-tuned checkpoint minus Qwen3 Base")
    draw_heatmap(axes[1], vs_q35, metric_labels, columns, "Fine-tuned checkpoint minus Qwen3.5 4B")
    save_figure(fig, "checkpoint_delta_heatmap")


def plot_data_composition(token_data: dict[str, Any]) -> None:
    datasets = ["humanevalplus", "mbppplus", "swesmith_py"]
    labels = [DATASET_LABELS[key] for key in datasets]
    rows = np.array([token_data["by_dataset"][key]["rows"] for key in datasets])
    tokens = np.array([token_data["by_dataset"][key]["total_tokens"] for key in datasets])
    medians = np.array([token_data["by_dataset"][key]["p50_tokens"] for key in datasets])
    p95s = np.array([token_data["by_dataset"][key]["p95_tokens"] for key in datasets])

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.4))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.18, wspace=0.22)
    fig.suptitle("Primary SFT data composition", x=0.075, y=0.965, ha="left", fontsize=19, fontweight="bold", color=INK)
    fig.text(0.075, 0.91, "1,069 cleaned trajectories; Qwen3 tokenizer; 7.58M context tokens", color=MUTED)

    x = np.arange(len(labels))
    width = 0.34
    bars_rows = axes[0].bar(x - width / 2, rows / rows.sum(), width, color=BLUE, label="Row share", zorder=3)
    bars_tokens = axes[0].bar(x + width / 2, tokens / tokens.sum(), width, color=AMBER, label="Token share", zorder=3)
    axes[0].bar_label(bars_rows, labels=[f"{value:.1%}" for value in rows / rows.sum()], padding=4, fontsize=9)
    axes[0].bar_label(bars_tokens, labels=[f"{value:.1%}" for value in tokens / tokens.sum()], padding=4, fontsize=9)
    axes[0].set_xticks(x, labels, color=INK)
    axes[0].set_ylim(0, 0.76)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_title("Examples versus token mass", loc="left", color=INK)
    axes[0].legend(loc="upper left")
    clean_axis(axes[0])

    bars_median = axes[1].bar(x - width / 2, medians, width, color=TEAL, label="Median", zorder=3)
    bars_p95 = axes[1].bar(x + width / 2, p95s, width, color=MAGENTA, label="P95", zorder=3)
    axes[1].bar_label(bars_median, labels=[f"{value / 1000:.1f}k" for value in medians], padding=4, fontsize=9)
    axes[1].bar_label(bars_p95, labels=[f"{value / 1000:.1f}k" for value in p95s], padding=4, fontsize=9)
    axes[1].set_xticks(x, labels, color=INK)
    axes[1].set_ylim(0, 28500)
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.0f}k"))
    axes[1].set_title("Sequence length by benchmark", loc="left", color=INK)
    axes[1].set_ylabel("Context tokens")
    axes[1].legend(loc="upper left")
    clean_axis(axes[1])
    save_figure(fig, "training_data_composition")


def plot_token_buckets(token_data: dict[str, Any]) -> None:
    datasets = ["humanevalplus", "mbppplus", "swesmith_py"]
    labels = [DATASET_LABELS[key] for key in datasets]
    bucket_order = ["1-4096", "4097-8192", "8193-16384", "16385-32768", ">32768"]
    bucket_labels = ["<=4k", "4-8k", "8-16k", "16-32k", ">32k"]
    colors = ["#B8D3DE", "#77AFC0", "#39899F", "#805DA9", MAGENTA]
    fig, ax = plt.subplots(figsize=(11.8, 6.7), constrained_layout=True)
    y = np.arange(len(datasets))
    left = np.zeros(len(datasets))
    totals = np.array([token_data["by_dataset"][key]["rows"] for key in datasets])
    for bucket, label, color in zip(bucket_order, bucket_labels, colors):
        counts = np.array([token_data["by_dataset"][key]["buckets"].get(bucket, 0) for key in datasets])
        shares = counts / totals
        bars = ax.barh(y, shares, left=left, height=0.60, color=color, label=label, zorder=3)
        for bar, count, share in zip(bars, counts, shares):
            if share >= 0.055:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, str(int(count)), ha="center", va="center", fontsize=9, color=INK, fontweight="bold")
        left += shares
    ax.set_yticks(y, labels, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Sequence-length bucket distribution", loc="left", pad=28, color=INK)
    ax.text(0, 1.025, "Counts are shown inside each segment; the two >32k rows are both SWE-smith", transform=ax.transAxes, color=MUTED)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    clean_axis(ax, "x")
    save_figure(fig, "token_length_buckets")


def plot_training_dynamics(epoch_logs: dict[int, list[dict[str, Any]]]) -> None:
    global_x = []
    losses = []
    learning_rates = []
    gradients_x = []
    gradients = []
    for epoch in range(1, 6):
        rows = epoch_logs[epoch]
        denominator = max(1, len(rows) - 1)
        for index, row in enumerate(rows):
            x = epoch - 1 + index / denominator
            global_x.append(x)
            losses.append(float(row["loss"]))
            learning_rates.append(float(row["learning_rate"]))
            if row.get("gradient_norm") is not None:
                gradients_x.append(x)
                gradients.append(float(row["gradient_norm"]))
    global_x_np = np.array(global_x)
    losses_np = np.array(losses)
    learning_rates_np = np.array(learning_rates)

    fig, axes = plt.subplots(3, 1, figsize=(13.2, 9.2), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.09, hspace=0.20)
    fig.suptitle("Five-epoch training dynamics", x=0.075, y=0.97, ha="left", fontsize=19, fontweight="bold", color=INK)
    fig.text(0.075, 0.925, "Full microbatch logs; loss uses a 25-microbatch rolling mean", color=MUTED)

    axes[0].plot(global_x_np, losses_np, color="#B8C1CC", linewidth=0.6, alpha=0.35)
    axes[0].plot(global_x_np, rolling_mean(losses_np, 25), color=BLUE, linewidth=2.1)
    axes[0].set_ylabel("SFT loss")
    axes[0].set_ylim(0, min(1.05, float(np.quantile(losses_np, 0.995)) * 1.08))
    clean_axis(axes[0])

    axes[1].plot(global_x_np, learning_rates_np * 1e5, color=AMBER, linewidth=2.2)
    axes[1].set_ylabel("LR (x1e-5)")
    axes[1].set_ylim(bottom=0)
    clean_axis(axes[1])

    axes[2].scatter(gradients_x, gradients, s=8, color=MAGENTA, alpha=0.35, edgecolors="none")
    axes[2].plot(gradients_x, rolling_mean(np.array(gradients), 25), color=MAGENTA, linewidth=2.0)
    axes[2].set_ylabel("Gradient norm")
    axes[2].set_ylim(0, min(1.4, float(np.quantile(gradients, 0.995)) * 1.08))
    clean_axis(axes[2])

    for ax in axes:
        for boundary in (1, 2, 3, 4):
            ax.axvline(boundary, color=GRID, linewidth=1.0, linestyle="--")
    axes[2].set_xlim(0, 5)
    axes[2].set_xticks(np.arange(0.5, 5.5, 1), ["Epoch 1", "Epoch 2", "Epoch 3", "Epoch 4", "Epoch 5"], color=INK)
    save_figure(fig, "training_dynamics_e1_e5")


def plot_loss_eval_divergence(models: dict[str, dict[str, Any]], epoch_stats: dict[int, dict[str, Any]]) -> None:
    epochs = np.arange(1, 6)
    losses = np.array([epoch_stats[index]["weighted_loss"] for index in epochs])
    evaluated_epochs = np.array([1, 3, 5])
    eval_keys = ["epoch_001", "epoch_003", "epoch_005"]
    overall_p1 = np.array([models[key]["aggregate"]["pass@1"] for key in eval_keys])
    overall_p3 = np.array([models[key]["aggregate"]["pass@3"] for key in eval_keys])
    smith_p1 = np.array([models[key]["by_dataset"]["swesmith_py"]["pass@1"] for key in eval_keys])
    smith_p3 = np.array([models[key]["by_dataset"]["swesmith_py"]["pass@3"] for key in eval_keys])

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.6))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.78, bottom=0.18, wspace=0.22)
    fig.suptitle("Training loss keeps falling after evaluation peaks", x=0.07, y=0.96, ha="left", fontsize=19, fontweight="bold", color=INK)
    fig.text(0.07, 0.89, "Epoch 3 is the best checkpoint despite lower train loss at Epoch 5", color=MUTED)

    axes[0].plot(epochs, losses, marker="o", linewidth=2.5, color=BLUE)
    for x, value in zip(epochs, losses):
        axes[0].annotate(f"{value:.3f}", (x, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    axes[0].set_title("Token-weighted train loss", loc="left", color=INK)
    axes[0].set_xticks(epochs)
    axes[0].set_ylim(0.14, 0.31)
    clean_axis(axes[0])

    for ax, title, p1_values, p3_values in (
        (axes[1], "Overall evaluation", overall_p1, overall_p3),
        (axes[2], "SWE-smith evaluation", smith_p1, smith_p3),
    ):
        ax.plot(evaluated_epochs, p1_values, marker="o", linewidth=2.5, color=AMBER, label="Pass@1")
        ax.plot(evaluated_epochs, p3_values, marker="o", linewidth=2.5, color=GREEN, label="Pass@3")
        for x, value in zip(evaluated_epochs, p1_values):
            ax.annotate(f"{value:.1%}", (x, value), xytext=(0, -15), textcoords="offset points", ha="center", fontsize=8.5)
        for x, value in zip(evaluated_epochs, p3_values):
            ax.annotate(f"{value:.1%}", (x, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xticks(evaluated_epochs)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        values = np.concatenate([p1_values, p3_values])
        ax.set_ylim(max(0, values.min() - 0.08), min(1, values.max() + 0.08))
        clean_axis(ax)
    axes[1].legend(loc="lower center", bbox_to_anchor=(1.1, -0.27), ncol=2)
    save_figure(fig, "loss_vs_evaluation_divergence")


def plot_failure_heatmap(models: dict[str, dict[str, Any]]) -> None:
    categories = ["resolved", "tests failed", "no edit", "max steps", "parse error", "context overflow", "other"]
    matrix = np.array([[failure_buckets(models[key]["verifier_status"])[category] / 270 for category in categories] for key in MODEL_ORDER])
    fig, ax = plt.subplots(figsize=(12.7, 6.1), constrained_layout=True)
    image = ax.imshow(matrix, cmap=LinearSegmentedColormap.from_list("failure", ["#F7F8FA", "#8CB6C1", "#225A6B"]), vmin=0, vmax=0.68, aspect="auto")
    ax.set_xticks(np.arange(len(categories)), categories, color=INK)
    ax.set_yticks(np.arange(len(MODEL_ORDER)), [MODEL_LABELS[key] for key in MODEL_ORDER], color=INK)
    ax.set_title("Verifier outcome heatmap", loc="left", pad=28, color=INK)
    ax.text(0, 1.025, "Share of 270 valid rollouts; cells show count and percentage", transform=ax.transAxes, color=MUTED)
    for i in range(matrix.shape[0]):
        buckets = failure_buckets(models[MODEL_ORDER[i]]["verifier_status"])
        for j, category in enumerate(categories):
            count = buckets[category]
            value = matrix[i, j]
            color = WHITE if value > 0.38 else INK
            ax.text(j, i, f"{count}\n{value:.0%}", ha="center", va="center", color=color, fontsize=8.5, fontweight="bold")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.025)
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    save_figure(fig, "failure_mode_heatmap")


def plot_capability_tradeoff(models: dict[str, dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 7.3), constrained_layout=True)
    points = {}
    for key in MODEL_ORDER:
        model = models[key]
        function_p1 = np.mean([model["by_dataset"][dataset]["pass@1"] for dataset in ("humanevalplus", "mbppplus")])
        smith_p1 = model["by_dataset"]["swesmith_py"]["pass@1"]
        points[key] = (function_p1, smith_p1)
        ax.scatter(function_p1, smith_p1, s=145, color=MODEL_COLORS[key], edgecolor=WHITE, linewidth=1.4, zorder=4)
        offsets = {
            "qwen35_4b": (9, 8),
            "qwen3_4b_baseline": (-78, -4),
            "epoch_001": (9, -13),
            "epoch_003": (9, 8),
            "epoch_005": (9, -14),
        }
        ax.annotate(MODEL_LABELS[key], (function_p1, smith_p1), xytext=offsets[key], textcoords="offset points", color=INK, fontsize=9.5, fontweight="bold")
    for start, end in zip(FT_ORDER[:-1], FT_ORDER[1:]):
        ax.annotate("", xy=points[end], xytext=points[start], arrowprops={"arrowstyle": "->", "color": "#8B96A3", "linewidth": 1.5, "shrinkA": 8, "shrinkB": 8})
    ax.set_xlabel("Function-level Pass@1 (mean of HumanEval+ and MBPP+)")
    ax.set_ylabel("Repository-level SWE-smith Pass@1")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(0.66, 0.82)
    ax.set_ylim(0.07, 0.45)
    ax.set_title("Capability trade-off map", loc="left", pad=28, color=INK)
    ax.text(0, 1.025, "Upper-right is better; arrows trace Qwen3 Base -> Epoch 1 -> Epoch 3 -> Epoch 5", transform=ax.transAxes, color=MUTED)
    clean_axis(ax, "both")
    save_figure(fig, "capability_tradeoff_map")


def plot_overall_uncertainty(models: dict[str, dict[str, Any]], uncertainty: dict[str, dict[str, tuple[float, float, float]]]) -> None:
    fig, ax = plt.subplots(figsize=(11.8, 6.8), constrained_layout=True)
    x = np.arange(len(MODEL_ORDER))
    offsets = {"pass@1": -0.14, "pass@3": 0.14}
    colors = {"pass@1": AMBER, "pass@3": GREEN}
    for metric in ("pass@1", "pass@3"):
        means = np.array([uncertainty[key][metric][0] for key in MODEL_ORDER])
        lows = np.array([uncertainty[key][metric][1] for key in MODEL_ORDER])
        highs = np.array([uncertainty[key][metric][2] for key in MODEL_ORDER])
        ax.errorbar(x + offsets[metric], means, yerr=np.vstack([means - lows, highs - means]), fmt="o", markersize=8, capsize=5, linewidth=1.8, color=colors[metric], label=metric.replace("pass", "Pass"), zorder=4)
    ax.set_xticks(x, [MODEL_LABELS[key] for key in MODEL_ORDER], color=INK)
    ax.set_ylim(0.37, 0.90)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Overall Pass@1 and Pass@3 with approximate 95% intervals", loc="left", pad=28, color=INK)
    ax.text(0, 1.025, "Normal intervals over 90 task-level Pass@k contributions; paired bootstrap requires raw per-task results", transform=ax.transAxes, color=MUTED)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    clean_axis(ax)
    save_figure(fig, "overall_uncertainty_intervals")


def main() -> None:
    configure_style()
    eval_data = load_json(EVAL_DATA_PATH)
    token_data = load_json(TOKEN_DATA_PATH)
    train_config = load_json(TRAIN_CONFIG_PATH)
    models = {model["key"]: model for model in eval_data["models"]}
    epoch_logs = load_training_logs()
    if sorted(epoch_logs) != [1, 2, 3, 4, 5]:
        raise RuntimeError(f"Expected epochs 1-5, found {sorted(epoch_logs)}")
    epoch_stats = {epoch: summarize_epoch(rows) for epoch, rows in epoch_logs.items()}
    consistency = {key: task_consistency(models[key]["aggregate"]) for key in MODEL_ORDER}
    uncertainty = {key: task_contribution_variance(consistency[key]) for key in MODEL_ORDER}

    plot_task_consistency(models, consistency)
    plot_delta_heatmap(models)
    plot_data_composition(token_data)
    plot_token_buckets(token_data)
    plot_training_dynamics(epoch_logs)
    plot_loss_eval_divergence(models, epoch_stats)
    plot_failure_heatmap(models)
    plot_capability_tradeoff(models)
    plot_overall_uncertainty(models, uncertainty)

    serializable_uncertainty = {
        key: {metric: {"mean": values[0], "low_95": values[1], "high_95": values[2]} for metric, values in metrics.items()}
        for key, metrics in uncertainty.items()
    }
    output = {
        "format": "lottie_v3_detailed_analysis_v1",
        "evaluation": {
            "models": MODEL_ORDER,
            "tasks_per_model": eval_data["task_count_per_model"],
            "rollouts_per_task": eval_data["rollouts_per_task"],
            "task_consistency_counts": consistency,
            "approximate_normal_intervals": serializable_uncertainty,
        },
        "training_data": {
            "primary_rows": token_data["overall"]["rows"],
            "context_tokens": token_data["overall"]["total_tokens"],
            "length_summary": token_data["overall"],
            "by_dataset": token_data["by_dataset"],
        },
        "training_run": {
            "configured_rows": train_config["dataset_rows"],
            "configured_context_tokens_per_epoch": train_config["context_tokens_per_epoch"],
            "configured_supervised_tokens_per_epoch": train_config["supervised_tokens_per_epoch"],
            "max_sequence_length": train_config["max_sequence_length"],
            "epoch_stats": epoch_stats,
            "total_context_tokens_e1_e5": sum(stats["context_tokens"] for stats in epoch_stats.values()),
            "total_supervised_tokens_e1_e5": sum(stats["supervised_tokens"] for stats in epoch_stats.values()),
        },
        "provenance_warnings": [
            "The packaged primary set has 1069 rows, while the training config records 1071 rows.",
            "The packaged tokenizer report contains two rows over 32768 tokens, while training was capped at 32768 tokens.",
            "Task-level sample results are not archived locally, so cross-model paired bootstrap and task-level win/loss analysis are unavailable.",
        ],
    }
    ANALYSIS_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"charts": 9, "epochs": len(epoch_stats), "output": str(OUTPUT_DIR), "analysis": str(ANALYSIS_PATH)}))


if __name__ == "__main__":
    main()
