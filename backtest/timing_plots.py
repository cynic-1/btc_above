"""
时间窗口实验可视化

热力图、边际效应图、增量贡献图
"""

import logging
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .timing_experiment import TimingWindowResult

logger = logging.getLogger(__name__)


def _build_grid_matrix(
    results: List[TimingWindowResult],
    value_fn,
) -> tuple:
    """
    从结果列表构建热力图矩阵

    Returns: (matrix, start_labels, stop_labels)
    """
    starts = sorted(set(r.start_minutes for r in results), reverse=True)
    stops = sorted(set(r.stop_minutes for r in results))

    matrix = np.full((len(starts), len(stops)), np.nan)
    start_idx = {s: i for i, s in enumerate(starts)}
    stop_idx = {s: i for i, s in enumerate(stops)}

    for r in results:
        i = start_idx.get(r.start_minutes)
        j = stop_idx.get(r.stop_minutes)
        if i is not None and j is not None:
            val = value_fn(r)
            if val is not None:
                matrix[i, j] = val

    start_labels = [f"T-{m}m" for m in starts]
    stop_labels = [f"T-{m}m" for m in stops]
    return matrix, start_labels, stop_labels


def _plot_single_heatmap(
    matrix: np.ndarray,
    start_labels: list,
    stop_labels: list,
    title: str,
    cmap: str,
    fmt: str,
    output_path: str,
):
    """绘制单张热力图"""
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(stop_labels)))
    ax.set_xticklabels(stop_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(start_labels)))
    ax.set_yticklabels(start_labels)
    ax.set_xlabel("Stop Time (before expiry)")
    ax.set_ylabel("Start Time (before expiry)")
    ax.set_title(title)

    # 标注数值
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isnan(val):
                text = fmt.format(val)
                color = "white" if abs(val - np.nanmean(matrix)) > np.nanstd(matrix) else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"热力图已保存: {output_path}")


def plot_heatmaps(results: List[TimingWindowResult], output_dir: str):
    """绘制 4 张热力图：PnL, Profit Factor, Sharpe, ROI"""
    os.makedirs(output_dir, exist_ok=True)

    configs = [
        ("PnL ($)", lambda r: r.total_pnl, "RdYlGn", "{:.0f}",
         os.path.join(output_dir, "heatmap_pnl.png")),
        ("Profit Factor", lambda r: min(r.profit_factor, 10.0) if r.profit_factor != float("inf") else 10.0,
         "RdYlGn", "{:.2f}",
         os.path.join(output_dir, "heatmap_profit_factor.png")),
        ("Sharpe Ratio", lambda r: r.sharpe, "RdYlGn", "{:.2f}",
         os.path.join(output_dir, "heatmap_sharpe.png")),
        ("ROI", lambda r: r.roi * 100, "RdYlGn", "{:.1f}%",
         os.path.join(output_dir, "heatmap_roi.png")),
    ]

    for title, value_fn, cmap, fmt, path in configs:
        matrix, start_labels, stop_labels = _build_grid_matrix(results, value_fn)
        _plot_single_heatmap(matrix, start_labels, stop_labels, title, cmap, fmt, path)


def plot_marginal_effects(results: List[TimingWindowResult], output_dir: str):
    """
    绘制 2 张柱状图：start 边际效应、stop 边际效应

    边际效应 = 固定另一个维度，对该维度各值的 PnL 取平均
    """
    os.makedirs(output_dir, exist_ok=True)

    # Start 边际效应
    start_pnl: Dict[int, List[float]] = {}
    stop_pnl: Dict[int, List[float]] = {}

    for r in results:
        if r.n_trades == 0:
            continue
        start_pnl.setdefault(r.start_minutes, []).append(r.total_pnl)
        stop_pnl.setdefault(r.stop_minutes, []).append(r.total_pnl)

    # Start 边际
    fig, ax = plt.subplots(figsize=(8, 5))
    starts = sorted(start_pnl.keys(), reverse=True)
    means = [np.mean(start_pnl[s]) for s in starts]
    labels = [f"T-{s}m" for s in starts]
    colors = ["green" if m > 0 else "red" for m in means]
    ax.bar(labels, means, color=colors, alpha=0.7)
    ax.set_xlabel("Start Time")
    ax.set_ylabel("Avg PnL ($)")
    ax.set_title("Start Time Marginal Effect (Avg PnL)")
    ax.axhline(y=0, color="black", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "marginal_start.png"), dpi=150)
    plt.close(fig)

    # Stop 边际
    fig, ax = plt.subplots(figsize=(8, 5))
    stops = sorted(stop_pnl.keys())
    means = [np.mean(stop_pnl[s]) for s in stops]
    labels = [f"T-{s}m" for s in stops]
    colors = ["green" if m > 0 else "red" for m in means]
    ax.bar(labels, means, color=colors, alpha=0.7)
    ax.set_xlabel("Stop Time")
    ax.set_ylabel("Avg PnL ($)")
    ax.set_title("Stop Time Marginal Effect (Avg PnL)")
    ax.axhline(y=0, color="black", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "marginal_stop.png"), dpi=150)
    plt.close(fig)

    logger.info("边际效应图已保存")


def plot_incremental_value(incremental_results: List[Dict], output_dir: str):
    """
    绘制每 30 分钟时间段的增量 PnL 贡献柱状图
    """
    os.makedirs(output_dir, exist_ok=True)

    # 过滤有数据的桶
    buckets = [r for r in incremental_results if r["n_obs_in_bucket"] > 0]
    if not buckets:
        logger.warning("无增量数据可绘图")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    labels = [r["bucket"] for r in buckets]
    values = [r["incremental_pnl"] for r in buckets]
    colors = ["green" if v > 0 else "red" for v in values]

    ax.bar(range(len(labels)), values, color=colors, alpha=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Incremental PnL ($)")
    ax.set_title("Incremental PnL by 30-min Bucket")
    ax.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "incremental_value.png"), dpi=150)
    plt.close(fig)
    logger.info("增量贡献图已保存")
