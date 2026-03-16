"""
收敛分析可视化

上半图: favorable_rate / win_rate 折线（ALL/YES/NO）
下半图: avg_mid_drift 柱状图（ALL/YES/NO）
"""

import logging
import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .convergence import ConvergenceResult, HoldingPeriodResult

logger = logging.getLogger(__name__)


def plot_convergence(result: ConvergenceResult, output_dir: str) -> str:
    """
    绘制收敛分析图表 → convergence_chart.png

    Returns: 输出文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "convergence_chart.png")

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # X 轴标签
    x_labels = [r.holding for r in result.all_results]
    x = np.arange(len(x_labels))

    # --- 上半图: 折线 ---
    colors = {"ALL": "#2196F3", "YES": "#4CAF50", "NO": "#F44336"}

    for label, results_list in [
        ("ALL", result.all_results),
        ("YES", result.yes_results),
        ("NO", result.no_results),
    ]:
        if not results_list:
            continue
        fav_rates = [r.favorable_rate * 100 for r in results_list]
        win_rates = [r.win_rate * 100 for r in results_list]
        c = colors[label]
        ax_top.plot(x, fav_rates, marker="o", color=c, linewidth=2,
                    label=f"FavR% {label}")
        ax_top.plot(x, win_rates, marker="s", color=c, linewidth=1.5,
                    linestyle="--", alpha=0.7, label=f"WinR% {label}")

    ax_top.axhline(y=50, color="gray", linewidth=0.8, linestyle=":")
    ax_top.set_ylabel("Rate (%)")
    ax_top.set_title("Favorable Rate vs Win Rate by Holding Period")
    ax_top.legend(loc="upper left", fontsize=8, ncol=2)
    ax_top.grid(axis="y", alpha=0.3)

    # --- 下半图: 柱状图 ---
    width = 0.25
    groups = [
        ("ALL", result.all_results),
        ("YES", result.yes_results),
        ("NO", result.no_results),
    ]
    offsets = [-width, 0, width]

    for (label, results_list), offset in zip(groups, offsets):
        if not results_list:
            continue
        drifts = [r.avg_mid_drift for r in results_list]
        bar_colors = [colors[label] if d >= 0 else "#BBBBBB" for d in drifts]
        ax_bot.bar(x + offset, drifts, width=width, color=bar_colors,
                   alpha=0.8, label=label, edgecolor="white", linewidth=0.5)

    ax_bot.axhline(y=0, color="black", linewidth=0.5)
    ax_bot.set_ylabel("Avg Mid Drift")
    ax_bot.set_title("Average Mid-Price Drift by Holding Period")
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(x_labels)
    ax_bot.set_xlabel("Holding Period")
    ax_bot.legend(fontsize=8)
    ax_bot.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"收敛图表已保存: {output_path}")
    return output_path
