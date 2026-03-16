#!/usr/bin/env python3
"""
价格收敛分析 CLI 入口

分析模型发现 edge 后，市场价格是否朝模型方向收敛。
对比不同持有期（30m/1h/2h/3h/6h/结算）的 PnL 表现。

用法:
    python3 run_convergence.py --detail-csv backtest_results/detail.csv
    python3 run_convergence.py --detail-csv backtest_results/detail.csv --entry-threshold 0.05
"""

import argparse
import csv
import logging
import os
import sys

from backtest.convergence import (
    ConvergenceResult,
    HoldingPeriodResult,
    HOLDINGS,
    run_convergence,
)
from backtest.convergence_chart import plot_convergence

logger = logging.getLogger(__name__)


def format_table_overall(result: ConvergenceResult) -> str:
    """格式化总体收敛表"""
    lines = []
    sep = "=" * 115
    lines.append(sep)
    lines.append(
        f"价格收敛分析 (threshold={result.entry_threshold}, "
        f"N={result.n_total_signals} signals, "
        f"shares={result.shares_per_trade})"
    )
    lines.append(sep)

    header = (
        f"{'Holding':<12}{'#Valid':>7}{'#Wins':>7}{'WinR%':>7}{'FavR%':>7}"
        f"{'AvgWin':>8}{'AvgLoss':>8}{'赔率':>7}"
        f"{'AvgPnL':>9}{'AvgDrift':>9}{'TotalPnL':>11}{'AvgEdge':>8}"
    )
    lines.append(header)
    lines.append("-" * 115)

    for r in result.all_results:
        wr = f"{r.win_rate * 100:.1f}%"
        fr = f"{r.favorable_rate * 100:.1f}%"
        pr = f"{r.payoff_ratio:.2f}" if r.payoff_ratio != float("inf") else "inf"
        avg = f"{'+' if r.avg_pnl >= 0 else ''}{r.avg_pnl:.4f}"
        drift = f"{'+' if r.avg_mid_drift >= 0 else ''}{r.avg_mid_drift:.4f}"
        total = f"${r.total_pnl:,.0f}"
        lines.append(
            f"{r.holding:<12}{r.n_with_exit:>7}{r.n_wins:>7}{wr:>7}{fr:>7}"
            f"{r.avg_win:>8.4f}{r.avg_loss:>8.4f}{pr:>7}"
            f"{avg:>9}{drift:>9}{total:>11}{r.avg_edge:>8.4f}"
        )

    lines.append(sep)
    return "\n".join(lines)


def format_table_direction(result: ConvergenceResult) -> str:
    """格式化方向拆分表"""
    lines = []
    sep = "=" * 83
    lines.append(sep)
    lines.append("方向拆分")
    lines.append(sep)

    header = (
        f"{'Holding':<12}{'Dir':<6}{'#Valid':>7}{'WinR%':>7}{'FavR%':>7}"
        f"{'AvgPnL':>9}{'赔率':>7}{'TotalPnL':>11}{'AvgEdge':>8}"
    )
    lines.append(header)
    lines.append("-" * 83)

    # 按持有期交错显示 YES/NO
    for i in range(len(result.all_results)):
        yes_r = result.yes_results[i] if i < len(result.yes_results) else None
        no_r = result.no_results[i] if i < len(result.no_results) else None

        for r, direction in [(yes_r, "YES"), (no_r, "NO")]:
            if r is None or r.n_signals == 0:
                continue
            holding = r.holding if direction == "YES" else ""
            wr = f"{r.win_rate * 100:.1f}%"
            fr = f"{r.favorable_rate * 100:.1f}%"
            pr = f"{r.payoff_ratio:.2f}" if r.payoff_ratio != float("inf") else "inf"
            avg = f"{'+' if r.avg_pnl >= 0 else ''}{r.avg_pnl:.4f}"
            total = f"${r.total_pnl:,.0f}"
            lines.append(
                f"{holding:<12}{direction:<6}{r.n_with_exit:>7}{wr:>7}{fr:>7}"
                f"{avg:>9}{pr:>7}{total:>11}{r.avg_edge:>8.4f}"
            )

    lines.append(sep)
    return "\n".join(lines)


def save_results(result: ConvergenceResult, output_dir: str) -> None:
    """保存 convergence.csv + summary.txt"""
    os.makedirs(output_dir, exist_ok=True)

    # convergence.csv
    csv_path = os.path.join(output_dir, "convergence.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "direction", "holding", "holding_minutes", "n_signals",
            "n_with_exit", "n_wins", "n_losses", "win_rate",
            "avg_win", "avg_loss", "payoff_ratio", "ev_per_trade",
            "avg_pnl", "total_pnl", "median_pnl", "avg_edge",
            "n_favorable", "favorable_rate", "avg_mid_drift",
            "median_mid_drift",
        ])
        for direction, results_list in [
            ("ALL", result.all_results),
            ("YES", result.yes_results),
            ("NO", result.no_results),
        ]:
            for r in results_list:
                writer.writerow([
                    direction, r.holding, r.holding_minutes, r.n_signals,
                    r.n_with_exit, r.n_wins, r.n_losses, f"{r.win_rate:.4f}",
                    f"{r.avg_win:.6f}", f"{r.avg_loss:.6f}",
                    f"{r.payoff_ratio:.4f}" if r.payoff_ratio != float("inf") else "inf",
                    f"{r.ev_per_trade:.6f}",
                    f"{r.avg_pnl:.6f}", f"{r.total_pnl:.2f}",
                    f"{r.median_pnl:.6f}", f"{r.avg_edge:.6f}",
                    r.n_favorable, f"{r.favorable_rate:.4f}",
                    f"{r.avg_mid_drift:.6f}", f"{r.median_mid_drift:.6f}",
                ])
    logger.info(f"CSV → {csv_path}")

    # summary.txt
    summary_path = os.path.join(output_dir, "convergence_summary.txt")
    with open(summary_path, "w") as f:
        f.write(format_table_overall(result))
        f.write("\n\n")
        f.write(format_table_direction(result))
        f.write("\n")
    logger.info(f"Summary → {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="价格收敛分析")
    parser.add_argument(
        "--detail-csv",
        default="backtest_results/detail.csv",
        help="detail.csv 路径 (default: backtest_results/detail.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="backtest_results",
        help="输出目录 (default: backtest_results)",
    )
    parser.add_argument(
        "--entry-threshold",
        type=float,
        default=0.03,
        help="入场阈值 (default: 0.03)",
    )
    parser.add_argument(
        "--shares-per-trade",
        type=int,
        default=200,
        help="每笔份额 (default: 200)",
    )
    parser.add_argument(
        "--chart",
        action="store_true",
        help="生成收敛分析图表",
    )
    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not os.path.exists(args.detail_csv):
        logger.error(f"文件不存在: {args.detail_csv}")
        sys.exit(1)

    # 运行分析
    result = run_convergence(
        csv_path=args.detail_csv,
        threshold=args.entry_threshold,
        shares_per_trade=args.shares_per_trade,
    )

    # 打印结果
    print()
    print(format_table_overall(result))
    print()
    print(format_table_direction(result))
    print()

    # 保存文件
    save_results(result, args.output_dir)

    # 图表
    if args.chart:
        chart_path = plot_convergence(result, args.output_dir)
        print(f"图表已保存: {chart_path}")

    print(f"结果已保存: {args.output_dir}/convergence.csv, "
          f"{args.output_dir}/convergence_summary.txt")


if __name__ == "__main__":
    main()
