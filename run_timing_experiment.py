#!/usr/bin/env python3
"""
时间窗口实验 CLI

用法:
    python run_timing_experiment.py --detail-csv backtest_results/detail.csv --output-dir timing_results
"""

import argparse
import csv
import logging
import os
import sys

from backtest.timing_experiment import (
    START_MINUTES,
    STOP_MINUTES,
    TimingWindowResult,
    compute_incremental_value,
    find_optimal_windows,
    load_observations_from_csv,
    run_timing_grid,
)
from backtest.timing_plots import (
    plot_heatmaps,
    plot_incremental_value,
    plot_marginal_effects,
)

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def format_pf(pf: float) -> str:
    """格式化 profit factor"""
    if pf == float("inf"):
        return "inf"
    return f"{pf:.2f}"


def print_results_table(results: list[TimingWindowResult]):
    """打印 ASCII 结果表"""
    header = (
        f"{'Rank':>4} {'Start':>8} {'Stop':>8} {'PnL':>10} {'Return%':>8} "
        f"{'PF':>6} {'Sharpe':>7} {'WinR%':>6} {'ROI%':>7} "
        f"{'MaxDD%':>7} {'#Trades':>7} {'Score':>6} {'Note':>5}"
    )
    print("\n" + "=" * len(header))
    print("时间窗口实验结果 (按综合评分排序)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for i, r in enumerate(results):
        sharpe_str = f"{r.sharpe:.2f}" if r.sharpe is not None else "N/A"
        note = "⚠低" if r.low_confidence else ""
        print(
            f"{i+1:>4} "
            f"{'T-'+str(r.start_minutes)+'m':>8} "
            f"{'T-'+str(r.stop_minutes)+'m':>8} "
            f"${r.total_pnl:>9,.0f} "
            f"{r.return_pct:>7.2f}% "
            f"{format_pf(r.profit_factor):>6} "
            f"{sharpe_str:>7} "
            f"{r.win_rate*100:>5.1f}% "
            f"{r.roi*100:>6.1f}% "
            f"{r.max_drawdown_pct:>6.2f}% "
            f"{r.n_trades:>7} "
            f"{r.composite_score:>5.3f} "
            f"{note:>5}"
        )

    print("=" * len(header))


def save_grid_csv(results: list[TimingWindowResult], output_path: str):
    """保存完整网格结果到 CSV"""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "start_minutes", "stop_minutes", "total_pnl", "return_pct",
            "profit_factor", "sharpe", "win_rate", "roi", "max_drawdown_pct",
            "n_trades", "n_markets", "brier", "composite_score", "low_confidence",
        ])
        for r in results:
            pf = r.profit_factor if r.profit_factor != float("inf") else 999.0
            writer.writerow([
                r.start_minutes, r.stop_minutes, f"{r.total_pnl:.2f}",
                f"{r.return_pct:.4f}", f"{pf:.4f}",
                f"{r.sharpe:.4f}" if r.sharpe is not None else "",
                f"{r.win_rate:.4f}", f"{r.roi:.4f}",
                f"{r.max_drawdown_pct:.4f}", r.n_trades, r.n_markets,
                f"{r.brier:.6f}", f"{r.composite_score:.4f}",
                r.low_confidence,
            ])
    logger.info(f"网格结果已保存: {output_path}")


def save_summary(
    top_results: list[TimingWindowResult],
    incremental: list[dict],
    output_path: str,
):
    """保存文本摘要"""
    lines = []
    lines.append("=" * 60)
    lines.append("时间窗口实验摘要")
    lines.append("=" * 60)
    lines.append("")

    if top_results:
        best = top_results[0]
        lines.append(f"推荐最优窗口: T-{best.start_minutes}m ~ T-{best.stop_minutes}m")
        lines.append(f"  PnL: ${best.total_pnl:,.0f}")
        lines.append(f"  Return: {best.return_pct:.2f}%")
        lines.append(f"  Profit Factor: {format_pf(best.profit_factor)}")
        lines.append(f"  Sharpe: {best.sharpe:.2f}" if best.sharpe else "  Sharpe: N/A")
        lines.append(f"  Win Rate: {best.win_rate*100:.1f}%")
        lines.append(f"  ROI: {best.roi*100:.1f}%")
        lines.append(f"  Max DD: {best.max_drawdown_pct:.2f}%")
        lines.append(f"  Trades: {best.n_trades}")
        lines.append(f"  综合评分: {best.composite_score:.3f}")
        if best.low_confidence:
            lines.append("  ⚠ 低置信度（交易数 < 20）")
        lines.append("")

    lines.append("Top-5 窗口:")
    for i, r in enumerate(top_results[:5]):
        sharpe_str = f"{r.sharpe:.2f}" if r.sharpe else "N/A"
        lines.append(
            f"  {i+1}. T-{r.start_minutes}m~T-{r.stop_minutes}m | "
            f"PnL=${r.total_pnl:,.0f} | PF={format_pf(r.profit_factor)} | "
            f"Sharpe={sharpe_str} | Score={r.composite_score:.3f}"
        )

    lines.append("")
    lines.append("增量贡献分析 (每 30 分钟桶):")
    positive_buckets = [r for r in incremental if r["incremental_pnl"] > 0 and r["n_obs_in_bucket"] > 0]
    negative_buckets = [r for r in incremental if r["incremental_pnl"] < 0 and r["n_obs_in_bucket"] > 0]

    if positive_buckets:
        lines.append("  正贡献桶:")
        for b in sorted(positive_buckets, key=lambda x: x["incremental_pnl"], reverse=True)[:5]:
            lines.append(f"    {b['bucket']}: +${b['incremental_pnl']:,.0f} ({b['n_obs_in_bucket']} obs)")

    if negative_buckets:
        lines.append("  负贡献桶:")
        for b in sorted(negative_buckets, key=lambda x: x["incremental_pnl"])[:5]:
            lines.append(f"    {b['bucket']}: ${b['incremental_pnl']:,.0f} ({b['n_obs_in_bucket']} obs)")

    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"摘要已保存: {output_path}")


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="时间窗口实验")
    parser.add_argument(
        "--detail-csv", default="backtest_results/detail.csv",
        help="detail.csv 路径",
    )
    parser.add_argument(
        "--output-dir", default="timing_results",
        help="输出目录",
    )
    parser.add_argument(
        "--entry-threshold", type=float, default=0.03,
        help="入场阈值 (默认 0.03)",
    )
    parser.add_argument(
        "--initial-capital", type=float, default=100_000.0,
        help="初始资金 (默认 100000)",
    )
    parser.add_argument(
        "--shares-per-trade", type=int, default=200,
        help="每笔交易份数 (默认 200)",
    )
    parser.add_argument(
        "--max-net-shares", type=int, default=10_000,
        help="单市场最大净仓位 (默认 10000)",
    )
    parser.add_argument(
        "--bucket-width", type=int, default=30,
        help="增量分析桶宽 (分钟, 默认 30)",
    )
    parser.add_argument(
        "--top-n", type=int, default=15,
        help="显示 top N 结果 (默认 15)",
    )
    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.detail_csv):
        logger.error(f"找不到输入文件: {args.detail_csv}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载观测数据
    logger.info(f"加载数据: {args.detail_csv}")
    observations = load_observations_from_csv(args.detail_csv)
    logger.info(f"加载完成: {len(observations)} 个观测")

    # 2. 运行时间网格实验
    logger.info("运行时间网格实验...")
    grid_results = run_timing_grid(
        observations,
        initial_capital=args.initial_capital,
        shares_per_trade=args.shares_per_trade,
        max_net_shares=args.max_net_shares,
        entry_threshold=args.entry_threshold,
    )

    # 3. 找最优窗口
    top_results = find_optimal_windows(grid_results, top_n=args.top_n)

    # 4. 增量贡献分析
    logger.info("运行增量贡献分析...")
    incremental = compute_incremental_value(
        observations,
        bucket_width_minutes=args.bucket_width,
        initial_capital=args.initial_capital,
        shares_per_trade=args.shares_per_trade,
        max_net_shares=args.max_net_shares,
        entry_threshold=args.entry_threshold,
    )

    # 5. 输出结果
    print_results_table(top_results)

    # 保存 CSV
    save_grid_csv(grid_results, os.path.join(args.output_dir, "grid_results.csv"))

    # 保存摘要
    save_summary(top_results, incremental, os.path.join(args.output_dir, "summary.txt"))

    # 6. 生成图表
    logger.info("生成图表...")
    plot_heatmaps(grid_results, args.output_dir)
    plot_marginal_effects(grid_results, args.output_dir)
    plot_incremental_value(incremental, args.output_dir)

    logger.info(f"全部完成！输出目录: {args.output_dir}")
    print(f"\n输出文件:")
    for f in sorted(os.listdir(args.output_dir)):
        print(f"  {args.output_dir}/{f}")


if __name__ == "__main__":
    main()
