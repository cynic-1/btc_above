#!/usr/bin/env python3
"""
Strike 过滤实验 CLI

用法:
    python run_strike_filter_experiment.py --detail-csv backtest_results/detail.csv
"""

import argparse
import csv
import logging
import os
import sys

from backtest.strike_filter_experiment import (
    StrikeFilterResult,
    run_strike_filter_experiment,
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


def print_comparison_table(results: list[StrikeFilterResult]):
    """打印 ASCII 对比表"""
    header = (
        f"{'Config':>12} {'PnL':>10} {'Return%':>8} {'PF':>6} "
        f"{'Sharpe':>7} {'WinR%':>6} {'ROI%':>7} "
        f"{'MaxDD%':>7} {'#Trades':>7} {'#Mkts':>6} {'Brier':>7} {'AvgK':>5}"
    )
    print("\n" + "=" * len(header))
    print("Strike 过滤实验: Baseline vs Nearest-N")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        label = "ALL" if r.n_nearest == 0 else f"nearest-{r.n_nearest}"
        sharpe_str = f"{r.sharpe:.2f}" if r.sharpe is not None else "N/A"
        print(
            f"{label:>12} "
            f"${r.total_pnl:>9,.0f} "
            f"{r.return_pct:>7.2f}% "
            f"{format_pf(r.profit_factor):>6} "
            f"{sharpe_str:>7} "
            f"{r.win_rate*100:>5.1f}% "
            f"{r.roi*100:>6.1f}% "
            f"{r.max_drawdown_pct:>6.2f}% "
            f"{r.n_trades:>7} "
            f"{r.n_markets:>6} "
            f"{r.brier:>6.4f} "
            f"{r.avg_strikes_per_date:>5.1f}"
        )

    print("=" * len(header))

    # 与 baseline 的对比
    if len(results) > 1:
        baseline = results[0]
        print("\n与 Baseline 对比:")
        for r in results[1:]:
            pnl_diff = r.total_pnl - baseline.total_pnl
            pnl_sign = "+" if pnl_diff >= 0 else ""
            label = f"nearest-{r.n_nearest}"
            parts = [f"  {label}: PnL {pnl_sign}${pnl_diff:,.0f}"]
            if baseline.profit_factor > 0 and r.profit_factor > 0:
                parts.append(f"PF {format_pf(r.profit_factor)} vs {format_pf(baseline.profit_factor)}")
            if r.sharpe is not None and baseline.sharpe is not None:
                parts.append(f"Sharpe {r.sharpe:.2f} vs {baseline.sharpe:.2f}")
            print(" | ".join(parts))


def save_comparison_csv(results: list[StrikeFilterResult], output_path: str):
    """保存对比结果到 CSV"""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n_nearest", "total_pnl", "return_pct", "profit_factor",
            "sharpe", "win_rate", "roi", "max_drawdown_pct",
            "n_trades", "n_markets", "brier", "n_event_dates",
            "avg_strikes_per_date",
        ])
        for r in results:
            pf = r.profit_factor if r.profit_factor != float("inf") else 999.0
            writer.writerow([
                r.n_nearest, f"{r.total_pnl:.2f}", f"{r.return_pct:.4f}",
                f"{pf:.4f}",
                f"{r.sharpe:.4f}" if r.sharpe is not None else "",
                f"{r.win_rate:.4f}", f"{r.roi:.4f}",
                f"{r.max_drawdown_pct:.4f}",
                r.n_trades, r.n_markets, f"{r.brier:.6f}",
                r.n_event_dates, f"{r.avg_strikes_per_date:.2f}",
            ])
    logger.info(f"对比结果已保存: {output_path}")


def save_summary(results: list[StrikeFilterResult], output_path: str):
    """保存文本摘要"""
    lines = []
    lines.append("=" * 60)
    lines.append("Strike 过滤实验摘要")
    lines.append("=" * 60)
    lines.append("")
    lines.append("假设: 只交易离当前 BTC 价格最近的 N 个 strike，")
    lines.append("      比交易所有可用 strike 效果更好。")
    lines.append("")

    if results:
        baseline = results[0]
        lines.append(f"Baseline (全部 strikes):")
        lines.append(f"  PnL: ${baseline.total_pnl:,.0f}")
        lines.append(f"  PF: {format_pf(baseline.profit_factor)}")
        lines.append(f"  Sharpe: {baseline.sharpe:.2f}" if baseline.sharpe else "  Sharpe: N/A")
        lines.append(f"  Trades: {baseline.n_trades}")
        lines.append(f"  Markets: {baseline.n_markets}")
        lines.append("")

    for r in results[1:]:
        pnl_diff = r.total_pnl - results[0].total_pnl
        pnl_sign = "+" if pnl_diff >= 0 else ""
        lines.append(f"Nearest-{r.n_nearest}:")
        lines.append(f"  PnL: ${r.total_pnl:,.0f} ({pnl_sign}${pnl_diff:,.0f} vs baseline)")
        lines.append(f"  PF: {format_pf(r.profit_factor)}")
        lines.append(f"  Sharpe: {r.sharpe:.2f}" if r.sharpe else "  Sharpe: N/A")
        lines.append(f"  Trades: {r.n_trades}")
        lines.append(f"  Markets: {r.n_markets}")
        lines.append(f"  Avg strikes/date: {r.avg_strikes_per_date:.1f}")
        lines.append("")

    # 结论
    best = max(results, key=lambda r: r.total_pnl)
    if best.n_nearest == 0:
        lines.append("结论: Baseline（全部 strikes）表现最优，过滤无益。")
    else:
        lines.append(f"结论: Nearest-{best.n_nearest} 表现最优 "
                      f"(PnL ${best.total_pnl:,.0f})。")

    lines.append("")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"摘要已保存: {output_path}")


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Strike 过滤实验")
    parser.add_argument(
        "--detail-csv", default="backtest_results/detail.csv",
        help="detail.csv 路径",
    )
    parser.add_argument(
        "--output-dir", default="strike_filter_results",
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
        "--n-nearest", type=str, default="1,2,3,4",
        help="要测试的 n 值，逗号分隔 (默认 1,2,3,4)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.detail_csv):
        logger.error(f"找不到输入文件: {args.detail_csv}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    n_nearest_list = [int(x.strip()) for x in args.n_nearest.split(",")]

    # 运行实验
    results = run_strike_filter_experiment(
        csv_path=args.detail_csv,
        n_nearest_list=n_nearest_list,
        initial_capital=args.initial_capital,
        shares_per_trade=args.shares_per_trade,
        max_net_shares=args.max_net_shares,
        entry_threshold=args.entry_threshold,
    )

    # 输出结果
    print_comparison_table(results)

    save_comparison_csv(results, os.path.join(args.output_dir, "comparison.csv"))
    save_summary(results, os.path.join(args.output_dir, "summary.txt"))

    logger.info(f"全部完成！输出目录: {args.output_dir}")
    print(f"\n输出文件:")
    for f in sorted(os.listdir(args.output_dir)):
        print(f"  {args.output_dir}/{f}")


if __name__ == "__main__":
    main()
