#!/usr/bin/env python3
"""
周末效应实验 CLI

用法:
    python run_weekend_experiment.py --detail-csv backtest_results/detail.csv
"""

import argparse
import csv
import logging
import os
import sys

from backtest.weekend_experiment import (
    DayGroupResult,
    run_weekend_experiment,
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


def format_opt(val, fmt: str = ".4f") -> str:
    """格式化可选值"""
    if val is None:
        return "N/A"
    return f"{val:{fmt}}"


def print_comparison_table(weekend: DayGroupResult, weekday: DayGroupResult):
    """打印 weekend vs weekday ASCII 对比表"""
    header = (
        f"{'Group':>10} {'#Days':>5} {'PnL':>10} {'Return%':>8} {'PF':>6} "
        f"{'Sharpe':>7} {'WinR%':>6} {'MaxDD%':>7} {'#Trades':>7} {'#Mkts':>6} "
        f"{'Brier':>7} {'AvgEdge':>8} {'MktCov%':>7} {'Spread':>7}"
    )
    print("\n" + "=" * len(header))
    print("周末效应实验: Weekend vs Weekday")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in [weekday, weekend]:
        sharpe_str = format_opt(r.sharpe, ".2f")
        brier_str = format_opt(r.brier, ".4f")
        edge_str = format_opt(r.avg_abs_edge, ".4f")
        spread_str = format_opt(r.avg_spread, ".4f")
        print(
            f"{r.group_name:>10} "
            f"{r.n_event_dates:>5} "
            f"${r.total_pnl:>9,.0f} "
            f"{r.return_pct:>7.2f}% "
            f"{format_pf(r.profit_factor):>6} "
            f"{sharpe_str:>7} "
            f"{r.win_rate*100:>5.1f}% "
            f"{r.max_drawdown_pct:>6.2f}% "
            f"{r.n_trades:>7} "
            f"{r.n_markets:>6} "
            f"{brier_str:>7} "
            f"{edge_str:>8} "
            f"{r.market_price_coverage*100:>6.1f}% "
            f"{spread_str:>7}"
        )

    print("=" * len(header))

    # 差异
    pnl_diff = weekend.total_pnl - weekday.total_pnl
    pnl_sign = "+" if pnl_diff >= 0 else ""
    print(f"\n差异 (weekend - weekday):")
    print(f"  PnL: {pnl_sign}${pnl_diff:,.0f}")
    if weekend.sharpe is not None and weekday.sharpe is not None:
        print(f"  Sharpe: {weekend.sharpe:.2f} vs {weekday.sharpe:.2f}")
    if weekend.brier is not None and weekday.brier is not None:
        brier_diff = weekend.brier - weekday.brier
        brier_sign = "+" if brier_diff >= 0 else ""
        print(f"  Brier: {brier_sign}{brier_diff:.4f} (越低越好)")


def print_daily_table(per_day_results: list[DayGroupResult]):
    """打印逐日明细表 Mon~Sun"""
    header = (
        f"{'Day':>10} {'#Days':>5} {'PnL':>10} {'Return%':>8} {'PF':>6} "
        f"{'Sharpe':>7} {'WinR%':>6} {'#Trades':>7} {'Brier':>7}"
    )
    print("\n" + "=" * len(header))
    print("逐日明细")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in per_day_results:
        if r.n_event_dates == 0:
            print(f"{r.group_name:>10} {'—':>5} {'—':>10} {'—':>8} {'—':>6} {'—':>7} {'—':>6} {'—':>7} {'—':>7}")
            continue
        sharpe_str = format_opt(r.sharpe, ".2f")
        brier_str = format_opt(r.brier, ".4f")
        print(
            f"{r.group_name:>10} "
            f"{r.n_event_dates:>5} "
            f"${r.total_pnl:>9,.0f} "
            f"{r.return_pct:>7.2f}% "
            f"{format_pf(r.profit_factor):>6} "
            f"{sharpe_str:>7} "
            f"{r.win_rate*100:>5.1f}% "
            f"{r.n_trades:>7} "
            f"{brier_str:>7}"
        )

    print("=" * len(header))


def save_comparison_csv(
    weekend: DayGroupResult,
    weekday: DayGroupResult,
    per_day_results: list[DayGroupResult],
    output_path: str,
):
    """保存对比结果到 CSV"""
    fieldnames = [
        "group_name", "n_event_dates", "event_dates",
        "brier", "avg_abs_edge", "market_price_coverage",
        "total_pnl", "return_pct", "profit_factor",
        "n_trades", "n_markets", "win_rate",
        "max_drawdown_pct", "sharpe", "avg_spread",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in [weekday, weekend] + per_day_results:
            pf = r.profit_factor if r.profit_factor != float("inf") else 999.0
            writer.writerow({
                "group_name": r.group_name,
                "n_event_dates": r.n_event_dates,
                "event_dates": ";".join(r.event_dates),
                "brier": f"{r.brier:.6f}" if r.brier is not None else "",
                "avg_abs_edge": f"{r.avg_abs_edge:.6f}" if r.avg_abs_edge is not None else "",
                "market_price_coverage": f"{r.market_price_coverage:.4f}",
                "total_pnl": f"{r.total_pnl:.2f}",
                "return_pct": f"{r.return_pct:.4f}",
                "profit_factor": f"{pf:.4f}",
                "n_trades": r.n_trades,
                "n_markets": r.n_markets,
                "win_rate": f"{r.win_rate:.4f}",
                "max_drawdown_pct": f"{r.max_drawdown_pct:.4f}",
                "sharpe": f"{r.sharpe:.4f}" if r.sharpe is not None else "",
                "avg_spread": f"{r.avg_spread:.6f}" if r.avg_spread is not None else "",
            })
    logger.info(f"对比结果已保存: {output_path}")


def save_summary(
    weekend: DayGroupResult,
    weekday: DayGroupResult,
    per_day_results: list[DayGroupResult],
    output_path: str,
):
    """保存文本摘要"""
    lines = []
    lines.append("=" * 60)
    lines.append("周末效应实验摘要")
    lines.append("=" * 60)
    lines.append("")
    lines.append("问题: 策略在周末（周六/周日）的表现是否与工作日有显著差异？")
    lines.append("")

    # Weekend
    lines.append(f"Weekend ({weekend.n_event_dates} 日: {', '.join(weekend.event_dates)}):")
    lines.append(f"  PnL: ${weekend.total_pnl:,.0f}")
    lines.append(f"  PF: {format_pf(weekend.profit_factor)}")
    lines.append(f"  Sharpe: {format_opt(weekend.sharpe, '.2f')}")
    lines.append(f"  Trades: {weekend.n_trades}")
    lines.append(f"  Markets: {weekend.n_markets}")
    lines.append(f"  Brier: {format_opt(weekend.brier, '.4f')}")
    lines.append(f"  市场价格覆盖率: {weekend.market_price_coverage*100:.1f}%")
    if weekend.avg_spread is not None:
        lines.append(f"  平均 Spread: {weekend.avg_spread:.4f}")
    lines.append("")

    # Weekday
    lines.append(f"Weekday ({weekday.n_event_dates} 日: {', '.join(weekday.event_dates)}):")
    lines.append(f"  PnL: ${weekday.total_pnl:,.0f}")
    lines.append(f"  PF: {format_pf(weekday.profit_factor)}")
    lines.append(f"  Sharpe: {format_opt(weekday.sharpe, '.2f')}")
    lines.append(f"  Trades: {weekday.n_trades}")
    lines.append(f"  Markets: {weekday.n_markets}")
    lines.append(f"  Brier: {format_opt(weekday.brier, '.4f')}")
    lines.append(f"  市场价格覆盖率: {weekday.market_price_coverage*100:.1f}%")
    if weekday.avg_spread is not None:
        lines.append(f"  平均 Spread: {weekday.avg_spread:.4f}")
    lines.append("")

    # 差异
    pnl_diff = weekend.total_pnl - weekday.total_pnl
    pnl_sign = "+" if pnl_diff >= 0 else ""
    lines.append("差异分析:")
    lines.append(f"  PnL 差异: {pnl_sign}${pnl_diff:,.0f}")

    # 每日 PnL
    if weekend.n_event_dates > 0 and weekday.n_event_dates > 0:
        wknd_per_day = weekend.total_pnl / weekend.n_event_dates
        wkdy_per_day = weekday.total_pnl / weekday.n_event_dates
        lines.append(f"  日均 PnL: Weekend ${wknd_per_day:,.0f} vs Weekday ${wkdy_per_day:,.0f}")
    lines.append("")

    # 逐日
    lines.append("逐日明细:")
    for r in per_day_results:
        if r.n_event_dates == 0:
            lines.append(f"  {r.group_name}: 无数据")
            continue
        per_day_pnl = r.total_pnl / r.n_event_dates if r.n_event_dates > 0 else 0
        lines.append(
            f"  {r.group_name} ({r.n_event_dates}日): "
            f"PnL=${r.total_pnl:,.0f} (日均${per_day_pnl:,.0f}), "
            f"PF={format_pf(r.profit_factor)}, "
            f"Trades={r.n_trades}"
        )
    lines.append("")

    # 结论
    if weekend.n_event_dates > 0 and weekday.n_event_dates > 0:
        wknd_per_day = weekend.total_pnl / weekend.n_event_dates
        wkdy_per_day = weekday.total_pnl / weekday.n_event_dates
        if wknd_per_day > wkdy_per_day * 1.2:
            lines.append("结论: 周末日均 PnL 明显高于工作日，不建议周末暂停。")
        elif wkdy_per_day > wknd_per_day * 1.2:
            lines.append("结论: 工作日日均 PnL 明显高于周末，可考虑周末降低仓位。")
        else:
            lines.append("结论: 周末与工作日表现差异不显著，无需区分对待。")
    else:
        lines.append("结论: 数据不足，无法得出可靠结论。")
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"摘要已保存: {output_path}")


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="周末效应实验")
    parser.add_argument(
        "--detail-csv", default="backtest_results/detail.csv",
        help="detail.csv 路径",
    )
    parser.add_argument(
        "--output-dir", default="weekend_results",
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
    args = parser.parse_args()

    if not os.path.exists(args.detail_csv):
        logger.error(f"找不到输入文件: {args.detail_csv}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # 运行实验
    weekend, weekday, per_day = run_weekend_experiment(
        csv_path=args.detail_csv,
        initial_capital=args.initial_capital,
        shares_per_trade=args.shares_per_trade,
        max_net_shares=args.max_net_shares,
        entry_threshold=args.entry_threshold,
    )

    # 输出结果
    print_comparison_table(weekend, weekday)
    print_daily_table(per_day)

    save_comparison_csv(
        weekend, weekday, per_day,
        os.path.join(args.output_dir, "comparison.csv"),
    )
    save_summary(
        weekend, weekday, per_day,
        os.path.join(args.output_dir, "summary.txt"),
    )

    logger.info(f"全部完成！输出目录: {args.output_dir}")
    print(f"\n输出文件:")
    for f in sorted(os.listdir(args.output_dir)):
        print(f"  {args.output_dir}/{f}")


if __name__ == "__main__":
    main()
