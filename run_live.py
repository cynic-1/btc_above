"""
实盘交易 CLI 入口

用法:
    python run_live.py --event-date 2026-03-09
    python run_live.py --event-date 2026-03-09 --dry-run
    python run_live.py --event-date 2026-03-09 --shares-per-trade 100 --order-type FOK
"""

import argparse
import sys

from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BTC above K 二元期权实盘交易系统",
    )
    parser.add_argument(
        "--event-date",
        required=True,
        help="事件日期 YYYY-MM-DD",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="模拟模式，不实际下单",
    )
    parser.add_argument(
        "--shares-per-trade",
        type=int,
        default=200,
        help="单次下单份数 (default: 200)",
    )
    parser.add_argument(
        "--max-net-shares",
        type=int,
        default=10_000,
        help="单市场净仓位限制 (default: 10000)",
    )
    parser.add_argument(
        "--max-total-cost",
        type=float,
        default=50_000.0,
        help="总金额限制 (default: 50000)",
    )
    parser.add_argument(
        "--entry-threshold",
        type=float,
        default=0.03,
        help="edge 入场阈值 (default: 0.03)",
    )
    parser.add_argument(
        "--order-type",
        choices=["GTC", "FOK"],
        default="GTC",
        help="订单类型 (default: GTC)",
    )
    parser.add_argument(
        "--order-cooldown",
        type=float,
        default=20.0,
        help="同市场下单冷却秒数 (default: 20)",
    )
    parser.add_argument(
        "--order-timeout",
        type=float,
        default=300.0,
        help="GTC 挂单超时取消秒数 (default: 300 = 5分钟)",
    )
    parser.add_argument(
        "--direction",
        choices=["both", "yes_only", "no_only"],
        default="both",
        help="方向过滤 (default: both)",
    )
    parser.add_argument(
        "--yes-threshold",
        type=float,
        default=None,
        help="YES 方向 edge 阈值 (default: 沿用 --entry-threshold)",
    )
    parser.add_argument(
        "--no-threshold",
        type=float,
        default=None,
        help="NO 方向 edge 阈值 (default: 沿用 --entry-threshold)",
    )
    parser.add_argument(
        "--max-spread",
        type=float,
        default=None,
        help="最大 bid-ask 价差过滤 (default: 不限)",
    )
    parser.add_argument(
        "--min-minutes-to-event",
        type=int,
        default=0,
        help="距事件最少分钟数 (default: 0)",
    )
    parser.add_argument(
        "--max-minutes-to-event",
        type=int,
        default=99999,
        help="距事件最多分钟数 (default: 99999)",
    )
    parser.add_argument(
        "--pricing-interval",
        type=float,
        default=10.0,
        help="定价间隔秒 (default: 10)",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=2000,
        help="MC 采样数 (default: 2000)",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="日志目录 (default: logs)",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()

    args = parse_args()

    # 延迟导入，确保 .env 已加载
    from pricing_core.utils.logger import setup_logger
    from live.config import LiveTradingConfig
    from live.engine import LiveTradingEngine

    setup_logger(log_dir=args.log_dir)

    config = LiveTradingConfig(
        event_date=args.event_date,
        dry_run=args.dry_run,
        shares_per_trade=args.shares_per_trade,
        max_net_shares=args.max_net_shares,
        max_total_cost=args.max_total_cost,
        entry_threshold=args.entry_threshold,
        order_type=args.order_type,
        order_cooldown_seconds=args.order_cooldown,
        order_timeout_seconds=args.order_timeout,
        pricing_interval_seconds=args.pricing_interval,
        mc_samples=args.mc_samples,
        log_dir=args.log_dir,
        direction_filter=args.direction,
        yes_threshold=args.yes_threshold,
        no_threshold=args.no_threshold,
        max_spread=args.max_spread,
        min_minutes_to_event=args.min_minutes_to_event,
        max_minutes_to_event=args.max_minutes_to_event,
    )

    engine = LiveTradingEngine(config)
    engine.start()


if __name__ == "__main__":
    main()
