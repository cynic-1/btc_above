"""
模型定价 vs Polymarket 对比图表 CLI 入口

用法:
    python run_charts.py --start 2026-01-01 --end 2026-03-01
    python run_charts.py --event-date 2026-02-15 --strike 90000
"""

import argparse
import logging

from pricing_core.utils.logger import setup_logger

from backtest.chart_engine import ChartConfig, ChartGenerator

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="模型定价 vs Polymarket 对比图表")
    parser.add_argument("--start", default="2026-01-01", help="起始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-03-01", help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--output-dir", default="charts", help="输出目录")
    parser.add_argument("--step-minutes", type=int, default=1, help="时间分辨率（分钟）")
    parser.add_argument("--lookback-hours", type=int, default=24, help="事件前多少小时开始")
    parser.add_argument("--mc-samples", type=int, default=2000, help="MC 采样数")
    parser.add_argument("--event-date", help="只生成特定日期")
    parser.add_argument("--strike", type=float, help="只生成特定 strike")
    parser.add_argument("--cache-dir", default="data/klines", help="K线缓存目录")
    parser.add_argument("--polymarket-cache-dir", default="data/polymarket", help="Polymarket 缓存目录")
    args = parser.parse_args()

    setup_logger()

    # 如果指定了 event-date，覆盖 start/end 范围
    start = args.start
    end = args.end
    if args.event_date:
        start = args.event_date
        # end 需要是 event_date 的下一天（_date_range 是 [start, end)）
        from datetime import datetime, timedelta
        dt = datetime.strptime(args.event_date, "%Y-%m-%d")
        end = (dt + timedelta(days=1)).strftime("%Y-%m-%d")

    config = ChartConfig(
        start_date=start,
        end_date=end,
        cache_dir=args.cache_dir,
        polymarket_cache_dir=args.polymarket_cache_dir,
        output_dir=args.output_dir,
        step_minutes=args.step_minutes,
        lookback_hours=args.lookback_hours,
        mc_samples=args.mc_samples,
    )

    generator = ChartGenerator(config=config)
    generator.run(
        filter_event_date=args.event_date,
        filter_strike=args.strike,
    )


if __name__ == "__main__":
    main()
