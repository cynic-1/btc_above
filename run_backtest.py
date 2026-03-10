"""
回测 CLI 入口

用法:
    python run_backtest.py --start 2026-01-01 --end 2026-03-01
    python run_backtest.py --download-only --start 2026-02-25 --end 2026-02-28
"""

import argparse
import glob
import logging
import sys

from pricing_core.utils.logger import setup_logger

from backtest.config import BacktestConfig
from backtest.engine import BacktestEngine
from backtest.report import generate_report

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="BTC 二元期权回测")
    parser.add_argument("--start", default="2026-02-07", help="起始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-03-01", help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--cache-dir", default="data/klines", help="K线缓存目录")
    parser.add_argument("--output-dir", default="backtest_results", help="输出目录")
    parser.add_argument("--mc-samples", type=int, default=2000, help="MC 采样数")
    parser.add_argument("--step-minutes", type=int, default=1, help="观测间隔（分钟）")
    parser.add_argument("--lookback-hours", type=int, default=24, help="事件前多少小时开始")
    parser.add_argument("--download-only", action="store_true", help="仅下载数据不跑回测")
    parser.add_argument("--no-market-prices", action="store_true", help="不使用 Polymarket 真实市场价格")
    parser.add_argument("--polymarket-cache-dir", default="data/polymarket", help="Polymarket 缓存目录")
    parser.add_argument("--preprocess-orderbook", action="store_true", help="预处理 Parquet 订单簿数据为 npz 缓存")
    parser.add_argument("--entry-threshold", type=float, default=0.03, help="入场 edge 阈值 (默认 0.03)")
    parser.add_argument("--no-orderbook", action="store_true", help="禁用订单簿缓存，使用 CLOB 价格历史")
    parser.add_argument("--orderbook-cache-dir", default="data/orderbook_cache", help="订单簿 npz 缓存目录")
    parser.add_argument("--orderbook-events-json", default="data/btc_above_events.json", help="事件映射 JSON")
    parser.add_argument("--no-position-limit", action="store_true", help="禁用单市场仓位上限")
    parser.add_argument("--direction", choices=["both", "yes_only", "no_only"],
                        default="both", help="方向过滤: both/yes_only/no_only")
    parser.add_argument("--cooldown", type=int, default=0,
                        help="交易冷却期（分钟），同一市场两次交易间最少等待时间 (默认 0=禁用)")
    args = parser.parse_args()

    setup_logger()

    # 预处理订单簿（一次性操作）
    if args.preprocess_orderbook:
        from backtest.orderbook_preprocessor import preprocess_parquet_files
        logger.info("预处理 Parquet 订单簿数据...")
        result = preprocess_parquet_files(
            cache_dir=args.orderbook_cache_dir,
            events_json=args.orderbook_events_json,
        )
        logger.info(f"预处理完成: {len(result)} 个市场, {sum(result.values())} 条记录")
        return

    extra_kwargs = {}
    if args.no_position_limit:
        extra_kwargs["max_net_shares"] = 10**9

    config = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        mc_samples=args.mc_samples,
        step_minutes=args.step_minutes,
        lookback_hours=args.lookback_hours,
        use_market_prices=not args.no_market_prices,
        polymarket_cache_dir=args.polymarket_cache_dir,
        entry_threshold=args.entry_threshold,
        use_orderbook=not args.no_orderbook,
        orderbook_cache_dir=args.orderbook_cache_dir,
        orderbook_events_json=args.orderbook_events_json,
        direction_filter=args.direction,
        cooldown_minutes=args.cooldown,
        **extra_kwargs,
    )

    engine = BacktestEngine(config=config)

    # 下载数据
    logger.info("检查/下载 K线数据...")
    engine.download_data()

    if config.use_market_prices:
        logger.info("检查/下载 Polymarket 市场数据...")
        engine.download_polymarket_data()

    if args.download_only:
        logger.info("仅下载模式，完成")
        return

    # 运行回测
    logger.info("开始回测...")
    result = engine.run()

    # 生成报告
    logger.info("生成报告...")
    metrics = generate_report(result, config)

    # 打印关键指标
    overall = metrics.get("overall", {})
    brier = overall.get("brier_score", "N/A")
    port = overall.get("portfolio", {})

    print(f"\n回测完成!")
    print(f"  Brier Score: {brier:.6f}" if isinstance(brier, float) else f"  Brier Score: {brier}")
    if port:
        print(f"  期初资金: ${port.get('initial_capital', 0):,.0f}")
        print(f"  总 PnL: ${port.get('total_pnl', 0):,.2f}")
        print(f"  总投入成本: ${port.get('total_cost', 0):,.2f}")
        print(f"  收益率: {port.get('total_return_pct', 0):.2f}%")
        print(f"  市场数: {port.get('n_markets', 0)}")
        print(f"  交易笔数: {port.get('n_trades', 0)}")
        print(f"  盈利市场: {port.get('win_markets', 0)}  亏损市场: {port.get('lose_markets', 0)}")
        print(f"  盈利因子: {port.get('profit_factor', 0):.3f}")
        print(f"  最大回撤: {port.get('max_drawdown_pct', 0):.2f}%")
    print(f"  结果目录: {config.output_dir}/")

    # 列出生成的报告文件
    dd_files = sorted(glob.glob(f"{config.output_dir}/dd_*.md"))
    report_files = sorted(glob.glob(f"{config.output_dir}/report_*.md"))
    if dd_files:
        print(f"  尽调清单: {dd_files[-1]}")
    if report_files:
        print(f"  策略评审: {report_files[-1]}")


if __name__ == "__main__":
    main()
