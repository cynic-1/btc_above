"""
Above 合约回测 CLI 入口

用法:
    python run_above_backtest.py --start-date 2026-03-01 --end-date 2026-03-24
    python run_above_backtest.py --start-date 2026-03-01 --end-date 2026-03-24 --download-only
    python run_above_backtest.py --start-date 2026-03-01 --end-date 2026-03-10 --iv-source default --default-sigma 0.55
"""

import argparse
import logging

from pricing_core.utils.logger import setup_logger

from above.backtest_engine import AboveBacktestEngine
from above.chart_engine import AboveChartGenerator
from above.models import AboveBacktestConfig
from above.report_generator import generate_report

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Above 合约回测 (BTC above K at ET noon)")

    # 日期范围
    parser.add_argument("--start-date", required=True, help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="结束日期 (YYYY-MM-DD, 不含)")

    # 观测设置
    parser.add_argument("--step-minutes", type=int, default=15,
                        help="观测间隔（分钟，默认 15）")
    parser.add_argument("--lookback-hours", type=int, default=24,
                        help="事件前回看小时数（默认 24）")

    # IV 来源
    parser.add_argument("--iv-source", choices=["dvol", "default"],
                        default="dvol", help="IV 来源")
    parser.add_argument("--vrp-k", type=float, default=1.0, help="VRP 缩放系数")
    parser.add_argument("--mu", type=float, default=0.0, help="漂移率 (默认 0.0)")
    parser.add_argument("--default-sigma", type=float, default=0.65,
                        help="默认年化波动率 (当 IV 不可用时)")

    # 交易参数
    parser.add_argument("--entry-threshold", type=float, default=0.03,
                        help="入场 edge 阈值")
    parser.add_argument("--shrinkage", type=float, default=0.6,
                        help="收缩系数 lambda")

    # 币种
    parser.add_argument("--symbol", default="BTC", choices=["BTC", "ETH"],
                        help="币种 (默认 BTC)")

    # 输出控制
    parser.add_argument("--no-market-prices", action="store_true",
                        help="不使用 Polymarket 市场价格")
    parser.add_argument("--charts", action="store_true",
                        help="生成图表（默认不生成，由 run_above_charts.py 负责）")
    parser.add_argument("--no-report", action="store_true",
                        help="不生成报告")
    parser.add_argument("--download-only", action="store_true",
                        help="仅下载数据不跑回测")

    # 目录
    parser.add_argument("--output-dir", default="above_backtest_results",
                        help="输出目录")
    parser.add_argument("--cache-dir", default="data/klines",
                        help="K线缓存目录")
    parser.add_argument("--iv-cache-dir", default="data/deribit_iv",
                        help="DVOL 缓存目录")
    parser.add_argument("--polymarket-cache-dir", default="data/polymarket",
                        help="Polymarket 缓存目录")

    args = parser.parse_args()
    setup_logger()

    config = AboveBacktestConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        symbol=args.symbol,
        cache_dir=args.cache_dir,
        iv_cache_dir=args.iv_cache_dir,
        output_dir=args.output_dir,
        step_minutes=args.step_minutes,
        lookback_hours=args.lookback_hours,
        iv_source=args.iv_source,
        default_sigma=args.default_sigma,
        mu=args.mu,
        vrp_k=args.vrp_k,
        entry_threshold=args.entry_threshold,
        shrinkage_lambda=args.shrinkage,
        use_market_prices=not args.no_market_prices,
        polymarket_cache_dir=args.polymarket_cache_dir,
    )

    engine = AboveBacktestEngine(config=config)

    print(f"\nAbove 合约回测: {config.start_date} ~ {config.end_date}")

    # 1. 下载数据
    logger.info("检查/下载 K线数据...")
    engine.download_data()

    logger.info("检查/下载 DVOL 历史数据...")
    engine.download_iv_data()

    if config.use_market_prices:
        logger.info("检查/下载 Polymarket 市场数据...")
        engine.download_polymarket_data()

    if args.download_only:
        logger.info("仅下载模式，完成")
        return

    # 2. 运行回测
    logger.info("开始回测...")
    observations = engine.run()

    if not observations:
        logger.error("无观测结果")
        return

    n_days = len(set(obs.event_date for obs in observations))
    n_strikes = len(set(K for obs in observations for K in obs.k_grid))
    print(f"  {n_days} 天, {len(observations)} 观测, {n_strikes} 个 strike")

    # 3. 序列化
    tag = f"{config.start_date}_{config.end_date}"
    AboveBacktestEngine.serialize_observations(observations, config.output_dir, tag)

    # 4. 报告
    if not args.no_report:
        report_path = generate_report(observations, config)
        print(f"  报告: {report_path}")

    # 5. 图表（默认不生成）
    if args.charts:
        chart_gen = AboveChartGenerator(
            output_dir=config.output_dir, symbol=config.symbol,
        )
        chart_gen.generate(observations)

    print(f"  结果目录: {config.output_dir}/")


if __name__ == "__main__":
    main()
