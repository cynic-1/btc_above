"""
触碰障碍期权回测 CLI 入口

用法:
    python run_touch_backtest.py --month 2026-03
    python run_touch_backtest.py --month 2026-03 --download-only
    python run_touch_backtest.py --month 2026-03 --step-minutes 30 --default-sigma 0.65
"""

import argparse
import logging

from pricing_core.utils.logger import setup_logger

from touch.backtest_engine import TouchBacktestEngine
from touch.chart_engine import TouchChartGenerator
from touch.models import TouchBacktestConfig
from touch.report_generator import generate_report

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="触碰障碍期权回测")
    parser.add_argument("--month", default="2026-03", help="回测月份 (YYYY-MM)")
    parser.add_argument("--cache-dir", default="data/klines", help="K线缓存目录")
    parser.add_argument("--iv-cache-dir", default="data/deribit_iv", help="DVOL 缓存目录")
    parser.add_argument("--output-dir", default="touch_backtest_results", help="输出目录")
    parser.add_argument("--step-minutes", type=int, default=60, help="观测间隔（分钟，默认 60）")
    parser.add_argument("--iv-source", choices=["dvol", "option_chain", "har_rv"],
                        default="dvol", help="IV 来源")
    parser.add_argument("--default-sigma", type=float, default=0.65,
                        help="默认年化波动率 (当 IV 不可用时)")
    parser.add_argument("--mu", type=float, default=0.0, help="漂移率 (默认 0.0)")
    parser.add_argument("--vrp-k", type=float, default=1.0, help="VRP 缩放系数")
    parser.add_argument("--entry-threshold", type=float, default=0.03, help="入场 edge 阈值")
    parser.add_argument("--shrinkage", type=float, default=0.6, help="收缩系数 lambda")
    parser.add_argument("--term-alpha", type=float, default=0.05,
                        help="DVOL 期限结构校正指数 (0=不校正, 0.05=温和, 0.10=较陡)")
    parser.add_argument("--symbol", default="BTC", choices=["BTC", "ETH"], help="币种 (默认 BTC)")
    parser.add_argument("--download-only", action="store_true", help="仅下载数据不跑回测")
    parser.add_argument("--no-market-prices", action="store_true", help="不使用 Polymarket 市场价格")
    parser.add_argument("--no-charts", action="store_true", help="不生成图表")
    parser.add_argument("--no-report", action="store_true", help="不生成报告")
    parser.add_argument("--polymarket-cache-dir", default="data/polymarket",
                        help="Polymarket 缓存目录")
    args = parser.parse_args()

    setup_logger()

    config = TouchBacktestConfig(
        month=args.month,
        symbol=args.symbol,
        cache_dir=args.cache_dir,
        iv_cache_dir=args.iv_cache_dir,
        output_dir=args.output_dir,
        step_minutes=args.step_minutes,
        iv_source=args.iv_source,
        default_sigma=args.default_sigma,
        mu=args.mu,
        vrp_k=args.vrp_k,
        entry_threshold=args.entry_threshold,
        shrinkage_lambda=args.shrinkage,
        term_structure_alpha=args.term_alpha,
        use_market_prices=not args.no_market_prices,
        polymarket_cache_dir=args.polymarket_cache_dir,
    )

    engine = TouchBacktestEngine(config=config)

    # 下载数据
    logger.info(f"检查/下载 K线数据 ({args.month})...")
    engine.download_data()

    logger.info("检查/下载 DVOL 历史数据...")
    engine.download_iv_data()

    if config.use_market_prices:
        logger.info("检查/下载 Polymarket 市场数据...")
        engine.download_polymarket_data()

    if args.download_only:
        logger.info("仅下载模式，完成")
        return

    # 运行回测
    logger.info("开始触碰障碍期权回测...")
    observations = engine.run()

    if not observations:
        logger.error("无观测结果")
        return

    # 计算指标
    metrics = engine.compute_metrics(observations)

    # 输出关键指标
    print(f"\n回测完成 ({args.month})")
    print(f"  观测数: {metrics.get('n_observations', 0)}")
    print(f"  预测数: {metrics.get('n_predictions', 0)}")

    brier = metrics.get("brier_score")
    if brier is not None:
        print(f"  Brier Score: {brier:.6f}")

    per_barrier = metrics.get("per_barrier", {})
    if per_barrier:
        print(f"\n  各 Barrier 详情:")
        for barrier in sorted(per_barrier.keys()):
            info = per_barrier[barrier]
            label_str = "YES" if info.get("label") == 1 else "NO"
            mean_pred = info.get("mean_pred", 0)
            brier_b = info.get("brier_score", None)
            mean_edge = info.get("mean_edge", None)
            print(f"    ${barrier:,.0f}: label={label_str}, "
                  f"mean_p={mean_pred:.4f}", end="")
            if brier_b is not None:
                print(f", brier={brier_b:.4f}", end="")
            if mean_edge is not None:
                print(f", mean_edge={mean_edge:+.4f}", end="")
            print()

    # 生成报告
    if not args.no_report:
        logger.info("生成报告...")
        report_path = generate_report(observations, config)
        print(f"\n  报告: {report_path}")

    # 生成图表
    if not args.no_charts:
        logger.info("生成图表...")
        chart_gen = TouchChartGenerator(output_dir=config.output_dir, symbol=config.symbol)
        chart_paths = chart_gen.generate(observations, args.month)
        print(f"  图表: {len(chart_paths)} 个文件 → {config.output_dir}/")

    print(f"  结果目录: {config.output_dir}/")


if __name__ == "__main__":
    main()
