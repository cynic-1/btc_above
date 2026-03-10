#!/usr/bin/env python3
"""
BTC 二元预测市场定价系统 - 入口脚本

用法:
    python run_pricing.py --date 2026-03-05
    python run_pricing.py --date 2026-03-05 --strikes 90000,95000,100000
    python run_pricing.py --date 2026-03-05 --mc-samples 50000
"""

import argparse
import logging
import sys
import time

from dotenv import load_dotenv

from pricing_core.config import PricingConfig
from pricing_core.pipeline import PricingPipeline
from pricing_core.utils.logger import setup_logger

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BTC 二元预测市场定价系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_pricing.py --date 2026-03-05
  python run_pricing.py --date 2026-03-05 --strikes 90000,95000,100000
  python run_pricing.py --date 2026-03-05 --mc-samples 50000 --shrinkage 0.7
        """,
    )
    parser.add_argument(
        "--date", required=True,
        help="事件日期 (YYYY-MM-DD)，结算时刻为 ET 12:00",
    )
    parser.add_argument(
        "--strikes", type=str, default=None,
        help="行权价列表（逗号分隔），默认自动生成",
    )
    parser.add_argument(
        "--mc-samples", type=int, default=None,
        help="Monte Carlo 采样数（默认 10000）",
    )
    parser.add_argument(
        "--shrinkage", type=float, default=None,
        help="收缩系数 lambda（默认 0.6）",
    )
    parser.add_argument(
        "--kelly-eta", type=float, default=None,
        help="Kelly 系数 eta（默认 0.2）",
    )
    parser.add_argument(
        "--log-dir", type=str, default=None,
        help="日志目录（默认 logs/）",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    # 初始化配置
    config = PricingConfig()
    if args.mc_samples:
        config.mc_samples = args.mc_samples
    if args.shrinkage is not None:
        config.shrinkage_lambda = args.shrinkage
    if args.kelly_eta is not None:
        config.kelly_eta = args.kelly_eta
    if args.log_dir:
        config.log_dir = args.log_dir

    # 初始化日志
    setup_logger(log_dir=config.log_dir)

    logger.info("=" * 60)
    logger.info("BTC 二元预测市场定价系统启动")
    logger.info(f"事件日期: {args.date}")
    logger.info(f"MC 采样数: {config.mc_samples}")
    logger.info(f"收缩系数: {config.shrinkage_lambda}")
    logger.info(f"Kelly 系数: {config.kelly_eta}")
    logger.info("=" * 60)

    # 解析行权价
    k_grid = None
    if args.strikes:
        k_grid = [float(k.strip()) for k in args.strikes.split(",")]
        logger.info(f"行权价网格: {k_grid}")

    # 执行定价
    pipeline = PricingPipeline(config=config)

    try:
        start = time.time()
        result = pipeline.run(event_date=args.date, k_grid=k_grid)
        elapsed = time.time() - start

        logger.info(f"定价完成，耗时 {elapsed:.2f}s")

        # 打印简洁摘要到标准输出
        print(f"\n{'='*60}")
        print(f"定价结果 - 事件日期: {args.date}")
        print(f"当前价格: {result.pricing_input.s0:.2f}")
        print(f"距到期: {result.pricing_input.minutes_to_expiry:.1f} 分钟")
        print(f"{'='*60}")
        print(f"{'Strike':>10} {'P(S>K)':>8} {'CI_lo':>8} {'CI_hi':>8} {'Edge':>8} {'Size':>8}")
        print(f"{'-'*60}")
        for sr in result.strike_results:
            print(
                f"{sr.strike:>10.1f} {sr.p_physical:>8.4f} {sr.ci_lower:>8.4f} "
                f"{sr.ci_upper:>8.4f} {sr.edge:>8.4f} {sr.position_size:>8.2f}"
            )
        print(f"{'='*60}\n")

        return 0

    except Exception as e:
        logger.error(f"定价失败: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
