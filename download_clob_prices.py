#!/usr/bin/env python3
"""
CLOB 价格历史批量下载脚本

从 data/btc_above_events.json 读取所有市场，
用 PolymarketTradeCache 批量下载/更新 CLOB 价格历史。

用法:
    python download_clob_prices.py           # 仅下载缺失的
    python download_clob_prices.py --force    # 强制重新下载全部
    python download_clob_prices.py --check    # 仅检查缓存状态
"""

import argparse
import logging
import sys

from backtest.orderbook_reader import load_markets_from_events_json
from backtest.polymarket_trades import PolymarketTradeCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="CLOB 价格历史批量下载")
    parser.add_argument(
        "--force", action="store_true",
        help="强制重新下载已有缓存",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="仅检查缓存状态，不下载",
    )
    parser.add_argument(
        "--events-json", default="data/btc_above_events.json",
        help="事件 JSON 文件路径 (默认: data/btc_above_events.json)",
    )
    parser.add_argument(
        "--cache-dir", default="data/polymarket",
        help="缓存目录 (默认: data/polymarket)",
    )
    args = parser.parse_args()

    # 加载市场映射
    logger.info(f"加载市场映射: {args.events_json}")
    markets = load_markets_from_events_json(args.events_json)
    if not markets:
        logger.error("未找到任何市场，退出")
        sys.exit(1)

    # 去重: condition_id → (yes_token_id, event_date, strike)
    cid_map: dict[str, tuple[str, str, float]] = {}
    for (event_date, strike), info in markets.items():
        cid_map[info.condition_id] = (info.yes_token_id, event_date, strike)

    total = len(cid_map)
    logger.info(f"共 {total} 个唯一合约 (来自 {len(markets)} 个市场映射)")

    # 初始化缓存
    cache = PolymarketTradeCache(cache_dir=args.cache_dir)

    # 检查缓存状态
    cached = 0
    missing_cids = []
    for cid, (token_id, event_date, strike) in sorted(
        cid_map.items(), key=lambda x: (x[1][1], x[1][2])
    ):
        if cache.has_trades(cid):
            cached += 1
        else:
            missing_cids.append(cid)

    logger.info(f"缓存状态: {cached}/{total} 已缓存, {len(missing_cids)} 缺失")

    if args.check:
        # 打印缺失详情
        if missing_cids:
            logger.info("缺失的合约:")
            for cid in missing_cids:
                token_id, event_date, strike = cid_map[cid]
                logger.info(f"  {event_date} K={strike:.0f} (cid={cid[:16]})")
        else:
            logger.info("所有合约缓存完整!")
        return

    # 确定需要下载的合约
    if args.force:
        download_cids = list(cid_map.keys())
        logger.info(f"强制模式: 重新下载全部 {len(download_cids)} 个合约")
    else:
        download_cids = missing_cids
        if not download_cids:
            logger.info("无需下载，全部缓存完整!")
            return
        logger.info(f"下载缺失的 {len(download_cids)} 个合约")

    # 批量下载
    import time

    success = 0
    for i, cid in enumerate(download_cids):
        token_id, event_date, strike = cid_map[cid]
        logger.info(f"[{i + 1}/{len(download_cids)}] {event_date} K={strike:.0f}")
        cache.download_prices(cid, token_id)
        if cache.has_trades(cid):
            success += 1
        if i < len(download_cids) - 1:
            time.sleep(0.3)  # 限速

    logger.info(f"下载完成: {success}/{len(download_cids)} 成功")

    # 最终状态
    final_cached = sum(1 for cid in cid_map if cache.has_trades(cid))
    logger.info(f"最终缓存状态: {final_cached}/{total}")


if __name__ == "__main__":
    main()
