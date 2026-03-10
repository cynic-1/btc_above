"""
订单簿 npz 缓存运行时查询

从预处理的 npz 文件加载 (timestamps_ms, best_bids, best_asks)，
提供与 PolymarketPriceLookup 接口兼容的 mid-price 查询
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .polymarket_discovery import PolymarketMarketInfo, parse_strike_from_question, snap_strike

logger = logging.getLogger(__name__)


@dataclass
class OrderbookQuote:
    """订单簿报价快照"""
    timestamp_ms: int
    best_bid: float
    best_ask: float
    mid_price: float


def load_markets_from_events_json(
    events_json: str = "data/btc_above_events.json",
) -> Dict[Tuple[str, float], PolymarketMarketInfo]:
    """
    从本地 btc_above_events.json 构建市场映射

    返回格式与 discover_markets_for_range() 一致:
    {(event_date, strike): PolymarketMarketInfo}
    """
    with open(events_json, "r") as f:
        events = json.load(f)

    result: Dict[Tuple[str, float], PolymarketMarketInfo] = {}

    for date_str, event in events.items():
        for mkt in event.get("markets", []):
            question = mkt.get("question", "")
            cid = mkt.get("conditionId", "")
            token_ids = mkt.get("clobTokenIds", [])

            if not cid or len(token_ids) < 2:
                continue

            strike = parse_strike_from_question(question)
            if strike is None:
                continue

            info = PolymarketMarketInfo(
                event_date=date_str,
                strike=strike,
                condition_id=cid,
                yes_token_id=token_ids[0],
                no_token_id=token_ids[1],
                question=question,
            )
            result[(date_str, strike)] = info

    logger.info(f"从 events.json 加载 {len(result)} 个市场映射")
    return result


class OrderbookPriceLookup:
    """
    与 PolymarketPriceLookup 接口兼容的订单簿查询

    从 npz 缓存加载数据，用 np.searchsorted 实现 O(log n) 查询
    """

    def __init__(self, cache_dir: str = "data/orderbook_cache"):
        self.cache_dir = cache_dir
        # {condition_id: (timestamps_ms, best_bids, best_asks)}
        self._data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def preload(self, condition_ids: List[str]) -> None:
        """预加载 npz 缓存到内存"""
        loaded = 0
        for cid in condition_ids:
            if cid in self._data:
                continue
            npz_path = os.path.join(self.cache_dir, f"{cid[:16]}.npz")
            if not os.path.exists(npz_path):
                continue
            try:
                data = np.load(npz_path)
                ts = data["timestamps_ms"]
                bids = data["best_bids"]
                asks = data["best_asks"]
                if len(ts) > 0:
                    self._data[cid] = (ts, bids, asks)
                    loaded += 1
            except Exception as e:
                logger.warning(f"加载 npz 失败 (cid={cid[:16]}): {e}")

        logger.info(f"订单簿预加载: {loaded} 个有数据的合约")

    def get_price_at(self, condition_id: str, timestamp_ms: int) -> Optional[float]:
        """
        查询 <= timestamp_ms 的最后 mid_price

        使用 np.searchsorted 实现，对 NumPy 数组比 Python bisect 更快
        """
        arrays = self._data.get(condition_id)
        if arrays is None:
            return None

        timestamps, bids, asks = arrays
        # searchsorted('right') 找到第一个 > timestamp_ms 的位置
        idx = int(np.searchsorted(timestamps, timestamp_ms, side='right')) - 1
        if idx < 0:
            return None

        bid = float(bids[idx])
        ask = float(asks[idx])
        # mid-price: 如果 bid 和 ask 都有效则取均值，否则取非零的那个
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        elif bid > 0:
            return bid
        elif ask > 0:
            return ask
        return None

    def get_quote_at(
        self, condition_id: str, timestamp_ms: int
    ) -> Optional[OrderbookQuote]:
        """返回完整报价 (timestamp_ms, best_bid, best_ask, mid_price)"""
        arrays = self._data.get(condition_id)
        if arrays is None:
            return None

        timestamps, bids, asks = arrays
        idx = int(np.searchsorted(timestamps, timestamp_ms, side='right')) - 1
        if idx < 0:
            return None

        bid = float(bids[idx])
        ask = float(asks[idx])
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
        elif bid > 0:
            mid = bid
        elif ask > 0:
            mid = ask
        else:
            return None

        return OrderbookQuote(
            timestamp_ms=int(timestamps[idx]),
            best_bid=bid,
            best_ask=ask,
            mid_price=mid,
        )

    def get_market_prices_at(
        self,
        markets: Dict[Tuple[str, float], PolymarketMarketInfo],
        event_date: str,
        timestamp_ms: int,
        k_grid: List[float],
        max_snap_diff: float = 250.0,
    ) -> Dict[float, float]:
        """
        与 PolymarketPriceLookup.get_market_prices_at() 接口完全一致

        Returns: {K: mid_price}
        """
        # 筛选该日期的市场
        date_markets: Dict[float, PolymarketMarketInfo] = {}
        for (d, strike), info in markets.items():
            if d == event_date:
                date_markets[strike] = info

        if not date_markets:
            return {}

        available_strikes = list(date_markets.keys())
        result: Dict[float, float] = {}

        for k in k_grid:
            snapped = snap_strike(k, available_strikes, max_diff=max_snap_diff)
            if snapped is None:
                continue

            info = date_markets[snapped]
            price = self.get_price_at(info.condition_id, timestamp_ms)
            if price is not None:
                result[k] = price

        return result

    def get_bid_ask_at(
        self,
        markets: Dict[Tuple[str, float], PolymarketMarketInfo],
        event_date: str,
        timestamp_ms: int,
        k_grid: List[float],
        max_snap_diff: float = 250.0,
    ) -> Dict[float, Tuple[float, float]]:
        """
        获取各 K 的 bid/ask 报价

        Returns: {K: (best_bid, best_ask)}
        """
        date_markets: Dict[float, PolymarketMarketInfo] = {}
        for (d, strike), info in markets.items():
            if d == event_date:
                date_markets[strike] = info

        if not date_markets:
            return {}

        available_strikes = list(date_markets.keys())
        result: Dict[float, Tuple[float, float]] = {}

        for k in k_grid:
            snapped = snap_strike(k, available_strikes, max_diff=max_snap_diff)
            if snapped is None:
                continue

            info = date_markets[snapped]
            quote = self.get_quote_at(info.condition_id, timestamp_ms)
            if quote is not None:
                result[k] = (quote.best_bid, quote.best_ask)

        return result

    def get_first_timestamp(self, condition_id: str) -> Optional[int]:
        """获取某合约最早的报价时间戳（chart_engine 需要）"""
        arrays = self._data.get(condition_id)
        if arrays is None:
            return None
        timestamps = arrays[0]
        if len(timestamps) == 0:
            return None
        return int(timestamps[0])
