"""
订单簿 npz 缓存运行时查询

从预处理的 npz 文件加载 (timestamps_ms, best_bids, best_asks)，
提供与 PolymarketPriceLookup 接口兼容的 mid-price 查询

数据质量过滤:
- max_stale_ms: 最大过期时间，超过则拒绝
- max_spread: 最大 bid-ask 价差，超过则拒绝
- zero_side_policy: bid 或 ask 为 0 时的处理策略
"""

import json
import logging
import os
from dataclasses import dataclass, field
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


@dataclass
class PriceFilterStats:
    """订单簿价格过滤统计"""
    total_queries: int = 0
    returned: int = 0
    filtered_stale: int = 0
    filtered_zero_side: int = 0
    filtered_wide_spread: int = 0
    no_data: int = 0


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

    数据质量过滤:
    - max_stale_ms: 快照过期阈值（毫秒），0 表示不检查
    - max_spread: bid-ask 最大价差，0 表示不检查
    - zero_side_policy:
        "legacy"  — bid=0 返回 ask, ask=0 返回 bid（原始行为）
        "reject"  — 任一边为 0 则拒绝
        "accept_settled" — 非零边 ≥ 0.995 时允许（已结算合约）
    """

    def __init__(
        self,
        cache_dir: str = "data/orderbook_cache",
        max_stale_ms: int = 3_600_000,
        max_spread: float = 0.15,
        zero_side_policy: str = "legacy",
    ):
        self.cache_dir = cache_dir
        self.max_stale_ms = max_stale_ms
        self.max_spread = max_spread
        self.zero_side_policy = zero_side_policy
        # {condition_id: (timestamps_ms, best_bids, best_asks)}
        self._data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.filter_stats = PriceFilterStats()

    def reset_stats(self) -> None:
        """重置过滤统计"""
        self.filter_stats = PriceFilterStats()

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

    def _validate_snapshot(
        self, timestamp_ms: int, snapshot_ts: int, bid: float, ask: float,
    ) -> Optional[str]:
        """
        校验快照质量

        Returns: None 表示通过，否则返回拒绝原因字符串
        """
        # 1. 过期检查
        if self.max_stale_ms > 0:
            age = timestamp_ms - snapshot_ts
            if age > self.max_stale_ms:
                return "stale"

        # 2. 零边检查
        if bid <= 0 or ask <= 0:
            if self.zero_side_policy == "reject":
                return "zero_side"
            elif self.zero_side_policy == "accept_settled":
                non_zero = bid if bid > 0 else ask
                if non_zero < 0.995:
                    return "zero_side"
                # 已结算合约（非零边 ≥ 0.995），放行
            # "legacy" 策略不在此拒绝

        # 3. 宽幅检查（仅当两边都有效时）
        if self.max_spread > 0 and bid > 0 and ask > 0:
            if (ask - bid) > self.max_spread:
                return "wide_spread"

        return None

    def _compute_mid(self, bid: float, ask: float) -> Optional[float]:
        """根据 zero_side_policy 计算 mid-price"""
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0

        # 零边情况
        if self.zero_side_policy == "legacy":
            if bid > 0:
                return bid
            if ask > 0:
                return ask
            return None
        elif self.zero_side_policy == "accept_settled":
            # _validate_snapshot 已放行 ≥ 0.995 的非零边
            non_zero = bid if bid > 0 else ask
            if non_zero > 0:
                return non_zero
            return None
        else:
            # "reject" — 不应到达这里（_validate_snapshot 已拒绝）
            return None

    def get_price_at(self, condition_id: str, timestamp_ms: int) -> Optional[float]:
        """
        查询 <= timestamp_ms 的最后 mid_price

        使用 np.searchsorted 实现，对 NumPy 数组比 Python bisect 更快
        """
        self.filter_stats.total_queries += 1

        arrays = self._data.get(condition_id)
        if arrays is None:
            self.filter_stats.no_data += 1
            return None

        timestamps, bids, asks = arrays
        # searchsorted('right') 找到第一个 > timestamp_ms 的位置
        idx = int(np.searchsorted(timestamps, timestamp_ms, side='right')) - 1
        if idx < 0:
            self.filter_stats.no_data += 1
            return None

        snapshot_ts = int(timestamps[idx])
        bid = float(bids[idx])
        ask = float(asks[idx])

        # 数据质量校验
        reject_reason = self._validate_snapshot(timestamp_ms, snapshot_ts, bid, ask)
        if reject_reason == "stale":
            self.filter_stats.filtered_stale += 1
            return None
        elif reject_reason == "zero_side":
            self.filter_stats.filtered_zero_side += 1
            return None
        elif reject_reason == "wide_spread":
            self.filter_stats.filtered_wide_spread += 1
            return None

        mid = self._compute_mid(bid, ask)
        if mid is not None:
            self.filter_stats.returned += 1
        else:
            self.filter_stats.no_data += 1
        return mid

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

        snapshot_ts = int(timestamps[idx])
        bid = float(bids[idx])
        ask = float(asks[idx])

        # 数据质量校验
        reject_reason = self._validate_snapshot(timestamp_ms, snapshot_ts, bid, ask)
        if reject_reason is not None:
            return None

        mid = self._compute_mid(bid, ask)
        if mid is None:
            return None

        return OrderbookQuote(
            timestamp_ms=snapshot_ts,
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
