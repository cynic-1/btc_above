"""
混合价格查询: Orderbook + CLOB 回退

per-query 检测 orderbook 数据新鲜度，过期时自动回退到 CLOB 价格历史。
接口与 OrderbookPriceLookup / PolymarketPriceLookup 完全兼容。
"""

import logging
from typing import Dict, List, Optional, Tuple

from .orderbook_reader import OrderbookPriceLookup, OrderbookQuote
from .polymarket_discovery import PolymarketMarketInfo, snap_strike
from .polymarket_trades import PolymarketPriceLookup

logger = logging.getLogger(__name__)


class HybridPriceLookup:
    """
    混合价格源: orderbook 新鲜时优先使用，过期时回退 CLOB

    过期检测: query_ts - data_ts > staleness_threshold_ms → 过期
    """

    def __init__(
        self,
        orderbook: OrderbookPriceLookup,
        clob: PolymarketPriceLookup,
        staleness_threshold_ms: int = 7_200_000,  # 2h
    ):
        self.orderbook = orderbook
        self.clob = clob
        self.staleness_threshold_ms = staleness_threshold_ms

    def _is_stale(self, condition_id: str, query_ts: int) -> bool:
        """检查 orderbook 数据是否过期"""
        quote = self.orderbook.get_quote_at(condition_id, query_ts)
        if quote is None:
            return True
        return (query_ts - quote.timestamp_ms) > self.staleness_threshold_ms

    def get_price_at(self, condition_id: str, timestamp_ms: int) -> Optional[float]:
        """查询价格，orderbook 过期时回退 CLOB"""
        if not self._is_stale(condition_id, timestamp_ms):
            return self.orderbook.get_price_at(condition_id, timestamp_ms)
        return self.clob.get_price_at(condition_id, timestamp_ms)

    def get_quote_at(
        self, condition_id: str, timestamp_ms: int
    ) -> Optional[OrderbookQuote]:
        """返回完整报价（仅 orderbook 新鲜时可用，否则从 CLOB 构造）"""
        if not self._is_stale(condition_id, timestamp_ms):
            return self.orderbook.get_quote_at(condition_id, timestamp_ms)
        # CLOB 无 bid/ask，用 price 构造近似 quote
        price = self.clob.get_price_at(condition_id, timestamp_ms)
        if price is None:
            return None
        return OrderbookQuote(
            timestamp_ms=timestamp_ms,
            best_bid=price,
            best_ask=price,
            mid_price=price,
        )

    def get_market_prices_at(
        self,
        markets: Dict[Tuple[str, float], PolymarketMarketInfo],
        event_date: str,
        timestamp_ms: int,
        k_grid: List[float],
        max_snap_diff: float = 250.0,
    ) -> Dict[float, float]:
        """获取事件日各 K 的市场价格（per-K 判断新鲜度）"""
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
        """获取各 K 的 bid/ask 报价"""
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
        """获取最早数据时间戳（取两源中更早的）"""
        ob_ts = self.orderbook.get_first_timestamp(condition_id)
        # CLOB 内部数据
        clob_data = self.clob._data.get(condition_id, [])
        clob_ts = clob_data[0][0] if clob_data else None

        if ob_ts is not None and clob_ts is not None:
            return min(ob_ts, clob_ts)
        return ob_ts or clob_ts

    def preload(self, condition_ids: List[str]) -> None:
        """预加载两个数据源"""
        self.orderbook.preload(condition_ids)
        self.clob.preload(condition_ids)
