"""
Polymarket 历史价格缓存 + 查询

使用 CLOB API /prices-history 获取 YES token 历史价格，
缓存为 gzip CSV，提供防前瞻偏差的历史价格查询（bisect_right）
"""

import csv
import gzip
import logging
import os
import time
from bisect import bisect_right
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests

from .polymarket_discovery import PolymarketMarketInfo, snap_strike

logger = logging.getLogger(__name__)


@dataclass
class PolymarketTrade:
    """价格记录点"""
    timestamp_ms: int
    price: float
    size: float  # prices-history 无 size，默认 0


class PolymarketTradeCache:
    """
    价格历史缓存

    每个 condition_id 存为一个 gzip CSV:
    data/polymarket/trades_{cid[:16]}.csv.gz

    使用 CLOB API: GET /prices-history?market={yes_token_id}&interval=all&fidelity=1
    """

    def __init__(self, cache_dir: str = "data/polymarket", clob_api: str = "https://clob.polymarket.com"):
        self.cache_dir = cache_dir
        self.clob_api = clob_api
        os.makedirs(cache_dir, exist_ok=True)

    def _file_path(self, condition_id: str) -> str:
        return os.path.join(self.cache_dir, f"trades_{condition_id[:16]}.csv.gz")

    def has_trades(self, condition_id: str) -> bool:
        return os.path.exists(self._file_path(condition_id))

    def download_prices(self, condition_id: str, yes_token_id: str) -> None:
        """
        从 CLOB API 下载价格历史并缓存

        GET /prices-history?market={yes_token_id}&interval=all&fidelity=1
        返回: {"history": [{"t": unix_ts, "p": price}, ...]}
        """
        path = self._file_path(condition_id)
        url = f"{self.clob_api}/prices-history"
        params = {
            "market": yes_token_id,
            "interval": "all",
            "fidelity": 1,
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"下载价格历史失败 (cid={condition_id[:16]}): {e}")
            return

        # 解析价格历史
        trades = []
        raw_history = data.get("history", []) if isinstance(data, dict) else []

        for entry in raw_history:
            try:
                # API 返回 unix 秒级时间戳，转为毫秒
                ts_sec = int(entry.get("t", 0))
                price = float(entry.get("p", 0))
                if ts_sec > 0 and 0 < price < 1:
                    trades.append((ts_sec * 1000, price, 0.0))
            except (ValueError, TypeError):
                continue

        if not trades:
            logger.info(f"无价格历史 (cid={condition_id[:16]})")
            # 写空文件标记已尝试
            with gzip.open(path, "wt", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp_ms", "price", "size"])
            return

        # 按时间排序
        trades.sort(key=lambda x: x[0])

        # 写入 gzip CSV
        with gzip.open(path, "wt", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_ms", "price", "size"])
            for ts, price, size in trades:
                writer.writerow([ts, price, size])

        logger.info(f"缓存价格历史: {len(trades)} 条 (cid={condition_id[:16]})")

    def ensure_trades(
        self,
        markets: Dict[Tuple[str, float], PolymarketMarketInfo],
    ) -> None:
        """批量检查/下载缺失的价格数据"""
        # 去重：condition_id → yes_token_id 映射
        cid_to_token: Dict[str, str] = {}
        for info in markets.values():
            cid_to_token[info.condition_id] = info.yes_token_id

        missing = {cid: tok for cid, tok in cid_to_token.items() if not self.has_trades(cid)}
        if not missing:
            logger.info(f"价格数据缓存完整 ({len(cid_to_token)} 个合约)")
            return

        logger.info(f"下载价格历史: {len(missing)}/{len(cid_to_token)} 个合约")
        for i, (cid, token_id) in enumerate(missing.items()):
            self.download_prices(cid, token_id)
            if i < len(missing) - 1:
                time.sleep(0.3)  # 限速

    def load_trades(self, condition_id: str) -> List[PolymarketTrade]:
        """从缓存加载交易记录（已按时间排序）"""
        path = self._file_path(condition_id)
        if not os.path.exists(path):
            return []

        trades = []
        with gzip.open(path, "rt") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    trades.append(PolymarketTrade(
                        timestamp_ms=int(row["timestamp_ms"]),
                        price=float(row["price"]),
                        size=float(row["size"]),
                    ))
                except (ValueError, KeyError):
                    continue

        return trades


class PolymarketPriceLookup:
    """
    防前瞻偏差的历史价格查询

    使用 bisect_right 只返回 <= timestamp_ms 的最后成交价（forward-fill）
    """

    def __init__(self, trade_cache: PolymarketTradeCache):
        self.trade_cache = trade_cache
        # {condition_id: [(ts_ms, price), ...]} 已排序
        self._data: Dict[str, List[Tuple[int, float]]] = {}

    def preload(self, condition_ids: List[str]) -> None:
        """预加载交易数据到内存"""
        loaded = 0
        for cid in condition_ids:
            if cid in self._data:
                continue
            trades = self.trade_cache.load_trades(cid)
            self._data[cid] = [(t.timestamp_ms, t.price) for t in trades]
            if self._data[cid]:
                loaded += 1

        logger.info(f"预加载价格数据: {loaded} 个有交易的合约")

    def get_price_at(self, condition_id: str, timestamp_ms: int) -> Optional[float]:
        """
        查询 <= timestamp_ms 的最后成交价

        bisect_right 保证不使用未来数据
        """
        data = self._data.get(condition_id, [])
        if not data:
            return None

        # bisect_right 找到第一个 > timestamp_ms 的位置
        idx = bisect_right(data, (timestamp_ms, float("inf"))) - 1
        if idx < 0:
            return None

        return data[idx][1]

    def get_market_prices_at(
        self,
        markets: Dict[Tuple[str, float], PolymarketMarketInfo],
        event_date: str,
        timestamp_ms: int,
        k_grid: List[float],
        max_snap_diff: float = 250.0,
    ) -> Dict[float, float]:
        """
        获取事件日各 K 的市场价格

        1. 筛选 event_date 的市场
        2. 对 k_grid 中每个 K snap 到最近的 Polymarket K
        3. 用 get_price_at 获取 YES token 价格

        Returns: {K: market_price}
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
