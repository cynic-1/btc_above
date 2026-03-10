"""
Binance 数据模块
提供 BTC/USDT 现货 1m K线数据获取
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import requests
import numpy as np

from .utils.helpers import TokenBucket

logger = logging.getLogger(__name__)

# Binance kline 返回的字段索引
_OPEN_TIME = 0
_OPEN = 1
_HIGH = 2
_LOW = 3
_CLOSE = 4
_VOLUME = 5
_CLOSE_TIME = 6


@dataclass
class Kline:
    """Binance K线数据"""
    open_time: int      # openTime (UTC ms)
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int     # closeTime (UTC ms)


class BinanceClient:
    """Binance REST API 客户端"""

    def __init__(self, base_url: str = "https://api.binance.com", max_rps: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self._limiter = TokenBucket(rate=max_rps)

    def get_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        limit: int = 1000,
    ) -> List[Kline]:
        """
        获取 K线数据

        Args:
            symbol: 交易对
            interval: K线间隔
            start_ms: 起始时间 (UTC ms)
            end_ms: 结束时间 (UTC ms)
            limit: 最大返回条数（上限 1000）

        Returns:
            K线列表，按时间升序
        """
        params: dict = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000),
        }
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms

        self._limiter.acquire()
        url = f"{self.base_url}/api/v3/klines"
        resp = self.session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        raw = resp.json()

        klines = [
            Kline(
                open_time=int(row[_OPEN_TIME]),
                open=float(row[_OPEN]),
                high=float(row[_HIGH]),
                low=float(row[_LOW]),
                close=float(row[_CLOSE]),
                volume=float(row[_VOLUME]),
                close_time=int(row[_CLOSE_TIME]),
            )
            for row in raw
        ]
        logger.debug(f"get_klines: {symbol} {interval}, 返回 {len(klines)} 条")
        return klines

    def get_klines_extended(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
    ) -> List[Kline]:
        """
        获取大量 K线（自动分页，突破 1000 条限制）

        Args:
            symbol: 交易对
            interval: K线间隔
            start_ms: 起始时间 (UTC ms)
            end_ms: 结束时间 (UTC ms)

        Returns:
            K线列表，按时间升序
        """
        all_klines: List[Kline] = []
        cursor = start_ms

        while True:
            batch = self.get_klines(
                symbol=symbol,
                interval=interval,
                start_ms=cursor,
                end_ms=end_ms,
                limit=1000,
            )
            if not batch:
                break
            all_klines.extend(batch)
            # 下一批从最后一条的下一个间隔开始
            cursor = batch[-1].open_time + 60_000  # 1m interval
            if end_ms is not None and cursor > end_ms:
                break
            if len(batch) < 1000:
                break

        return all_klines

    def get_close_at_event(self, event_open_ms: int, symbol: str = "BTCUSDT") -> float:
        """
        获取事件时刻 1m K线的 close 价格

        Args:
            event_open_ms: 事件时刻对应的 kline openTime (UTC ms)
            symbol: 交易对

        Returns:
            该 1m K线的 close 价格

        Raises:
            ValueError: 未找到对应 K线
        """
        klines = self.get_klines(
            symbol=symbol,
            interval="1m",
            start_ms=event_open_ms,
            end_ms=event_open_ms + 60_000,
            limit=1,
        )
        if not klines:
            raise ValueError(f"未找到 openTime={event_open_ms} 的 K线")

        kline = klines[0]
        if kline.open_time != event_open_ms:
            raise ValueError(
                f"K线 openTime 不匹配: 期望 {event_open_ms}, 得到 {kline.open_time}"
            )

        logger.info(f"事件 K线: openTime={kline.open_time}, close={kline.close}")
        return kline.close

    def get_current_price(self, symbol: str = "BTCUSDT") -> float:
        """获取当前价格（最新成交价）"""
        self._limiter.acquire()
        url = f"{self.base_url}/api/v3/ticker/price"
        resp = self.session.get(url, params={"symbol": symbol}, timeout=5)
        resp.raise_for_status()
        return float(resp.json()["price"])

    def get_close_prices(self, klines: List[Kline]) -> np.ndarray:
        """从 K线列表提取 close 价格数组"""
        return np.array([k.close for k in klines])
