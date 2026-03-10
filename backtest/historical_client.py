"""
历史数据客户端
实现与 BinanceClient 相同的接口，但使用缓存数据防止前视偏差
"""

import bisect
import logging
from typing import List, Optional

import numpy as np

from pricing_core.binance_data import Kline

logger = logging.getLogger(__name__)


class HistoricalBinanceClient:
    """
    防前视偏差的历史 Binance 数据客户端

    通过 set_now() 设置模拟时刻，所有查询只返回 open_time < now_utc_ms 的数据。
    唯一例外: get_close_at_event() 允许访问"未来"数据，仅用于标签生成。
    """

    def __init__(self):
        self._klines: List[Kline] = []
        self._open_times: List[int] = []  # 有序，用于 bisect
        self._now_utc_ms: int = 0

    def preload(self, klines: List[Kline]) -> None:
        """
        预加载全部 K线数据到内存

        Args:
            klines: 按 open_time 升序排列的 K线列表
        """
        self._klines = sorted(klines, key=lambda k: k.open_time)
        self._open_times = [k.open_time for k in self._klines]
        logger.info(f"预加载 {len(self._klines)} 条 K线, "
                     f"范围 [{self._open_times[0]} ~ {self._open_times[-1]}]")

    def set_now(self, now_utc_ms: int) -> None:
        """设置当前模拟时刻"""
        self._now_utc_ms = now_utc_ms

    def get_current_price(self, symbol: str = "BTCUSDT") -> float:
        """
        返回 now_utc_ms 前最近 kline 的 close

        即 open_time < now_utc_ms 的最后一条 K线
        """
        idx = bisect.bisect_left(self._open_times, self._now_utc_ms) - 1
        if idx < 0:
            raise ValueError(f"now_utc_ms={self._now_utc_ms} 之前无数据")
        return self._klines[idx].close

    def get_klines_extended(
        self,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
    ) -> List[Kline]:
        """
        获取指定范围的 K线，硬截断 open_time < now_utc_ms

        Args:
            start_ms: 起始时间 (UTC ms)
            end_ms: 结束时间 (UTC ms)，会被 now_utc_ms 截断

        Returns:
            K线列表
        """
        # 截断: 不返回 now_utc_ms 及之后的数据
        effective_end = min(end_ms, self._now_utc_ms - 1) if end_ms else self._now_utc_ms - 1

        if start_ms is not None:
            left = bisect.bisect_left(self._open_times, start_ms)
        else:
            left = 0

        right = bisect.bisect_right(self._open_times, effective_end)
        return self._klines[left:right]

    def get_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        limit: int = 1000,
    ) -> List[Kline]:
        """兼容 BinanceClient.get_klines 接口"""
        klines = self.get_klines_extended(
            start_ms=start_ms, end_ms=end_ms,
            symbol=symbol, interval=interval,
        )
        return klines[:limit]

    def get_close_at_event(self, event_open_ms: int, symbol: str = "BTCUSDT") -> float:
        """
        获取事件时刻 K线的 close 价格（结算价）

        注意: 这是唯一允许访问"未来"数据的方法，仅用于标签
        """
        idx = bisect.bisect_left(self._open_times, event_open_ms)
        if idx < len(self._klines) and self._klines[idx].open_time == event_open_ms:
            return self._klines[idx].close
        raise ValueError(f"未找到 openTime={event_open_ms} 的 K线")

    def get_close_prices(self, klines: List[Kline]) -> np.ndarray:
        """从 K线列表提取 close 价格数组"""
        return np.array([k.close for k in klines])
