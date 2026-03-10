"""
历史数据客户端测试
"""

import numpy as np
import pytest

from pricing_core.binance_data import Kline
from backtest.historical_client import HistoricalBinanceClient


def _make_klines(n: int = 100, start_ms: int = 1_000_000, interval_ms: int = 60_000) -> list:
    """生成 n 条连续 K线"""
    klines = []
    for i in range(n):
        open_time = start_ms + i * interval_ms
        price = 90000.0 + i * 10  # 线性递增
        klines.append(Kline(
            open_time=open_time,
            open=price,
            high=price + 5,
            low=price - 5,
            close=price + 2,
            volume=100.0,
            close_time=open_time + interval_ms - 1,
        ))
    return klines


class TestHistoricalBinanceClient:
    def setup_method(self):
        self.klines = _make_klines(100)
        self.client = HistoricalBinanceClient()
        self.client.preload(self.klines)

    def test_preload(self):
        assert len(self.client._klines) == 100

    def test_set_now_and_get_current_price(self):
        # 设置 now 在第 50 条之后
        now_ms = self.klines[50].open_time + 30_000  # 在第50条的中间
        self.client.set_now(now_ms)
        price = self.client.get_current_price()
        # 应该返回第 50 条的 close
        assert price == self.klines[50].close

    def test_get_current_price_no_data(self):
        self.client.set_now(0)  # 在所有数据之前
        with pytest.raises(ValueError):
            self.client.get_current_price()

    def test_get_klines_extended_truncated(self):
        """确认不返回 now_utc_ms 及之后的数据"""
        now_ms = self.klines[50].open_time
        self.client.set_now(now_ms)
        result = self.client.get_klines_extended(
            start_ms=self.klines[0].open_time,
            end_ms=self.klines[99].open_time,
        )
        # 所有返回的 K线 open_time 必须 < now_ms
        for k in result:
            assert k.open_time < now_ms

    def test_get_klines_extended_range(self):
        now_ms = self.klines[99].open_time + 60_000  # 在最后一条之后
        self.client.set_now(now_ms)
        result = self.client.get_klines_extended(
            start_ms=self.klines[10].open_time,
            end_ms=self.klines[20].open_time,
        )
        assert len(result) == 11  # [10, 20] 包含两端

    def test_get_close_at_event(self):
        """结算价可以访问"未来"数据"""
        self.client.set_now(self.klines[0].open_time)  # 设 now 在最早
        # 但仍能获取最后一条的结算价
        settlement = self.client.get_close_at_event(self.klines[99].open_time)
        assert settlement == self.klines[99].close

    def test_get_close_at_event_not_found(self):
        with pytest.raises(ValueError):
            self.client.get_close_at_event(999999)

    def test_get_close_prices(self):
        prices = self.client.get_close_prices(self.klines[:5])
        assert isinstance(prices, np.ndarray)
        assert len(prices) == 5

    def test_get_klines_respects_limit(self):
        now_ms = self.klines[99].open_time + 60_000
        self.client.set_now(now_ms)
        result = self.client.get_klines(
            start_ms=self.klines[0].open_time,
            end_ms=self.klines[99].open_time,
            limit=5,
        )
        assert len(result) == 5
