"""
K线缓存模块测试
"""

import gzip
import os
import tempfile

import pandas as pd
import pytest

from pricing_core.binance_data import Kline
from backtest.data_cache import KlineCache, _date_range, _date_to_utc_ms


class TestDateUtils:
    def test_date_to_utc_ms(self):
        ms = _date_to_utc_ms("2026-01-01")
        # 2026-01-01 00:00:00 UTC
        assert ms == 1767225600000

    def test_date_range(self):
        dates = _date_range("2026-01-01", "2026-01-04")
        assert dates == ["2026-01-01", "2026-01-02", "2026-01-03"]

    def test_date_range_same_day(self):
        dates = _date_range("2026-01-01", "2026-01-01")
        assert dates == []


class TestKlineCache:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache = KlineCache(cache_dir=self.tmpdir)

    def test_file_path(self):
        path = self.cache._file_path("2026-01-15")
        assert path.endswith("BTCUSDT_1m_2026-01-15.csv.gz")

    def test_day_exists_false(self):
        assert not self.cache._day_exists("2026-01-15")

    def test_save_and_load_day(self):
        """写入缓存文件后能正确读取"""
        date = "2026-01-15"
        # 手动写一个缓存文件
        rows = []
        base_ms = _date_to_utc_ms(date)
        for i in range(10):
            rows.append({
                "open_time": base_ms + i * 60_000,
                "open": 90000.0 + i,
                "high": 90010.0 + i,
                "low": 89990.0 + i,
                "close": 90005.0 + i,
                "volume": 100.0,
                "close_time": base_ms + i * 60_000 + 59999,
            })
        df = pd.DataFrame(rows)
        path = self.cache._file_path(date)
        df.to_csv(path, index=False, compression="gzip")

        assert self.cache._day_exists(date)

        klines = self.cache.load_day(date)
        assert len(klines) == 10
        assert klines[0].open_time == base_ms
        assert isinstance(klines[0], Kline)

    def test_load_range_ms(self):
        """按毫秒范围加载"""
        date = "2026-01-15"
        base_ms = _date_to_utc_ms(date)
        rows = []
        for i in range(100):
            rows.append({
                "open_time": base_ms + i * 60_000,
                "open": 90000.0,
                "high": 90010.0,
                "low": 89990.0,
                "close": 90005.0,
                "volume": 100.0,
                "close_time": base_ms + i * 60_000 + 59999,
            })
        df = pd.DataFrame(rows)
        df.to_csv(self.cache._file_path(date), index=False, compression="gzip")

        # 加载前 50 条
        klines = self.cache.load_range_ms(base_ms, base_ms + 49 * 60_000)
        assert len(klines) == 50

    def test_load_day_not_found(self):
        with pytest.raises(FileNotFoundError):
            self.cache.load_day("2099-01-01")
