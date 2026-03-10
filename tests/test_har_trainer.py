"""
HAR 训练器测试
"""

import tempfile

import numpy as np
import pandas as pd
import pytest

from pricing_core.models import HARCoefficients
from backtest.data_cache import KlineCache, _date_to_utc_ms
from backtest.har_trainer import HARTrainer


def _create_synthetic_cache(cache_dir: str, start_date: str, n_days: int = 35):
    """创建合成 K线缓存用于测试"""
    cache = KlineCache(cache_dir=cache_dir)
    from datetime import datetime, timedelta

    dt = datetime.strptime(start_date, "%Y-%m-%d")
    for d in range(n_days):
        date_str = (dt + timedelta(days=d)).strftime("%Y-%m-%d")
        base_ms = _date_to_utc_ms(date_str)

        rows = []
        # 每天 1440 条 1m K线
        np.random.seed(42 + d)
        price = 90000.0
        for i in range(1440):
            price += np.random.normal(0, 10)
            rows.append({
                "open_time": base_ms + i * 60_000,
                "open": price,
                "high": price + abs(np.random.normal(0, 5)),
                "low": price - abs(np.random.normal(0, 5)),
                "close": price + np.random.normal(0, 3),
                "volume": 100.0,
                "close_time": base_ms + i * 60_000 + 59999,
            })
        df = pd.DataFrame(rows)
        df.to_csv(cache._file_path(date_str), index=False, compression="gzip")

    return cache


class TestHARTrainer:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache = _create_synthetic_cache(self.tmpdir, "2026-01-01", n_days=35)
        self.trainer = HARTrainer(
            cache=self.cache,
            train_days=30,
            retrain_interval=7,
            ridge_alpha=0.01,
        )

    def test_get_coeffs_returns_coefficients(self):
        """能够训练出 HAR 系数"""
        coeffs = self.trainer.get_coeffs("2026-02-05")
        assert isinstance(coeffs, HARCoefficients)
        # 系数应该有变化（不是默认值）
        assert not (coeffs.b1 == 0.25 and coeffs.b2 == 0.25 and
                    coeffs.b3 == 0.25 and coeffs.b4 == 0.25)

    def test_cache_key_alignment(self):
        """同一 retrain_interval 内的日期共享缓存"""
        key1 = self.trainer._cache_key("2026-01-08")
        key2 = self.trainer._cache_key("2026-01-10")
        assert key1 == key2  # 同一个 7 天区间

    def test_coeffs_cached(self):
        """第二次调用应使用缓存"""
        c1 = self.trainer.get_coeffs("2026-02-05")
        c2 = self.trainer.get_coeffs("2026-02-05")
        assert c1 is c2  # 同一个对象

    def test_insufficient_data_returns_default(self):
        """数据不足时返回默认系数"""
        tmpdir2 = tempfile.mkdtemp()
        cache2 = KlineCache(cache_dir=tmpdir2)
        trainer2 = HARTrainer(cache=cache2, train_days=30)
        coeffs = trainer2.get_coeffs("2026-02-05")
        assert coeffs.b1 == 0.25  # 默认值
