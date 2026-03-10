"""实时定价引擎单元测试"""

import pytest
import numpy as np

from pricing_core.binance_data import Kline
from pricing_core.models import DistParams, HARCoefficients
from live.config import LiveTradingConfig
from live.pricing_engine import LivePricingEngine


@pytest.fixture
def config():
    return LiveTradingConfig(
        mc_samples=500,
        dist_refit_minutes=30,
        vrp_k=1.0,
    )


@pytest.fixture
def engine(config):
    return LivePricingEngine(config)


def _make_klines(n: int, base_price: float = 85000.0) -> list:
    """生成 n 条 K线测试数据"""
    klines = []
    # 添加微小随机波动
    rng = np.random.RandomState(42)
    for i in range(n):
        close = base_price + rng.randn() * 50
        klines.append(Kline(
            open_time=i * 60_000,
            open=close - 1,
            high=close + 5,
            low=close - 5,
            close=close,
            volume=100.0,
            close_time=i * 60_000 + 59999,
        ))
    return klines


class TestComputePrices:
    """定价计算测试"""

    def test_basic_pricing(self, engine):
        """基本定价应返回合理概率"""
        klines = _make_klines(1500)  # ~25h 数据
        s0 = 85000.0
        k_list = [84000.0, 85000.0, 86000.0]
        # 事件时间设为 6h 后
        now_utc_ms = 0
        # 2026-03-09 ET 12:00 → UTC 17:00 → 大约 6h 后
        # 使用固定时间避免 DST 问题
        event_date = "2026-03-09"

        probs = engine.compute_prices(
            event_date=event_date,
            now_utc_ms=now_utc_ms,
            s0=s0,
            klines=klines,
            k_list=k_list,
        )

        assert len(probs) == 3
        # 84000 < 85000 → 概率应较高
        assert probs[84000.0] > probs[85000.0]
        # 86000 > 85000 → 概率应较低
        assert probs[85000.0] > probs[86000.0]
        # 所有概率在 [0, 1]
        for p in probs.values():
            assert 0.0 <= p <= 1.0

    def test_insufficient_data(self, engine):
        """数据不足应返回空"""
        klines = _make_klines(30)
        probs = engine.compute_prices(
            event_date="2026-03-09",
            now_utc_ms=0,
            s0=85000.0,
            klines=klines,
            k_list=[85000.0],
        )
        assert probs == {}

    def test_empty_k_list(self, engine):
        """空 K 列表应返回空"""
        klines = _make_klines(1500)
        probs = engine.compute_prices(
            event_date="2026-03-09",
            now_utc_ms=0,
            s0=85000.0,
            klines=klines,
            k_list=[],
        )
        assert probs == {}

    def test_dist_cache(self, engine):
        """分布参数应被缓存"""
        klines = _make_klines(1500)
        k_list = [85000.0]

        # 第一次调用
        engine.compute_prices("2026-03-09", 0, 85000.0, klines, k_list)
        assert engine._cached_dist_params is not None

        # 缓存时间记录
        first_cache_time = engine._cached_dist_time_ms

        # 第二次调用（同一时间 → 不重拟合）
        engine.compute_prices("2026-03-09", 0, 85000.0, klines, k_list)
        assert engine._cached_dist_time_ms == first_cache_time

    def test_reset_cache(self, engine):
        """reset_cache 应清除缓存"""
        engine._cached_dist_params = DistParams(df=5.0, loc=0.0, scale=1.0)
        engine._cached_dist_time_ms = 12345

        engine.reset_cache()

        assert engine._cached_dist_params is None
        assert engine._cached_dist_time_ms == 0


class TestTrainHAR:
    """HAR 训练测试"""

    def test_train_with_enough_data(self, engine):
        """足够数据应训练出系数"""
        klines = _make_klines(3000)
        engine.train_har(klines)

        assert engine._har_coeffs is not None
        assert engine._har_coeffs.b0 != HARCoefficients().b0 or \
               engine._har_coeffs.b1 != HARCoefficients().b1

    def test_train_insufficient_data(self, engine):
        """数据不足应使用默认系数"""
        klines = _make_klines(100)
        engine.train_har(klines)

        coeffs = engine._har_coeffs
        assert coeffs is not None
        # 默认等权重
        assert coeffs.b1 == 0.25
        assert coeffs.b2 == 0.25
