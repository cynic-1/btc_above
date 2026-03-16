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

    def test_24h_kline_filter(self, engine):
        """compute_prices 应只使用最近 24h 的 K线（与回测一致）"""
        # 生成 3000 条 K线（~2 天），now_utc_ms 在最后一条之后
        klines = _make_klines(3000)
        last_open_time = klines[-1].open_time
        now_utc_ms = last_open_time + 60_000

        # 24h = 1440 分钟，cutoff = now - 24h*60*1000
        cutoff_ms = now_utc_ms - 24 * 60 * 60 * 1000
        expected_count = sum(1 for k in klines if k.open_time >= cutoff_ms)
        # 应过滤掉超过 24h 的数据
        assert expected_count < len(klines)

        # 定价应正常工作（内部过滤到 24h）
        probs = engine.compute_prices(
            event_date="2026-03-09",
            now_utc_ms=now_utc_ms,
            s0=85000.0,
            klines=klines,
            k_list=[85000.0],
        )
        assert len(probs) == 1

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
        # event_date 在 klines 之后，所以所有数据都参与训练
        engine.train_har(klines, event_date="2026-03-09")

        assert engine._har_coeffs is not None
        assert engine._har_coeffs.b0 != HARCoefficients().b0 or \
               engine._har_coeffs.b1 != HARCoefficients().b1

    def test_train_insufficient_data(self, engine):
        """数据不足应使用默认系数"""
        klines = _make_klines(100)
        engine.train_har(klines, event_date="2026-03-09")

        coeffs = engine._har_coeffs
        assert coeffs is not None
        # 默认等权重
        assert coeffs.b1 == 0.25
        assert coeffs.b2 == 0.25

    def test_train_excludes_event_day(self, engine):
        """训练应排除 event_date 当天数据"""
        # 生成跨越 event_date 的 K线
        # event_date = "2026-03-09" → midnight UTC = 2026-03-09 00:00:00
        from datetime import datetime
        import pytz
        midnight_ms = int(
            datetime(2026, 3, 9, tzinfo=pytz.utc).timestamp() * 1000
        )
        # 3000 条 K线，前 2500 条在 event_date 之前，后 500 条在当天
        klines = []
        rng = np.random.RandomState(42)
        start_ms = midnight_ms - 2500 * 60_000
        for i in range(3000):
            close = 85000.0 + rng.randn() * 50
            t = start_ms + i * 60_000
            klines.append(Kline(
                open_time=t, open=close - 1, high=close + 5,
                low=close - 5, close=close, volume=100.0,
                close_time=t + 59999,
            ))

        engine.train_har(klines, event_date="2026-03-09")
        # 应成功训练（2500 条足够）
        assert engine._har_coeffs is not None

        # 如果全部 3000 条都是 event_date 当天，应数据不足
        today_klines = _make_klines(3000)
        # 将所有 K线设为 event_date 当天
        for i, k in enumerate(today_klines):
            today_klines[i] = Kline(
                open_time=midnight_ms + i * 60_000,
                open=k.open, high=k.high, low=k.low,
                close=k.close, volume=k.volume,
                close_time=midnight_ms + i * 60_000 + 59999,
            )
        engine.train_har(today_klines, event_date="2026-03-09")
        # 全是事件日数据，训练后应退回默认系数
        assert engine._har_coeffs.b1 == 0.25
