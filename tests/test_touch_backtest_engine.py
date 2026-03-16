"""
触碰障碍期权回测引擎集成测试

使用合成 K线数据跑完整月度回测
"""

import math
import os
import tempfile

import pytest

from pricing_core.binance_data import Kline
from pricing_core.time_utils import month_boundaries_utc_ms

from touch.backtest_engine import TouchBacktestEngine
from touch.barrier_pricing import one_touch_up, one_touch_down
from touch.iv_source import DeribitIVCache
from touch.models import TouchBacktestConfig, TouchObservationResult


def _make_synthetic_klines(
    start_ms: int,
    end_ms: int,
    base_price: float = 83000.0,
    amplitude: float = 3000.0,
):
    """
    生成合成 1m K线数据

    使用正弦波模拟价格波动，确保有明确的高点和低点
    """
    import numpy as np

    klines = []
    minute_ms = 60_000
    current_ms = start_ms
    n = int((end_ms - start_ms) / minute_ms) + 1

    # 使用正弦波: 价格围绕 base_price 波动
    t = np.linspace(0, 4 * np.pi, n)  # ~2 个完整周期
    prices = base_price + amplitude * np.sin(t)

    # 添加一些随机噪音
    rng = np.random.RandomState(42)
    noise = rng.normal(0, 200, n)
    prices += noise

    for i in range(n):
        close = prices[i]
        high = close + abs(rng.normal(0, 100))
        low = close - abs(rng.normal(0, 100))
        klines.append(Kline(
            open_time=start_ms + i * minute_ms,
            open=close + rng.normal(0, 50),
            high=max(high, close),
            low=min(low, close),
            close=close,
            volume=rng.exponential(100),
            close_time=start_ms + (i + 1) * minute_ms - 1,
        ))

    return klines


class TestTouchBacktestEngine:
    """回测引擎测试"""

    def test_running_extremes(self):
        """验证 running_high / running_low 预计算"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TouchBacktestConfig(
                month="2026-03",
                cache_dir=os.path.join(tmpdir, "klines"),
                iv_cache_dir=os.path.join(tmpdir, "iv"),
                output_dir=os.path.join(tmpdir, "output"),
                use_market_prices=False,
                step_minutes=1440,  # 每天一次（加快测试）
            )
            engine = TouchBacktestEngine(config)

            # 手动注入 K线数据
            engine._kline_times = [1000, 2000, 3000, 4000, 5000]
            engine._kline_highs = [100.0, 105.0, 103.0, 110.0, 108.0]
            engine._kline_lows = [95.0, 98.0, 97.0, 102.0, 100.0]
            engine._kline_closes = [98.0, 103.0, 100.0, 108.0, 105.0]

            engine._precompute_running_extremes()

            # running_high = cumulative max of highs
            assert engine._running_highs == [100.0, 105.0, 105.0, 110.0, 110.0]
            # running_low = cumulative min of lows
            assert engine._running_lows == [95.0, 95.0, 95.0, 95.0, 95.0]

    def test_kline_index_lookup(self):
        """验证 bisect 查找不使用未来数据"""
        config = TouchBacktestConfig(use_market_prices=False)
        engine = TouchBacktestEngine(config)

        engine._kline_times = [1000, 2000, 3000, 4000, 5000]

        assert engine._get_kline_index_at(1000) == 0
        assert engine._get_kline_index_at(1500) == 0  # 在 1000~2000 之间，返回 0
        assert engine._get_kline_index_at(2000) == 1
        assert engine._get_kline_index_at(5000) == 4
        assert engine._get_kline_index_at(6000) == 4  # 超出范围，返回最后一个
        assert engine._get_kline_index_at(500) == 0   # 之前，返回第一个

    def test_sigma_fallback(self):
        """IV 回退链: DVOL → 默认值"""
        config = TouchBacktestConfig(
            default_sigma=0.65,
            iv_source="dvol",
            term_structure_alpha=0.0,  # 禁用期限结构校正，专注测试回退链
            use_market_prices=False,
        )
        engine = TouchBacktestEngine(config)

        # 无 DVOL 数据 → 使用默认值
        sigma = engine._get_sigma_at(1000)
        assert sigma == 0.65

        # 有 DVOL 数据
        engine.iv_cache.set_data([(500, 0.70), (1500, 0.55)])
        sigma = engine._get_sigma_at(1000)
        assert sigma == 0.70  # 使用 500 时刻的值

    def test_sigma_vrp_scaling(self):
        """VRP 缩放"""
        config = TouchBacktestConfig(
            vrp_k=1.2,
            iv_source="dvol",
            term_structure_alpha=0.0,  # 禁用期限结构校正，专注测试 VRP
            use_market_prices=False,
        )
        engine = TouchBacktestEngine(config)
        engine.iv_cache.set_data([(1000, 0.50)])

        sigma = engine._get_sigma_at(1000)
        assert abs(sigma - 0.50 * 1.2) < 1e-10

    def test_sigma_option_chain_fallback_to_dvol(self):
        """option_chain 模式: IV 源不可用时回退到 DVOL"""
        config = TouchBacktestConfig(
            iv_source="option_chain",
            default_sigma=0.65,
            term_structure_alpha=0.0,  # 禁用期限结构校正，专注测试回退链
            use_market_prices=False,
        )
        engine = TouchBacktestEngine(config)
        # _iv_source 为 None（未初始化 DeribitClient）→ 应回退

        # 无 DVOL → 使用默认值
        sigma = engine._get_sigma_at(1000)
        assert sigma == 0.65

        # 有 DVOL → 使用 DVOL
        engine.iv_cache.set_data([(500, 0.55)])
        sigma = engine._get_sigma_at(1000)
        assert sigma == 0.55

    def test_sigma_option_chain_with_mock(self):
        """option_chain 模式: 模拟 ATM IV 获取"""
        config = TouchBacktestConfig(
            iv_source="option_chain",
            vrp_k=1.1,
            default_sigma=0.65,
            use_market_prices=False,
        )
        engine = TouchBacktestEngine(config)

        # 模拟: 手动注入缓存的 ATM IV
        from touch.iv_source import DeribitIVSource
        engine._iv_source = DeribitIVSource(deribit_client=None)  # 无 client
        engine._cached_atm_iv = 0.58  # 模拟从期权链获取的 ATM IV
        engine._cached_atm_iv_ms = 1000

        sigma = engine._get_sigma_at(1000)
        assert abs(sigma - 0.58 * 1.1) < 1e-10  # 0.638

    def test_term_structure_correction_month_start(self):
        """期限结构校正: 月初 (30天剩余) → 几乎无校正"""
        config = TouchBacktestConfig(
            month="2026-03",
            iv_source="dvol",
            term_structure_alpha=0.05,
            use_market_prices=False,
        )
        engine = TouchBacktestEngine(config)

        # 用月初实际时间戳
        from pricing_core.time_utils import month_boundaries_utc_ms
        month_start_ms, _ = month_boundaries_utc_ms("2026-03")
        engine.iv_cache.set_data([(month_start_ms - 1000, 0.50)])  # DVOL = 50%

        # 月初: 剩余 ~31 天 ≈ 30 天 → factor ≈ (30/31)^0.05 ≈ 0.998
        sigma = engine._get_sigma_at(month_start_ms)
        assert 0.498 < sigma < 0.502  # 几乎不变

    def test_term_structure_correction_month_end(self):
        """期限结构校正: 月末 (7天剩余) → 明显放大"""
        config = TouchBacktestConfig(
            month="2026-03",
            iv_source="dvol",
            term_structure_alpha=0.05,
            use_market_prices=False,
        )
        engine = TouchBacktestEngine(config)
        engine.iv_cache.set_data([(0, 0.50)])

        # 月末 7 天前: 2026-03-24 → factor = (30/7)^0.05 ≈ 1.075
        from pricing_core.time_utils import month_boundaries_utc_ms
        _, month_end_ms = month_boundaries_utc_ms("2026-03")
        t_7d_before = month_end_ms - 7 * 86400 * 1000

        sigma = engine._get_sigma_at(t_7d_before)
        # 0.50 * (30/7)^0.05 ≈ 0.50 * 1.075 = 0.5375
        assert 0.53 < sigma < 0.55

    def test_term_structure_correction_disabled(self):
        """alpha=0 → 不校正"""
        config = TouchBacktestConfig(
            month="2026-03",
            iv_source="dvol",
            term_structure_alpha=0.0,
            use_market_prices=False,
        )
        engine = TouchBacktestEngine(config)
        engine.iv_cache.set_data([(0, 0.50)])

        from pricing_core.time_utils import month_boundaries_utc_ms
        _, month_end_ms = month_boundaries_utc_ms("2026-03")
        t_3d_before = month_end_ms - 3 * 86400 * 1000

        sigma = engine._get_sigma_at(t_3d_before)
        assert sigma == 0.50  # 无校正

    def test_term_structure_clamp(self):
        """剩余 < 0.5 天时不再放大（防止极端值）"""
        config = TouchBacktestConfig(
            month="2026-03",
            iv_source="dvol",
            term_structure_alpha=0.10,
            use_market_prices=False,
        )
        engine = TouchBacktestEngine(config)
        engine.iv_cache.set_data([(0, 0.50)])

        from pricing_core.time_utils import month_boundaries_utc_ms
        _, month_end_ms = month_boundaries_utc_ms("2026-03")
        # 月末前 1 小时
        t_1h_before = month_end_ms - 3600 * 1000

        sigma = engine._get_sigma_at(t_1h_before)
        # clamp 到 0.5 天 → (30/0.5)^0.10 = 60^0.10 ≈ 1.506
        # sigma ≈ 0.50 * 1.506 = 0.753，不会无穷大
        assert sigma < 1.0

    def test_option_chain_no_term_correction(self):
        """option_chain 模式不做期限结构校正"""
        config = TouchBacktestConfig(
            month="2026-03",
            iv_source="option_chain",
            term_structure_alpha=0.05,
            vrp_k=1.0,
            use_market_prices=False,
        )
        engine = TouchBacktestEngine(config)
        from touch.iv_source import DeribitIVSource
        engine._iv_source = DeribitIVSource(deribit_client=None)
        engine._cached_atm_iv = 0.48
        engine._cached_atm_iv_ms = 0

        from pricing_core.time_utils import month_boundaries_utc_ms
        _, month_end_ms = month_boundaries_utc_ms("2026-03")
        t_7d_before = month_end_ms - 7 * 86400 * 1000

        sigma = engine._get_sigma_at(t_7d_before)
        # option_chain 已是正确期限，不做期限结构校正
        assert sigma == 0.48

    def test_full_synthetic_backtest(self):
        """合成数据完整回测"""
        month_start_ms, month_end_ms = month_boundaries_utc_ms("2026-03")

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TouchBacktestConfig(
                month="2026-03",
                cache_dir=os.path.join(tmpdir, "klines"),
                iv_cache_dir=os.path.join(tmpdir, "iv"),
                output_dir=os.path.join(tmpdir, "output"),
                step_minutes=720,  # 12 小时一次（加快测试）
                default_sigma=0.65,
                use_market_prices=False,
            )
            engine = TouchBacktestEngine(config)

            # 生成合成 K线
            klines = _make_synthetic_klines(
                start_ms=month_start_ms,
                end_ms=month_end_ms,
                base_price=83000.0,
                amplitude=3000.0,
            )

            # 注入数据
            engine._kline_times = [k.open_time for k in klines]
            engine._kline_highs = [k.high for k in klines]
            engine._kline_lows = [k.low for k in klines]
            engine._kline_closes = [k.close for k in klines]

            engine._precompute_running_extremes()

            # 设置 DVOL 缓存
            engine.iv_cache.set_data([
                (month_start_ms, 0.65),
                (month_start_ms + 10 * 86400_000, 0.60),
                (month_start_ms + 20 * 86400_000, 0.70),
            ])

            # 运行回测
            observations = engine.run()

            assert len(observations) > 0

            # 验证结果
            for obs in observations:
                assert obs.s0 > 0
                assert obs.running_high >= obs.s0 or obs.running_high >= obs.running_low
                assert obs.T_remaining_years >= 0
                assert obs.sigma > 0
                assert len(obs.predictions) > 0

                for barrier, p in obs.predictions.items():
                    assert 0.0 <= p <= 1.0

                # 已触碰的 barrier 应该有 P = 1.0
                for barrier, touched in obs.already_touched.items():
                    if touched:
                        assert obs.predictions[barrier] == 1.0

    def test_metrics_computation(self):
        """指标计算测试"""
        config = TouchBacktestConfig(use_market_prices=False)
        engine = TouchBacktestEngine(config)

        observations = [
            TouchObservationResult(
                obs_utc_ms=1000,
                s0=83000.0,
                running_high=83500.0,
                running_low=82500.0,
                T_remaining_years=0.08,
                sigma=0.65,
                barriers=[80000.0, 90000.0],
                predictions={80000.0: 0.9, 90000.0: 0.4},
                labels={80000.0: 1, 90000.0: 0},
                market_prices={},
                already_touched={80000.0: False, 90000.0: False},
            ),
            TouchObservationResult(
                obs_utc_ms=2000,
                s0=84000.0,
                running_high=84500.0,
                running_low=82000.0,
                T_remaining_years=0.06,
                sigma=0.60,
                barriers=[80000.0, 90000.0],
                predictions={80000.0: 0.95, 90000.0: 0.3},
                labels={80000.0: 1, 90000.0: 0},
                market_prices={},
                already_touched={80000.0: True, 90000.0: False},
            ),
        ]

        metrics = engine.compute_metrics(observations)

        assert metrics["n_observations"] == 2
        assert metrics["n_predictions"] == 4
        assert "brier_score" in metrics
        assert "per_barrier" in metrics
        assert 80000.0 in metrics["per_barrier"]
        assert 90000.0 in metrics["per_barrier"]

    def test_month_boundaries(self):
        """月份边界计算"""
        start_ms, end_ms = month_boundaries_utc_ms("2026-03")

        # 2026-03-01 00:00:00 UTC
        from datetime import datetime
        from pricing_core.time_utils import UTC
        start_dt = datetime.fromtimestamp(start_ms / 1000, tz=UTC)
        assert start_dt.year == 2026
        assert start_dt.month == 3
        assert start_dt.day == 1
        assert start_dt.hour == 0

        # 2026-03-31 23:59:59 UTC
        end_dt = datetime.fromtimestamp(end_ms / 1000, tz=UTC)
        assert end_dt.month == 3
        assert end_dt.day == 31
        assert end_dt.hour == 23
        assert end_dt.minute == 59
