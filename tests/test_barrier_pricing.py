"""
barrier_pricing 解析公式单元测试

覆盖：
- 已知值验证
- 边界条件 (K=S0, T=0, sigma=0)
- 单调性（K 越远 → 概率越低）
- 对称性（K=S0 时 P=1）
- 已触碰处理
"""

import math
import pytest
from scipy.stats import norm

from touch.barrier_pricing import (
    one_touch_up,
    one_touch_down,
    one_touch,
    price_touch_barriers,
)


class TestOneTouchUp:
    """上触碰概率测试"""

    def test_at_the_money(self):
        """K = S0 → P = 1.0"""
        p = one_touch_up(100.0, 100.0, 0.20, 1.0)
        assert p == 1.0

    def test_below_spot(self):
        """K < S0 → P = 1.0（已经在 barrier 之上）"""
        p = one_touch_up(100.0, 90.0, 0.20, 1.0)
        assert p == 1.0

    def test_zero_time(self):
        """T = 0, K > S0 → P = 0"""
        p = one_touch_up(100.0, 110.0, 0.20, 0.0)
        assert p == 0.0

    def test_zero_sigma(self):
        """sigma = 0, mu = 0 → 价格不变，K > S0 → P = 0"""
        p = one_touch_up(100.0, 110.0, 0.0, 1.0, mu=0.0)
        assert p == 0.0

    def test_zero_sigma_positive_drift(self):
        """sigma = 0, mu > 0 → 确定性上涨"""
        # S_T = 100 * exp(0.5 * 1) = 100 * 1.6487 = 164.87 > 110
        p = one_touch_up(100.0, 110.0, 0.0, 1.0, mu=0.5)
        assert p == 1.0

    def test_zero_sigma_insufficient_drift(self):
        """sigma = 0, mu > 0 但不足以触碰"""
        # S_T = 100 * exp(0.01 * 1) = 101.005 < 110
        p = one_touch_up(100.0, 110.0, 0.0, 1.0, mu=0.01)
        assert p == 0.0

    def test_known_value(self):
        """已知参数验证（手动计算）"""
        s0, K, sigma, T, mu = 100.0, 110.0, 0.20, 1.0, 0.0
        nu = mu - sigma ** 2 / 2  # -0.02
        m = math.log(K / s0)      # 0.09531

        d1 = (nu * T - m) / (sigma * math.sqrt(T))
        d2 = (-nu * T - m) / (sigma * math.sqrt(T))
        exp_term = math.exp(2 * nu * m / sigma ** 2)

        expected = norm.cdf(d1) + exp_term * norm.cdf(d2)
        actual = one_touch_up(s0, K, sigma, T, mu)

        assert abs(actual - expected) < 1e-10

    def test_monotonicity_in_strike(self):
        """K 越高 → 触碰概率越低"""
        s0, sigma, T = 100.0, 0.30, 1.0
        p1 = one_touch_up(s0, 105.0, sigma, T)
        p2 = one_touch_up(s0, 110.0, sigma, T)
        p3 = one_touch_up(s0, 120.0, sigma, T)
        p4 = one_touch_up(s0, 150.0, sigma, T)

        assert p1 > p2 > p3 > p4

    def test_monotonicity_in_time(self):
        """T 越长 → 触碰概率越高"""
        s0, K, sigma = 100.0, 110.0, 0.30
        p1 = one_touch_up(s0, K, sigma, 0.1)
        p2 = one_touch_up(s0, K, sigma, 0.5)
        p3 = one_touch_up(s0, K, sigma, 1.0)
        p4 = one_touch_up(s0, K, sigma, 5.0)

        assert p1 < p2 < p3 < p4

    def test_monotonicity_in_sigma(self):
        """sigma 越高 → 触碰概率越高"""
        s0, K, T = 100.0, 120.0, 1.0
        p1 = one_touch_up(s0, K, 0.10, T)
        p2 = one_touch_up(s0, K, 0.30, T)
        p3 = one_touch_up(s0, K, 0.50, T)

        assert p1 < p2 < p3

    def test_probability_bounds(self):
        """概率在 [0, 1] 范围内"""
        test_cases = [
            (100, 200, 0.50, 0.1),
            (100, 101, 0.80, 2.0),
            (100, 300, 0.10, 0.01),
            (100, 100.01, 0.01, 0.001),
        ]
        for s0, K, sigma, T in test_cases:
            p = one_touch_up(s0, K, sigma, T)
            assert 0.0 <= p <= 1.0, f"Out of bounds: p={p} for (s0={s0}, K={K})"

    def test_high_vol_near_certainty(self):
        """高波动率 + 长时间 + 近 barrier → 接近 1"""
        p = one_touch_up(100.0, 101.0, 0.80, 2.0)
        assert p > 0.95

    def test_deep_otm_low_prob(self):
        """远 OTM + 低波动率 + 短时间 → 接近 0"""
        p = one_touch_up(100.0, 200.0, 0.10, 0.01)
        assert p < 0.01


class TestOneTouchDown:
    """下触碰概率测试"""

    def test_at_the_money(self):
        """K = S0 → P = 1.0"""
        p = one_touch_down(100.0, 100.0, 0.20, 1.0)
        assert p == 1.0

    def test_above_spot(self):
        """K > S0 → P = 1.0（已经在 barrier 之下）"""
        p = one_touch_down(100.0, 110.0, 0.20, 1.0)
        assert p == 1.0

    def test_zero_time(self):
        """T = 0, K < S0 → P = 0"""
        p = one_touch_down(100.0, 90.0, 0.20, 0.0)
        assert p == 0.0

    def test_zero_barrier(self):
        """K = 0 → P = 0"""
        p = one_touch_down(100.0, 0.0, 0.20, 1.0)
        assert p == 0.0

    def test_monotonicity_in_strike(self):
        """K 越低 → 下触碰概率越低"""
        s0, sigma, T = 100.0, 0.30, 1.0
        p1 = one_touch_down(s0, 95.0, sigma, T)
        p2 = one_touch_down(s0, 90.0, sigma, T)
        p3 = one_touch_down(s0, 80.0, sigma, T)
        p4 = one_touch_down(s0, 50.0, sigma, T)

        assert p1 > p2 > p3 > p4

    def test_monotonicity_in_time(self):
        """T 越长 → 下触碰概率越高"""
        s0, K, sigma = 100.0, 90.0, 0.30
        p1 = one_touch_down(s0, K, sigma, 0.1)
        p2 = one_touch_down(s0, K, sigma, 0.5)
        p3 = one_touch_down(s0, K, sigma, 1.0)

        assert p1 < p2 < p3

    def test_known_value(self):
        """已知参数验证"""
        s0, K, sigma, T, mu = 100.0, 90.0, 0.20, 1.0, 0.0
        nu = mu - sigma ** 2 / 2
        m = math.log(K / s0)

        d1 = (m - nu * T) / (sigma * math.sqrt(T))
        d2 = (m + nu * T) / (sigma * math.sqrt(T))
        exp_term = math.exp(2 * nu * m / sigma ** 2)

        expected = norm.cdf(d1) + exp_term * norm.cdf(d2)
        actual = one_touch_down(s0, K, sigma, T, mu)

        assert abs(actual - expected) < 1e-10


class TestOneTouch:
    """自动判断方向测试"""

    def test_auto_up(self):
        """K > S0 → 自动选择上触碰"""
        p = one_touch(100.0, 110.0, 0.30, 1.0)
        p_up = one_touch_up(100.0, 110.0, 0.30, 1.0)
        assert abs(p - p_up) < 1e-10

    def test_auto_down(self):
        """K < S0 → 自动选择下触碰"""
        p = one_touch(100.0, 90.0, 0.30, 1.0)
        p_down = one_touch_down(100.0, 90.0, 0.30, 1.0)
        assert abs(p - p_down) < 1e-10

    def test_auto_atm(self):
        """K = S0 → P = 1.0"""
        p = one_touch(100.0, 100.0, 0.30, 1.0)
        assert p == 1.0


class TestPriceTouchBarriers:
    """批量定价测试"""

    def test_basic_batch(self):
        """基本批量定价"""
        results = price_touch_barriers(
            s0=80000.0,
            barriers=[70000.0, 75000.0, 85000.0, 90000.0, 95000.0],
            sigma=0.65,
            T=30 / 365.25,
            mu=0.0,
            running_high=82000.0,
            running_low=78000.0,
        )
        assert len(results) == 5

        # 检查方向
        assert results[0].direction == "down"  # 70000 < 80000
        assert results[1].direction == "down"  # 75000 < 80000
        assert results[2].direction == "up"    # 85000 > 80000
        assert results[3].direction == "up"    # 90000 > 80000
        assert results[4].direction == "up"    # 95000 > 80000

        # 所有概率在合理范围
        for r in results:
            assert 0.0 <= r.p_touch <= 1.0

    def test_already_touched_up(self):
        """已触碰（上）→ P = 1.0"""
        results = price_touch_barriers(
            s0=80000.0,
            barriers=[85000.0],
            sigma=0.65,
            T=20 / 365.25,
            mu=0.0,
            running_high=86000.0,  # 已超过 85000
            running_low=75000.0,
        )
        assert results[0].p_touch == 1.0
        assert results[0].already_touched is True

    def test_already_touched_down(self):
        """已触碰（下）→ P = 1.0"""
        results = price_touch_barriers(
            s0=80000.0,
            barriers=[75000.0],
            sigma=0.65,
            T=20 / 365.25,
            mu=0.0,
            running_high=82000.0,
            running_low=74000.0,  # 已低于 75000
        )
        assert results[0].p_touch == 1.0
        assert results[0].already_touched is True

    def test_not_touched(self):
        """未触碰 → 计算概率"""
        results = price_touch_barriers(
            s0=80000.0,
            barriers=[90000.0],
            sigma=0.65,
            T=20 / 365.25,
            mu=0.0,
            running_high=82000.0,  # 未达 90000
            running_low=78000.0,
        )
        assert results[0].already_touched is False
        assert 0.0 < results[0].p_touch < 1.0

    def test_btc_realistic_params(self):
        """BTC 实际参数测试（sigma=65%, 30天, 当前 83k）"""
        s0 = 83000.0
        sigma = 0.65
        T = 30 / 365.25  # 30 天
        barriers = [75000, 80000, 85000, 90000, 95000, 100000, 110000]

        results = price_touch_barriers(
            s0=s0,
            barriers=[float(b) for b in barriers],
            sigma=sigma,
            T=T,
            mu=0.0,
            running_high=s0,
            running_low=s0,
        )

        # 越远的 barrier → 概率越低
        up_results = [r for r in results if r.direction == "up"]
        for i in range(len(up_results) - 1):
            assert up_results[i].p_touch >= up_results[i + 1].p_touch

        down_results = [r for r in results if r.direction == "down"]
        # down_results 按 barrier 递减排列（75000, 80000），
        # barrier 越低 → 概率越低
        for i in range(len(down_results) - 1):
            assert down_results[i].p_touch <= down_results[i + 1].p_touch
