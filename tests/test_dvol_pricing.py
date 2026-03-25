"""
Above 合约解析定价单元测试

测试 GBM 下 P(S_T > K) = Phi(d2) 的各种边界条件和数值正确性
"""

import math

import pytest
from scipy.stats import norm

from above.dvol_pricing import price_above_strikes, prob_above_k_gbm


class TestProbAboveKGBM:
    """prob_above_k_gbm 测试"""

    def test_atm_zero_drift(self):
        """ATM (s0 = K), mu=0 → p 略低于 0.5（因为 -sigma^2/2 漂移）"""
        p = prob_above_k_gbm(s0=87000, K=87000, sigma=0.65, T=1 / 365)
        # d2 = (0 + (0 - 0.65^2/2) * (1/365)) / (0.65 * sqrt(1/365))
        # 负的 -sigma^2/2 漂移使 p < 0.5
        assert 0.45 < p < 0.50, f"ATM p={p}, 应略低于 0.5"

    def test_atm_manual_calculation(self):
        """ATM 手动验证: 与 scipy 直接计算对比"""
        s0 = 87000.0
        K = 87000.0
        sigma = 0.65
        T = 1.0 / 365
        mu = 0.0

        d2 = (math.log(s0 / K) + (mu - sigma**2 / 2) * T) / (sigma * math.sqrt(T))
        expected = norm.cdf(d2)

        actual = prob_above_k_gbm(s0, K, sigma, T, mu)
        assert abs(actual - expected) < 1e-10

    def test_deep_itm(self):
        """Deep ITM: s0 >> K → p 接近 1.0"""
        p = prob_above_k_gbm(s0=100000, K=50000, sigma=0.65, T=1 / 365)
        assert p > 0.99, f"Deep ITM p={p}, 应接近 1.0"

    def test_deep_otm(self):
        """Deep OTM: s0 << K → p 接近 0.0"""
        p = prob_above_k_gbm(s0=50000, K=100000, sigma=0.65, T=1 / 365)
        assert p < 0.01, f"Deep OTM p={p}, 应接近 0.0"

    def test_t_zero_above(self):
        """T=0, s0 > K → 确定性 1.0"""
        p = prob_above_k_gbm(s0=88000, K=87000, sigma=0.65, T=0)
        assert p == 1.0

    def test_t_zero_below(self):
        """T=0, s0 < K → 确定性 0.0"""
        p = prob_above_k_gbm(s0=86000, K=87000, sigma=0.65, T=0)
        assert p == 0.0

    def test_t_zero_equal(self):
        """T=0, s0 = K → 确定性 0.0 (not strictly above)"""
        p = prob_above_k_gbm(s0=87000, K=87000, sigma=0.65, T=0)
        assert p == 0.0

    def test_sigma_zero_above(self):
        """sigma=0, s0 > K → 确定性 1.0（确定性路径 S_T = s0）"""
        p = prob_above_k_gbm(s0=88000, K=87000, sigma=0, T=1 / 365)
        assert p == 1.0

    def test_sigma_zero_below(self):
        """sigma=0, s0 < K → 确定性 0.0"""
        p = prob_above_k_gbm(s0=86000, K=87000, sigma=0, T=1 / 365)
        assert p == 0.0

    def test_sigma_zero_with_drift(self):
        """sigma=0, 正漂移使 S_T > K"""
        # S_T = s0 * exp(mu * T)
        # 如果 mu 足够大使 S_T > K
        p = prob_above_k_gbm(s0=87000, K=87500, sigma=0, T=1, mu=0.1)
        # S_T = 87000 * exp(0.1) = 87000 * 1.105 = 96135 > 87500
        assert p == 1.0

    def test_negative_strike(self):
        """K <= 0 → 必定 above"""
        p = prob_above_k_gbm(s0=87000, K=-1000, sigma=0.65, T=1 / 365)
        assert p == 1.0

        p = prob_above_k_gbm(s0=87000, K=0, sigma=0.65, T=1 / 365)
        assert p == 1.0

    def test_monotone_in_strike(self):
        """p_above 关于 K 单调递减"""
        strikes = [80000, 82000, 84000, 86000, 88000, 90000]
        probs = [
            prob_above_k_gbm(87000, K, 0.65, 1 / 365)
            for K in strikes
        ]
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1], (
                f"非单调: p({strikes[i]})={probs[i]} < p({strikes[i+1]})={probs[i+1]}"
            )

    def test_monotone_in_time(self):
        """ATM: T 越大 p 越接近 0.5（不确定性增加，但受漂移影响不严格）"""
        # 对于 OTM (s0 < K), T 越大越容易上去 → p 增加
        p_short = prob_above_k_gbm(87000, 90000, 0.65, 1 / 365)
        p_long = prob_above_k_gbm(87000, 90000, 0.65, 30 / 365)
        assert p_long > p_short

    def test_positive_drift_increases_p(self):
        """正漂移应增大 p_above"""
        p_zero = prob_above_k_gbm(87000, 87000, 0.65, 1 / 365, mu=0.0)
        p_pos = prob_above_k_gbm(87000, 87000, 0.65, 1 / 365, mu=1.0)
        assert p_pos > p_zero


class TestPriceAboveStrikes:
    """price_above_strikes 批量定价测试"""

    def test_batch_consistency(self):
        """批量结果与单个调用一致"""
        k_grid = [84000, 86000, 88000, 90000]
        results = price_above_strikes(
            s0=87000, k_grid=k_grid, sigma=0.65, T=1 / 365
        )

        assert len(results) == len(k_grid)
        for r in results:
            expected = prob_above_k_gbm(87000, r.strike, 0.65, 1 / 365)
            assert abs(r.p_above - expected) < 1e-10

    def test_result_fields(self):
        """结果字段默认值正确"""
        results = price_above_strikes(
            s0=87000, k_grid=[85000], sigma=0.65, T=1 / 365
        )
        assert len(results) == 1
        r = results[0]
        assert r.strike == 85000
        assert 0 < r.p_above < 1
        assert r.p_trade == 0.0
        assert r.edge == 0.0

    def test_empty_grid(self):
        """空网格返回空列表"""
        results = price_above_strikes(
            s0=87000, k_grid=[], sigma=0.65, T=1 / 365
        )
        assert results == []
