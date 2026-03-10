"""pricing 模块测试"""

import numpy as np
import pytest

from pricing_core.pricing import simulate_ST, prob_above_K, confidence_interval, price_strikes
from pricing_core.models import BasisParams, DistParams


class TestSimulateST:

    def test_output_shape(self):
        ST = simulate_ST(
            s0=90000, rv_hat=1e-4,
            dist_params=DistParams(df=5, loc=0, scale=1),
            basis_params=BasisParams(mu_b=0, sigma_b=0),
            n=1000,
        )
        assert ST.shape == (1000,)

    def test_centered_around_s0(self):
        """大量采样均值应接近 S0"""
        rng = np.random.default_rng(42)
        ST = simulate_ST(
            s0=90000, rv_hat=1e-8,  # 极小方差
            dist_params=DistParams(df=100, loc=0, scale=1),
            basis_params=BasisParams(mu_b=0, sigma_b=0),
            n=50000, rng=rng,
        )
        assert np.mean(ST) == pytest.approx(90000, rel=0.01)

    def test_basis_shift(self):
        """基差使均值偏移"""
        rng = np.random.default_rng(42)
        ST = simulate_ST(
            s0=90000, rv_hat=1e-8,
            dist_params=DistParams(df=100, loc=0, scale=1),
            basis_params=BasisParams(mu_b=100, sigma_b=0),
            n=50000, rng=rng,
        )
        assert np.mean(ST) == pytest.approx(90100, rel=0.01)


class TestProbAboveK:

    def test_all_above(self):
        """所有样本都大于 K -> p = 1"""
        ST = np.array([100, 200, 300, 400, 500], dtype=float)
        probs = prob_above_K(ST, [50])
        assert probs[0] == 1.0

    def test_none_above(self):
        """所有样本都小于 K -> p = 0"""
        ST = np.array([100, 200, 300], dtype=float)
        probs = prob_above_K(ST, [1000])
        assert probs[0] == 0.0

    def test_half_above(self):
        """一半大于 K"""
        ST = np.array([1, 2, 3, 4], dtype=float)
        probs = prob_above_K(ST, [2.5])
        assert probs[0] == pytest.approx(0.5)

    def test_multiple_strikes(self):
        """多个行权价一次计算"""
        ST = np.arange(1, 101, dtype=float)
        probs = prob_above_K(ST, [25, 50, 75])
        assert probs[0] == pytest.approx(0.75)
        assert probs[1] == pytest.approx(0.50)
        assert probs[2] == pytest.approx(0.25)

    def test_convergence_to_normal_cdf(self):
        """大样本 MC 收敛到正态 CDF 解析解"""
        from scipy import stats
        rng = np.random.default_rng(42)
        s0 = 90000
        sigma = 0.01  # 1% vol
        # 直接生成正态分布 S_T（绕过 Student-t）
        ST = s0 * np.exp(rng.normal(0, sigma, 100000))

        K = s0  # ATM
        probs = prob_above_K(ST, [K])
        # 解析解: P(S > K) = P(ln(S/S0) > 0) ≈ 0.5（对数正态对称）
        assert probs[0] == pytest.approx(0.5, abs=0.01)


class TestConfidenceInterval:

    def test_ci_contains_point_estimate(self):
        """置信区间包含点估计"""
        rng = np.random.default_rng(42)
        ST = rng.normal(90000, 1000, 10000)
        K = 90000
        p = np.mean(ST > K)
        ci_lo, ci_hi = confidence_interval(ST, K, rng=np.random.default_rng(42))
        assert ci_lo <= p <= ci_hi

    def test_ci_narrows_with_samples(self):
        """更多样本 -> 更窄的置信区间"""
        rng1 = np.random.default_rng(42)
        ST_small = rng1.normal(90000, 1000, 1000)
        ci_lo1, ci_hi1 = confidence_interval(ST_small, 90000, rng=np.random.default_rng(42))

        rng2 = np.random.default_rng(42)
        ST_large = rng2.normal(90000, 1000, 50000)
        ci_lo2, ci_hi2 = confidence_interval(ST_large, 90000, rng=np.random.default_rng(42))

        width1 = ci_hi1 - ci_lo1
        width2 = ci_hi2 - ci_lo2
        assert width2 < width1


class TestPriceStrikes:

    def test_returns_results_for_each_strike(self):
        results = price_strikes(
            s0=90000, rv_hat=1e-4,
            dist_params=DistParams(df=5, loc=0, scale=1),
            basis_params=BasisParams(),
            k_grid=[89000, 90000, 91000],
            n_mc=5000, n_bootstrap=100,
        )
        assert len(results) == 3
        assert results[0].strike == 89000
        assert results[1].strike == 90000
        assert results[2].strike == 91000

    def test_probability_ordering(self):
        """P(S>K) 随 K 增大而递减"""
        results = price_strikes(
            s0=90000, rv_hat=1e-4,
            dist_params=DistParams(df=5, loc=0, scale=1),
            basis_params=BasisParams(),
            k_grid=[88000, 89000, 90000, 91000, 92000],
            n_mc=10000, n_bootstrap=100,
            rng=np.random.default_rng(42),
        )
        probs = [r.p_physical for r in results]
        # 应该大致单调递减
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1] - 0.05  # 允许少量 MC 噪声
