"""vol_forecast 模块测试"""

import numpy as np
import pytest

from pricing_core.vol_forecast import (
    compute_log_returns,
    compute_rv,
    har_features,
    har_predict,
    har_fit,
    intraday_seasonality_factor,
    compute_hourly_rv_profile,
    get_path_hours,
)
from pricing_core.models import HARFeatures, HARCoefficients


class TestComputeLogReturns:

    def test_basic(self):
        prices = np.array([100.0, 101.0, 99.0])
        returns = compute_log_returns(prices)
        assert len(returns) == 2
        assert returns[0] == pytest.approx(np.log(101 / 100))
        assert returns[1] == pytest.approx(np.log(99 / 101))

    def test_constant_prices(self):
        prices = np.ones(10) * 100
        returns = compute_log_returns(prices)
        np.testing.assert_allclose(returns, 0.0)


class TestComputeRV:

    def test_rv_sum_of_squares(self):
        returns = np.array([0.01, -0.02, 0.015, -0.01, 0.005])
        rv = compute_rv(returns, window=3)
        expected = 0.015**2 + 0.01**2 + 0.005**2
        assert rv == pytest.approx(expected)

    def test_rv_full_window(self):
        returns = np.array([0.01, -0.02])
        rv = compute_rv(returns, window=10)  # 窗口 > 数据长度
        expected = 0.01**2 + 0.02**2
        assert rv == pytest.approx(expected)


class TestHARFeatures:

    def test_extracts_features(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.001, 1500)
        features = har_features(returns)
        assert features.rv_30m > 0
        assert features.rv_2h > 0
        assert features.rv_6h > 0
        assert features.rv_24h > 0
        # 更长窗口包含更多 squared returns
        assert features.rv_24h >= features.rv_6h


class TestHARPredict:

    def test_default_coefficients(self):
        features = HARFeatures(rv_30m=1e-6, rv_2h=2e-6, rv_6h=3e-6, rv_24h=4e-6)
        rv_hat = har_predict(features)
        expected = 0.25 * (1e-6 + 2e-6 + 3e-6 + 4e-6)
        assert rv_hat == pytest.approx(expected)

    def test_custom_coefficients(self):
        features = HARFeatures(rv_30m=1e-5, rv_2h=2e-5, rv_6h=3e-5, rv_24h=4e-5)
        coeffs = HARCoefficients(b0=1e-6, b1=0.5, b2=0.3, b3=0.1, b4=0.1)
        rv_hat = har_predict(features, coeffs)
        expected = 1e-6 + 0.5*1e-5 + 0.3*2e-5 + 0.1*3e-5 + 0.1*4e-5
        assert rv_hat == pytest.approx(expected)

    def test_non_negative(self):
        features = HARFeatures(rv_30m=0, rv_2h=0, rv_6h=0, rv_24h=0)
        rv_hat = har_predict(features)
        assert rv_hat > 0  # 下界 1e-12


class TestHARFit:

    def test_recovers_known_coefficients(self):
        """用已知系数生成数据，验证拟合能恢复"""
        rng = np.random.default_rng(42)
        n = 500
        X = rng.uniform(1e-6, 1e-4, (n, 4))
        true_b = np.array([1e-6, 0.4, 0.3, 0.2, 0.1])
        y = X @ true_b[1:] + true_b[0] + rng.normal(0, 1e-7, n)

        coeffs = har_fit(X, y)
        assert coeffs.b1 == pytest.approx(0.4, abs=0.05)
        assert coeffs.b2 == pytest.approx(0.3, abs=0.05)


class TestIntraDaySeasonality:

    def test_uniform_factor_is_one(self):
        """均匀波动率 -> 因子 = 1"""
        hourly_rv = np.ones(24)
        factor = intraday_seasonality_factor(hourly_rv, list(range(24)))
        assert factor == pytest.approx(1.0)

    def test_high_vol_hours(self):
        """高波动时段 -> 因子 > 1"""
        hourly_rv = np.ones(24)
        hourly_rv[14:18] = 3.0  # 14-17 UTC 高波动
        factor = intraday_seasonality_factor(hourly_rv, [14, 15, 16, 17])
        assert factor > 1.0

    def test_low_vol_hours(self):
        """低波动时段 -> 因子 < 1"""
        hourly_rv = np.ones(24) * 2.0
        hourly_rv[2:6] = 0.5  # 凌晨低波动
        factor = intraday_seasonality_factor(hourly_rv, [2, 3, 4, 5])
        assert factor < 1.0


class TestGetPathHours:

    def test_same_hour(self):
        assert get_path_hours(10, 10) == [10]

    def test_forward(self):
        assert get_path_hours(10, 13) == [10, 11, 12, 13]

    def test_wrap_around(self):
        hours = get_path_hours(22, 2)
        assert hours == [22, 23, 0, 1, 2]
