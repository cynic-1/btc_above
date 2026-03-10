"""distribution 模块测试"""

import numpy as np
import pytest
from scipy import stats

from pricing_core.distribution import (
    fit_student_t,
    sample_return,
    build_empirical_cdf,
    compute_standardized_residuals,
)
from pricing_core.models import DistParams


class TestFitStudentT:

    def test_recovers_parameters(self):
        """能从 Student-t 样本恢复参数"""
        rng = np.random.default_rng(42)
        true_df = 5.0
        samples = stats.t.rvs(df=true_df, loc=0, scale=1, size=5000, random_state=rng)

        params = fit_student_t(samples)
        assert params.df == pytest.approx(true_df, rel=0.3)  # df 估计允许 30% 偏差
        assert params.loc == pytest.approx(0, abs=0.1)
        assert params.scale == pytest.approx(1, rel=0.2)

    def test_returns_dist_params(self):
        """返回 DistParams 类型"""
        samples = np.random.default_rng(42).standard_normal(100)
        params = fit_student_t(samples)
        assert isinstance(params, DistParams)
        assert params.df > 0
        assert params.scale > 0


class TestSampleReturn:

    def test_output_shape(self):
        """输出形状正确"""
        params = DistParams(df=5, loc=0, scale=1)
        R = sample_return(rv_hat=1e-4, dist_params=params, n=1000)
        assert R.shape == (1000,)

    def test_variance_scaling(self):
        """方差随 rv_hat 缩放"""
        params = DistParams(df=100, loc=0, scale=1)  # 大 df 近似正态
        rng = np.random.default_rng(42)

        R1 = sample_return(rv_hat=1e-4, dist_params=params, n=50000, rng=rng)
        rng2 = np.random.default_rng(42)
        R2 = sample_return(rv_hat=4e-4, dist_params=params, n=50000, rng=rng2)

        # R2 的方差应约为 R1 的 4 倍
        ratio = np.var(R2) / np.var(R1)
        assert ratio == pytest.approx(4.0, rel=0.3)


class TestBuildEmpiricalCDF:

    def test_cdf_range(self):
        """CDF 范围在 [0, 1]"""
        samples = np.random.default_rng(42).standard_normal(100)
        vals, cdf = build_empirical_cdf(samples)
        assert cdf[0] > 0
        assert cdf[-1] == pytest.approx(1.0)
        assert np.all(np.diff(cdf) >= 0)  # 单调递增

    def test_sorted_values(self):
        """值已排序"""
        samples = np.array([3, 1, 2, 5, 4])
        vals, cdf = build_empirical_cdf(samples)
        np.testing.assert_array_equal(vals, [1, 2, 3, 4, 5])


class TestComputeStandardizedResiduals:

    def test_basic(self):
        log_returns = np.array([0.01, -0.02, 0.015])
        realized_vols = np.array([0.01, 0.02, 0.01])
        z = compute_standardized_residuals(log_returns, realized_vols)
        expected = np.array([1.0, -1.0, 1.5])
        np.testing.assert_allclose(z, expected)

    def test_filters_zero_vol(self):
        """过滤波动率为零的样本"""
        log_returns = np.array([0.01, 0.02, 0.03])
        realized_vols = np.array([0.01, 0.0, 0.01])
        z = compute_standardized_residuals(log_returns, realized_vols)
        assert len(z) == 2  # 中间那个被过滤
