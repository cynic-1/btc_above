"""
校准分析模块测试
"""

import numpy as np
import pytest

from backtest.calibration import (
    CalibrationResult,
    IsotonicCalibrator,
    run_calibration_analysis,
)
from backtest.models import ObservationResult

try:
    import sklearn
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

skip_no_sklearn = pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn 未安装")


class TestIsotonicCalibrator:
    @skip_no_sklearn
    def test_fit_and_transform(self):
        """基本拟合与转换"""
        cal = IsotonicCalibrator()
        preds = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        labels = np.array([0, 0, 0, 1, 1, 1])
        assert cal.fit(preds, labels) is True
        result = cal.transform(preds)
        assert result is not None
        assert len(result) == 6

    def test_transform_without_fit(self):
        """未拟合时 transform 返回 None"""
        cal = IsotonicCalibrator()
        result = cal.transform(np.array([0.5]))
        assert result is None

    @skip_no_sklearn
    def test_output_bounded(self):
        """校准后输出在 [0.01, 0.99] 范围内"""
        cal = IsotonicCalibrator()
        preds = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
        labels = np.array([0, 0, 1, 1, 1])
        cal.fit(preds, labels)
        result = cal.transform(np.array([0.0, 0.5, 1.0]))
        assert result is not None
        assert all(0.01 <= v <= 0.99 for v in result)

    def test_fit_without_sklearn(self):
        """无 sklearn 时 fit 返回 False（或 True 如果已安装）"""
        cal = IsotonicCalibrator()
        preds = np.array([0.1, 0.9])
        labels = np.array([0, 1])
        result = cal.fit(preds, labels)
        assert isinstance(result, bool)


def _make_obs_list(n=20):
    """生成足够多的观测用于校准分析"""
    obs_list = []
    np.random.seed(42)
    for i in range(n):
        settlement = 91000.0 if i % 2 == 0 else 89000.0
        label = 1 if settlement > 90000.0 else 0
        p = 0.7 if label == 1 else 0.3
        p += np.random.normal(0, 0.1)
        p = max(0.05, min(0.95, p))
        obs_list.append(ObservationResult(
            event_date=f"2026-02-{i+1:02d}",
            obs_minutes=60,
            now_utc_ms=1000 + i * 100,
            s0=90000.0,
            settlement_price=settlement,
            k_grid=[90000.0],
            predictions={90000.0: p},
            labels={90000.0: label},
        ))
    return obs_list


class TestRunCalibrationAnalysis:
    def test_basic_flow(self):
        """基本流程：返回 CalibrationResult"""
        obs = _make_obs_list(20)
        result = run_calibration_analysis(obs)
        if result is None:
            pytest.skip("sklearn 不可用")
        assert isinstance(result, CalibrationResult)
        assert result.ece_before >= 0
        assert result.ece_after >= 0
        assert result.brier_before >= 0
        assert result.brier_after >= 0
        assert result.n_train > 0
        assert result.n_test > 0

    def test_insufficient_data(self):
        """数据不足时返回 None"""
        obs = _make_obs_list(3)
        result = run_calibration_analysis(obs)
        assert result is None

    def test_train_frac(self):
        """不同训练比例"""
        obs = _make_obs_list(20)
        result = run_calibration_analysis(obs, train_frac=0.8)
        if result is None:
            pytest.skip("sklearn 不可用")
        assert result.n_train > result.n_test
