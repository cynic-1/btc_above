"""
Walk-forward 验证模块测试
"""

import pytest

from backtest.models import ObservationResult
from backtest.walk_forward import (
    WalkForwardResult,
    WalkForwardValidator,
    WalkForwardWindow,
)


def _make_obs(event_date, now_ms, settlement, label, p_model=0.7, market_price=0.5):
    """构造单个观测"""
    return ObservationResult(
        event_date=event_date,
        obs_minutes=60,
        now_utc_ms=now_ms,
        s0=90000.0,
        settlement_price=settlement,
        k_grid=[90000.0],
        predictions={90000.0: p_model},
        labels={90000.0: label},
        market_prices={90000.0: market_price},
    )


class TestWalkForwardValidator:
    def test_generate_windows_basic(self):
        """基本窗口生成"""
        v = WalkForwardValidator(train_days=7, test_days=3, step_days=3)
        windows = v.generate_windows("2026-02-01", "2026-02-20")
        assert len(windows) > 0
        # 每个窗口是 4 元组
        for w in windows:
            assert len(w) == 4

    def test_generate_windows_too_short(self):
        """数据太短无法生成窗口"""
        v = WalkForwardValidator(train_days=14, test_days=7, step_days=7)
        windows = v.generate_windows("2026-02-01", "2026-02-05")
        assert len(windows) == 0

    def test_run_empty_observations(self):
        """空观测列表"""
        v = WalkForwardValidator()
        result = v.run([])
        assert isinstance(result, WalkForwardResult)
        assert len(result.windows) == 0

    def test_run_single_date(self):
        """单日期，无法生成窗口"""
        obs = [_make_obs("2026-02-15", 1000, 91000.0, 1)]
        v = WalkForwardValidator()
        result = v.run(obs)
        assert isinstance(result, WalkForwardResult)

    def test_run_with_data(self):
        """多日期数据，验证返回结构"""
        obs = []
        for d in range(1, 25):
            date = f"2026-02-{d:02d}"
            settlement = 91000.0 if d % 2 == 0 else 89000.0
            label = 1 if settlement > 90000.0 else 0
            p_model = 0.8 if label == 1 else 0.2
            obs.append(_make_obs(date, 1000 + d, settlement, label, p_model))

        v = WalkForwardValidator(train_days=7, test_days=7, step_days=7)
        result = v.run(obs)
        assert isinstance(result, WalkForwardResult)
        assert len(result.windows) > 0

        for w in result.windows:
            assert isinstance(w, WalkForwardWindow)
            assert w.window_id > 0

    def test_aggregate_metrics(self):
        """验证汇总指标"""
        obs = []
        for d in range(1, 25):
            date = f"2026-02-{d:02d}"
            settlement = 91000.0 if d % 2 == 0 else 89000.0
            label = 1 if settlement > 90000.0 else 0
            obs.append(_make_obs(date, 1000 + d, settlement, label, 0.7, 0.5))

        v = WalkForwardValidator(train_days=7, test_days=7, step_days=7)
        result = v.run(obs)
        if result.windows:
            assert result.aggregate_pnl is not None

    def test_adaptive_windows_short_data(self):
        """短数据自动缩小窗口"""
        v = WalkForwardValidator(train_days=14, test_days=7, step_days=7)
        windows = v.generate_windows("2026-02-01", "2026-02-15")
        # 15 天 < 14 + 2*7 = 28，应自适应
        assert len(windows) >= 1
        assert v._actual_params["adapted"] is True

    def test_actual_params_in_result(self):
        """结果中包含实际窗口参数"""
        obs = []
        for d in range(1, 15):
            date = f"2026-02-{d:02d}"
            obs.append(_make_obs(date, 1000 + d, 91000.0, 1, 0.7, 0.5))

        v = WalkForwardValidator(train_days=14, test_days=7, step_days=7)
        result = v.run(obs)
        assert result.actual_params is not None
        assert "train_days" in result.actual_params
        assert "adapted" in result.actual_params
