"""
时间窗口实验测试
"""

import csv
import os
import tempfile

import pytest

from backtest.models import ObservationResult
from backtest.timing_experiment import (
    TimingWindowResult,
    _compute_composite_score,
    _filter_observations,
    find_optimal_windows,
    load_observations_from_csv,
    run_timing_grid,
)


def _make_obs(
    event_date: str = "2026-02-10",
    obs_minutes: int = 60,
    s0: float = 100000.0,
    settlement: float = 99000.0,
    strikes: list = None,
    predictions: dict = None,
    labels: dict = None,
    market_prices: dict = None,
) -> ObservationResult:
    """构造测试用 ObservationResult"""
    if strikes is None:
        strikes = [99000.0, 100000.0, 101000.0]
    if predictions is None:
        predictions = {k: 0.5 for k in strikes}
    if labels is None:
        labels = {k: int(settlement > k) for k in strikes}
    if market_prices is None:
        market_prices = {k: 0.5 for k in strikes}

    return ObservationResult(
        event_date=event_date,
        obs_minutes=obs_minutes,
        now_utc_ms=0,
        s0=s0,
        settlement_price=settlement,
        k_grid=strikes,
        predictions=predictions,
        labels=labels,
        market_prices=market_prices,
    )


class TestLoadObservationsFromCsv:
    def test_basic_load(self, tmp_path):
        """基本 CSV 加载"""
        csv_path = tmp_path / "detail.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "event_date", "obs_minutes", "strike", "s0", "settlement",
                "p_physical", "label", "ci_lower", "ci_upper", "market_price",
            ])
            writer.writerow(["2026-02-10", "60", "99000", "100000", "99500", "0.6", "1", "", "", "0.55"])
            writer.writerow(["2026-02-10", "60", "100000", "100000", "99500", "0.4", "0", "", "", "0.45"])
            writer.writerow(["2026-02-10", "30", "99000", "100100", "99500", "0.65", "1", "", "", "0.58"])

        obs = load_observations_from_csv(str(csv_path))

        assert len(obs) == 2  # 2 个唯一 (date, minutes) 组合
        # 第一个观测 (30m 排在 60m 前面因为按 key 排序)
        obs_30 = [o for o in obs if o.obs_minutes == 30][0]
        assert obs_30.event_date == "2026-02-10"
        assert len(obs_30.k_grid) == 1
        assert obs_30.predictions[99000.0] == pytest.approx(0.65)
        assert obs_30.market_prices[99000.0] == pytest.approx(0.58)

    def test_missing_market_price(self, tmp_path):
        """市场价格缺失时不加入 market_prices dict"""
        csv_path = tmp_path / "detail.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "event_date", "obs_minutes", "strike", "s0", "settlement",
                "p_physical", "label", "ci_lower", "ci_upper", "market_price",
            ])
            writer.writerow(["2026-02-10", "60", "99000", "100000", "99500", "0.6", "1", "", "", ""])

        obs = load_observations_from_csv(str(csv_path))
        assert len(obs) == 1
        assert len(obs[0].market_prices) == 0

    def test_multiple_dates(self, tmp_path):
        """多个事件日"""
        csv_path = tmp_path / "detail.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "event_date", "obs_minutes", "strike", "s0", "settlement",
                "p_physical", "label", "ci_lower", "ci_upper", "market_price",
            ])
            writer.writerow(["2026-02-10", "60", "99000", "100000", "99500", "0.6", "1", "", "", "0.5"])
            writer.writerow(["2026-02-11", "60", "99000", "100000", "100500", "0.7", "1", "", "", "0.6"])

        obs = load_observations_from_csv(str(csv_path))
        assert len(obs) == 2
        dates = {o.event_date for o in obs}
        assert dates == {"2026-02-10", "2026-02-11"}


class TestFilterObservations:
    def test_basic_filter(self):
        """基本窗口过滤"""
        obs = [
            _make_obs(obs_minutes=720),
            _make_obs(obs_minutes=360),
            _make_obs(obs_minutes=60),
            _make_obs(obs_minutes=10),
            _make_obs(obs_minutes=1),
        ]
        filtered = _filter_observations(obs, start_minutes=360, stop_minutes=10)
        minutes = [o.obs_minutes for o in filtered]
        assert sorted(minutes) == [60, 360]

    def test_start_equals_stop_empty(self):
        """start == stop 时无结果"""
        obs = [_make_obs(obs_minutes=60)]
        filtered = _filter_observations(obs, start_minutes=60, stop_minutes=60)
        assert len(filtered) == 0

    def test_empty_window(self):
        """窗口内无观测"""
        obs = [_make_obs(obs_minutes=720)]
        filtered = _filter_observations(obs, start_minutes=60, stop_minutes=10)
        assert len(filtered) == 0

    def test_boundary_inclusive_exclusive(self):
        """边界条件: stop < obs_minutes <= start"""
        obs = [
            _make_obs(obs_minutes=60),
            _make_obs(obs_minutes=10),
        ]
        # obs_minutes=60: 10 < 60 <= 60 → True
        # obs_minutes=10: 10 < 10 → False
        filtered = _filter_observations(obs, start_minutes=60, stop_minutes=10)
        assert len(filtered) == 1
        assert filtered[0].obs_minutes == 60


class TestCompositeScore:
    def test_perfect_result(self):
        """高指标的结果应有高评分"""
        r = TimingWindowResult(
            start_minutes=360, stop_minutes=0,
            total_pnl=10000, return_pct=10.0,
            profit_factor=3.0, sharpe=2.0,
            roi=0.5, max_drawdown_pct=1.0,
            n_trades=50,
        )
        score = _compute_composite_score(r)
        assert score > 0.5

    def test_bad_result(self):
        """差指标的结果应有低评分"""
        r = TimingWindowResult(
            start_minutes=360, stop_minutes=0,
            total_pnl=-5000, return_pct=-5.0,
            profit_factor=0.5, sharpe=-1.0,
            roi=-0.3, max_drawdown_pct=8.0,
            n_trades=50,
        )
        score = _compute_composite_score(r)
        assert score < 0.3

    def test_none_sharpe(self):
        """Sharpe 为 None 时不报错"""
        r = TimingWindowResult(
            start_minutes=60, stop_minutes=0,
            sharpe=None, profit_factor=2.0,
            return_pct=5.0, roi=0.2,
            max_drawdown_pct=2.0,
        )
        score = _compute_composite_score(r)
        assert score >= 0


class TestRunTimingGrid:
    def _make_test_observations(self):
        """构造多时间点的测试观测"""
        obs = []
        for date in ["2026-02-10", "2026-02-11", "2026-02-12"]:
            settlement = 99500.0
            for minutes in [720, 360, 180, 120, 60, 30, 10, 5, 1]:
                obs.append(_make_obs(
                    event_date=date,
                    obs_minutes=minutes,
                    settlement=settlement,
                    market_prices={
                        99000.0: 0.55,
                        100000.0: 0.45,
                        101000.0: 0.35,
                    },
                ))
        return obs

    def test_grid_produces_results(self):
        """网格实验产生结果"""
        obs = self._make_test_observations()
        results = run_timing_grid(
            obs,
            start_grid=[720, 60],
            stop_grid=[0, 10],
        )
        # 有效组合: (720,0), (720,10), (60,0), (60,10)
        assert len(results) == 4

    def test_invalid_combinations_skipped(self):
        """start <= stop 的组合被跳过"""
        obs = self._make_test_observations()
        results = run_timing_grid(
            obs,
            start_grid=[60],
            stop_grid=[60, 120],
        )
        assert len(results) == 0

    def test_results_have_scores(self):
        """每个结果都有综合评分"""
        obs = self._make_test_observations()
        results = run_timing_grid(
            obs,
            start_grid=[720],
            stop_grid=[0],
        )
        assert len(results) == 1
        assert isinstance(results[0].composite_score, float)


class TestFindOptimalWindows:
    def test_sort_by_score(self):
        """按评分排序"""
        results = [
            TimingWindowResult(start_minutes=60, stop_minutes=0, composite_score=0.3, n_trades=25),
            TimingWindowResult(start_minutes=360, stop_minutes=0, composite_score=0.8, n_trades=50),
            TimingWindowResult(start_minutes=180, stop_minutes=10, composite_score=0.5, n_trades=30),
        ]
        top = find_optimal_windows(results, top_n=2)
        assert len(top) == 2
        assert top[0].composite_score == 0.8
        assert top[1].composite_score == 0.5

    def test_skip_zero_trades(self):
        """过滤无交易的窗口"""
        results = [
            TimingWindowResult(start_minutes=60, stop_minutes=0, composite_score=0.9, n_trades=0),
            TimingWindowResult(start_minutes=360, stop_minutes=0, composite_score=0.5, n_trades=10),
        ]
        top = find_optimal_windows(results)
        assert len(top) == 1
        assert top[0].start_minutes == 360
