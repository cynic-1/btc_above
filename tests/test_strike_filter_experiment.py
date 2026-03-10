"""
Strike 过滤实验测试
"""

import csv

import pytest

from backtest.models import ObservationResult
from backtest.strike_filter_experiment import (
    StrikeFilterResult,
    _compute_result,
    filter_observations_by_strikes,
    run_strike_filter_experiment,
    select_nearest_strikes,
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
        strikes = [98000.0, 99000.0, 100000.0, 101000.0, 102000.0]
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


class TestSelectNearestStrikes:
    def test_basic_selection(self):
        """选择距 S0 最近的 2 个 strike"""
        obs = [
            _make_obs(obs_minutes=720, s0=100000.0),
            _make_obs(obs_minutes=60, s0=100100.0),
        ]
        result = select_nearest_strikes(obs, n=2)
        assert "2026-02-10" in result
        # S0=100000, strikes=[98000,99000,100000,101000,102000]
        # 距离: 2000, 1000, 0, 1000, 2000
        # 最近 2 个: 100000, 99000 或 101000（距离相同取先出现的）
        selected = result["2026-02-10"]
        assert len(selected) == 2
        assert 100000.0 in selected

    def test_uses_max_obs_minutes(self):
        """使用最大 obs_minutes 的观测的 S0"""
        obs = [
            _make_obs(obs_minutes=60, s0=99000.0),   # 不应使用
            _make_obs(obs_minutes=720, s0=101500.0),  # 应使用
        ]
        result = select_nearest_strikes(obs, n=2)
        selected = result["2026-02-10"]
        # S0=101500, 最近: 101000(500), 102000(500)
        assert 101000.0 in selected
        assert 102000.0 in selected

    def test_n_equals_1(self):
        """n=1 只选最近 1 个"""
        obs = [_make_obs(obs_minutes=720, s0=100000.0)]
        result = select_nearest_strikes(obs, n=1)
        assert len(result["2026-02-10"]) == 1
        assert result["2026-02-10"][0] == 100000.0

    def test_n_exceeds_available(self):
        """n 大于可用 strike 数时返回全部"""
        obs = [_make_obs(obs_minutes=720, s0=100000.0)]
        result = select_nearest_strikes(obs, n=10)
        assert len(result["2026-02-10"]) == 5  # 只有 5 个 strike

    def test_multiple_dates(self):
        """多个日期分别选择"""
        obs = [
            _make_obs(event_date="2026-02-10", obs_minutes=720, s0=100000.0),
            _make_obs(event_date="2026-02-11", obs_minutes=720, s0=98500.0),
        ]
        result = select_nearest_strikes(obs, n=1)
        assert result["2026-02-10"] == [100000.0]
        # S0=98500, 最近: 98000(500) vs 99000(500)，排序稳定取第一个
        assert len(result["2026-02-11"]) == 1


class TestFilterObservationsByStrikes:
    def test_basic_filter(self):
        """过滤后只保留选中的 strikes"""
        obs = [_make_obs(obs_minutes=60)]
        strike_map = {"2026-02-10": [100000.0, 101000.0]}

        filtered = filter_observations_by_strikes(obs, strike_map)
        assert len(filtered) == 1
        assert filtered[0].k_grid == [100000.0, 101000.0]
        assert set(filtered[0].predictions.keys()) == {100000.0, 101000.0}
        assert set(filtered[0].labels.keys()) == {100000.0, 101000.0}

    def test_preserves_all_time_points(self):
        """过滤仅限制 strike，不限制时间点"""
        obs = [
            _make_obs(obs_minutes=720),
            _make_obs(obs_minutes=360),
            _make_obs(obs_minutes=60),
        ]
        strike_map = {"2026-02-10": [100000.0]}

        filtered = filter_observations_by_strikes(obs, strike_map)
        assert len(filtered) == 3
        for o in filtered:
            assert o.k_grid == [100000.0]

    def test_missing_date_skipped(self):
        """strike_map 中没有的日期被跳过"""
        obs = [_make_obs(event_date="2026-02-12", obs_minutes=60)]
        strike_map = {"2026-02-10": [100000.0]}

        filtered = filter_observations_by_strikes(obs, strike_map)
        assert len(filtered) == 0

    def test_no_matching_strikes(self):
        """选中的 strike 不在 obs 的 k_grid 中时跳过"""
        obs = [_make_obs(obs_minutes=60, strikes=[90000.0, 91000.0])]
        strike_map = {"2026-02-10": [100000.0]}

        filtered = filter_observations_by_strikes(obs, strike_map)
        assert len(filtered) == 0

    def test_preserves_market_prices(self):
        """过滤后保留对应的 market_prices"""
        obs = [_make_obs(
            obs_minutes=60,
            market_prices={98000.0: 0.8, 99000.0: 0.6, 100000.0: 0.5,
                           101000.0: 0.4, 102000.0: 0.2},
        )]
        strike_map = {"2026-02-10": [99000.0, 100000.0]}

        filtered = filter_observations_by_strikes(obs, strike_map)
        assert filtered[0].market_prices == {99000.0: 0.6, 100000.0: 0.5}

    def test_original_unchanged(self):
        """过滤不修改原始观测"""
        obs = [_make_obs(obs_minutes=60)]
        original_k_grid = list(obs[0].k_grid)
        strike_map = {"2026-02-10": [100000.0]}

        filter_observations_by_strikes(obs, strike_map)
        assert obs[0].k_grid == original_k_grid


class TestComputeResult:
    def test_empty_observations(self):
        """空观测返回零值结果"""
        result = _compute_result(
            [], n_nearest=2,
            initial_capital=100000, shares_per_trade=200,
            max_net_shares=10000, entry_threshold=0.03,
        )
        assert result.n_nearest == 2
        assert result.total_pnl == 0.0
        assert result.n_trades == 0

    def test_result_has_correct_n_nearest(self):
        """n_nearest 正确设置"""
        obs = [_make_obs(
            obs_minutes=60,
            predictions={98000.0: 0.9, 99000.0: 0.7, 100000.0: 0.5,
                         101000.0: 0.3, 102000.0: 0.1},
            market_prices={98000.0: 0.5, 99000.0: 0.5, 100000.0: 0.5,
                           101000.0: 0.5, 102000.0: 0.5},
        )]
        result = _compute_result(
            obs, n_nearest=3,
            initial_capital=100000, shares_per_trade=200,
            max_net_shares=10000, entry_threshold=0.03,
        )
        assert result.n_nearest == 3


class TestRunStrikeFilterExperiment:
    def test_full_pipeline(self, tmp_path):
        """完整 pipeline: CSV -> 实验结果"""
        csv_path = tmp_path / "detail.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "event_date", "obs_minutes", "strike", "s0", "settlement",
                "p_physical", "label", "ci_lower", "ci_upper", "market_price",
            ])
            # 3 个 strikes, 2 个时间点
            for minutes in [720, 60]:
                for strike, pred, label, mp in [
                    (99000, 0.7, 1, 0.5),
                    (100000, 0.5, 0, 0.5),
                    (101000, 0.2, 0, 0.5),
                ]:
                    writer.writerow([
                        "2026-02-10", minutes, strike, 100000, 99500,
                        pred, label, "", "", mp,
                    ])

        results = run_strike_filter_experiment(
            csv_path=str(csv_path),
            n_nearest_list=[1, 2],
        )

        # baseline + 2 个配置 = 3 个结果
        assert len(results) == 3
        assert results[0].n_nearest == 0  # baseline
        assert results[1].n_nearest == 1
        assert results[2].n_nearest == 2

    def test_baseline_has_most_trades(self, tmp_path):
        """Baseline 应该有最多（或相同）的交易数"""
        csv_path = tmp_path / "detail.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "event_date", "obs_minutes", "strike", "s0", "settlement",
                "p_physical", "label", "ci_lower", "ci_upper", "market_price",
            ])
            for minutes in [720, 360, 60]:
                for strike, pred, mp in [
                    (98000, 0.9, 0.5),
                    (99000, 0.7, 0.5),
                    (100000, 0.5, 0.5),
                    (101000, 0.2, 0.5),
                ]:
                    label = 1 if 99500 > strike else 0
                    writer.writerow([
                        "2026-02-10", minutes, strike, 100000, 99500,
                        pred, label, "", "", mp,
                    ])

        results = run_strike_filter_experiment(
            csv_path=str(csv_path),
            n_nearest_list=[1, 2],
        )

        baseline_trades = results[0].n_trades
        for r in results[1:]:
            assert r.n_trades <= baseline_trades
