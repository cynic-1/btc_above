"""
周末效应实验测试
"""

import csv

import pytest

from backtest.models import ObservationResult
from backtest.weekend_experiment import (
    DayGroupResult,
    compute_group_result,
    filter_by_dates,
    is_weekend,
    run_weekend_experiment,
    weekday_index,
    weekday_name,
)


def _make_obs(
    event_date: str = "2026-02-21",
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


class TestIsWeekend:
    def test_saturday(self):
        # 2026-02-21 是周六
        assert is_weekend("2026-02-21") is True

    def test_sunday(self):
        # 2026-02-22 是周日
        assert is_weekend("2026-02-22") is True

    def test_monday(self):
        # 2026-02-23 是周一
        assert is_weekend("2026-02-23") is False

    def test_friday(self):
        # 2026-02-27 是周五
        assert is_weekend("2026-02-27") is False

    def test_wednesday(self):
        # 2026-02-25 是周三
        assert is_weekend("2026-02-25") is False


class TestWeekdayName:
    def test_saturday(self):
        assert weekday_name("2026-02-21") == "Saturday"

    def test_sunday(self):
        assert weekday_name("2026-02-22") == "Sunday"

    def test_monday(self):
        assert weekday_name("2026-02-23") == "Monday"

    def test_friday(self):
        assert weekday_name("2026-02-27") == "Friday"


class TestWeekdayIndex:
    def test_monday_is_0(self):
        assert weekday_index("2026-02-23") == 0

    def test_sunday_is_6(self):
        assert weekday_index("2026-02-22") == 6

    def test_saturday_is_5(self):
        assert weekday_index("2026-02-21") == 5


class TestFilterByDates:
    def test_basic_filter(self):
        obs = [
            _make_obs(event_date="2026-02-21"),
            _make_obs(event_date="2026-02-22"),
            _make_obs(event_date="2026-02-23"),
        ]
        result = filter_by_dates(obs, {"2026-02-21", "2026-02-22"})
        assert len(result) == 2
        assert all(o.event_date in {"2026-02-21", "2026-02-22"} for o in result)

    def test_empty_dates_set(self):
        obs = [_make_obs(event_date="2026-02-21")]
        result = filter_by_dates(obs, set())
        assert len(result) == 0

    def test_no_matching_dates(self):
        obs = [_make_obs(event_date="2026-02-21")]
        result = filter_by_dates(obs, {"2026-03-01"})
        assert len(result) == 0

    def test_multiple_obs_same_date(self):
        """同一日期多个观测都应保留"""
        obs = [
            _make_obs(event_date="2026-02-21", obs_minutes=720),
            _make_obs(event_date="2026-02-21", obs_minutes=60),
            _make_obs(event_date="2026-02-21", obs_minutes=10),
        ]
        result = filter_by_dates(obs, {"2026-02-21"})
        assert len(result) == 3


class TestComputeGroupResult:
    def test_empty_observations(self):
        result = compute_group_result([], "test")
        assert result.group_name == "test"
        assert result.n_event_dates == 0
        assert result.total_pnl == 0.0
        assert result.n_trades == 0
        assert result.brier is None

    def test_basic_result(self):
        obs = [_make_obs(
            event_date="2026-02-21",
            obs_minutes=60,
            predictions={98000.0: 0.9, 99000.0: 0.7, 100000.0: 0.5,
                         101000.0: 0.3, 102000.0: 0.1},
            market_prices={98000.0: 0.5, 99000.0: 0.5, 100000.0: 0.5,
                           101000.0: 0.5, 102000.0: 0.5},
        )]
        result = compute_group_result(obs, "weekend")
        assert result.group_name == "weekend"
        assert result.n_event_dates == 1
        assert result.event_dates == ["2026-02-21"]
        assert result.brier is not None
        assert result.market_price_coverage == 1.0

    def test_no_market_prices(self):
        """无市场价格时 coverage 应为 0"""
        obs = [_make_obs(
            event_date="2026-02-21",
            obs_minutes=60,
            market_prices={},
        )]
        result = compute_group_result(obs, "test")
        assert result.market_price_coverage == 0.0
        assert result.avg_abs_edge is None

    def test_with_spread_data(self):
        obs = [_make_obs(event_date="2026-02-21", obs_minutes=60)]
        spread_data = {
            ("2026-02-21", 60, 100000.0): (0.48, 0.52),
            ("2026-02-21", 60, 101000.0): (0.45, 0.55),
        }
        result = compute_group_result(obs, "test", spread_data=spread_data)
        assert result.avg_spread is not None
        assert result.avg_spread == pytest.approx(0.07)  # avg(0.04, 0.10) = 0.07

    def test_multiple_dates(self):
        obs = [
            _make_obs(event_date="2026-02-21", obs_minutes=60),
            _make_obs(event_date="2026-02-22", obs_minutes=60),
        ]
        result = compute_group_result(obs, "weekend")
        assert result.n_event_dates == 2
        assert "2026-02-21" in result.event_dates
        assert "2026-02-22" in result.event_dates


class TestRunWeekendExperiment:
    def _write_csv(self, path, rows):
        """写入测试 CSV"""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "event_date", "obs_minutes", "strike", "s0", "settlement",
                "p_physical", "label", "ci_lower", "ci_upper",
                "market_price", "market_bid", "market_ask",
            ])
            for row in rows:
                writer.writerow(row)

    def test_full_pipeline(self, tmp_path):
        """完整 pipeline: CSV -> 实验结果"""
        csv_path = tmp_path / "detail.csv"
        rows = []
        # 周六 2026-02-21
        for minutes in [720, 60]:
            for strike, pred, label, mp in [
                (99000, 0.7, 1, 0.5),
                (100000, 0.5, 0, 0.5),
            ]:
                rows.append([
                    "2026-02-21", minutes, strike, 100000, 99500,
                    pred, label, "", "", mp, "", "",
                ])
        # 周一 2026-02-23
        for minutes in [720, 60]:
            for strike, pred, label, mp in [
                (99000, 0.7, 1, 0.5),
                (100000, 0.5, 0, 0.5),
            ]:
                rows.append([
                    "2026-02-23", minutes, strike, 100000, 99500,
                    pred, label, "", "", mp, "", "",
                ])
        self._write_csv(csv_path, rows)

        weekend, weekday, per_day = run_weekend_experiment(csv_path=str(csv_path))

        assert weekend.group_name == "weekend"
        assert weekday.group_name == "weekday"
        assert weekend.n_event_dates == 1  # 周六
        assert weekday.n_event_dates == 1  # 周一
        assert len(per_day) == 7

    def test_per_day_has_correct_names(self, tmp_path):
        """逐日结果名称正确"""
        csv_path = tmp_path / "detail.csv"
        rows = []
        for strike, pred, label, mp in [(99000, 0.7, 1, 0.5)]:
            rows.append([
                "2026-02-21", 60, strike, 100000, 99500,
                pred, label, "", "", mp, "", "",
            ])
        self._write_csv(csv_path, rows)

        _, _, per_day = run_weekend_experiment(csv_path=str(csv_path))

        names = [r.group_name for r in per_day]
        assert names == [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday",
        ]

    def test_weekend_dates_classified_correctly(self, tmp_path):
        """周末日期正确分到 weekend 组"""
        csv_path = tmp_path / "detail.csv"
        rows = []
        # 周六 2/21, 周日 2/22, 周一 2/23, 周二 2/24
        for date in ["2026-02-21", "2026-02-22", "2026-02-23", "2026-02-24"]:
            rows.append([
                date, 60, 99000, 100000, 99500,
                0.7, 1, "", "", 0.5, "", "",
            ])
        self._write_csv(csv_path, rows)

        weekend, weekday, per_day = run_weekend_experiment(csv_path=str(csv_path))

        assert weekend.n_event_dates == 2  # 周六+周日
        assert weekday.n_event_dates == 2  # 周一+周二
        assert set(weekend.event_dates) == {"2026-02-21", "2026-02-22"}
        assert set(weekday.event_dates) == {"2026-02-23", "2026-02-24"}

    def test_spread_data_loaded(self, tmp_path):
        """CSV 中的 bid/ask 数据被正确加载"""
        csv_path = tmp_path / "detail.csv"
        rows = [
            # 有 bid/ask 数据
            ["2026-02-21", 60, 99000, 100000, 99500,
             0.7, 1, "", "", 0.5, 0.48, 0.52],
            # 无 bid/ask 数据
            ["2026-02-23", 60, 99000, 100000, 99500,
             0.7, 1, "", "", 0.5, "", ""],
        ]
        self._write_csv(csv_path, rows)

        weekend, weekday, _ = run_weekend_experiment(csv_path=str(csv_path))

        assert weekend.avg_spread is not None
        assert weekend.avg_spread == pytest.approx(0.04)  # 0.52 - 0.48
        assert weekday.avg_spread is None

    def test_empty_group(self, tmp_path):
        """只有周末数据时，weekday 组为空"""
        csv_path = tmp_path / "detail.csv"
        rows = [
            ["2026-02-21", 60, 99000, 100000, 99500,
             0.7, 1, "", "", 0.5, "", ""],
        ]
        self._write_csv(csv_path, rows)

        weekend, weekday, _ = run_weekend_experiment(csv_path=str(csv_path))

        assert weekend.n_event_dates == 1
        assert weekday.n_event_dates == 0
        assert weekday.total_pnl == 0.0
        assert weekday.n_trades == 0
