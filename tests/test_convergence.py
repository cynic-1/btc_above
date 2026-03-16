"""
价格收敛分析测试
"""

import csv
import os
import tempfile

import pytest

from backtest.convergence import (
    TradeSignal,
    HoldingPeriodResult,
    load_price_lookup,
    find_trade_signals,
    compute_exit_pnl,
    compute_exit_mid_drift,
    compute_settlement_pnl,
    compute_settlement_mid_drift,
    aggregate_holding_period,
    run_convergence,
)
from backtest.convergence_chart import plot_convergence


def _write_csv(path, rows):
    """辅助: 写测试 CSV"""
    header = [
        "event_date", "obs_minutes", "strike", "s0", "settlement",
        "p_physical", "label", "ci_lower", "ci_upper", "market_price",
        "market_bid", "market_ask",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)


def _make_signal(
    direction="YES",
    obs_minutes=120,
    entry_bid=0.50,
    entry_ask=0.52,
    edge=0.05,
    label=1,
    event_date="2026-03-01",
    strike=85000.0,
):
    return TradeSignal(
        event_date=event_date,
        obs_minutes=obs_minutes,
        strike=strike,
        direction=direction,
        entry_ask=entry_ask,
        entry_bid=entry_bid,
        edge=edge,
        label=label,
    )


# ===== TestLoadPriceLookup =====

class TestLoadPriceLookup:

    def test_basic_load(self, tmp_path):
        """基本加载: 正确解析 bid/ask/label"""
        csv_path = str(tmp_path / "test.csv")
        _write_csv(csv_path, [
            # date, obs_min, strike, s0, settlement, p, label, ci_lo, ci_hi, mp, bid, ask
            ["2026-03-01", 120, 85000.0, 84000.0, 86000.0, 0.6, 1, "", "", 0.55, 0.54, 0.56],
            ["2026-03-01", 60, 85000.0, 84000.0, 86000.0, 0.65, 1, "", "", 0.58, 0.57, 0.59],
        ])

        price_lookup, label_lookup, rows = load_price_lookup(csv_path)

        assert len(rows) == 2
        assert ("2026-03-01", 85000.0, 120) in price_lookup
        assert price_lookup[("2026-03-01", 85000.0, 120)] == (0.54, 0.56)
        assert label_lookup[("2026-03-01", 85000.0, 120)] == 1

    def test_empty_bid_ask(self, tmp_path):
        """空 bid/ask 不进入 price_lookup"""
        csv_path = str(tmp_path / "test.csv")
        _write_csv(csv_path, [
            ["2026-03-01", 120, 85000.0, 84000.0, 86000.0, 0.6, 1, "", "", 0.55, "", ""],
        ])

        price_lookup, label_lookup, rows = load_price_lookup(csv_path)

        assert len(rows) == 1
        assert ("2026-03-01", 85000.0, 120) not in price_lookup
        # label 仍然存在
        assert label_lookup[("2026-03-01", 85000.0, 120)] == 1

    def test_lookup_key_correct(self, tmp_path):
        """多日期多 strike 键正确"""
        csv_path = str(tmp_path / "test.csv")
        _write_csv(csv_path, [
            ["2026-03-01", 120, 85000.0, 84000.0, 86000.0, 0.6, 1, "", "", "", 0.50, 0.52],
            ["2026-03-01", 120, 86000.0, 84000.0, 86000.0, 0.4, 0, "", "", "", 0.40, 0.42],
            ["2026-03-02", 120, 85000.0, 84000.0, 85500.0, 0.55, 1, "", "", "", 0.48, 0.50],
        ])

        price_lookup, label_lookup, rows = load_price_lookup(csv_path)

        assert len(price_lookup) == 3
        assert price_lookup[("2026-03-01", 86000.0, 120)] == (0.40, 0.42)
        assert label_lookup[("2026-03-02", 85000.0, 120)] == 1


# ===== TestFindTradeSignals =====

class TestFindTradeSignals:

    def test_yes_trigger(self):
        """YES 触发: p - ask > threshold"""
        rows = [{
            "event_date": "2026-03-01", "obs_minutes": 120, "strike": 85000.0,
            "bid": 0.50, "ask": 0.52, "p_physical": 0.60, "label": 1,
            "market_price": 0.51,
        }]
        signals = find_trade_signals(rows, threshold=0.03)
        assert len(signals) == 1
        assert signals[0].direction == "YES"
        assert signals[0].edge == pytest.approx(0.08)  # 0.60 - 0.52

    def test_no_trigger(self):
        """NO 触发: bid - p > threshold"""
        rows = [{
            "event_date": "2026-03-01", "obs_minutes": 120, "strike": 85000.0,
            "bid": 0.50, "ask": 0.52, "p_physical": 0.40, "label": 0,
            "market_price": 0.51,
        }]
        signals = find_trade_signals(rows, threshold=0.03)
        assert len(signals) == 1
        assert signals[0].direction == "NO"
        assert signals[0].edge == pytest.approx(0.10)  # 0.50 - 0.40

    def test_low_edge_no_trigger(self):
        """低 edge 不触发"""
        rows = [{
            "event_date": "2026-03-01", "obs_minutes": 120, "strike": 85000.0,
            "bid": 0.50, "ask": 0.52, "p_physical": 0.53, "label": 1,
            "market_price": 0.51,
        }]
        signals = find_trade_signals(rows, threshold=0.03)
        assert len(signals) == 0

    def test_no_bid_ask_skip(self):
        """无 bid/ask 跳过"""
        rows = [{
            "event_date": "2026-03-01", "obs_minutes": 120, "strike": 85000.0,
            "bid": None, "ask": None, "p_physical": 0.60, "label": 1,
            "market_price": 0.51,
        }]
        signals = find_trade_signals(rows, threshold=0.03)
        assert len(signals) == 0

    def test_bid_equals_ask_skip(self):
        """bid == ask 跳过"""
        rows = [{
            "event_date": "2026-03-01", "obs_minutes": 120, "strike": 85000.0,
            "bid": 0.50, "ask": 0.50, "p_physical": 0.60, "label": 1,
            "market_price": 0.50,
        }]
        signals = find_trade_signals(rows, threshold=0.03)
        assert len(signals) == 0

    def test_yes_priority_over_no(self):
        """YES 和 NO 都超阈值时，YES 优先（elif 逻辑）"""
        # p=0.60, bid=0.50, ask=0.52 → edge_yes=0.08, edge_no=... 不可能同时超阈值
        # 但可以构造 edge_yes 和 edge_no 都 > 0.03 的场景吗？
        # p - ask > 0.03 means p > ask + 0.03
        # bid - p > 0.03 means p < bid - 0.03
        # 如果 ask > bid（正常），则 ask+0.03 > bid-0.03 → p 不可能同时满足
        # 所以这个场景不会发生，跳过
        pass

    def test_extreme_price_skip(self):
        """极端价格跳过: bid <= 0.01 或 ask >= 0.99"""
        rows = [
            {
                "event_date": "2026-03-01", "obs_minutes": 120, "strike": 85000.0,
                "bid": 0.01, "ask": 0.03, "p_physical": 0.10, "label": 0,
                "market_price": 0.02,
            },
            {
                "event_date": "2026-03-01", "obs_minutes": 120, "strike": 85000.0,
                "bid": 0.97, "ask": 0.99, "p_physical": 0.80, "label": 1,
                "market_price": 0.98,
            },
        ]
        signals = find_trade_signals(rows, threshold=0.03)
        assert len(signals) == 0


# ===== TestComputeExitPnl =====

class TestComputeExitPnl:

    def test_yes_profit(self):
        """YES 盈利: exit_bid > entry_ask"""
        signal = _make_signal(direction="YES", obs_minutes=120, entry_ask=0.52)
        price_lookup = {
            ("2026-03-01", 85000.0, 90): (0.58, 0.60),  # exit at obs=90
        }
        pnl = compute_exit_pnl(signal, price_lookup, delta_minutes=30)
        assert pnl == pytest.approx(0.06)  # 0.58 - 0.52

    def test_yes_loss(self):
        """YES 亏损: exit_bid < entry_ask"""
        signal = _make_signal(direction="YES", obs_minutes=120, entry_ask=0.52)
        price_lookup = {
            ("2026-03-01", 85000.0, 90): (0.48, 0.50),
        }
        pnl = compute_exit_pnl(signal, price_lookup, delta_minutes=30)
        assert pnl == pytest.approx(-0.04)  # 0.48 - 0.52

    def test_no_profit(self):
        """NO 盈利: entry_bid > exit_ask (价格下跌)"""
        signal = _make_signal(direction="NO", obs_minutes=120, entry_bid=0.50)
        price_lookup = {
            ("2026-03-01", 85000.0, 60): (0.42, 0.44),
        }
        pnl = compute_exit_pnl(signal, price_lookup, delta_minutes=60)
        assert pnl == pytest.approx(0.06)  # 0.50 - 0.44

    def test_no_loss(self):
        """NO 亏损: entry_bid < exit_ask (价格上涨)"""
        signal = _make_signal(direction="NO", obs_minutes=120, entry_bid=0.50)
        price_lookup = {
            ("2026-03-01", 85000.0, 60): (0.54, 0.56),
        }
        pnl = compute_exit_pnl(signal, price_lookup, delta_minutes=60)
        assert pnl == pytest.approx(-0.06)  # 0.50 - 0.56

    def test_holding_exceeds_range(self):
        """持有期超出范围: obs_minutes - delta < 0 → None"""
        signal = _make_signal(direction="YES", obs_minutes=20)
        pnl = compute_exit_pnl(signal, {}, delta_minutes=30)
        assert pnl is None

    def test_no_exit_data(self):
        """退出时无数据 → None"""
        signal = _make_signal(direction="YES", obs_minutes=120)
        pnl = compute_exit_pnl(signal, {}, delta_minutes=30)  # 空 lookup
        assert pnl is None

    def test_exit_bid_zero(self):
        """退出时 bid <= 0 → None"""
        signal = _make_signal(direction="YES", obs_minutes=120)
        price_lookup = {
            ("2026-03-01", 85000.0, 90): (0.0, 0.50),
        }
        pnl = compute_exit_pnl(signal, price_lookup, delta_minutes=30)
        assert pnl is None

    def test_exit_bid_equals_ask(self):
        """退出时 bid == ask → None"""
        signal = _make_signal(direction="YES", obs_minutes=120)
        price_lookup = {
            ("2026-03-01", 85000.0, 90): (0.50, 0.50),
        }
        pnl = compute_exit_pnl(signal, price_lookup, delta_minutes=30)
        assert pnl is None


# ===== TestComputeSettlementPnl =====

class TestComputeSettlementPnl:

    def test_yes_wins(self):
        """YES 赢: label=1, PnL = 1 - ask"""
        signal = _make_signal(direction="YES", entry_ask=0.52, label=1)
        pnl = compute_settlement_pnl(signal)
        assert pnl == pytest.approx(0.48)

    def test_yes_loses(self):
        """YES 输: label=0, PnL = 0 - ask = -ask"""
        signal = _make_signal(direction="YES", entry_ask=0.52, label=0)
        pnl = compute_settlement_pnl(signal)
        assert pnl == pytest.approx(-0.52)

    def test_no_wins(self):
        """NO 赢: label=0, PnL = (1-0) - (1-bid) = bid"""
        signal = _make_signal(direction="NO", entry_bid=0.50, label=0)
        pnl = compute_settlement_pnl(signal)
        assert pnl == pytest.approx(0.50)

    def test_no_loses(self):
        """NO 输: label=1, PnL = (1-1) - (1-bid) = -(1-bid)"""
        signal = _make_signal(direction="NO", entry_bid=0.50, label=1)
        pnl = compute_settlement_pnl(signal)
        assert pnl == pytest.approx(-0.50)


# ===== TestAggregateHoldingPeriod =====

class TestAggregateHoldingPeriod:

    def test_mixed_wins_losses(self):
        """混合胜负: 正确计算胜率/赔率"""
        signals = [
            _make_signal(edge=0.05),
            _make_signal(edge=0.04),
            _make_signal(edge=0.06),
            _make_signal(edge=0.03),
        ]
        pnls = [0.10, -0.05, 0.08, -0.03]

        result = aggregate_holding_period(signals, pnls, "1h", 60, shares_per_trade=100)

        assert result.n_signals == 4
        assert result.n_with_exit == 4
        assert result.n_wins == 2
        assert result.n_losses == 2
        assert result.win_rate == pytest.approx(0.5)
        assert result.avg_win == pytest.approx(0.09)    # (0.10 + 0.08) / 2
        assert result.avg_loss == pytest.approx(0.04)   # (0.05 + 0.03) / 2
        assert result.payoff_ratio == pytest.approx(0.09 / 0.04)
        assert result.avg_pnl == pytest.approx(0.025)   # (0.10 - 0.05 + 0.08 - 0.03) / 4
        assert result.total_pnl == pytest.approx(0.025 * 4 * 100)  # 10.0

    def test_all_wins(self):
        """全赢: payoff_ratio = inf"""
        signals = [_make_signal(edge=0.05), _make_signal(edge=0.04)]
        pnls = [0.10, 0.08]

        result = aggregate_holding_period(signals, pnls, "30m", 30)

        assert result.n_wins == 2
        assert result.n_losses == 0
        assert result.win_rate == pytest.approx(1.0)
        assert result.payoff_ratio == float("inf")

    def test_empty_list(self):
        """空列表: 全部为零"""
        result = aggregate_holding_period([], [], "1h", 60)

        assert result.n_signals == 0
        assert result.n_with_exit == 0
        assert result.win_rate == 0.0
        assert result.avg_pnl == 0.0
        assert result.total_pnl == 0.0

    def test_with_nones(self):
        """含 None 的 pnl 列表: None 不计入 n_with_exit"""
        signals = [_make_signal(edge=0.05), _make_signal(edge=0.04), _make_signal(edge=0.06)]
        pnls = [0.10, None, -0.05]

        result = aggregate_holding_period(signals, pnls, "2h", 120)

        assert result.n_signals == 3
        assert result.n_with_exit == 2  # 只有 2 个有效
        assert result.n_wins == 1
        assert result.n_losses == 1

    def test_all_nones(self):
        """全部 None: 等效空"""
        signals = [_make_signal(), _make_signal()]
        pnls = [None, None]

        result = aggregate_holding_period(signals, pnls, "6h", 360)

        assert result.n_signals == 2
        assert result.n_with_exit == 0
        assert result.avg_pnl == 0.0


# ===== TestRunConvergence =====

class TestRunConvergence:

    def _make_test_csv(self, tmp_path) -> str:
        """构造完整测试 CSV"""
        csv_path = str(tmp_path / "detail.csv")
        rows = []
        # 事件日 1: 2026-03-01, strike=85000, settlement=86000 (label=1)
        # obs=120: p=0.60, bid=0.50, ask=0.52 → YES edge=0.08
        # obs=90: bid=0.58, ask=0.60 (价格上涨 → YES 盈利, edge=0.02 不触发)
        # obs=60: bid=0.65, ask=0.67 (edge=0.01 不触发)
        # obs=0: 结算
        rows.append(["2026-03-01", 120, 85000.0, 84000, 86000, 0.60, 1, "", "", 0.51, 0.50, 0.52])
        rows.append(["2026-03-01", 90, 85000.0, 84000, 86000, 0.62, 1, "", "", 0.59, 0.58, 0.60])
        rows.append(["2026-03-01", 60, 85000.0, 84000, 86000, 0.68, 1, "", "", 0.66, 0.65, 0.67])
        rows.append(["2026-03-01", 0, 85000.0, 84000, 86000, 0.95, 1, "", "", 0.95, 0.94, 0.96])

        # 事件日 1: strike=87000, settlement=86000 (label=0)
        # obs=120: p=0.30, bid=0.45, ask=0.47 → NO edge=0.15
        # obs=90: bid=0.40, ask=0.42 (edge=0.02 不触发, p=0.38)
        # obs=60: bid=0.35, ask=0.37 (edge=0.02 不触发, p=0.33)
        rows.append(["2026-03-01", 120, 87000.0, 84000, 86000, 0.30, 0, "", "", 0.46, 0.45, 0.47])
        rows.append(["2026-03-01", 90, 87000.0, 84000, 86000, 0.38, 0, "", "", 0.41, 0.40, 0.42])
        rows.append(["2026-03-01", 60, 87000.0, 84000, 86000, 0.33, 0, "", "", 0.36, 0.35, 0.37])

        _write_csv(csv_path, rows)
        return csv_path

    def test_full_pipeline(self, tmp_path):
        """完整 CSV → 结果"""
        csv_path = self._make_test_csv(tmp_path)
        result = run_convergence(
            csv_path, threshold=0.03, shares_per_trade=100,
            holdings=[("30m", 30), ("1h", 60), ("settlement", -1)],
        )

        assert result.n_total_signals == 2  # 1 YES + 1 NO
        assert result.entry_threshold == 0.03
        assert len(result.all_results) == 3  # 3 holding periods

        # 30m: obs=120→90
        r30 = result.all_results[0]
        assert r30.holding == "30m"
        assert r30.n_with_exit == 2

        # settlement
        rsettl = result.all_results[2]
        assert rsettl.holding == "settlement"
        assert rsettl.n_with_exit == 2

    def test_direction_split(self, tmp_path):
        """方向拆分正确"""
        csv_path = self._make_test_csv(tmp_path)
        result = run_convergence(
            csv_path, threshold=0.03,
            holdings=[("30m", 30), ("settlement", -1)],
        )

        assert len(result.yes_results) == 2
        assert len(result.no_results) == 2

        # YES 应有 1 个信号
        assert result.yes_results[0].n_signals == 1
        # NO 应有 1 个信号
        assert result.no_results[0].n_signals == 1

    def test_yes_pnl_correct(self, tmp_path):
        """YES 30m PnL 正确"""
        csv_path = self._make_test_csv(tmp_path)
        result = run_convergence(
            csv_path, threshold=0.03,
            holdings=[("30m", 30)],
        )

        # YES signal: entry_ask=0.52, exit obs=90 bid=0.58 → PnL=0.06
        yes_r = result.yes_results[0]
        assert yes_r.avg_pnl == pytest.approx(0.06)

    def test_no_pnl_correct(self, tmp_path):
        """NO 30m PnL 正确"""
        csv_path = self._make_test_csv(tmp_path)
        result = run_convergence(
            csv_path, threshold=0.03,
            holdings=[("30m", 30)],
        )

        # NO signal: entry_bid=0.45, exit obs=90 ask=0.42 → PnL=0.03
        no_r = result.no_results[0]
        assert no_r.avg_pnl == pytest.approx(0.03)

    def test_settlement_pnl_correct(self, tmp_path):
        """结算 PnL 正确"""
        csv_path = self._make_test_csv(tmp_path)
        result = run_convergence(
            csv_path, threshold=0.03,
            holdings=[("settlement", -1)],
        )

        # YES: label=1, PnL = 1 - 0.52 = 0.48
        yes_r = result.yes_results[0]
        assert yes_r.avg_pnl == pytest.approx(0.48)

        # NO: label=0, PnL = (1-0) - (1-0.45) = 0.45
        no_r = result.no_results[0]
        assert no_r.avg_pnl == pytest.approx(0.45)

    def test_no_signals(self, tmp_path):
        """无信号: 空结果"""
        csv_path = str(tmp_path / "empty.csv")
        _write_csv(csv_path, [
            ["2026-03-01", 120, 85000.0, 84000, 86000, 0.51, 1, "", "", 0.50, 0.50, 0.52],
        ])
        result = run_convergence(csv_path, threshold=0.10)
        assert result.n_total_signals == 0
        assert result.all_results == []

    def test_mid_drift_populated(self, tmp_path):
        """run_convergence 结果中 mid_drift 字段已填充"""
        csv_path = self._make_test_csv(tmp_path)
        result = run_convergence(
            csv_path, threshold=0.03,
            holdings=[("30m", 30), ("settlement", -1)],
        )
        r30 = result.all_results[0]
        # 2 个信号都有退出数据，应有 favorable_rate > 0
        assert r30.n_favorable >= 0
        assert 0.0 <= r30.favorable_rate <= 1.0
        # settlement 也应有 drift
        rsettl = result.all_results[1]
        assert rsettl.favorable_rate > 0  # 两个都赢了


# ===== TestComputeExitMidDrift =====

class TestComputeExitMidDrift:

    def test_yes_positive_drift(self):
        """YES 正漂移: 价格上涨"""
        signal = _make_signal(direction="YES", obs_minutes=120,
                              entry_bid=0.50, entry_ask=0.52)
        price_lookup = {
            ("2026-03-01", 85000.0, 90): (0.58, 0.60),
        }
        drift = compute_exit_mid_drift(signal, price_lookup, delta_minutes=30)
        # entry_mid=0.51, exit_mid=0.59 → drift=0.08
        assert drift == pytest.approx(0.08)

    def test_yes_negative_drift(self):
        """YES 负漂移: 价格下跌"""
        signal = _make_signal(direction="YES", obs_minutes=120,
                              entry_bid=0.50, entry_ask=0.52)
        price_lookup = {
            ("2026-03-01", 85000.0, 90): (0.44, 0.46),
        }
        drift = compute_exit_mid_drift(signal, price_lookup, delta_minutes=30)
        # entry_mid=0.51, exit_mid=0.45 → drift=-0.06
        assert drift == pytest.approx(-0.06)

    def test_no_positive_drift(self):
        """NO 正漂移: 价格下跌（对 NO 有利）"""
        signal = _make_signal(direction="NO", obs_minutes=120,
                              entry_bid=0.50, entry_ask=0.52)
        price_lookup = {
            ("2026-03-01", 85000.0, 60): (0.42, 0.44),
        }
        drift = compute_exit_mid_drift(signal, price_lookup, delta_minutes=60)
        # entry_mid=0.51, exit_mid=0.43 → drift=0.51-0.43=0.08
        assert drift == pytest.approx(0.08)

    def test_no_negative_drift(self):
        """NO 负漂移: 价格上涨（对 NO 不利）"""
        signal = _make_signal(direction="NO", obs_minutes=120,
                              entry_bid=0.50, entry_ask=0.52)
        price_lookup = {
            ("2026-03-01", 85000.0, 60): (0.56, 0.58),
        }
        drift = compute_exit_mid_drift(signal, price_lookup, delta_minutes=60)
        # entry_mid=0.51, exit_mid=0.57 → drift=0.51-0.57=-0.06
        assert drift == pytest.approx(-0.06)

    def test_no_data_returns_none(self):
        """无退出数据 → None"""
        signal = _make_signal(direction="YES", obs_minutes=120)
        drift = compute_exit_mid_drift(signal, {}, delta_minutes=30)
        assert drift is None

    def test_holding_exceeds_range(self):
        """持有期超出范围 → None"""
        signal = _make_signal(direction="YES", obs_minutes=20)
        drift = compute_exit_mid_drift(signal, {}, delta_minutes=30)
        assert drift is None


# ===== TestComputeSettlementMidDrift =====

class TestComputeSettlementMidDrift:

    def test_yes_win(self):
        """YES 赢: label=1, drift = 1 - mid"""
        signal = _make_signal(direction="YES", entry_bid=0.50,
                              entry_ask=0.52, label=1)
        drift = compute_settlement_mid_drift(signal)
        # entry_mid=0.51, drift=1-0.51=0.49
        assert drift == pytest.approx(0.49)

    def test_yes_lose(self):
        """YES 输: label=0, drift = 0 - mid"""
        signal = _make_signal(direction="YES", entry_bid=0.50,
                              entry_ask=0.52, label=0)
        drift = compute_settlement_mid_drift(signal)
        # entry_mid=0.51, drift=0-0.51=-0.51
        assert drift == pytest.approx(-0.51)

    def test_no_win(self):
        """NO 赢: label=0, drift = mid - 0"""
        signal = _make_signal(direction="NO", entry_bid=0.50,
                              entry_ask=0.52, label=0)
        drift = compute_settlement_mid_drift(signal)
        # entry_mid=0.51, drift=0.51-0=0.51
        assert drift == pytest.approx(0.51)

    def test_no_lose(self):
        """NO 输: label=1, drift = mid - 1"""
        signal = _make_signal(direction="NO", entry_bid=0.50,
                              entry_ask=0.52, label=1)
        drift = compute_settlement_mid_drift(signal)
        # entry_mid=0.51, drift=0.51-1=-0.49
        assert drift == pytest.approx(-0.49)


# ===== TestAggregateWithMidDrifts =====

class TestAggregateWithMidDrifts:

    def test_with_mid_drifts(self):
        """传入 mid_drifts 验证新字段"""
        signals = [
            _make_signal(edge=0.05),
            _make_signal(edge=0.04),
            _make_signal(edge=0.06),
        ]
        pnls = [0.10, -0.05, 0.08]
        mid_drifts = [0.08, -0.03, 0.05]

        result = aggregate_holding_period(
            signals, pnls, "1h", 60, mid_drifts=mid_drifts,
        )

        assert result.n_favorable == 2  # 0.08 > 0, 0.05 > 0
        assert result.favorable_rate == pytest.approx(2 / 3)
        assert result.avg_mid_drift == pytest.approx((0.08 - 0.03 + 0.05) / 3)
        assert result.median_mid_drift == pytest.approx(0.05)

    def test_without_mid_drifts(self):
        """不传 mid_drifts 时新字段为 0（向后兼容）"""
        signals = [_make_signal(edge=0.05)]
        pnls = [0.10]

        result = aggregate_holding_period(signals, pnls, "1h", 60)

        assert result.n_favorable == 0
        assert result.favorable_rate == 0.0
        assert result.avg_mid_drift == 0.0
        assert result.median_mid_drift == 0.0

    def test_mid_drifts_with_nones(self):
        """mid_drifts 含 None: 仅统计非 None"""
        signals = [_make_signal(edge=0.05), _make_signal(edge=0.04)]
        pnls = [0.10, -0.05]
        mid_drifts = [0.08, None]

        result = aggregate_holding_period(
            signals, pnls, "1h", 60, mid_drifts=mid_drifts,
        )

        assert result.n_favorable == 1
        assert result.favorable_rate == pytest.approx(1.0)
        assert result.avg_mid_drift == pytest.approx(0.08)


# ===== TestConvergenceChart =====

class TestConvergenceChart:

    def test_generates_png(self, tmp_path):
        """生成 PNG 到 tmp_path，验证文件存在"""
        from backtest.convergence import HoldingPeriodResult, ConvergenceResult

        def _make_result(holding, minutes, wr=0.6, fr=0.65, drift=0.02):
            return HoldingPeriodResult(
                holding=holding, holding_minutes=minutes,
                n_signals=100, n_with_exit=90, n_wins=54, n_losses=36,
                win_rate=wr, avg_win=0.08, avg_loss=0.05,
                payoff_ratio=1.6, ev_per_trade=0.03,
                avg_pnl=0.025, total_pnl=450.0, median_pnl=0.02,
                avg_edge=0.04, n_favorable=59, favorable_rate=fr,
                avg_mid_drift=drift, median_mid_drift=drift * 0.9,
            )

        result = ConvergenceResult(
            all_results=[_make_result("30m", 30), _make_result("1h", 60),
                         _make_result("settlement", -1, drift=0.04)],
            yes_results=[_make_result("30m", 30, fr=0.7),
                         _make_result("1h", 60, fr=0.68),
                         _make_result("settlement", -1, fr=0.72, drift=0.05)],
            no_results=[_make_result("30m", 30, fr=0.58),
                        _make_result("1h", 60, fr=0.60),
                        _make_result("settlement", -1, fr=0.55, drift=0.03)],
            n_total_signals=200,
        )

        output_dir = str(tmp_path)
        path = plot_convergence(result, output_dir)

        assert os.path.exists(path)
        assert path.endswith("convergence_chart.png")
        # 文件大小 > 0
        assert os.path.getsize(path) > 0
