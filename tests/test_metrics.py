"""
回测指标模块测试
"""

import numpy as np
import pytest

from backtest.metrics import (
    brier_score,
    calibration_curve,
    compute_all_metrics,
    compute_annualized_sharpe,
    compute_annualized_sortino,
    compute_auc,
    compute_calmar,
    compute_direction_analysis,
    compute_drawdown_details,
    compute_ece,
    compute_edge_quintile_stats,
    compute_opinion_fee,
    compute_price_range_stats,
    compute_risk_metrics,
    compute_sharpe,
    compute_sortino,
    log_loss,
    run_adversarial_tests,
    run_capacity_analysis,
    run_cost_sensitivity,
    run_latency_sensitivity,
    simulate_pnl,
    simulate_portfolio,
)
from backtest.models import ObservationResult


class TestBrierScore:
    def test_perfect_predictions(self):
        preds = np.array([1.0, 0.0, 1.0, 0.0])
        labels = np.array([1, 0, 1, 0])
        assert brier_score(preds, labels) == pytest.approx(0.0)

    def test_worst_predictions(self):
        preds = np.array([0.0, 1.0])
        labels = np.array([1, 0])
        assert brier_score(preds, labels) == pytest.approx(1.0)

    def test_coin_flip(self):
        preds = np.array([0.5, 0.5, 0.5, 0.5])
        labels = np.array([1, 0, 1, 0])
        assert brier_score(preds, labels) == pytest.approx(0.25)


class TestLogLoss:
    def test_near_perfect(self):
        preds = np.array([0.999, 0.001])
        labels = np.array([1, 0])
        assert log_loss(preds, labels) < 0.01

    def test_coin_flip(self):
        preds = np.array([0.5, 0.5])
        labels = np.array([1, 0])
        expected = -np.log(0.5)
        assert log_loss(preds, labels) == pytest.approx(expected)

    def test_no_nan_on_extreme(self):
        """概率截断防止 log(0)"""
        preds = np.array([0.0, 1.0])
        labels = np.array([1, 0])
        result = log_loss(preds, labels)
        assert np.isfinite(result)


class TestCalibrationCurve:
    def test_basic_shape(self):
        preds = np.random.uniform(0, 1, 100)
        labels = np.random.randint(0, 2, 100)
        centers, freq, counts = calibration_curve(preds, labels, n_bins=5)
        assert len(centers) == 5
        assert len(freq) == 5
        assert len(counts) == 5
        assert counts.sum() == 100

    def test_perfect_calibration(self):
        """完美校准：预测值 = 实际频率"""
        preds = np.array([0.1] * 100 + [0.9] * 100)
        labels = np.array([0] * 90 + [1] * 10 + [1] * 90 + [0] * 10)
        centers, freq, counts = calibration_curve(preds, labels, n_bins=10)
        # bin [0.1, 0.2) → center=0.15, index=1
        low_bin_idx = 1
        # bin [0.9, 1.0] → center=0.95, index=9
        high_bin_idx = 9
        assert freq[low_bin_idx] == pytest.approx(0.1, abs=0.05)
        assert freq[high_bin_idx] == pytest.approx(0.9, abs=0.05)


class TestOpinionFee:
    def test_at_half(self):
        # 0.06 * 0.5 * 0.5 + 0.0025 = 0.015 + 0.0025 = 0.0175
        assert compute_opinion_fee(0.5) == pytest.approx(0.0175)

    def test_at_extreme(self):
        # 0.06 * 0.1 * 0.9 + 0.0025 = 0.0054 + 0.0025 = 0.0079
        assert compute_opinion_fee(0.1) == pytest.approx(0.0079)


class TestSimulatePnl:
    def _make_obs(self, p_phys, label, strike=90000.0):
        return ObservationResult(
            event_date="2026-02-01",
            obs_minutes=60,
            now_utc_ms=1000,
            s0=89000.0,
            settlement_price=91000.0,
            k_grid=[strike],
            predictions={strike: p_phys},
            labels={strike: label},
        )

    def test_no_trades_below_threshold(self):
        """edge 不够大则不交易"""
        # shrinkage_lambda=1.0: p_trade = p_phys = 0.52
        # edge = 0.52 - 0.5 = 0.02, < threshold 0.03
        obs = self._make_obs(0.52, 1)
        result = simulate_pnl([obs])
        assert result["n_trades"] == 0

    def test_winning_trade(self):
        """高概率 + 正确标签 → 盈利"""
        # p_trade = 0.9, edge = 0.9 - 0.5 = 0.4, fee=0
        obs = self._make_obs(0.9, 1)
        result = simulate_pnl([obs])
        assert result["n_trades"] > 0
        assert result["total_pnl"] > 0

    def test_losing_trade(self):
        """高概率 + 错误标签 → 亏损"""
        # p_trade = 0.9, edge = 0.4, BUY YES but label=0
        obs = self._make_obs(0.9, 0)
        result = simulate_pnl([obs])
        assert result["n_trades"] > 0
        assert result["total_pnl"] < 0


class TestComputeAllMetrics:
    def test_basic_flow(self):
        obs = ObservationResult(
            event_date="2026-02-01",
            obs_minutes=60,
            now_utc_ms=1000,
            s0=89000.0,
            settlement_price=91000.0,
            k_grid=[88000.0, 90000.0],
            predictions={88000.0: 0.8, 90000.0: 0.55},
            labels={88000.0: 1, 90000.0: 1},
        )
        metrics = compute_all_metrics([obs])
        assert "overall" in metrics
        assert "by_time_bucket" in metrics
        assert "T-1h~10m" in metrics["by_time_bucket"]
        assert "brier_score" in metrics["overall"]
        assert "portfolio" in metrics["overall"]


class TestSimulatePortfolio:
    def _make_obs(self, p_phys, label, strike=90000.0, market_price=0.5,
                  obs_minutes=60, now_utc_ms=1000, spread=0.02):
        bid = market_price - spread / 2
        ask = market_price + spread / 2
        return ObservationResult(
            event_date="2026-02-15",
            obs_minutes=obs_minutes,
            now_utc_ms=now_utc_ms,
            s0=89000.0,
            settlement_price=91000.0 if label == 1 else 89000.0,
            k_grid=[strike],
            predictions={strike: p_phys},
            labels={strike: label},
            market_prices={strike: market_price},
            market_bid_ask={strike: (bid, ask)},
        )

    def test_no_trade_below_threshold(self):
        """edge 不够不开仓"""
        obs = self._make_obs(0.52, 1, market_price=0.50)
        result = simulate_portfolio([obs], entry_threshold=0.03)
        assert result["n_trades"] == 0
        assert result["total_pnl"] == 0.0

    def test_buy_yes_wins(self):
        """模型高于市场 → BUY YES, 结算 YES → 盈利"""
        # model=0.80, bid=0.49, ask=0.51, edge=0.80-0.51=0.29 > 0.03
        # 买 200 份 YES @ ask=$0.51, 成本=$102, 结算 YES → 赔付 $200, pnl=$98
        obs = self._make_obs(0.80, 1, market_price=0.50)
        result = simulate_portfolio([obs], shares_per_trade=200)
        assert result["n_trades"] == 1
        assert result["n_markets"] == 1
        assert result["total_pnl"] == pytest.approx(98.0)

    def test_buy_no_wins(self):
        """市场高于模型 → BUY NO, 结算 NO → 盈利"""
        # model=0.20, bid=0.49, ask=0.51, edge=bid-model=0.49-0.20=0.29 > 0.03
        # 买 200 份 NO @ (1-bid)=$0.51, 成本=$102, 结算 NO → 赔付 $200, pnl=$98
        obs = self._make_obs(0.20, 0, market_price=0.50)
        result = simulate_portfolio([obs], shares_per_trade=200)
        assert result["n_trades"] == 1
        assert result["total_pnl"] == pytest.approx(98.0)

    def test_net_position_limit(self):
        """多个同方向观测，验证净仓位不超限"""
        # max_net_shares=400, shares_per_trade=200 → 最多 2 次买入
        obs_list = []
        for i in range(5):
            obs_list.append(self._make_obs(
                0.80, 1, market_price=0.50,
                obs_minutes=60 - i,
                now_utc_ms=1000 + i,
            ))
        result = simulate_portfolio(
            obs_list,
            shares_per_trade=200,
            max_net_shares=400,
        )
        # 只能买 2 次（200+200=400 = max_net_shares）
        assert result["n_trades"] == 2
        assert result["markets"][0]["yes_shares"] == 400

    def test_skip_when_bid_equals_ask(self):
        """bid==ask（无真实价差）时应跳过交易"""
        # market_bid_ask = (0.50, 0.50) → bid==ask → skip
        obs = ObservationResult(
            event_date="2026-02-15",
            obs_minutes=60,
            now_utc_ms=1000,
            s0=89000.0,
            settlement_price=91000.0,
            k_grid=[90000.0],
            predictions={90000.0: 0.80},
            labels={90000.0: 1},
            market_prices={90000.0: 0.50},
            market_bid_ask={90000.0: (0.50, 0.50)},
        )
        result = simulate_portfolio([obs], shares_per_trade=200)
        assert result["n_trades"] == 0

    def test_skip_when_no_bid_ask(self):
        """无 market_bid_ask 时应跳过交易"""
        obs = ObservationResult(
            event_date="2026-02-15",
            obs_minutes=60,
            now_utc_ms=1000,
            s0=89000.0,
            settlement_price=91000.0,
            k_grid=[90000.0],
            predictions={90000.0: 0.80},
            labels={90000.0: 1},
            market_prices={90000.0: 0.50},
            market_bid_ask={},
        )
        result = simulate_portfolio([obs], shares_per_trade=200)
        assert result["n_trades"] == 0

    def test_market_summary(self):
        """验证返回 markets 结构字段完整"""
        obs = self._make_obs(0.80, 1, market_price=0.50)
        result = simulate_portfolio([obs], shares_per_trade=200)

        assert "markets" in result
        assert len(result["markets"]) == 1
        mkt = result["markets"][0]
        assert mkt["event_date"] == "2026-02-15"
        assert mkt["strike"] == 90000.0
        assert mkt["settlement"] == "YES"
        assert mkt["yes_shares"] == 200
        assert mkt["yes_avg_price"] == pytest.approx(0.51)  # ask = 0.50 + 0.01
        assert mkt["no_shares"] == 0
        assert mkt["pnl"] == pytest.approx(98.0)  # 200 * (1 - 0.51)
        assert len(mkt["trades"]) == 1

        # 顶级字段
        assert "equity_curve" in result
        assert "event_pnls" in result
        assert "total_cost" in result
        assert "profit_factor" in result


# ===================================================================
# Unit 1: 风险指标
# ===================================================================

class TestComputeSharpe:
    def test_basic(self):
        pnls = [{"pnl": 100}, {"pnl": 200}, {"pnl": -50}, {"pnl": 150}]
        result = compute_sharpe(pnls, 100_000.0)
        assert result is not None
        assert isinstance(result, float)

    def test_not_annualized(self):
        """per-event Sharpe 不应乘 sqrt(365)"""
        pnls = [{"pnl": 100}, {"pnl": 200}, {"pnl": -50}, {"pnl": 150}]
        result = compute_sharpe(pnls, 100_000.0)
        # per-event Sharpe 量级应远小于年化 (< 10)
        assert abs(result) < 10

    def test_insufficient_data(self):
        assert compute_sharpe([{"pnl": 100}], 100_000.0) is None

    def test_zero_std(self):
        pnls = [{"pnl": 100}, {"pnl": 100}, {"pnl": 100}]
        assert compute_sharpe(pnls, 100_000.0) is None


class TestAnnualizedSharpe:
    def test_insufficient_periods(self):
        """n < 60 → None"""
        pnls = [{"pnl": 100 + i} for i in range(20)]
        assert compute_annualized_sharpe(pnls, 100_000.0) is None

    def test_sufficient_periods(self):
        """n >= 60 → 有值"""
        pnls = [{"pnl": 100 + (i % 5) * 10 - 20} for i in range(65)]
        result = compute_annualized_sharpe(pnls, 100_000.0)
        assert result is not None


class TestAnnualizedSortino:
    def test_insufficient_periods(self):
        pnls = [{"pnl": 100}, {"pnl": -50}]
        assert compute_annualized_sortino(pnls, 100_000.0) is None


class TestComputeSortino:
    def test_basic(self):
        pnls = [{"pnl": 100}, {"pnl": -50}, {"pnl": 200}, {"pnl": -30}]
        result = compute_sortino(pnls, 100_000.0)
        assert result is not None
        assert isinstance(result, float)

    def test_no_downside(self):
        """全部盈利无下行偏差"""
        pnls = [{"pnl": 100}, {"pnl": 200}, {"pnl": 300}]
        assert compute_sortino(pnls, 100_000.0) is None


class TestComputeCalmar:
    def test_basic(self):
        result = compute_calmar(10.0, 5.0, 30)
        assert result is not None
        assert result > 0

    def test_zero_drawdown(self):
        assert compute_calmar(10.0, 0.0, 30) is None

    def test_zero_days(self):
        assert compute_calmar(10.0, 5.0, 0) is None


class TestComputeRiskMetrics:
    def test_returns_dict(self):
        portfolio = {
            "event_pnls": [{"pnl": 100}, {"pnl": -50}, {"pnl": 200}],
            "initial_capital": 100_000.0,
            "total_return_pct": 0.25,
            "max_drawdown_pct": 0.05,
        }
        result = compute_risk_metrics(portfolio)
        assert "sharpe" in result
        assert "sortino" in result
        assert "calmar" in result


# ===================================================================
# Unit 2: ECE / AUC
# ===================================================================

class TestComputeECE:
    def test_perfect_calibration(self):
        """完美校准 ECE 应接近 0"""
        preds = np.array([0.1] * 50 + [0.9] * 50)
        labels = np.array([0] * 45 + [1] * 5 + [1] * 45 + [0] * 5)
        ece = compute_ece(preds, labels)
        assert ece < 0.05

    def test_bad_calibration(self):
        """全部预测 0.5 但全是 1 → ECE 大"""
        preds = np.array([0.5] * 100)
        labels = np.array([1] * 100)
        ece = compute_ece(preds, labels)
        assert ece > 0.3

    def test_empty(self):
        assert compute_ece(np.array([]), np.array([])) == 0.0


class TestComputeAUC:
    def test_perfect_separation(self):
        preds = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
        labels = np.array([0, 0, 0, 1, 1, 1])
        result = compute_auc(preds, labels)
        if result is None:
            pytest.skip("sklearn 不可用")
        assert result == pytest.approx(1.0)

    def test_single_class(self):
        preds = np.array([0.5, 0.6])
        labels = np.array([1, 1])
        result = compute_auc(preds, labels)
        assert result is None


# ===================================================================
# Unit 3: Edge 五分位
# ===================================================================

class TestEdgeQuintile:
    def test_empty_portfolio(self):
        result = compute_edge_quintile_stats({"markets": []})
        assert result == []

    def test_with_trades(self):
        portfolio = {
            "markets": [
                {
                    "settlement": "YES",
                    "trades": [
                        {"direction": "YES", "shares": 100, "model_price": 0.8, "market_price": 0.5, "cost": 50.0},
                        {"direction": "YES", "shares": 100, "model_price": 0.7, "market_price": 0.5, "cost": 50.0},
                        {"direction": "YES", "shares": 100, "model_price": 0.6, "market_price": 0.5, "cost": 50.0},
                        {"direction": "NO", "shares": 100, "model_price": 0.3, "market_price": 0.5, "cost": 50.0},
                        {"direction": "NO", "shares": 100, "model_price": 0.2, "market_price": 0.5, "cost": 50.0},
                    ],
                }
            ]
        }
        result = compute_edge_quintile_stats(portfolio)
        assert len(result) == 5
        for q in result:
            assert "quintile" in q
            assert "pnl" in q
            assert "n_trades" in q


# ===================================================================
# Unit 4: 对抗测试
# ===================================================================

class TestAdversarialTests:
    def _make_obs(self, event_date, obs_minutes, p_model, label,
                  market_price=0.5, strike=90000.0):
        settlement = 91000.0 if label == 1 else 89000.0
        return ObservationResult(
            event_date=event_date,
            obs_minutes=obs_minutes,
            now_utc_ms=1000,
            s0=90000.0,
            settlement_price=settlement,
            k_grid=[strike],
            predictions={strike: p_model},
            labels={strike: label},
            market_prices={strike: market_price},
        )

    def test_returns_three_tests(self):
        obs = [
            self._make_obs("2026-02-15", 60, 0.80, 1),
            self._make_obs("2026-02-15", 5, 0.85, 1),
            self._make_obs("2026-02-16", 120, 0.20, 0),
        ]
        result = run_adversarial_tests(obs)
        assert "no_last_10m" in result
        assert "no_top5_markets" in result
        assert "no_top_quintile" in result

    def test_survived_field(self):
        obs = [
            self._make_obs("2026-02-15", 60, 0.80, 1),
            self._make_obs("2026-02-16", 30, 0.20, 0),
        ]
        result = run_adversarial_tests(obs)
        for key in result:
            assert "survived" in result[key]
            assert isinstance(result[key]["survived"], bool)


# ===================================================================
# compute_all_metrics 新字段
# ===================================================================

class TestComputeAllMetricsNewFields:
    def test_new_fields_present(self):
        obs = ObservationResult(
            event_date="2026-02-01",
            obs_minutes=60,
            now_utc_ms=1000,
            s0=89000.0,
            settlement_price=91000.0,
            k_grid=[88000.0, 90000.0],
            predictions={88000.0: 0.8, 90000.0: 0.55},
            labels={88000.0: 1, 90000.0: 1},
            market_prices={88000.0: 0.5, 90000.0: 0.5},
        )
        metrics = compute_all_metrics([obs])
        overall = metrics["overall"]
        assert "risk_metrics" in overall
        assert "ece" in overall
        assert "edge_quintiles" in overall
        assert "adversarial" in overall
        # 新增字段
        assert "price_range_stats" in overall
        assert "drawdown_details" in overall
        assert "cost_sensitivity" in overall
        assert "latency_sensitivity" in overall
        assert "capacity_analysis" in overall

    def test_risk_metrics_structure(self):
        """验证风险指标包含 per-event 和年化字段"""
        obs = ObservationResult(
            event_date="2026-02-01",
            obs_minutes=60,
            now_utc_ms=1000,
            s0=89000.0,
            settlement_price=91000.0,
            k_grid=[88000.0],
            predictions={88000.0: 0.8},
            labels={88000.0: 1},
            market_prices={88000.0: 0.5},
        )
        metrics = compute_all_metrics([obs])
        risk = metrics["overall"]["risk_metrics"]
        assert "n_periods" in risk
        assert "calmar_short_period" in risk
        assert "annualized_sharpe" in risk
        assert "annualized_sortino" in risk


# ===================================================================
# Unit 2: 按价格区间分层
# ===================================================================

class TestPriceRangeStats:
    def test_itm_atm_otm(self):
        """不同 moneyness 分组"""
        obs_itm = ObservationResult(
            event_date="2026-02-01", obs_minutes=60, now_utc_ms=1000,
            s0=92000.0, settlement_price=91000.0,
            k_grid=[90000.0],
            predictions={90000.0: 0.8}, labels={90000.0: 1},
            market_prices={90000.0: 0.5},
        )
        obs_otm = ObservationResult(
            event_date="2026-02-02", obs_minutes=60, now_utc_ms=2000,
            s0=88000.0, settlement_price=89000.0,
            k_grid=[90000.0],
            predictions={90000.0: 0.3}, labels={90000.0: 0},
            market_prices={90000.0: 0.5},
        )
        portfolio = simulate_portfolio([obs_itm, obs_otm])
        result = compute_price_range_stats([obs_itm, obs_otm], portfolio)
        assert len(result) == 3
        names = [r["range"] for r in result]
        assert "ITM" in names
        assert "ATM" in names
        assert "OTM" in names

    def test_empty(self):
        result = compute_price_range_stats([], {"markets": []})
        assert len(result) == 3
        for r in result:
            assert r["n_obs"] == 0


# ===================================================================
# Unit 3: 回撤详情
# ===================================================================

class TestDrawdownDetails:
    def test_no_drawdown(self):
        """单调递增无回撤"""
        curve = [100, 110, 120, 130]
        dd = compute_drawdown_details(curve)
        assert dd["max_dd_duration_events"] == 0
        assert dd["max_consecutive_losses"] == 0

    def test_with_drawdown(self):
        """有回撤"""
        curve = [100, 110, 105, 95, 100, 115]
        dd = compute_drawdown_details(curve)
        assert dd["max_dd_duration_events"] > 0
        assert dd["max_consecutive_losses"] >= 1
        assert len(dd["dd_periods"]) > 0

    def test_short_curve(self):
        dd = compute_drawdown_details([100])
        assert dd["max_dd_duration_events"] == 0

    def test_unrecovered(self):
        """末端仍在回撤"""
        curve = [100, 110, 90, 85]
        dd = compute_drawdown_details(curve)
        assert len(dd["dd_periods"]) > 0


# ===================================================================
# Unit 4: 手续费敏感性
# ===================================================================

class TestCostSensitivity:
    def test_returns_three_scenarios(self):
        obs = ObservationResult(
            event_date="2026-02-01", obs_minutes=60, now_utc_ms=1000,
            s0=89000.0, settlement_price=91000.0,
            k_grid=[90000.0],
            predictions={90000.0: 0.8}, labels={90000.0: 1},
            market_prices={90000.0: 0.5},
        )
        result = run_cost_sensitivity([obs])
        assert len(result) == 3
        mults = [r["fee_mult"] for r in result]
        assert 0.5 in mults
        assert 1.0 in mults
        assert 2.0 in mults

    def test_higher_fee_lower_pnl(self):
        obs = ObservationResult(
            event_date="2026-02-01", obs_minutes=60, now_utc_ms=1000,
            s0=89000.0, settlement_price=91000.0,
            k_grid=[90000.0],
            predictions={90000.0: 0.8}, labels={90000.0: 1},
            market_prices={90000.0: 0.5},
        )
        result = run_cost_sensitivity([obs])
        # fee x2 的 PnL 应 <= fee x1 的 PnL
        pnl_by_mult = {r["fee_mult"]: r["pnl"] for r in result}
        assert pnl_by_mult[2.0] <= pnl_by_mult[1.0]


# ===================================================================
# Unit 6: 延迟敏感性
# ===================================================================

class TestLatencySensitivity:
    def test_returns_four_buckets(self):
        obs_list = [
            ObservationResult(
                event_date="2026-02-01", obs_minutes=m, now_utc_ms=1000,
                s0=89000.0, settlement_price=91000.0,
                k_grid=[90000.0],
                predictions={90000.0: 0.8}, labels={90000.0: 1},
                market_prices={90000.0: 0.5},
            )
            for m in [5, 20, 45, 120]
        ]
        result = run_latency_sensitivity(obs_list)
        assert len(result) == 4
        buckets = [r["bucket"] for r in result]
        assert ">60min" in buckets
        assert "<10min" in buckets


# ===================================================================
# Unit 7: 容量扩展
# ===================================================================

class TestCapacityAnalysis:
    def test_returns_four_scenarios(self):
        obs = ObservationResult(
            event_date="2026-02-01", obs_minutes=60, now_utc_ms=1000,
            s0=89000.0, settlement_price=91000.0,
            k_grid=[90000.0],
            predictions={90000.0: 0.8}, labels={90000.0: 1},
            market_prices={90000.0: 0.5},
        )
        result = run_capacity_analysis([obs])
        assert len(result) == 4
        spts = [r["shares_per_trade"] for r in result]
        assert 100 in spts
        assert 200 in spts
        assert 500 in spts
        assert 1000 in spts

    def test_linear_scaling(self):
        """无限仓位下 PnL 应近似线性缩放"""
        obs = ObservationResult(
            event_date="2026-02-01", obs_minutes=60, now_utc_ms=1000,
            s0=89000.0, settlement_price=91000.0,
            k_grid=[90000.0],
            predictions={90000.0: 0.8}, labels={90000.0: 1},
            market_prices={90000.0: 0.5},
        )
        result = run_capacity_analysis([obs], max_net_shares=100_000)
        pnl_by_spt = {r["shares_per_trade"]: r["pnl"] for r in result}
        # 200 份 PnL 应约为 100 份的 2 倍
        if pnl_by_spt[100] != 0:
            ratio = pnl_by_spt[200] / pnl_by_spt[100]
            assert ratio == pytest.approx(2.0, abs=0.1)


# ===================================================================
# 方向分析 (§4.6)
# ===================================================================

class TestDirectionAnalysisBasic:
    """验证 traded / not_traded 分组正确"""

    def test_traded_not_traded_split(self):
        # edge = |0.8 - 0.5| = 0.3 > 0.03 → traded
        obs_traded = ObservationResult(
            event_date="2026-02-01", obs_minutes=60, now_utc_ms=1000,
            s0=89000.0, settlement_price=91000.0,
            k_grid=[90000.0],
            predictions={90000.0: 0.8}, labels={90000.0: 1},
            market_prices={90000.0: 0.5},
        )
        # edge = |0.51 - 0.50| = 0.01 < 0.03 → not_traded
        obs_not_traded = ObservationResult(
            event_date="2026-02-02", obs_minutes=60, now_utc_ms=2000,
            s0=89000.0, settlement_price=91000.0,
            k_grid=[90000.0],
            predictions={90000.0: 0.51}, labels={90000.0: 1},
            market_prices={90000.0: 0.50},
        )
        result = compute_direction_analysis([obs_traded, obs_not_traded])
        assert result["n_traded"] == 1
        assert result["n_not_traded"] == 1
        assert result["n_with_market_price"] == 2

    def test_conditional_brier_computed(self):
        obs = ObservationResult(
            event_date="2026-02-01", obs_minutes=60, now_utc_ms=1000,
            s0=89000.0, settlement_price=91000.0,
            k_grid=[90000.0],
            predictions={90000.0: 0.8}, labels={90000.0: 1},
            market_prices={90000.0: 0.5},
        )
        result = compute_direction_analysis([obs])
        assert "traded_model_brier" in result
        assert "traded_market_brier" in result
        # (0.8-1)^2 = 0.04
        assert result["traded_model_brier"] == pytest.approx(0.04)
        # (0.5-1)^2 = 0.25
        assert result["traded_market_brier"] == pytest.approx(0.25)


class TestDirectionAnalysisBuyYesNoSplit:
    """验证 BUY YES / BUY NO 经济学计算"""

    def test_buy_yes_economics(self):
        # p=0.8 > mp=0.5 → BUY YES, label=1 → win
        obs = ObservationResult(
            event_date="2026-02-01", obs_minutes=60, now_utc_ms=1000,
            s0=89000.0, settlement_price=91000.0,
            k_grid=[90000.0],
            predictions={90000.0: 0.8}, labels={90000.0: 1},
            market_prices={90000.0: 0.5},
        )
        result = compute_direction_analysis([obs])
        by = result["buy_yes"]
        assert by["n"] == 1
        assert by["avg_cost"] == pytest.approx(0.5)
        assert by["avg_model"] == pytest.approx(0.8)
        assert by["win_rate"] == pytest.approx(1.0)
        # pnl_per_share = label - market_price = 1 - 0.5 = 0.5
        assert by["pnl_per_share"] == pytest.approx(0.5)

    def test_buy_no_economics(self):
        # p=0.2 < mp=0.5 → BUY NO, label=0 → win
        obs = ObservationResult(
            event_date="2026-02-01", obs_minutes=60, now_utc_ms=1000,
            s0=89000.0, settlement_price=89000.0,
            k_grid=[90000.0],
            predictions={90000.0: 0.2}, labels={90000.0: 0},
            market_prices={90000.0: 0.5},
        )
        result = compute_direction_analysis([obs])
        bn = result["buy_no"]
        assert bn["n"] == 1
        assert bn["avg_cost"] == pytest.approx(0.5)  # 1 - 0.5
        assert bn["avg_model"] == pytest.approx(0.8)  # 1 - 0.2
        assert bn["win_rate"] == pytest.approx(1.0)
        # pnl_per_share = mp - label = 0.5 - 0 = 0.5
        assert bn["pnl_per_share"] == pytest.approx(0.5)

    def test_mixed_directions(self):
        obs_yes = ObservationResult(
            event_date="2026-02-01", obs_minutes=60, now_utc_ms=1000,
            s0=89000.0, settlement_price=91000.0,
            k_grid=[90000.0],
            predictions={90000.0: 0.8}, labels={90000.0: 1},
            market_prices={90000.0: 0.5},
        )
        obs_no = ObservationResult(
            event_date="2026-02-02", obs_minutes=60, now_utc_ms=2000,
            s0=89000.0, settlement_price=89000.0,
            k_grid=[90000.0],
            predictions={90000.0: 0.2}, labels={90000.0: 0},
            market_prices={90000.0: 0.5},
        )
        result = compute_direction_analysis([obs_yes, obs_no])
        assert result["buy_yes"]["n"] == 1
        assert result["buy_no"]["n"] == 1


class TestDirectionAnalysisMarketLevelAccuracy:
    """验证市场级别方向正确率"""

    def test_correct_yes_direction(self):
        # 两个时点都 BUY YES 同一市场，label=1 → 正确
        obs1 = ObservationResult(
            event_date="2026-02-01", obs_minutes=60, now_utc_ms=1000,
            s0=89000.0, settlement_price=91000.0,
            k_grid=[90000.0],
            predictions={90000.0: 0.8}, labels={90000.0: 1},
            market_prices={90000.0: 0.5},
        )
        obs2 = ObservationResult(
            event_date="2026-02-01", obs_minutes=30, now_utc_ms=2000,
            s0=89500.0, settlement_price=91000.0,
            k_grid=[90000.0],
            predictions={90000.0: 0.85}, labels={90000.0: 1},
            market_prices={90000.0: 0.55},
        )
        result = compute_direction_analysis([obs1, obs2])
        # 同一 event_date + strike → 1 个市场
        assert result["market_level_yes_total"] == 1
        assert result["market_level_yes_correct"] == 1
        assert result["market_level_yes_accuracy"] == pytest.approx(1.0)

    def test_incorrect_no_direction(self):
        # BUY NO 但 label=1 → 方向错误
        obs = ObservationResult(
            event_date="2026-02-01", obs_minutes=60, now_utc_ms=1000,
            s0=89000.0, settlement_price=91000.0,
            k_grid=[90000.0],
            predictions={90000.0: 0.2}, labels={90000.0: 1},
            market_prices={90000.0: 0.5},
        )
        result = compute_direction_analysis([obs])
        assert result["market_level_no_total"] == 1
        assert result["market_level_no_correct"] == 0
        assert result["market_level_no_accuracy"] == pytest.approx(0.0)


class TestDirectionAnalysisBootstrap:
    """验证 bootstrap 结果结构 + p_value 范围"""

    def test_bootstrap_structure(self):
        obs = ObservationResult(
            event_date="2026-02-01", obs_minutes=60, now_utc_ms=1000,
            s0=89000.0, settlement_price=91000.0,
            k_grid=[90000.0],
            predictions={90000.0: 0.8}, labels={90000.0: 1},
            market_prices={90000.0: 0.5},
        )
        result = compute_direction_analysis([obs])
        bs = result["bootstrap"]
        assert "real_pnl_per_share_sum" in bs
        assert "random_mean" in bs
        assert "random_ci_lower" in bs
        assert "random_ci_upper" in bs
        assert "p_value" in bs
        assert "reverse_pnl" in bs

    def test_p_value_range(self):
        obs = ObservationResult(
            event_date="2026-02-01", obs_minutes=60, now_utc_ms=1000,
            s0=89000.0, settlement_price=91000.0,
            k_grid=[90000.0],
            predictions={90000.0: 0.8}, labels={90000.0: 1},
            market_prices={90000.0: 0.5},
        )
        result = compute_direction_analysis([obs])
        assert 0.0 <= result["bootstrap"]["p_value"] <= 1.0

    def test_reverse_pnl_sign(self):
        """反向 PnL 应与正向相反"""
        obs = ObservationResult(
            event_date="2026-02-01", obs_minutes=60, now_utc_ms=1000,
            s0=89000.0, settlement_price=91000.0,
            k_grid=[90000.0],
            predictions={90000.0: 0.8}, labels={90000.0: 1},
            market_prices={90000.0: 0.5},
        )
        result = compute_direction_analysis([obs])
        bs = result["bootstrap"]
        assert bs["reverse_pnl"] == pytest.approx(-bs["real_pnl_per_share_sum"])


class TestDirectionAnalysisNoMarketPrice:
    """无市场价格时返回空结果"""

    def test_no_market_price(self):
        obs = ObservationResult(
            event_date="2026-02-01", obs_minutes=60, now_utc_ms=1000,
            s0=89000.0, settlement_price=91000.0,
            k_grid=[90000.0],
            predictions={90000.0: 0.8}, labels={90000.0: 1},
            # 无 market_prices
        )
        result = compute_direction_analysis([obs])
        assert result["n_traded"] == 0
        assert result["n_not_traded"] == 0
        assert result["n_with_market_price"] == 0
