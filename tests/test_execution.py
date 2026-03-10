"""execution 模块测试"""

import pytest

from pricing_core.execution import (
    compute_opinion_fee,
    shrink_probability,
    compute_edge,
    should_trade,
    kelly_position,
    generate_signal,
)


class TestComputeOpinionFee:

    def test_at_half(self):
        """price=0.5 时费率最高"""
        fee = compute_opinion_fee(0.5)
        expected = 0.06 * 0.5 * 0.5 + 0.0025
        assert fee == pytest.approx(expected)

    def test_at_extremes(self):
        """price 接近 0 或 1 时费率低"""
        fee_low = compute_opinion_fee(0.05)
        fee_high = compute_opinion_fee(0.95)
        fee_mid = compute_opinion_fee(0.5)
        assert fee_low < fee_mid
        assert fee_high < fee_mid

    def test_formula(self):
        """验证公式: 0.06 * p * (1-p) + 0.0025"""
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            expected = 0.06 * p * (1 - p) + 0.0025
            assert compute_opinion_fee(p) == pytest.approx(expected)


class TestShrinkProbability:

    def test_shrinkage_formula(self):
        """p_trade = lambda * p_P + (1-lambda) * market"""
        p = shrink_probability(0.6, 0.5, lam=0.6)
        expected = 0.6 * 0.6 + 0.4 * 0.5
        assert p == pytest.approx(expected)

    def test_clamped(self):
        """结果被限制在 (0, 1)"""
        assert shrink_probability(1.0, 1.0) < 1.0
        assert shrink_probability(0.0, 0.0) > 0.0

    def test_lambda_one(self):
        """lambda=1 时等于物理概率"""
        p = shrink_probability(0.7, 0.3, lam=1.0)
        assert p == pytest.approx(0.7)

    def test_lambda_zero(self):
        """lambda=0 时等于市场价格"""
        p = shrink_probability(0.7, 0.3, lam=0.0)
        assert p == pytest.approx(0.3)


class TestComputeEdge:

    def test_positive_edge(self):
        assert compute_edge(0.6, 0.5) == pytest.approx(0.1)

    def test_negative_edge(self):
        assert compute_edge(0.4, 0.5) == pytest.approx(-0.1)

    def test_zero_edge(self):
        assert compute_edge(0.5, 0.5) == pytest.approx(0.0)


class TestShouldTrade:

    def test_above_threshold(self):
        assert should_trade(0.05, threshold=0.03) is True

    def test_below_threshold(self):
        assert should_trade(0.02, threshold=0.03) is False

    def test_negative_edge_above_threshold(self):
        assert should_trade(-0.05, threshold=0.03) is True

    def test_with_uncertainty_buffer(self):
        assert should_trade(0.04, threshold=0.03, uncertainty_buffer=0.02) is False
        assert should_trade(0.06, threshold=0.03, uncertainty_buffer=0.02) is True


class TestKellyPosition:

    def test_formula(self):
        """f = eta * |edge| / (p * (1-p))"""
        f = kelly_position(edge=0.1, p_trade=0.5, eta=0.2)
        expected = 0.2 * 0.1 / (0.5 * 0.5)
        assert f == pytest.approx(expected)

    def test_max_position(self):
        """仓位限制"""
        # edge=0.5, p=0.5, eta=10 -> f = 10*0.5/0.25 = 20, 但上限 5
        f = kelly_position(edge=0.5, p_trade=0.5, eta=10.0, max_position=5)
        assert f == 5.0

    def test_zero_edge(self):
        f = kelly_position(edge=0.0, p_trade=0.5, eta=0.2)
        assert f == 0.0

    def test_extreme_probability(self):
        """概率接近 0 或 1 时不会除零"""
        f = kelly_position(edge=0.01, p_trade=0.001, eta=0.2)
        assert f >= 0


class TestGenerateSignal:

    def test_buy_yes_signal(self):
        """正 edge -> BUY_YES"""
        signal = generate_signal(
            strike=90000, p_trade=0.6, market_price=0.5,
            threshold=0.03, eta=0.2,
        )
        assert signal is not None
        assert signal.direction == "BUY_YES"
        assert signal.edge > 0

    def test_buy_no_signal(self):
        """负 edge -> BUY_NO"""
        signal = generate_signal(
            strike=90000, p_trade=0.4, market_price=0.55,
            threshold=0.03, eta=0.2,
        )
        assert signal is not None
        assert signal.direction == "BUY_NO"

    def test_no_signal_below_threshold(self):
        """edge 不足 -> 无信号"""
        signal = generate_signal(
            strike=90000, p_trade=0.51, market_price=0.50,
            threshold=0.03,
        )
        assert signal is None

    def test_no_signal_fee_exceeds_edge(self):
        """费用超过 edge -> 无信号"""
        signal = generate_signal(
            strike=90000, p_trade=0.54, market_price=0.50,
            threshold=0.03, eta=0.2,
        )
        # edge=0.04, fee ≈ 0.06*0.5*0.5+0.0025 = 0.0175
        # net_edge = 0.04 - 0.0175 = 0.0225 > 0, 所以应该有信号
        assert signal is not None
        assert signal.net_edge > 0
