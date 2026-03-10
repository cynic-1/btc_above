"""仓位管理器单元测试"""

import pytest

from live.config import LiveTradingConfig
from live.models import Position, Signal
from live.position_manager import PositionManager


@pytest.fixture
def config():
    return LiveTradingConfig(
        max_net_shares=1000,
        max_total_cost=10_000.0,
        shares_per_trade=200,
    )


@pytest.fixture
def manager(config):
    return PositionManager(config)


def _make_signal(
    direction: str = "YES",
    side: str = "BUY",
    shares: int = 200,
    price: float = 0.55,
    strike: float = 85000.0,
    condition_id: str = "cond_1",
) -> Signal:
    return Signal(
        strike=strike,
        condition_id=condition_id,
        token_id=f"token_{direction.lower()}",
        direction=direction,
        side=side,
        model_price=0.60,
        market_price=price,
        edge=0.05,
        shares=shares,
        price=price,
        timestamp_ms=0,
    )


class TestCanTrade:
    """交易许可检查测试"""

    def test_allow_first_trade(self, manager):
        """首次交易应允许"""
        sig = _make_signal()
        can, reason = manager.can_trade(sig)
        assert can is True
        assert reason == "OK"

    def test_reject_over_net_shares(self, manager):
        """超过净仓位限制应拒绝"""
        # 先记录 900 shares
        sig1 = _make_signal(shares=900)
        manager.record_trade(sig1, "matched")

        # 再尝试 200 → 净仓 1100 > 1000
        sig2 = _make_signal(shares=200)
        can, reason = manager.can_trade(sig2)
        assert can is False
        assert "净仓位超限" in reason

    def test_reject_over_total_cost(self, manager):
        """超过总成本限制应拒绝"""
        # 先记录成本 $9000
        sig1 = _make_signal(shares=900, price=10.0)  # 高价模拟
        manager.record_trade(sig1, "matched")

        # 再尝试 $2000 → 总 $11000 > $10000
        sig2 = _make_signal(shares=200, price=10.0)
        can, reason = manager.can_trade(sig2)
        assert can is False
        assert "总成本超限" in reason

    def test_no_shares_net_with_both_sides(self, manager):
        """YES + NO 应互相抵消"""
        sig_yes = _make_signal(direction="YES", shares=500)
        manager.record_trade(sig_yes, "matched")

        sig_no = _make_signal(direction="NO", shares=400)
        manager.record_trade(sig_no, "matched")

        # 净仓 = 500 - 400 = 100，再买 900 YES → 1000，刚好不超
        sig3 = _make_signal(direction="YES", shares=900)
        can, reason = manager.can_trade(sig3)
        assert can is True


class TestRecordTrade:
    """仓位记录测试"""

    def test_record_yes_buy(self, manager):
        """记录 YES 买入"""
        sig = _make_signal(direction="YES", shares=200, price=0.55)
        manager.record_trade(sig, "matched")

        pos = manager.get_position("cond_1")
        assert pos.yes_shares == 200
        assert pos.yes_cost == pytest.approx(110.0)
        assert pos.no_shares == 0
        assert pos.net_shares == 200

    def test_record_no_buy(self, manager):
        """记录 NO 买入"""
        sig = _make_signal(direction="NO", shares=100, price=0.40)
        manager.record_trade(sig, "matched")

        pos = manager.get_position("cond_1")
        assert pos.no_shares == 100
        assert pos.no_cost == pytest.approx(40.0)
        assert pos.net_shares == -100

    def test_failed_trade_not_recorded(self, manager):
        """失败的交易不应记录"""
        sig = _make_signal(shares=200)
        manager.record_trade(sig, "failed")

        pos = manager.get_position("cond_1")
        assert pos.yes_shares == 0
        assert manager.get_total_cost() == 0.0

    def test_multiple_markets(self, manager):
        """多市场仓位独立追踪"""
        sig1 = _make_signal(condition_id="cond_1", shares=200)
        sig2 = _make_signal(condition_id="cond_2", shares=300, strike=90000.0)
        manager.record_trade(sig1, "matched")
        manager.record_trade(sig2, "matched")

        all_pos = manager.get_all_positions()
        assert len(all_pos) == 2
        assert all_pos["cond_1"].yes_shares == 200
        assert all_pos["cond_2"].yes_shares == 300


class TestSummary:
    """汇总测试"""

    def test_empty_summary(self, manager):
        assert manager.summary() == "无持仓"

    def test_with_positions(self, manager):
        sig = _make_signal(shares=200, price=0.55)
        manager.record_trade(sig, "matched")

        summary = manager.summary()
        assert "总成本" in summary
        assert "K=85000" in summary
