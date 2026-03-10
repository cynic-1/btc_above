"""主编排引擎单元测试"""

import time
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from pricing_core.binance_data import Kline

from live.config import LiveTradingConfig
from live.engine import LiveTradingEngine
from live.models import MarketInfo, OrderBookState, OrderBookLevel, Signal


@pytest.fixture
def config():
    return LiveTradingConfig(
        event_date="2026-03-09",
        dry_run=True,
        shares_per_trade=200,
        entry_threshold=0.03,
        pricing_interval_seconds=1.0,
        mc_samples=500,
    )


@pytest.fixture
def engine(config):
    return LiveTradingEngine(config)


def _make_market(strike: float = 85000.0) -> MarketInfo:
    return MarketInfo(
        event_date="2026-03-09",
        strike=strike,
        condition_id=f"cond_{int(strike)}",
        yes_token_id=f"yes_{int(strike)}",
        no_token_id=f"no_{int(strike)}",
        question=f"Will Bitcoin be above ${strike:,.0f} on March 9?",
        tick_size="0.01",
        neg_risk=False,
    )


class TestComputeSignals:
    """信号计算测试"""

    def test_no_signal_without_data(self, engine):
        """无数据时应返回空"""
        engine._markets = [_make_market()]
        engine._market_by_token = {"yes_85000": _make_market()}

        # Mock binance feed: 无数据
        engine._binance_feed = MagicMock()
        engine._binance_feed.get_current_price.return_value = 0.0
        engine._binance_feed.get_klines.return_value = []

        signals = engine._compute_signals(
            now_utc_ms=int(time.time() * 1000),
            event_utc_ms=int(time.time() * 1000) + 3600_000,
        )
        assert signals == []

    def test_signal_with_edge(self, engine):
        """有足够 edge 时应生成信号"""
        market = _make_market(85000.0)
        engine._markets = [market]
        engine._market_by_token = {market.yes_token_id: market}

        # Mock binance feed
        engine._binance_feed = MagicMock()
        engine._binance_feed.get_current_price.return_value = 85000.0
        # 生成简单 klines
        klines = []
        import numpy as np
        rng = np.random.RandomState(42)
        for i in range(1500):
            close = 85000.0 + rng.randn() * 50
            klines.append(Kline(
                open_time=i * 60_000,
                open=close - 1, high=close + 5, low=close - 5,
                close=close, volume=100.0, close_time=i * 60_000 + 59999,
            ))
        engine._binance_feed.get_klines.return_value = klines

        # Mock pricing engine: 返回高概率
        engine._pricing_engine = MagicMock()
        engine._pricing_engine.compute_prices.return_value = {85000.0: 0.70}

        # Mock orderbook: ask 价远低于模型价
        ob = OrderBookState(
            asset_id=market.yes_token_id,
            bids=[OrderBookLevel(price=0.55, size=1000)],
            asks=[OrderBookLevel(price=0.58, size=1000)],
            best_bid=0.55,
            best_ask=0.58,
        )
        engine._polymarket_ws = MagicMock()
        engine._polymarket_ws.get_orderbook.return_value = ob

        signals = engine._compute_signals(
            now_utc_ms=int(time.time() * 1000),
            event_utc_ms=int(time.time() * 1000) + 3600_000,
        )

        # 模型价 0.70, 收缩后 p_trade = 0.6*0.70 + 0.4*0.565 = 0.646
        # edge_yes = 0.646 - 0.58 = 0.066 > 0.03 → 应有信号
        assert len(signals) >= 1
        assert signals[0].direction == "YES"
        assert signals[0].edge > 0.03

    def test_no_signal_small_edge(self, engine):
        """edge 不够时不应生成信号"""
        market = _make_market(85000.0)
        engine._markets = [market]
        engine._market_by_token = {market.yes_token_id: market}

        engine._binance_feed = MagicMock()
        engine._binance_feed.get_current_price.return_value = 85000.0
        import numpy as np
        rng = np.random.RandomState(42)
        klines = []
        for i in range(1500):
            close = 85000.0 + rng.randn() * 50
            klines.append(Kline(
                open_time=i * 60_000,
                open=close - 1, high=close + 5, low=close - 5,
                close=close, volume=100.0, close_time=i * 60_000 + 59999,
            ))
        engine._binance_feed.get_klines.return_value = klines

        # 模型价 ≈ 市场价 → 无 edge
        engine._pricing_engine = MagicMock()
        engine._pricing_engine.compute_prices.return_value = {85000.0: 0.57}

        ob = OrderBookState(
            asset_id=market.yes_token_id,
            bids=[OrderBookLevel(price=0.55, size=1000)],
            asks=[OrderBookLevel(price=0.58, size=1000)],
            best_bid=0.55,
            best_ask=0.58,
        )
        engine._polymarket_ws = MagicMock()
        engine._polymarket_ws.get_orderbook.return_value = ob

        signals = engine._compute_signals(
            now_utc_ms=int(time.time() * 1000),
            event_utc_ms=int(time.time() * 1000) + 3600_000,
        )

        assert len(signals) == 0


class TestExecuteSignal:
    """信号执行测试"""

    def test_dry_run(self, engine):
        """dry-run 模式不应实际下单"""
        market = _make_market()
        engine._markets = [market]
        engine._market_by_token = {market.yes_token_id: market}
        engine._position_manager = MagicMock()
        engine._position_manager.can_trade.return_value = (True, "OK")

        sig = Signal(
            strike=85000.0,
            condition_id="cond_85000",
            token_id="yes_85000",
            direction="YES",
            side="BUY",
            model_price=0.65,
            market_price=0.58,
            edge=0.07,
            shares=200,
            price=0.58,
            timestamp_ms=0,
        )

        engine._execute_signal(sig)

        # 应记录交易
        assert len(engine._trade_records) == 1
        assert engine._trade_records[0].status == "dry-run"
        # 应更新仓位
        engine._position_manager.record_trade.assert_called_once()

    def test_position_rejected(self, engine):
        """仓位拒绝时不应下单"""
        engine._position_manager = MagicMock()
        engine._position_manager.can_trade.return_value = (False, "超限")

        sig = Signal(
            strike=85000.0,
            condition_id="cond_85000",
            token_id="yes_85000",
            direction="YES",
            side="BUY",
            model_price=0.65,
            market_price=0.58,
            edge=0.07,
            shares=200,
            price=0.58,
        )

        engine._execute_signal(sig)
        assert len(engine._trade_records) == 0


class TestHealthCheck:
    """健康检查测试"""

    def test_health_check(self, engine):
        """健康检查应返回完整状态"""
        engine._binance_feed = MagicMock()
        engine._binance_feed.get_current_price.return_value = 85000.0
        engine._binance_feed.is_connected.return_value = True
        engine._polymarket_ws = MagicMock()
        engine._polymarket_ws.is_connected.return_value = True
        engine._position_manager = MagicMock()
        engine._position_manager.get_total_cost.return_value = 1000.0
        engine._markets = [_make_market()]

        health = engine._health_check(mins_left=120.0)

        assert health["btc_price"] == 85000.0
        assert health["binance_ok"] is True
        assert health["polymarket_ok"] is True
        assert health["total_cost"] == 1000.0
        assert health["markets"] == 1
