"""pipeline 模块测试（集成测试，使用 mock 数据源）"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from pricing_core.config import PricingConfig
from pricing_core.models import BasisParams, DistParams
from pricing_core.binance_data import BinanceClient, Kline
from pricing_core.pipeline import PricingPipeline, generate_trade_signals


def _make_mock_klines(n=1500, base_price=90000.0, start_ms=None):
    """生成模拟 K线数据"""
    rng = np.random.default_rng(42)
    prices = base_price + np.cumsum(rng.normal(0, 10, n))
    if start_ms is None:
        start_ms = 1700000000000

    klines = []
    for i in range(n):
        p = float(prices[i])
        klines.append(Kline(
            open_time=start_ms + i * 60000,
            open=p - 5,
            high=p + 15,
            low=p - 15,
            close=p,
            volume=100.0,
            close_time=start_ms + i * 60000 + 59999,
        ))
    return klines


class TestPricingPipeline:

    def _make_pipeline(self):
        """创建带 mock 客户端的管线"""
        config = PricingConfig()
        config.mc_samples = 5000  # 测试用小样本

        mock_binance = MagicMock(spec=BinanceClient)
        mock_klines = _make_mock_klines()
        mock_binance.get_current_price.return_value = 90000.0
        mock_binance.get_klines_extended.return_value = mock_klines
        mock_binance.get_close_prices.return_value = np.array([k.close for k in mock_klines])

        pipeline = PricingPipeline(
            config=config,
            binance_client=mock_binance,
        )
        return pipeline

    def test_run_produces_results(self):
        """运行管线产生定价结果"""
        pipeline = self._make_pipeline()
        result = pipeline.run(
            event_date="2026-03-05",
            k_grid=[89000, 90000, 91000],
        )
        assert len(result.strike_results) == 3
        assert result.mc_samples == 5000
        assert result.pricing_input.s0 == 90000.0

    def test_probabilities_sum_reasonable(self):
        """概率值在合理范围"""
        pipeline = self._make_pipeline()
        result = pipeline.run(
            event_date="2026-03-05",
            k_grid=[85000, 90000, 95000],
        )
        for sr in result.strike_results:
            assert 0 <= sr.p_physical <= 1
            assert sr.ci_lower <= sr.p_physical <= sr.ci_upper

    def test_with_market_prices(self):
        """提供市场价格时计算 edge"""
        pipeline = self._make_pipeline()
        result = pipeline.run(
            event_date="2026-03-05",
            k_grid=[89000, 90000, 91000],
            market_prices={89000: 0.7, 90000: 0.5, 91000: 0.3},
        )
        for sr in result.strike_results:
            assert sr.p_trade > 0  # 收缩后概率已填充


class TestGenerateTradeSignals:

    def test_generates_signals(self):
        """从定价结果生成交易信号"""
        pipeline = TestPricingPipeline()._make_pipeline()
        result = pipeline.run(
            event_date="2026-03-05",
            k_grid=[89000, 90000, 91000],
            market_prices={89000: 0.7, 90000: 0.5, 91000: 0.3},
        )

        signals = generate_trade_signals(
            result,
            market_prices={89000: 0.7, 90000: 0.5, 91000: 0.3},
        )
        # 不保证有信号，但函数不应出错
        assert isinstance(signals, list)
