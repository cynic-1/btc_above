"""LiveTradingConfig 单元测试"""

import os
import pytest
from unittest.mock import patch

from live.config import LiveTradingConfig


class TestLiveTradingConfig:
    """配置加载测试"""

    def test_default_values(self):
        """默认值应正确设置"""
        config = LiveTradingConfig()
        assert config.polymarket_chain_id == 137
        assert config.shares_per_trade == 200
        assert config.max_net_shares == 10_000
        assert config.max_total_cost == 50_000.0
        assert config.entry_threshold == 0.03
        assert config.order_type == "GTC"
        assert config.mc_samples == 2000
        assert config.dist_refit_minutes == 30
        assert config.pricing_interval_seconds == 10.0
        assert config.vrp_k == 1.0
        assert config.dry_run is False
        assert config.har_train_days == 30
        assert config.har_ridge_alpha == 0.01
        assert config.order_cooldown_seconds == 60.0
        assert config.polymarket_signature_type == 2

    def test_env_override(self):
        """环境变量应覆盖默认值"""
        env = {
            "PM_HOST": "https://test-clob.polymarket.com",
            "PM_PRIVATE_KEY": "0xabc123",
            "PM_FUNDER": "0xfunder",
            "LOG_DIR": "/tmp/test_logs",
        }
        with patch.dict(os.environ, env, clear=False):
            config = LiveTradingConfig()
            assert config.polymarket_host == "https://test-clob.polymarket.com"
            assert config.polymarket_private_key == "0xabc123"
            assert config.polymarket_funder == "0xfunder"
            assert config.log_dir == "/tmp/test_logs"

    def test_explicit_override(self):
        """显式参数应覆盖环境变量"""
        config = LiveTradingConfig(
            shares_per_trade=100,
            max_net_shares=5000,
            entry_threshold=0.05,
            dry_run=True,
            event_date="2026-03-09",
        )
        assert config.shares_per_trade == 100
        assert config.max_net_shares == 5000
        assert config.entry_threshold == 0.05
        assert config.dry_run is True
        assert config.event_date == "2026-03-09"

    def test_binance_defaults(self):
        """Binance 配置默认值"""
        config = LiveTradingConfig()
        assert "stream.binance.com" in config.binance_ws_url
        assert "btcusdt@kline_1m" in config.binance_ws_url
        assert config.binance_max_rps == 10.0

    def test_polymarket_ws_default(self):
        """Polymarket WS 默认 URL"""
        config = LiveTradingConfig()
        assert "ws-subscriptions-clob" in config.polymarket_ws_url
