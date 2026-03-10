"""
回测引擎测试
"""

import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from pricing_core.binance_data import Kline
from pricing_core.models import HARCoefficients

from backtest.config import BacktestConfig
from backtest.data_cache import KlineCache, _date_to_utc_ms
from backtest.engine import BacktestEngine
from backtest.historical_client import HistoricalBinanceClient
from backtest.models import BacktestResult


def _create_full_cache(cache_dir: str, start_date: str, n_days: int):
    """创建完整的合成缓存（用于引擎测试）"""
    cache = KlineCache(cache_dir=cache_dir)

    dt = datetime.strptime(start_date, "%Y-%m-%d")
    np.random.seed(42)
    price = 90000.0

    for d in range(n_days):
        date_str = (dt + timedelta(days=d)).strftime("%Y-%m-%d")
        base_ms = _date_to_utc_ms(date_str)

        rows = []
        for i in range(1440):
            price += np.random.normal(0, 5)
            rows.append({
                "open_time": base_ms + i * 60_000,
                "open": price,
                "high": price + 5,
                "low": price - 5,
                "close": price + np.random.normal(0, 2),
                "volume": 100.0,
                "close_time": base_ms + i * 60_000 + 59999,
            })
        df = pd.DataFrame(rows)
        df.to_csv(cache._file_path(date_str), index=False, compression="gzip")

    return cache


class TestBacktestEngine:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.output_dir = tempfile.mkdtemp()
        # 创建 40 天的数据: 30 天 HAR 训练 + ~10 天回测
        self.cache = _create_full_cache(
            self.tmpdir,
            "2026-01-01",
            n_days=43,
        )

    def test_engine_short_backtest(self):
        """短日期范围的端到端回测"""
        config = BacktestConfig(
            start_date="2026-02-01",
            end_date="2026-02-03",  # 只测 2 天
            cache_dir=self.tmpdir,
            output_dir=self.output_dir,
            step_minutes=60,        # 每 60 分钟观测一次加速
            lookback_hours=1,       # 只看 1h 加速
            k_offsets=[0, 500],     # 减少 K 数量
            mc_samples=1000,        # 减少 MC 采样加速
            har_train_days=30,
        )
        engine = BacktestEngine(config=config, cache=self.cache)
        result = engine.run()

        assert isinstance(result, BacktestResult)
        assert result.start_date == "2026-02-01"
        assert result.end_date == "2026-02-03"
        # 2 个事件日 × 1h lookback / 60min step = 1 观测/天 = 2
        assert len(result.observations) == 2
        assert len(result.event_outcomes) == 2

        # 检查观测结果
        obs = result.observations[0]
        assert obs.event_date == "2026-02-01"
        assert obs.obs_minutes == 60
        assert len(obs.k_grid) == 2
        assert len(obs.predictions) == 2
        assert len(obs.labels) == 2
        # 预测值在 [0, 1] 范围
        for p in obs.predictions.values():
            assert 0 <= p <= 1

    def test_engine_builds_historical_client(self):
        """引擎正确构建历史客户端"""
        config = BacktestConfig(
            start_date="2026-02-01",
            end_date="2026-02-02",
            cache_dir=self.tmpdir,
            har_train_days=30,
        )
        engine = BacktestEngine(config=config, cache=self.cache)
        client = engine._build_historical_client()
        assert isinstance(client, HistoricalBinanceClient)
        assert len(client._klines) > 0

    def test_no_lookahead_bias(self):
        """验证无前视偏差: 观测时刻在事件之前"""
        config = BacktestConfig(
            start_date="2026-02-01",
            end_date="2026-02-02",
            cache_dir=self.tmpdir,
            step_minutes=360,       # 每 6h 步进
            lookback_hours=6,       # 只看 6h
            k_offsets=[0],
            mc_samples=500,
            har_train_days=30,
        )
        engine = BacktestEngine(config=config, cache=self.cache)
        result = engine.run()

        for obs in result.observations:
            # 所有观测的 now_utc_ms 应该在事件之前
            from pricing_core.time_utils import et_noon_to_utc_ms
            event_ms = et_noon_to_utc_ms(obs.event_date)
            assert obs.now_utc_ms < event_ms
            assert obs.now_utc_ms == event_ms - obs.obs_minutes * 60_000
