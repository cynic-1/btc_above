"""
Above 合约回测集成测试

使用合成数据验证回测引擎核心逻辑:
- 结算标签正确性
- 观测数量
- Brier score 可计算
"""

import os
import gzip
import csv
import tempfile
from typing import List

import numpy as np
import pytest

from above.backtest_engine import AboveBacktestEngine
from above.dvol_pricing import prob_above_k_gbm
from above.models import AboveBacktestConfig, AboveObservation
from backtest.metrics import brier_score


def _write_kline_csv(path: str, open_time: int, close: float) -> None:
    """写入单行 K线 gzip CSV（用于测试）"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    # 生成一天 1440 分钟的 K线
    for i in range(1440):
        t = open_time + i * 60_000
        rows.append({
            "open_time": t,
            "open": close,
            "high": close + 50,
            "low": close - 50,
            "close": close,
            "volume": 100.0,
            "close_time": t + 59_999,
        })
    with gzip.open(path, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "open_time", "open", "high", "low", "close", "volume", "close_time",
        ])
        writer.writeheader()
        writer.writerows(rows)


def _write_dvol_csv(path: str, data: List[tuple]) -> None:
    """写入 DVOL gzip CSV"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ms", "volatility"])
        for ts, vol in data:
            writer.writerow([ts, vol])


class TestAboveBacktestEngine:
    """回测引擎集成测试"""

    def _setup_synthetic_data(self, tmpdir: str, prices: dict):
        """
        创建合成测试数据

        Args:
            tmpdir: 临时目录
            prices: {date_str: close_price}
        """
        kline_dir = os.path.join(tmpdir, "klines")
        iv_dir = os.path.join(tmpdir, "iv")
        os.makedirs(kline_dir, exist_ok=True)
        os.makedirs(iv_dir, exist_ok=True)

        # 写 K线
        for date_str, close in prices.items():
            from datetime import datetime
            from pricing_core.time_utils import UTC
            dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)
            day_start_ms = int(dt.timestamp() * 1000)
            path = os.path.join(kline_dir, f"BTCUSDT_1m_{date_str}.csv.gz")
            _write_kline_csv(path, day_start_ms, close)

        # 写 DVOL: 固定 55% IV
        dvol_data = []
        for date_str in prices:
            from datetime import datetime
            from pricing_core.time_utils import UTC
            dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)
            ts = int(dt.timestamp() * 1000)
            dvol_data.append((ts, 55.0))  # 百分比形式
        _write_dvol_csv(os.path.join(iv_dir, "dvol_btc.csv.gz"), dvol_data)

        return kline_dir, iv_dir

    def test_settlement_labels_above(self):
        """测试结算标签: settlement > K → label = 1"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # BTC 两天都是 87000
            prices = {
                "2026-03-04": 87000.0,
                "2026-03-05": 87000.0,
            }
            kline_dir, iv_dir = self._setup_synthetic_data(tmpdir, prices)

            config = AboveBacktestConfig(
                start_date="2026-03-05",
                end_date="2026-03-06",
                cache_dir=kline_dir,
                iv_cache_dir=iv_dir,
                step_minutes=60,
                lookback_hours=4,
                iv_source="dvol",
                use_market_prices=False,
                use_fixed_strikes=True,
                k_offsets=[-1000, 0, 1000],
                output_dir=os.path.join(tmpdir, "output"),
            )

            engine = AboveBacktestEngine(config)
            observations = engine.run()

            assert len(observations) > 0, "应有观测结果"

            # 检查标签
            for obs in observations:
                settlement = obs.settlement_price
                assert settlement == 87000.0, f"结算价应为 87000, 得到 {settlement}"
                for K, label in obs.labels.items():
                    if K < settlement:
                        assert label == 1, f"K={K} < settlement={settlement}, 标签应为 1"
                    else:
                        assert label == 0, f"K={K} >= settlement={settlement}, 标签应为 0"

    def test_observation_count(self):
        """测试观测数量: lookback=4h, step=60min → 每天 5 个观测 (包含起点)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            prices = {
                "2026-03-04": 87000.0,
                "2026-03-05": 87000.0,
                "2026-03-06": 87000.0,
            }
            kline_dir, iv_dir = self._setup_synthetic_data(tmpdir, prices)

            config = AboveBacktestConfig(
                start_date="2026-03-05",
                end_date="2026-03-07",
                cache_dir=kline_dir,
                iv_cache_dir=iv_dir,
                step_minutes=60,
                lookback_hours=4,
                iv_source="dvol",
                use_market_prices=False,
                use_fixed_strikes=True,
                k_offsets=[0],
                output_dir=os.path.join(tmpdir, "output"),
            )

            engine = AboveBacktestEngine(config)
            observations = engine.run()

            # 2 天 x (4h / 60min + 1) = 2 x 5 = 10
            n_days = len(set(obs.event_date for obs in observations))
            assert n_days == 2, f"应有 2 天, 得到 {n_days}"
            assert len(observations) == 10, (
                f"应有 10 个观测 (2天 x 5步), 得到 {len(observations)}"
            )

    def test_brier_score_computable(self):
        """验证 Brier score 可从观测结果计算"""
        with tempfile.TemporaryDirectory() as tmpdir:
            prices = {
                "2026-03-04": 87000.0,
                "2026-03-05": 87000.0,
            }
            kline_dir, iv_dir = self._setup_synthetic_data(tmpdir, prices)

            config = AboveBacktestConfig(
                start_date="2026-03-05",
                end_date="2026-03-06",
                cache_dir=kline_dir,
                iv_cache_dir=iv_dir,
                step_minutes=60,
                lookback_hours=2,
                iv_source="dvol",
                use_market_prices=False,
                use_fixed_strikes=True,
                k_offsets=[-2000, -1000, 0, 1000, 2000],
                output_dir=os.path.join(tmpdir, "output"),
            )

            engine = AboveBacktestEngine(config)
            observations = engine.run()

            assert len(observations) > 0

            # 收集预测和标签
            preds, labels = [], []
            for obs in observations:
                for K in obs.k_grid:
                    if K in obs.predictions and K in obs.labels:
                        preds.append(obs.predictions[K])
                        labels.append(obs.labels[K])

            assert len(preds) > 0, "应有预测数据"

            bs = brier_score(np.array(preds), np.array(labels))
            assert 0 <= bs <= 1, f"Brier score 应在 [0,1], 得到 {bs}"

    def test_serialization_roundtrip(self):
        """测试序列化 + 反序列化"""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs = AboveObservation(
                event_date="2026-03-05",
                obs_utc_ms=1741186800000,
                event_utc_ms=1741190400000,
                s0=87000.0,
                sigma=0.55,
                T_years=0.001,
                settlement_price=87000.0,
                k_grid=[85000, 87000, 89000],
                predictions={85000: 0.95, 87000: 0.48, 89000: 0.05},
                labels={85000: 1, 87000: 0, 89000: 0},
                market_prices={},
            )

            path = AboveBacktestEngine.serialize_observations(
                [obs], tmpdir, "test"
            )
            loaded = AboveBacktestEngine.load_observations(path)

            assert len(loaded) == 1
            assert loaded[0].event_date == "2026-03-05"
            assert loaded[0].s0 == 87000.0
            assert loaded[0].predictions[85000] == 0.95
            assert loaded[0].labels[87000] == 0

    def test_k_grid_from_fixed_offsets(self):
        """测试固定偏移 K grid 构建"""
        with tempfile.TemporaryDirectory() as tmpdir:
            prices = {
                "2026-03-04": 87000.0,
                "2026-03-05": 87000.0,
            }
            kline_dir, iv_dir = self._setup_synthetic_data(tmpdir, prices)

            config = AboveBacktestConfig(
                start_date="2026-03-05",
                end_date="2026-03-06",
                cache_dir=kline_dir,
                iv_cache_dir=iv_dir,
                step_minutes=240,  # 大步长减少观测
                lookback_hours=1,
                iv_source="dvol",
                use_market_prices=False,
                use_fixed_strikes=True,
                k_offsets=[-1000, 0, 1000],
                output_dir=os.path.join(tmpdir, "output"),
            )

            engine = AboveBacktestEngine(config)
            observations = engine.run()

            assert len(observations) > 0
            k_grid = observations[0].k_grid
            assert len(k_grid) == 3
            # 基准 = round(87000/500)*500 = 87000
            assert 86000 in k_grid
            assert 87000 in k_grid
            assert 88000 in k_grid
