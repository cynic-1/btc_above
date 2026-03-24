"""观测缓存序列化/反序列化测试"""

import pickle
import pytest

from backtest.models import BacktestResult, EventOutcome, ObservationResult
from backtest.observation_cache import CACHE_VERSION, load_observations, save_observations


def _make_sample_result() -> BacktestResult:
    """构造测试用 BacktestResult"""
    obs1 = ObservationResult(
        event_date="2026-03-01",
        obs_minutes=60,
        now_utc_ms=1000000,
        s0=85000.0,
        settlement_price=85500.0,
        k_grid=[84000.0, 85000.0, 86000.0],
        predictions={84000.0: 0.85, 85000.0: 0.55, 86000.0: 0.20},
        labels={84000.0: 1, 85000.0: 1, 86000.0: 0},
        market_prices={84000.0: 0.82, 85000.0: 0.50},
        market_bid_ask={84000.0: (0.80, 0.84), 85000.0: (0.48, 0.52)},
    )
    obs2 = ObservationResult(
        event_date="2026-03-01",
        obs_minutes=30,
        now_utc_ms=1800000,
        s0=85100.0,
        settlement_price=85500.0,
        k_grid=[84000.0, 85000.0, 86000.0],
        predictions={84000.0: 0.90, 85000.0: 0.60, 86000.0: 0.25},
        labels={84000.0: 1, 85000.0: 1, 86000.0: 0},
        market_prices={84000.0: 0.88},
        market_bid_ask={84000.0: (0.86, 0.90)},
    )
    event = EventOutcome(
        event_date="2026-03-01",
        event_utc_ms=2000000,
        settlement_price=85500.0,
        labels={84000.0: 1, 85000.0: 1, 86000.0: 0},
    )
    return BacktestResult(
        start_date="2026-03-01",
        end_date="2026-03-02",
        observations=[obs1, obs2],
        event_outcomes=[event],
    )


class TestObservationCache:
    """观测缓存测试"""

    def test_save_load_roundtrip(self, tmp_path):
        """保存-加载往返: 数据完整保留"""
        original = _make_sample_result()
        path = save_observations(original, str(tmp_path))
        loaded = load_observations(path)

        assert loaded.start_date == original.start_date
        assert loaded.end_date == original.end_date
        assert len(loaded.observations) == 2
        assert len(loaded.event_outcomes) == 1

        obs = loaded.observations[0]
        assert obs.event_date == "2026-03-01"
        assert obs.obs_minutes == 60
        assert obs.predictions[85000.0] == pytest.approx(0.55)
        assert obs.market_bid_ask[84000.0] == (0.80, 0.84)

        ev = loaded.event_outcomes[0]
        assert ev.settlement_price == 85500.0
        assert ev.labels[86000.0] == 0

    def test_version_mismatch(self, tmp_path):
        """版本不匹配时抛出 ValueError"""
        path = tmp_path / "bad_version.pkl"
        with open(path, "wb") as f:
            pickle.dump({"version": 999, "observations": []}, f)
        with pytest.raises(ValueError, match="版本不匹配"):
            load_observations(str(path))

    def test_file_not_found(self):
        """文件不存在时抛出 FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            load_observations("/nonexistent/path.pkl")

    def test_default_tag_uses_dates(self, tmp_path):
        """默认 tag 使用 start_end 日期"""
        result = _make_sample_result()
        path = save_observations(result, str(tmp_path))
        assert "observations_2026-03-01_2026-03-02.pkl" in path

    def test_custom_tag(self, tmp_path):
        """自定义 tag"""
        result = _make_sample_result()
        path = save_observations(result, str(tmp_path), tag="custom_run")
        assert "observations_custom_run.pkl" in path

    def test_market_bid_ask_preserved(self, tmp_path):
        """确保 market_bid_ask 元组在序列化后保留"""
        original = _make_sample_result()
        path = save_observations(original, str(tmp_path))
        loaded = load_observations(path)

        obs = loaded.observations[0]
        ba = obs.market_bid_ask.get(84000.0)
        assert ba is not None
        assert len(ba) == 2
        assert ba[0] == pytest.approx(0.80)
        assert ba[1] == pytest.approx(0.84)

    def test_summary_not_serialized(self, tmp_path):
        """summary 字段不序列化（回放时重新计算）"""
        original = _make_sample_result()
        original.summary = {"some": "data"}
        path = save_observations(original, str(tmp_path))
        loaded = load_observations(path)
        assert loaded.summary is None
