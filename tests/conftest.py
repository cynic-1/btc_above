"""
共享测试 fixture
"""

from backtest.metrics import compute_all_metrics
from backtest.models import BacktestResult, EventOutcome, ObservationResult


def make_test_backtest_data(obs2_minutes: int = 30):
    """
    构造回测测试数据

    Args:
        obs2_minutes: 第二个观测的 obs_minutes (默认 30，用 400 可测试 T-12h~6h 桶)
    """
    obs1 = ObservationResult(
        event_date="2026-02-15",
        obs_minutes=60,
        now_utc_ms=1000,
        s0=89000.0,
        settlement_price=91000.0,
        k_grid=[90000.0, 92000.0],
        predictions={90000.0: 0.8, 92000.0: 0.3},
        labels={90000.0: 1, 92000.0: 0},
        market_prices={90000.0: 0.5, 92000.0: 0.6},
    )
    obs2 = ObservationResult(
        event_date="2026-02-16",
        obs_minutes=obs2_minutes,
        now_utc_ms=2000,
        s0=90000.0,
        settlement_price=89000.0,
        k_grid=[90000.0],
        predictions={90000.0: 0.4},
        labels={90000.0: 0},
        market_prices={90000.0: 0.5},
    )
    result = BacktestResult(
        start_date="2026-02-15",
        end_date="2026-02-16",
        observations=[obs1, obs2],
        event_outcomes=[
            EventOutcome(event_date="2026-02-15", event_utc_ms=1000, settlement_price=91000.0),
            EventOutcome(event_date="2026-02-16", event_utc_ms=2000, settlement_price=89000.0),
        ],
    )
    metrics = compute_all_metrics([obs1, obs2])
    return result, metrics
