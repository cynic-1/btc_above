"""
Strike 过滤实验：验证只交易离 BTC 价格最近的 N 个 strike 是否优于全部 strike

从 detail.csv 加载已有回测数据，按 T-720m 时刻的 BTC 价格选择最近 N 个 strike，
过滤后重跑 simulate_portfolio 对比。
"""

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .metrics import brier_score, compute_sharpe, simulate_portfolio
from .models import ObservationResult
from .timing_experiment import load_observations_from_csv

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StrikeFilterResult:
    """单个 n_nearest 配置的指标"""
    n_nearest: int  # 0 表示 baseline（全部 strikes）
    total_pnl: float = 0.0
    return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    sharpe: Optional[float] = None
    n_trades: int = 0
    n_markets: int = 0
    roi: float = 0.0
    brier: float = 0.0
    n_event_dates: int = 0
    avg_strikes_per_date: float = 0.0


def select_nearest_strikes(
    observations: List[ObservationResult],
    n: int = 2,
) -> Dict[str, List[float]]:
    """
    按 event_date 分组，选择距 S0 最近的 N 个 strike

    对每个 event_date，取 obs_minutes 最大的观测（近似 T-720m 入场时刻），
    用其 s0 作为参考价格，按 |K - S0| 排序取前 n 个。

    Returns:
        {event_date: [selected_strikes]}
    """
    # 按 event_date 分组，找每日最大 obs_minutes 的观测
    date_max_obs: Dict[str, ObservationResult] = {}
    for obs in observations:
        existing = date_max_obs.get(obs.event_date)
        if existing is None or obs.obs_minutes > existing.obs_minutes:
            date_max_obs[obs.event_date] = obs

    strike_map: Dict[str, List[float]] = {}
    for date, obs in date_max_obs.items():
        s0 = obs.s0
        # 按距离排序，取前 n 个
        sorted_strikes = sorted(obs.k_grid, key=lambda k: abs(k - s0))
        strike_map[date] = sorted_strikes[:n]
        logger.debug(
            f"{date}: S0={s0:.0f}, 选中 strikes={strike_map[date]} "
            f"(共 {len(obs.k_grid)} 个可用)"
        )

    return strike_map


def filter_observations_by_strikes(
    observations: List[ObservationResult],
    strike_map: Dict[str, List[float]],
) -> List[ObservationResult]:
    """
    按 strike_map 过滤观测，仅保留选中的 strikes

    对每个 obs，只保留 strike_map[obs.event_date] 中的 strikes。
    如果过滤后无 strike 则跳过该观测。
    """
    filtered = []
    for obs in observations:
        selected = strike_map.get(obs.event_date)
        if not selected:
            continue

        selected_set = set(selected)
        new_k_grid = [k for k in obs.k_grid if k in selected_set]
        if not new_k_grid:
            continue

        new_obs = ObservationResult(
            event_date=obs.event_date,
            obs_minutes=obs.obs_minutes,
            now_utc_ms=obs.now_utc_ms,
            s0=obs.s0,
            settlement_price=obs.settlement_price,
            k_grid=new_k_grid,
            predictions={k: v for k, v in obs.predictions.items() if k in selected_set},
            labels={k: v for k, v in obs.labels.items() if k in selected_set},
            confidence_intervals={k: v for k, v in obs.confidence_intervals.items() if k in selected_set},
            market_prices={k: v for k, v in obs.market_prices.items() if k in selected_set},
            elapsed_seconds=obs.elapsed_seconds,
        )
        filtered.append(new_obs)

    return filtered


def _compute_result(
    observations: List[ObservationResult],
    n_nearest: int,
    initial_capital: float,
    shares_per_trade: int,
    max_net_shares: int,
    entry_threshold: float,
) -> StrikeFilterResult:
    """计算单个配置的完整指标"""
    result = StrikeFilterResult(n_nearest=n_nearest)

    if not observations:
        return result

    result.n_event_dates = len(set(obs.event_date for obs in observations))
    total_strikes = sum(len(obs.k_grid) for obs in observations)
    result.avg_strikes_per_date = total_strikes / len(observations) if observations else 0

    portfolio = simulate_portfolio(
        observations,
        initial_capital=initial_capital,
        shares_per_trade=shares_per_trade,
        max_net_shares=max_net_shares,
        entry_threshold=entry_threshold,
    )

    result.total_pnl = portfolio["total_pnl"]
    result.return_pct = portfolio["total_return_pct"]
    result.max_drawdown_pct = portfolio["max_drawdown_pct"]
    result.profit_factor = portfolio["profit_factor"]
    result.n_trades = portfolio["n_trades"]
    result.n_markets = portfolio["n_markets"]

    if portfolio["n_markets"] > 0:
        result.win_rate = portfolio["win_markets"] / portfolio["n_markets"]

    if portfolio["total_cost"] > 0:
        result.roi = portfolio["total_pnl"] / portfolio["total_cost"]

    result.sharpe = compute_sharpe(portfolio["event_pnls"], initial_capital)

    # Brier score
    preds, labels = [], []
    for obs in observations:
        for k in obs.k_grid:
            p = obs.predictions.get(k)
            y = obs.labels.get(k)
            if p is not None and y is not None:
                preds.append(p)
                labels.append(y)
    if preds:
        result.brier = float(brier_score(np.array(preds), np.array(labels)))

    return result


def run_strike_filter_experiment(
    csv_path: str,
    n_nearest_list: List[int] = None,
    initial_capital: float = 100_000.0,
    shares_per_trade: int = 200,
    max_net_shares: int = 10_000,
    entry_threshold: float = 0.03,
) -> List[StrikeFilterResult]:
    """
    运行 strike 过滤实验

    对比 baseline（全部 strikes）和不同 n_nearest 的表现。

    Args:
        csv_path: detail.csv 路径
        n_nearest_list: 要测试的 n 值列表，默认 [1, 2, 3, 4]

    Returns:
        结果列表，第一项为 baseline (n_nearest=0)
    """
    if n_nearest_list is None:
        n_nearest_list = [1, 2, 3, 4]

    logger.info(f"加载数据: {csv_path}")
    observations = load_observations_from_csv(csv_path)
    logger.info(f"加载完成: {len(observations)} 个观测")

    results = []

    # Baseline: 全部 strikes
    logger.info("计算 baseline（全部 strikes）...")
    baseline = _compute_result(
        observations, n_nearest=0,
        initial_capital=initial_capital,
        shares_per_trade=shares_per_trade,
        max_net_shares=max_net_shares,
        entry_threshold=entry_threshold,
    )
    results.append(baseline)

    # 各 n_nearest 配置
    for n in n_nearest_list:
        logger.info(f"计算 nearest-{n} strikes...")
        strike_map = select_nearest_strikes(observations, n=n)
        filtered = filter_observations_by_strikes(observations, strike_map)
        logger.info(f"  过滤后 {len(filtered)} 个观测（原 {len(observations)}）")

        r = _compute_result(
            filtered, n_nearest=n,
            initial_capital=initial_capital,
            shares_per_trade=shares_per_trade,
            max_net_shares=max_net_shares,
            entry_threshold=entry_threshold,
        )
        results.append(r)

    return results
