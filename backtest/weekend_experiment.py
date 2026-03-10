"""
周末效应实验：检验策略表现是否存在周末效应

从 detail.csv 加载已有回测数据，按 event_date 的星期几分组（weekend / weekday），
分别跑 simulate_portfolio 对比各组指标。
"""

import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .metrics import brier_score, compute_sharpe, simulate_portfolio
from .models import ObservationResult
from .timing_experiment import load_observations_from_csv

logger = logging.getLogger(__name__)

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@dataclass
class DayGroupResult:
    """单组（weekend/weekday/单日）的统计结果"""
    group_name: str        # "weekend" / "weekday" / "Monday" etc.
    n_event_dates: int = 0
    event_dates: List[str] = field(default_factory=list)
    # 信号质量
    brier: Optional[float] = None
    avg_abs_edge: Optional[float] = None
    market_price_coverage: float = 0.0  # 有市场价格的观测占比
    # 交易表现
    total_pnl: float = 0.0
    return_pct: float = 0.0
    profit_factor: float = 0.0
    n_trades: int = 0
    n_markets: int = 0
    win_rate: float = 0.0
    # 风险
    max_drawdown_pct: float = 0.0
    sharpe: Optional[float] = None
    # 市场微观
    avg_spread: Optional[float] = None  # avg(ask - bid)


def is_weekend(event_date: str) -> bool:
    """周六=5, 周日=6"""
    dt = datetime.strptime(event_date, "%Y-%m-%d")
    return dt.weekday() >= 5


def weekday_name(event_date: str) -> str:
    """返回 Monday/Tuesday/.../Sunday"""
    dt = datetime.strptime(event_date, "%Y-%m-%d")
    return WEEKDAY_NAMES[dt.weekday()]


def weekday_index(event_date: str) -> int:
    """返回 0=Monday ~ 6=Sunday"""
    dt = datetime.strptime(event_date, "%Y-%m-%d")
    return dt.weekday()


def filter_by_dates(
    observations: List[ObservationResult],
    dates_set: Set[str],
) -> List[ObservationResult]:
    """按 event_date 集合过滤观测"""
    return [obs for obs in observations if obs.event_date in dates_set]


def _load_spread_data(csv_path: str) -> Dict[Tuple[str, int, float], Tuple[float, float]]:
    """
    从 CSV 读取 bid/ask 数据

    Returns:
        {(event_date, obs_minutes, strike): (bid, ask)}
    """
    spreads: Dict[Tuple[str, int, float], Tuple[float, float]] = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bid_str = row.get("market_bid", "")
            ask_str = row.get("market_ask", "")
            if bid_str and ask_str:
                try:
                    bid = float(bid_str)
                    ask = float(ask_str)
                    if bid > 0 and ask > 0:
                        key = (row["event_date"], int(row["obs_minutes"]), float(row["strike"]))
                        spreads[key] = (bid, ask)
                except (ValueError, KeyError):
                    pass
    return spreads


def _compute_avg_spread(
    observations: List[ObservationResult],
    spread_data: Dict[Tuple[str, int, float], Tuple[float, float]],
) -> Optional[float]:
    """从预加载的 bid/ask 数据计算平均 spread"""
    spreads = []
    for obs in observations:
        for k in obs.k_grid:
            key = (obs.event_date, obs.obs_minutes, k)
            ba = spread_data.get(key)
            if ba:
                spreads.append(ba[1] - ba[0])  # ask - bid
    return float(np.mean(spreads)) if spreads else None


def compute_group_result(
    observations: List[ObservationResult],
    group_name: str,
    initial_capital: float = 100_000.0,
    shares_per_trade: int = 200,
    max_net_shares: int = 10_000,
    entry_threshold: float = 0.03,
    spread_data: Optional[Dict] = None,
) -> DayGroupResult:
    """对一组观测计算全部指标"""
    result = DayGroupResult(group_name=group_name)

    if not observations:
        return result

    event_dates = sorted(set(obs.event_date for obs in observations))
    result.n_event_dates = len(event_dates)
    result.event_dates = event_dates

    # === 信号质量 ===
    preds, labels = [], []
    edges = []
    total_obs = 0
    obs_with_mp = 0
    for obs in observations:
        for k in obs.k_grid:
            total_obs += 1
            p = obs.predictions.get(k)
            y = obs.labels.get(k)
            mp = obs.market_prices.get(k)
            if p is not None and y is not None:
                preds.append(p)
                labels.append(y)
            if mp is not None:
                obs_with_mp += 1
                if p is not None:
                    edges.append(abs(p - mp))

    if preds:
        result.brier = float(brier_score(np.array(preds), np.array(labels)))
    if edges:
        result.avg_abs_edge = float(np.mean(edges))
    result.market_price_coverage = obs_with_mp / total_obs if total_obs > 0 else 0.0

    # === 交易表现 ===
    portfolio = simulate_portfolio(
        observations,
        initial_capital=initial_capital,
        shares_per_trade=shares_per_trade,
        max_net_shares=max_net_shares,
        entry_threshold=entry_threshold,
    )

    result.total_pnl = portfolio["total_pnl"]
    result.return_pct = portfolio["total_return_pct"]
    result.profit_factor = portfolio["profit_factor"]
    result.n_trades = portfolio["n_trades"]
    result.n_markets = portfolio["n_markets"]
    result.max_drawdown_pct = portfolio["max_drawdown_pct"]

    if portfolio["n_markets"] > 0:
        result.win_rate = portfolio["win_markets"] / portfolio["n_markets"]

    result.sharpe = compute_sharpe(portfolio["event_pnls"], initial_capital)

    # === 市场微观 ===
    if spread_data:
        result.avg_spread = _compute_avg_spread(observations, spread_data)

    return result


def run_weekend_experiment(
    csv_path: str,
    initial_capital: float = 100_000.0,
    shares_per_trade: int = 200,
    max_net_shares: int = 10_000,
    entry_threshold: float = 0.03,
) -> Tuple[DayGroupResult, DayGroupResult, List[DayGroupResult]]:
    """
    运行周末效应实验

    Returns:
        (weekend_result, weekday_result, per_day_results)
        per_day_results: 7 个元素 (Mon~Sun)，无数据的日期结果为空
    """
    logger.info(f"加载数据: {csv_path}")
    observations = load_observations_from_csv(csv_path)
    logger.info(f"加载完成: {len(observations)} 个观测")

    # 加载 bid/ask 数据用于 spread 计算
    logger.info("加载 bid/ask 数据...")
    spread_data = _load_spread_data(csv_path)
    logger.info(f"bid/ask 数据: {len(spread_data)} 条")

    sim_kwargs = dict(
        initial_capital=initial_capital,
        shares_per_trade=shares_per_trade,
        max_net_shares=max_net_shares,
        entry_threshold=entry_threshold,
        spread_data=spread_data,
    )

    # 按 event_date 分类
    all_dates = sorted(set(obs.event_date for obs in observations))
    weekend_dates = set(d for d in all_dates if is_weekend(d))
    weekday_dates = set(d for d in all_dates if not is_weekend(d))
    logger.info(f"日期分布: {len(weekday_dates)} 工作日, {len(weekend_dates)} 周末日")

    # weekend vs weekday
    logger.info("计算 weekend 组...")
    weekend_obs = filter_by_dates(observations, weekend_dates)
    weekend_result = compute_group_result(weekend_obs, "weekend", **sim_kwargs)

    logger.info("计算 weekday 组...")
    weekday_obs = filter_by_dates(observations, weekday_dates)
    weekday_result = compute_group_result(weekday_obs, "weekday", **sim_kwargs)

    # 逐日 7 组
    per_day_results = []
    # 按 weekday index 分组
    day_dates: Dict[int, Set[str]] = {i: set() for i in range(7)}
    for d in all_dates:
        day_dates[weekday_index(d)].add(d)

    for i in range(7):
        name = WEEKDAY_NAMES[i]
        dates = day_dates[i]
        if dates:
            logger.info(f"计算 {name} 组 ({len(dates)} 日)...")
            day_obs = filter_by_dates(observations, dates)
            day_result = compute_group_result(day_obs, name, **sim_kwargs)
        else:
            day_result = DayGroupResult(group_name=name)
        per_day_results.append(day_result)

    return weekend_result, weekday_result, per_day_results
