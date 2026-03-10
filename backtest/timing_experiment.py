"""
时间窗口实验：找出策略最优启动/停止时间

从 detail.csv 重建 ObservationResult，按 (start_minutes, stop_minutes) 窗口过滤后
调用 simulate_portfolio() 计算各窗口的交易表现。
"""

import csv
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .metrics import brier_score, compute_sharpe, simulate_portfolio
from .models import ObservationResult

logger = logging.getLogger(__name__)

# 时间网格
START_MINUTES = [720, 360, 180, 120, 60, 30, 10]
STOP_MINUTES = [0, 1, 2, 5, 10, 15, 30, 60]


@dataclass
class TimingWindowResult:
    """单个时间窗口的所有指标"""
    start_minutes: int
    stop_minutes: int
    total_pnl: float = 0.0
    return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    brier: float = 0.0
    sharpe: Optional[float] = None
    n_trades: int = 0
    n_markets: int = 0
    roi: float = 0.0  # PnL / total_cost
    composite_score: float = 0.0
    low_confidence: bool = False


def load_observations_from_csv(csv_path: str) -> List[ObservationResult]:
    """
    从 detail.csv 重建 ObservationResult 列表

    CSV 列: event_date, obs_minutes, strike, s0, settlement,
            p_physical, label, ci_lower, ci_upper, market_price

    同一 (event_date, obs_minutes) 的行合并为一个 ObservationResult
    """
    # 按 (event_date, obs_minutes) 分组
    groups: Dict[Tuple[str, int], Dict] = {}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            event_date = row["event_date"]
            obs_minutes = int(row["obs_minutes"])
            key = (event_date, obs_minutes)

            if key not in groups:
                groups[key] = {
                    "event_date": event_date,
                    "obs_minutes": obs_minutes,
                    "s0": float(row["s0"]),
                    "settlement": float(row["settlement"]),
                    "k_grid": [],
                    "predictions": {},
                    "labels": {},
                    "confidence_intervals": {},
                    "market_prices": {},
                }

            g = groups[key]
            strike = float(row["strike"])
            g["k_grid"].append(strike)
            g["predictions"][strike] = float(row["p_physical"])
            g["labels"][strike] = int(row["label"])

            # 置信区间
            ci_lo = row.get("ci_lower", "")
            ci_hi = row.get("ci_upper", "")
            if ci_lo and ci_hi:
                g["confidence_intervals"][strike] = (float(ci_lo), float(ci_hi))

            # 市场价格
            mp = row.get("market_price", "")
            if mp:
                g["market_prices"][strike] = float(mp)

    # 转换为 ObservationResult
    observations = []
    for key in sorted(groups.keys()):
        g = groups[key]
        obs = ObservationResult(
            event_date=g["event_date"],
            obs_minutes=g["obs_minutes"],
            now_utc_ms=0,  # CSV 中无此字段，不影响 simulate_portfolio
            s0=g["s0"],
            settlement_price=g["settlement"],
            k_grid=g["k_grid"],
            predictions=g["predictions"],
            labels=g["labels"],
            confidence_intervals=g["confidence_intervals"],
            market_prices=g["market_prices"],
        )
        observations.append(obs)

    logger.info(f"从 CSV 加载 {len(observations)} 个观测 "
                f"({len(groups)} 个唯一 (date, minutes) 组合)")
    return observations


def _filter_observations(
    observations: List[ObservationResult],
    start_minutes: int,
    stop_minutes: int,
) -> List[ObservationResult]:
    """按时间窗口过滤观测：保留 stop_minutes < obs_minutes <= start_minutes"""
    return [
        obs for obs in observations
        if stop_minutes < obs.obs_minutes <= start_minutes
    ]


def _compute_window_result(
    observations: List[ObservationResult],
    start_minutes: int,
    stop_minutes: int,
    initial_capital: float,
    shares_per_trade: int,
    max_net_shares: int,
    entry_threshold: float,
) -> TimingWindowResult:
    """计算单个窗口的完整指标"""
    filtered = _filter_observations(observations, start_minutes, stop_minutes)

    result = TimingWindowResult(
        start_minutes=start_minutes,
        stop_minutes=stop_minutes,
    )

    if not filtered:
        result.low_confidence = True
        return result

    # 运行组合模拟
    portfolio = simulate_portfolio(
        filtered,
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

    # 胜率（按市场计）
    if portfolio["n_markets"] > 0:
        result.win_rate = portfolio["win_markets"] / portfolio["n_markets"]

    # ROI = PnL / total_cost
    if portfolio["total_cost"] > 0:
        result.roi = portfolio["total_pnl"] / portfolio["total_cost"]

    # Sharpe
    result.sharpe = compute_sharpe(
        portfolio["event_pnls"], initial_capital
    )

    # Brier score（仅对有市场价格的观测）
    preds, labels = [], []
    for obs in filtered:
        for k in obs.k_grid:
            p = obs.predictions.get(k)
            y = obs.labels.get(k)
            if p is not None and y is not None:
                preds.append(p)
                labels.append(y)
    if preds:
        result.brier = float(brier_score(np.array(preds), np.array(labels)))

    # 低交易数警告
    if result.n_trades < 20:
        result.low_confidence = True

    # 综合评分
    result.composite_score = _compute_composite_score(result)

    return result


def _compute_composite_score(r: TimingWindowResult) -> float:
    """
    综合评分: Sharpe(0.3) + profit_factor(0.25) + return%(0.2) + ROI(0.15) + -max_dd(0.1)

    各项归一化到合理范围后加权求和
    """
    score = 0.0

    # Sharpe: 典型范围 -2 ~ 3，归一化
    if r.sharpe is not None:
        score += 0.3 * max(-1.0, min(r.sharpe, 3.0)) / 3.0

    # Profit factor: 归一化，cap 在 5
    pf = r.profit_factor if r.profit_factor != float("inf") else 5.0
    pf = max(0.0, min(pf, 5.0))
    score += 0.25 * pf / 5.0

    # Return %: 归一化，cap 在 20%
    ret = max(-10.0, min(r.return_pct, 20.0))
    score += 0.2 * (ret + 10.0) / 30.0

    # ROI: cap 在 1.0
    roi = max(-0.5, min(r.roi, 1.0))
    score += 0.15 * (roi + 0.5) / 1.5

    # Max drawdown penalty: 越小越好
    dd = max(0.0, min(r.max_drawdown_pct, 10.0))
    score += 0.1 * (1.0 - dd / 10.0)

    return score


def run_timing_grid(
    observations: List[ObservationResult],
    start_grid: List[int] = None,
    stop_grid: List[int] = None,
    initial_capital: float = 100_000.0,
    shares_per_trade: int = 200,
    max_net_shares: int = 10_000,
    entry_threshold: float = 0.03,
) -> List[TimingWindowResult]:
    """
    遍历所有有效 (start, stop) 组合，计算各窗口表现

    有效条件: start > stop
    """
    if start_grid is None:
        start_grid = START_MINUTES
    if stop_grid is None:
        stop_grid = STOP_MINUTES

    results = []
    total = sum(1 for s in start_grid for e in stop_grid if s > e)
    done = 0

    for start in start_grid:
        for stop in stop_grid:
            if start <= stop:
                continue

            result = _compute_window_result(
                observations, start, stop,
                initial_capital, shares_per_trade,
                max_net_shares, entry_threshold,
            )
            results.append(result)
            done += 1

            if done % 10 == 0:
                logger.info(f"  进度: {done}/{total} 窗口已完成")

    logger.info(f"时间网格实验完成: {len(results)} 个窗口")
    return results


def find_optimal_windows(
    results: List[TimingWindowResult],
    top_n: int = 15,
) -> List[TimingWindowResult]:
    """按综合评分排序，返回 top-N 窗口"""
    # 过滤掉无交易的窗口
    valid = [r for r in results if r.n_trades > 0]
    valid.sort(key=lambda r: r.composite_score, reverse=True)
    return valid[:top_n]


def compute_incremental_value(
    observations: List[ObservationResult],
    bucket_width_minutes: int = 30,
    initial_capital: float = 100_000.0,
    shares_per_trade: int = 200,
    max_net_shares: int = 10_000,
    entry_threshold: float = 0.03,
) -> List[Dict]:
    """
    逐桶计算增量 PnL 贡献

    将时间轴按 bucket_width_minutes 分桶，对每桶：
    - 计算"包含该桶"的 PnL（全量）
    - 计算"排除该桶"的 PnL
    - 增量 = 全量 - 排除
    """
    # 找出 obs_minutes 的范围
    all_minutes = set(obs.obs_minutes for obs in observations)
    if not all_minutes:
        return []

    max_min = max(all_minutes)

    # 全量 PnL
    full_portfolio = simulate_portfolio(
        observations,
        initial_capital=initial_capital,
        shares_per_trade=shares_per_trade,
        max_net_shares=max_net_shares,
        entry_threshold=entry_threshold,
    )
    full_pnl = full_portfolio["total_pnl"]

    # 按桶计算
    results = []
    bucket_start = 0
    while bucket_start < max_min:
        bucket_end = bucket_start + bucket_width_minutes
        bucket_label = f"T-{bucket_end}m ~ T-{bucket_start}m"

        # 排除该桶的观测
        excluded = [
            obs for obs in observations
            if not (bucket_start < obs.obs_minutes <= bucket_end)
        ]

        if len(excluded) == len(observations):
            # 该桶无观测，增量为 0
            results.append({
                "bucket": bucket_label,
                "bucket_start": bucket_start,
                "bucket_end": bucket_end,
                "n_obs_in_bucket": 0,
                "full_pnl": full_pnl,
                "excluded_pnl": full_pnl,
                "incremental_pnl": 0.0,
            })
        else:
            excluded_portfolio = simulate_portfolio(
                excluded,
                initial_capital=initial_capital,
                shares_per_trade=shares_per_trade,
                max_net_shares=max_net_shares,
                entry_threshold=entry_threshold,
            )
            excluded_pnl = excluded_portfolio["total_pnl"]
            n_in_bucket = len(observations) - len(excluded)

            results.append({
                "bucket": bucket_label,
                "bucket_start": bucket_start,
                "bucket_end": bucket_end,
                "n_obs_in_bucket": n_in_bucket,
                "full_pnl": full_pnl,
                "excluded_pnl": excluded_pnl,
                "incremental_pnl": full_pnl - excluded_pnl,
            })

        bucket_start = bucket_end

    return results
