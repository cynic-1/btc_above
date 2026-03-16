"""
价格收敛分析 — 中途卖出的胜率与赔率

分析模型发现 edge 后，市场价格是否朝模型方向收敛。
对比不同持有期（30m/1h/2h/3h/6h/结算）的 PnL 表现。
"""

import csv
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 持有期定义: (名称, 分钟数)  -1 表示持有到结算
HOLDINGS = [
    ("30m", 30),
    ("1h", 60),
    ("2h", 120),
    ("3h", 180),
    ("4h", 240),
    ("6h", 360),
    ("settlement", -1),
]


@dataclass
class TradeSignal:
    """入场信号"""
    event_date: str
    obs_minutes: int
    strike: float
    direction: str       # "YES" / "NO"
    entry_ask: float     # YES ask at entry
    entry_bid: float     # YES bid at entry
    edge: float
    label: int           # 结算标签


@dataclass
class HoldingPeriodResult:
    """单个持有期的统计"""
    holding: str         # "30m" / "1h" / ... / "settlement"
    holding_minutes: int  # 30 / 60 / ... / -1 for settlement
    n_signals: int       # 入场信号总数
    n_with_exit: int     # 有退出价格数据的信号数
    n_wins: int          # PnL > 0
    n_losses: int        # PnL <= 0
    win_rate: float      # 胜率
    avg_win: float       # 赢时平均盈利 (per share)
    avg_loss: float      # 亏时平均亏损 (per share, 正数)
    payoff_ratio: float  # 赔率 = avg_win / avg_loss
    ev_per_trade: float  # 期望收益 per share
    avg_pnl: float       # 平均 PnL per share
    total_pnl: float     # 总 PnL (× shares_per_trade)
    median_pnl: float    # 中位数 PnL per share
    avg_edge: float      # 入场时平均 edge
    n_favorable: int = 0          # mid 向交易方向移动的次数
    favorable_rate: float = 0.0   # 有利移动率
    avg_mid_drift: float = 0.0    # 平均 mid 漂移（交易方向为正）
    median_mid_drift: float = 0.0 # 中位数 mid 漂移


@dataclass
class ConvergenceResult:
    """完整分析结果"""
    all_results: List[HoldingPeriodResult] = field(default_factory=list)
    yes_results: List[HoldingPeriodResult] = field(default_factory=list)
    no_results: List[HoldingPeriodResult] = field(default_factory=list)
    n_total_signals: int = 0
    entry_threshold: float = 0.03
    shares_per_trade: int = 200


def load_price_lookup(
    csv_path: str,
) -> Tuple[Dict[Tuple[str, float, int], Tuple[float, float]],
           Dict[Tuple[str, float, int], int],
           List[dict]]:
    """
    读 detail.csv → 价格查询表 + 标签表 + 原始行列表

    Returns:
        price_lookup: (event_date, strike, obs_minutes) → (bid, ask)
        label_lookup: (event_date, strike, obs_minutes) → label
        rows: 原始行数据列表
    """
    price_lookup: Dict[Tuple[str, float, int], Tuple[float, float]] = {}
    label_lookup: Dict[Tuple[str, float, int], int] = {}
    rows: List[dict] = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            event_date = row["event_date"]
            try:
                strike = float(row["strike"])
                obs_minutes = int(row["obs_minutes"])
            except (ValueError, KeyError):
                continue

            # 解析 bid/ask
            bid_str = row.get("market_bid", "")
            ask_str = row.get("market_ask", "")
            bid = float(bid_str) if bid_str else None
            ask = float(ask_str) if ask_str else None

            # 解析 p_physical, label, market_price
            p_str = row.get("p_physical", "")
            label_str = row.get("label", "")
            mp_str = row.get("market_price", "")

            p_physical = float(p_str) if p_str else None
            label = int(label_str) if label_str else None
            market_price = float(mp_str) if mp_str else None

            key = (event_date, strike, obs_minutes)
            if bid is not None and ask is not None:
                price_lookup[key] = (bid, ask)
            if label is not None:
                label_lookup[key] = label

            rows.append({
                "event_date": event_date,
                "strike": strike,
                "obs_minutes": obs_minutes,
                "bid": bid,
                "ask": ask,
                "p_physical": p_physical,
                "label": label,
                "market_price": market_price,
            })

    logger.info(f"加载 {len(rows)} 行, {len(price_lookup)} 条价格, "
                f"{len(label_lookup)} 条标签")
    return price_lookup, label_lookup, rows


def find_trade_signals(
    rows: List[dict],
    threshold: float = 0.03,
) -> List[TradeSignal]:
    """
    逐行判断入场信号（与 simulate_portfolio 逻辑一致）

    YES: p_physical - ask > threshold
    NO (elif): bid - p_physical > threshold
    """
    signals: List[TradeSignal] = []

    for row in rows:
        p = row.get("p_physical")
        bid = row.get("bid")
        ask = row.get("ask")
        label = row.get("label")

        if p is None or bid is None or ask is None or label is None:
            continue
        if bid <= 0 or ask <= 0 or bid == ask:
            continue
        if bid <= 0.01 or ask >= 0.99:
            continue

        edge_yes = p - ask
        edge_no = bid - p

        if edge_yes > threshold:
            signals.append(TradeSignal(
                event_date=row["event_date"],
                obs_minutes=row["obs_minutes"],
                strike=row["strike"],
                direction="YES",
                entry_ask=ask,
                entry_bid=bid,
                edge=edge_yes,
                label=label,
            ))
        elif edge_no > threshold:
            signals.append(TradeSignal(
                event_date=row["event_date"],
                obs_minutes=row["obs_minutes"],
                strike=row["strike"],
                direction="NO",
                entry_ask=ask,
                entry_bid=bid,
                edge=edge_no,
                label=label,
            ))

    logger.info(f"发现 {len(signals)} 个入场信号 (threshold={threshold})")
    return signals


def compute_exit_pnl(
    signal: TradeSignal,
    price_lookup: Dict[Tuple[str, float, int], Tuple[float, float]],
    delta_minutes: int,
) -> Optional[float]:
    """
    计算中途退出 PnL (per share)

    退出时间 = obs_minutes - delta_minutes（更接近结算 = 时间更晚）
    YES: SELL YES at exit_bid → PnL = exit_bid - entry_ask
    NO: SELL NO at (1 - exit_ask) → PnL = entry_bid - exit_ask
    """
    exit_obs = signal.obs_minutes - delta_minutes
    if exit_obs < 0:
        return None  # 距结算不足，无法持有这么久

    key = (signal.event_date, signal.strike, exit_obs)
    exit_data = price_lookup.get(key)
    if exit_data is None:
        return None
    exit_bid, exit_ask = exit_data
    if exit_bid is None or exit_ask is None:
        return None
    if exit_bid <= 0 or exit_ask <= 0 or exit_bid == exit_ask:
        return None

    if signal.direction == "YES":
        return exit_bid - signal.entry_ask
    else:  # NO
        return signal.entry_bid - exit_ask


def compute_settlement_pnl(signal: TradeSignal) -> float:
    """
    持有到结算的 PnL (per share)

    YES: label - entry_ask (赢得 $1 或 $0, 减去买入成本)
    NO: (1 - label) - (1 - entry_bid) (赢得 $1 或 $0, 减去买入成本)
    """
    if signal.direction == "YES":
        return signal.label - signal.entry_ask
    else:  # NO
        return (1 - signal.label) - (1 - signal.entry_bid)


def compute_exit_mid_drift(
    signal: TradeSignal,
    price_lookup: Dict[Tuple[str, float, int], Tuple[float, float]],
    delta_minutes: int,
) -> Optional[float]:
    """
    计算中途退出的 mid-price 漂移（不含 spread 成本）

    YES: exit_mid - entry_mid（正=市场认同）
    NO: entry_mid - exit_mid（正=市场认同）
    """
    exit_obs = signal.obs_minutes - delta_minutes
    if exit_obs < 0:
        return None

    key = (signal.event_date, signal.strike, exit_obs)
    exit_data = price_lookup.get(key)
    if exit_data is None:
        return None
    exit_bid, exit_ask = exit_data
    if exit_bid is None or exit_ask is None:
        return None
    if exit_bid <= 0 or exit_ask <= 0 or exit_bid == exit_ask:
        return None

    entry_mid = (signal.entry_bid + signal.entry_ask) / 2
    exit_mid = (exit_bid + exit_ask) / 2

    if signal.direction == "YES":
        return exit_mid - entry_mid
    else:  # NO
        return entry_mid - exit_mid


def compute_settlement_mid_drift(signal: TradeSignal) -> float:
    """
    持有到结算的 mid-price 漂移

    YES: label - entry_mid（正=市场认同）
    NO: entry_mid - label（正=市场认同）
    """
    entry_mid = (signal.entry_bid + signal.entry_ask) / 2
    if signal.direction == "YES":
        return signal.label - entry_mid
    else:  # NO
        return entry_mid - signal.label


def aggregate_holding_period(
    signals: List[TradeSignal],
    pnls: List[Optional[float]],
    holding_name: str,
    holding_minutes: int,
    shares_per_trade: int = 200,
    mid_drifts: Optional[List[Optional[float]]] = None,
) -> HoldingPeriodResult:
    """
    从 (signal, pnl) 对聚合统计 → HoldingPeriodResult

    pnls 中 None 表示该信号无退出数据（不计入 n_with_exit）
    """
    valid_pnls = [p for p in pnls if p is not None]
    valid_signals = [s for s, p in zip(signals, pnls) if p is not None]

    n_signals = len(signals)
    n_with_exit = len(valid_pnls)

    if n_with_exit == 0:
        return HoldingPeriodResult(
            holding=holding_name,
            holding_minutes=holding_minutes,
            n_signals=n_signals,
            n_with_exit=0,
            n_wins=0,
            n_losses=0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            payoff_ratio=0.0,
            ev_per_trade=0.0,
            avg_pnl=0.0,
            total_pnl=0.0,
            median_pnl=0.0,
            avg_edge=0.0,
        )

    pnl_arr = np.array(valid_pnls)
    wins = pnl_arr[pnl_arr > 0]
    losses = pnl_arr[pnl_arr <= 0]

    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = n_wins / n_with_exit

    avg_win = float(np.mean(wins)) if n_wins > 0 else 0.0
    avg_loss = float(np.mean(np.abs(losses))) if n_losses > 0 else 0.0

    if avg_loss > 0:
        payoff_ratio = avg_win / avg_loss
    else:
        payoff_ratio = float("inf") if avg_win > 0 else 0.0

    avg_pnl = float(np.mean(pnl_arr))
    ev_per_trade = win_rate * avg_win - (1 - win_rate) * avg_loss
    total_pnl = avg_pnl * n_with_exit * shares_per_trade
    median_pnl = float(np.median(pnl_arr))
    avg_edge = float(np.mean([s.edge for s in valid_signals]))

    # mid-drift 统计
    n_favorable = 0
    favorable_rate = 0.0
    avg_mid_drift = 0.0
    median_mid_drift = 0.0
    if mid_drifts is not None:
        valid_drifts = [d for d in mid_drifts if d is not None]
        if valid_drifts:
            drift_arr = np.array(valid_drifts)
            n_favorable = int(np.sum(drift_arr > 0))
            favorable_rate = n_favorable / len(valid_drifts)
            avg_mid_drift = float(np.mean(drift_arr))
            median_mid_drift = float(np.median(drift_arr))

    return HoldingPeriodResult(
        holding=holding_name,
        holding_minutes=holding_minutes,
        n_signals=n_signals,
        n_with_exit=n_with_exit,
        n_wins=n_wins,
        n_losses=n_losses,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        payoff_ratio=payoff_ratio,
        ev_per_trade=ev_per_trade,
        avg_pnl=avg_pnl,
        total_pnl=total_pnl,
        median_pnl=median_pnl,
        avg_edge=avg_edge,
        n_favorable=n_favorable,
        favorable_rate=favorable_rate,
        avg_mid_drift=avg_mid_drift,
        median_mid_drift=median_mid_drift,
    )


def run_convergence(
    csv_path: str,
    threshold: float = 0.03,
    shares_per_trade: int = 200,
    holdings: Optional[List[Tuple[str, int]]] = None,
) -> ConvergenceResult:
    """
    完整收敛分析流程

    1. 加载 detail.csv → price_lookup
    2. 寻找入场信号
    3. 对每个持有期计算 PnL 并聚合
    4. 拆分 YES / NO 方向
    """
    if holdings is None:
        holdings = HOLDINGS

    # 加载数据
    price_lookup, label_lookup, rows = load_price_lookup(csv_path)

    # 寻找入场信号
    signals = find_trade_signals(rows, threshold)
    if not signals:
        logger.warning("未发现任何入场信号")
        return ConvergenceResult(
            n_total_signals=0,
            entry_threshold=threshold,
            shares_per_trade=shares_per_trade,
        )

    # 方向拆分
    yes_signals = [s for s in signals if s.direction == "YES"]
    no_signals = [s for s in signals if s.direction == "NO"]

    def _compute_for_signals(
        sigs: List[TradeSignal],
    ) -> List[HoldingPeriodResult]:
        results = []
        for name, minutes in holdings:
            if minutes == -1:
                # 持有到结算
                pnls = [compute_settlement_pnl(s) for s in sigs]
                drifts = [compute_settlement_mid_drift(s) for s in sigs]
            else:
                pnls = [
                    compute_exit_pnl(s, price_lookup, minutes) for s in sigs
                ]
                drifts = [
                    compute_exit_mid_drift(s, price_lookup, minutes)
                    for s in sigs
                ]
            result = aggregate_holding_period(
                sigs, pnls, name, minutes, shares_per_trade,
                mid_drifts=drifts,
            )
            results.append(result)
        return results

    all_results = _compute_for_signals(signals)
    yes_results = _compute_for_signals(yes_signals) if yes_signals else []
    no_results = _compute_for_signals(no_signals) if no_signals else []

    logger.info(f"收敛分析完成: {len(signals)} 信号 "
                f"(YES={len(yes_signals)}, NO={len(no_signals)})")

    return ConvergenceResult(
        all_results=all_results,
        yes_results=yes_results,
        no_results=no_results,
        n_total_signals=len(signals),
        entry_threshold=threshold,
        shares_per_trade=shares_per_trade,
    )
