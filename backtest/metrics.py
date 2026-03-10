"""
回测评估指标
Brier score, log loss, calibration, PnL 模拟
"""

import logging
from copy import copy
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 概率截断范围，防止 log(0)
_EPS = 1e-6


def brier_score(preds: np.ndarray, labels: np.ndarray) -> float:
    """
    Brier score = mean((p - y)²)
    越低越好，0 为完美
    """
    return float(np.mean((preds - labels) ** 2))


def log_loss(preds: np.ndarray, labels: np.ndarray) -> float:
    """
    对数损失 = -mean(y·log(p) + (1-y)·log(1-p))
    越低越好
    """
    p = np.clip(preds, _EPS, 1 - _EPS)
    ll = -(labels * np.log(p) + (1 - labels) * np.log(1 - p))
    return float(np.mean(ll))


def calibration_curve(
    preds: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    校准曲线：按预测概率分桶，计算实际频率

    Returns:
        (bin_centers, actual_freq, counts)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    actual_freq = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (preds >= lo) & (preds <= hi)
        else:
            mask = (preds >= lo) & (preds < hi)
        counts[i] = int(np.sum(mask))
        if counts[i] > 0:
            actual_freq[i] = float(np.mean(labels[mask]))

    return bin_centers, actual_freq, counts


def compute_opinion_fee(price: float) -> float:
    """Opinion 手续费率"""
    return 0.06 * price * (1 - price) + 0.0025


def simulate_pnl(
    observations: list,
    shrinkage_lambda: float = 1.0,
    kelly_eta: float = 0.2,
    entry_threshold: float = 0.03,
    default_market_price: float = 0.5,
    fee_rate: float = 0.0,
) -> Dict:
    """
    模拟 PnL

    优先使用 obs.market_prices 中的真实 Polymarket 价格，
    无真实价格时回退到 default_market_price

    对每次观测的每个 K：
    1. p_trade = lambda * p_P + (1-lambda) * market_price (lambda=1.0 时直接用模型价)
    2. edge = p_trade - market_price
    3. 如果 |edge| > threshold → 下注
    4. 盈亏 = position * (payout - cost) - position * fee_rate

    Returns:
        {total_pnl, n_trades, win_rate, avg_edge, trades: [...]}
    """
    trades: List[Dict] = []
    total_pnl = 0.0
    wins = 0

    for obs in observations:
        for k in obs.k_grid:
            p_phys = obs.predictions.get(k)
            label = obs.labels.get(k)
            if p_phys is None or label is None:
                continue

            # 使用真实市场价格（如有），否则回退
            market_price = obs.market_prices.get(k, default_market_price)

            # p_trade（shrinkage_lambda=1.0 时直接用模型概率）
            p_trade = shrinkage_lambda * p_phys + (1 - shrinkage_lambda) * market_price
            p_trade = max(0.001, min(0.999, p_trade))

            edge = p_trade - market_price

            if abs(edge) <= entry_threshold:
                continue

            # Kelly 仓位
            variance = p_trade * (1 - p_trade)
            if variance < 1e-6:
                continue
            position = kelly_eta * abs(edge) / variance
            position = min(position, 1000.0)

            # 方向 & 盈亏
            if edge > 0:
                # BUY YES: 花 market_price, 赢得 1
                cost = market_price
                payout = 1.0 if label == 1 else 0.0
            else:
                # BUY NO: 花 (1 - market_price), 赢得 1
                cost = 1 - market_price
                payout = 1.0 if label == 0 else 0.0

            profit = position * (payout - cost) - position * fee_rate
            total_pnl += profit
            if profit > 0:
                wins += 1

            trades.append({
                "event_date": obs.event_date,
                "obs_minutes": obs.obs_minutes,
                "strike": k,
                "direction": "YES" if edge > 0 else "NO",
                "p_physical": p_phys,
                "p_trade": p_trade,
                "market_price": market_price,
                "edge": edge,
                "position": position,
                "label": label,
                "profit": profit,
            })

    n_trades = len(trades)
    return {
        "total_pnl": total_pnl,
        "n_trades": n_trades,
        "win_rate": wins / n_trades if n_trades > 0 else 0.0,
        "avg_edge": float(np.mean([t["edge"] for t in trades])) if trades else 0.0,
        "trades": trades,
    }


def simulate_portfolio(
    observations: list,
    initial_capital: float = 100_000.0,
    shares_per_trade: int = 200,
    max_net_shares: int = 10_000,
    entry_threshold: float = 0.03,
    direction_filter: str = "both",
    cooldown_minutes: int = 0,
) -> Dict:
    """
    固定份额组合模拟

    逻辑:
    - 每次交易固定 shares_per_trade 份
    - BUY YES: edge = p_model - ask > threshold, 以 ask 价买入
    - BUY NO: edge = bid - p_model > threshold, 以 (1-bid) 价买入
    - 无 bid/ask 时回退到 mid-price
    - 单市场 (event_date, strike) 净仓位 ≤ max_net_shares
    - 事件到期按标签结算
    """
    from collections import defaultdict

    allow_yes = direction_filter != "no_only"
    allow_no = direction_filter != "yes_only"

    equity_curve = [initial_capital]
    event_pnls: List[Dict] = []

    # 按 event_date 分组
    by_date: Dict[str, list] = defaultdict(list)
    for obs in observations:
        by_date[obs.event_date].append(obs)

    # 所有市场详情
    all_markets: List[Dict] = []
    total_cost = 0.0
    total_trades = 0

    for event_date in sorted(by_date.keys()):
        date_obs = sorted(by_date[event_date], key=lambda o: o.now_utc_ms)
        if not date_obs:
            continue

        settlement = date_obs[0].settlement_price

        # 当天各 strike 的持仓
        positions: Dict[float, Dict] = {}
        # 冷却期: 记录每个 strike 最后一次交易的 obs_minutes
        last_trade_minutes: Dict[float, int] = {}

        for obs in date_obs:
            for k in obs.k_grid:
                p_model = obs.predictions.get(k)
                market_price = obs.market_prices.get(k)
                if p_model is None or market_price is None:
                    continue
                if market_price <= 0.01 or market_price >= 0.99:
                    continue

                # 冷却期检查: obs_minutes 递减，elapsed = last - current
                if cooldown_minutes > 0 and k in last_trade_minutes:
                    elapsed = last_trade_minutes[k] - obs.obs_minutes
                    if elapsed < cooldown_minutes:
                        continue

                # 确定 bid/ask 执行价（有订单簿时用 bid/ask，否则回退到 mid）
                ba = obs.market_bid_ask.get(k)
                if ba and ba[0] > 0 and ba[1] > 0:
                    bid, ask = ba[0], ba[1]
                else:
                    bid, ask = market_price, market_price

                if k not in positions:
                    positions[k] = {
                        "yes_shares": 0,
                        "yes_cost": 0.0,
                        "no_shares": 0,
                        "no_cost": 0.0,
                        "trades": [],
                    }
                pos = positions[k]
                net = pos["yes_shares"] - pos["no_shares"]

                # BUY YES: 以 ask 价买入, edge = p_model - ask
                edge_yes = p_model - ask
                if allow_yes and edge_yes > entry_threshold and ask < 0.99:
                    if net + shares_per_trade <= max_net_shares:
                        cost = shares_per_trade * ask
                        pos["yes_shares"] += shares_per_trade
                        pos["yes_cost"] += cost
                        pos["trades"].append({
                            "obs_minutes": obs.obs_minutes,
                            "direction": "YES",
                            "shares": shares_per_trade,
                            "model_price": p_model,
                            "market_price": market_price,
                            "exec_price": ask,
                            "cost": cost,
                        })
                        last_trade_minutes[k] = obs.obs_minutes

                # BUY NO: 以 (1-bid) 价买入, edge = bid - p_model
                elif allow_no and bid - p_model > entry_threshold and bid > 0.01:
                    net_no = pos["no_shares"] - pos["yes_shares"]
                    if net_no + shares_per_trade <= max_net_shares:
                        cost = shares_per_trade * (1.0 - bid)
                        pos["no_shares"] += shares_per_trade
                        pos["no_cost"] += cost
                        pos["trades"].append({
                            "obs_minutes": obs.obs_minutes,
                            "direction": "NO",
                            "shares": shares_per_trade,
                            "model_price": p_model,
                            "market_price": market_price,
                            "exec_price": 1.0 - bid,
                            "cost": cost,
                        })
                        last_trade_minutes[k] = obs.obs_minutes

        # === 结算 ===
        date_pnl = 0.0
        for k, pos in positions.items():
            if pos["yes_shares"] == 0 and pos["no_shares"] == 0:
                continue

            label = 1 if settlement > k else 0
            yes_payout = pos["yes_shares"] * (1.0 if label == 1 else 0.0)
            no_payout = pos["no_shares"] * (1.0 if label == 0 else 0.0)
            pnl = (yes_payout - pos["yes_cost"]) + (no_payout - pos["no_cost"])
            date_pnl += pnl

            market_info = {
                "title": f"BTC > ${k:,.0f} on {event_date}",
                "event_date": event_date,
                "strike": k,
                "settlement": "YES" if label == 1 else "NO",
                "yes_shares": pos["yes_shares"],
                "yes_avg_price": pos["yes_cost"] / pos["yes_shares"] if pos["yes_shares"] > 0 else 0.0,
                "no_shares": pos["no_shares"],
                "no_avg_price": pos["no_cost"] / pos["no_shares"] if pos["no_shares"] > 0 else 0.0,
                "pnl": pnl,
                "trades": pos["trades"],
            }
            all_markets.append(market_info)
            total_cost += pos["yes_cost"] + pos["no_cost"]
            total_trades += len(pos["trades"])

        event_pnls.append({"event_date": event_date, "pnl": date_pnl})
        equity_curve.append(equity_curve[-1] + date_pnl)

    # === 汇总指标 ===
    total_pnl = sum(e["pnl"] for e in event_pnls)
    win_markets = sum(1 for m in all_markets if m["pnl"] > 0)
    lose_markets = sum(1 for m in all_markets if m["pnl"] <= 0)

    gross_profit = sum(m["pnl"] for m in all_markets if m["pnl"] > 0)
    gross_loss = abs(sum(m["pnl"] for m in all_markets if m["pnl"] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # 最大回撤
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    return {
        "initial_capital": initial_capital,
        "total_pnl": total_pnl,
        "total_cost": total_cost,
        "total_return_pct": total_pnl / initial_capital * 100 if initial_capital > 0 else 0.0,
        "n_markets": len(all_markets),
        "n_trades": total_trades,
        "win_markets": win_markets,
        "lose_markets": lose_markets,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_dd * 100,
        "markets": all_markets,
        "equity_curve": equity_curve,
        "event_pnls": event_pnls,
    }


def _bucket_obs_minutes(obs_min: int) -> str:
    """将 obs_minutes 归入时间段桶"""
    if obs_min > 720:
        return "T-24h~12h"
    elif obs_min > 360:
        return "T-12h~6h"
    elif obs_min > 180:
        return "T-6h~3h"
    elif obs_min > 60:
        return "T-3h~1h"
    elif obs_min > 10:
        return "T-1h~10m"
    else:
        return "T-10m~0"


# ---------------------------------------------------------------------------
# Unit 1: 风险指标 (Sharpe / Sortino / Calmar)
# ---------------------------------------------------------------------------


def compute_sharpe(
    event_pnls: list,
    initial_capital: float,
    annualization: int = 365,
) -> Optional[float]:
    """
    基于事件级 PnL 计算 per-event Sharpe ratio（不年化）

    event_pnls: simulate_portfolio 返回的 event_pnls 列表
    initial_capital: 初始资金，用于计算收益率

    返回 per-event Sharpe = mean(r) / std(r)，不做年化。
    """
    if len(event_pnls) < 2:
        return None
    returns = np.array([e["pnl"] / initial_capital for e in event_pnls])
    std = float(np.std(returns, ddof=1))
    if std == 0:
        return None
    mean_r = float(np.mean(returns))
    return mean_r / std


def compute_annualized_sharpe(
    event_pnls: list,
    initial_capital: float,
    min_periods: int = 60,
    annualization: int = 365,
) -> Optional[float]:
    """
    年化 Sharpe，仅当 n_periods >= min_periods 时计算

    样本不足时返回 None，避免短期数据膨胀。
    """
    n = len(event_pnls)
    if n < max(2, min_periods):
        return None
    per_event = compute_sharpe(event_pnls, initial_capital)
    if per_event is None:
        return None
    return per_event * np.sqrt(annualization)


def compute_sortino(
    event_pnls: list,
    initial_capital: float,
    annualization: int = 365,
) -> Optional[float]:
    """
    Per-event Sortino ratio（不年化）

    下行偏差 = sqrt(mean(min(r, 0)²))
    """
    if len(event_pnls) < 2:
        return None
    returns = np.array([e["pnl"] / initial_capital for e in event_pnls])
    downside = np.minimum(returns, 0.0)
    downside_dev = float(np.sqrt(np.mean(downside ** 2)))
    if downside_dev == 0:
        return None
    mean_r = float(np.mean(returns))
    return mean_r / downside_dev


def compute_annualized_sortino(
    event_pnls: list,
    initial_capital: float,
    min_periods: int = 60,
    annualization: int = 365,
) -> Optional[float]:
    """
    年化 Sortino，仅当 n_periods >= min_periods 时计算
    """
    n = len(event_pnls)
    if n < max(2, min_periods):
        return None
    per_event = compute_sortino(event_pnls, initial_capital)
    if per_event is None:
        return None
    return per_event * np.sqrt(annualization)


def compute_calmar(
    total_return_pct: float,
    max_dd_pct: float,
    n_days: int,
    annualization: int = 365,
) -> Optional[float]:
    """
    Calmar ratio = 年化收益率 / 最大回撤

    total_return_pct: 总收益率 (%)
    max_dd_pct: 最大回撤 (%)
    n_days: 回测天数

    当 n_days < 90 时，标注为短期估计（不影响计算，由报告层标注）。
    """
    if max_dd_pct <= 0 or n_days <= 0:
        return None
    annualized_return_pct = total_return_pct / n_days * annualization
    return annualized_return_pct / max_dd_pct


def compute_risk_metrics(portfolio: dict) -> dict:
    """
    便捷封装：从组合模拟结果一次性计算风险指标

    Returns:
        {
            "sharpe": per-event Sharpe (不年化),
            "sortino": per-event Sortino (不年化),
            "calmar": 年化 Calmar,
            "annualized_sharpe": 年化 Sharpe (仅 n>=60 时),
            "annualized_sortino": 年化 Sortino (仅 n>=60 时),
            "n_periods": 事件天数,
            "calmar_short_period": bool, 当 n_days < 90 时为 True,
        }
    """
    event_pnls = portfolio.get("event_pnls", [])
    initial_capital = portfolio.get("initial_capital", 1.0)
    n_days = len(event_pnls)

    sharpe = compute_sharpe(event_pnls, initial_capital)
    sortino = compute_sortino(event_pnls, initial_capital)
    ann_sharpe = compute_annualized_sharpe(event_pnls, initial_capital)
    ann_sortino = compute_annualized_sortino(event_pnls, initial_capital)

    calmar = compute_calmar(
        portfolio.get("total_return_pct", 0.0),
        portfolio.get("max_drawdown_pct", 0.0),
        n_days,
    )

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "annualized_sharpe": ann_sharpe,
        "annualized_sortino": ann_sortino,
        "n_periods": n_days,
        "calmar_short_period": n_days < 90,
    }


# ---------------------------------------------------------------------------
# Unit 2: ECE 和 AUC
# ---------------------------------------------------------------------------


def compute_ece(
    preds: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (期望校准误差)

    利用 calibration_curve() 分桶结果：
    ECE = sum(count_i / N * |bin_center_i - actual_freq_i|)
    只统计非空桶
    """
    n = len(preds)
    if n == 0:
        return 0.0

    centers, actual_freq, counts = calibration_curve(preds, labels, n_bins)
    ece = 0.0
    for c, freq, cnt in zip(centers, actual_freq, counts):
        if cnt == 0:
            continue
        ece += (cnt / n) * abs(c - freq)
    return ece


def compute_auc(
    preds: np.ndarray,
    labels: np.ndarray,
) -> Optional[float]:
    """
    AUC-ROC。如果 sklearn 不可用或标签只有一个类别则返回 None
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        logger.warning("sklearn 未安装，跳过 AUC 计算")
        return None

    unique = np.unique(labels)
    if len(unique) < 2:
        logger.warning("标签只有一个类别 (%s)，无法计算 AUC", unique)
        return None

    return float(roc_auc_score(labels, preds))


# ---------------------------------------------------------------------------
# Unit 3: Edge 五分位统计
# ---------------------------------------------------------------------------


def compute_edge_quintile_stats(portfolio: dict) -> list:
    """
    按 |edge| = |model_price - market_price| 分成 5 个等频桶，统计每桶指标

    Returns:
        长度为 5 的列表，每个元素:
        {quintile, pnl, profit_factor, win_rate, avg_pnl, n_trades}
    """
    # 收集所有交易并计算单笔 PnL
    trade_records: List[Dict] = []

    for market in portfolio.get("markets", []):
        # 由 settlement 推导 label
        label = 1 if market["settlement"] == "YES" else 0

        for t in market.get("trades", []):
            exec_p = t.get("exec_price", t["market_price"])
            abs_edge = abs(t["model_price"] - exec_p)
            # 计算单笔 PnL
            if t["direction"] == "YES":
                pnl = t["shares"] * (1.0 if label == 1 else 0.0) - t["cost"]
            else:
                pnl = t["shares"] * (1.0 if label == 0 else 0.0) - t["cost"]

            trade_records.append({"abs_edge": abs_edge, "pnl": pnl})

    if not trade_records:
        return []

    # 按 |edge| 排序
    trade_records.sort(key=lambda r: r["abs_edge"])

    # 等频分成 5 桶
    n = len(trade_records)
    n_bins = min(5, n)  # 交易数不足 5 笔时减少桶数
    results: List[Dict] = []

    for q in range(n_bins):
        lo = q * n // n_bins
        hi = (q + 1) * n // n_bins
        bucket = trade_records[lo:hi]
        if not bucket:
            continue

        pnls = [r["pnl"] for r in bucket]
        total_pnl = sum(pnls)
        wins = sum(1 for p in pnls if p > 0)
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))

        results.append({
            "quintile": q + 1,
            "pnl": total_pnl,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
            "win_rate": wins / len(bucket),
            "avg_pnl": total_pnl / len(bucket),
            "n_trades": len(bucket),
        })

    return results


# ---------------------------------------------------------------------------
# Unit 4: 对抗/稳健性测试
# ---------------------------------------------------------------------------

_EMPTY_ADVERSARIAL = {
    "pnl": 0, "return_pct": 0, "max_dd_pct": 0, "profit_factor": 0, "survived": False,
}


def run_adversarial_tests(
    observations: list,
    initial_capital: float = 100_000.0,
    shares_per_trade: int = 200,
    max_net_shares: int = 10_000,
    entry_threshold: float = 0.03,
    full_portfolio: Optional[Dict] = None,
    direction_filter: str = "both",
    cooldown_minutes: int = 0,
) -> Dict[str, Dict]:
    """
    三种对抗测试：过滤部分观测后重跑 simulate_portfolio

    1. 去掉 T-10m~0: 过滤 obs_minutes <= 10 的观测
    2. 去掉前 5 市场: 先跑完整 simulation 找 top-5 市场, 过滤对应观测
    3. 去掉最高信号分位: 按 |edge| 找 top-20%, 过滤后重跑

    Args:
        full_portfolio: 已计算的完整组合结果（可选，避免重复 simulate）

    Returns:
        {"no_last_10m": {...}, "no_top5_markets": {...}, "no_top_quintile": {...}}
        每项: {pnl, return_pct, max_dd_pct, profit_factor, survived}
    """
    sim_kwargs = dict(
        initial_capital=initial_capital,
        shares_per_trade=shares_per_trade,
        max_net_shares=max_net_shares,
        entry_threshold=entry_threshold,
        direction_filter=direction_filter,
        cooldown_minutes=cooldown_minutes,
    )

    def _summarize(portfolio: Dict) -> Dict:
        pnl = portfolio["total_pnl"]
        pf = portfolio["profit_factor"]
        return {
            "pnl": pnl,
            "return_pct": portfolio["total_return_pct"],
            "max_dd_pct": portfolio["max_drawdown_pct"],
            "profit_factor": pf,
            "survived": pnl > 0 and (pf > 1.0 if pf != float("inf") else True),
        }

    def _filter_obs_strikes(obs_list: list, exclude_keys: set, key_fn) -> list:
        """过滤掉指定组合的观测"""
        filtered = []
        for o in obs_list:
            remaining_k = [k for k in o.k_grid if key_fn(o, k) not in exclude_keys]
            if remaining_k:
                o2 = copy(o)
                o2.k_grid = remaining_k
                filtered.append(o2)
        return filtered

    def _run_filtered(filtered_obs: list) -> Dict:
        if filtered_obs:
            p = simulate_portfolio(filtered_obs, **sim_kwargs)
            return _summarize(p)
        return _EMPTY_ADVERSARIAL.copy()

    results: Dict[str, Dict] = {}

    # 1. 去掉 T-10m~0
    results["no_last_10m"] = _run_filtered(
        [o for o in observations if o.obs_minutes > 10]
    )

    # 2. 去掉前 5 市场 (by PnL)
    if full_portfolio is None:
        full_portfolio = simulate_portfolio(observations, **sim_kwargs)
    top5_keys = set()
    if full_portfolio["markets"]:
        sorted_markets = sorted(full_portfolio["markets"], key=lambda m: m["pnl"], reverse=True)
        for m in sorted_markets[:5]:
            top5_keys.add((m["event_date"], m["strike"]))

    if top5_keys:
        filtered2 = _filter_obs_strikes(
            observations, top5_keys, lambda o, k: (o.event_date, k)
        )
        results["no_top5_markets"] = _run_filtered(filtered2)
    else:
        results["no_top5_markets"] = _EMPTY_ADVERSARIAL.copy()

    # 3. 去掉最高信号分位 (top 20% by |edge|)
    edge_records = []
    for o in observations:
        for k in o.k_grid:
            p_model = o.predictions.get(k)
            mp = o.market_prices.get(k)
            if p_model is not None and mp is not None:
                edge_records.append((o.event_date, k, o.obs_minutes, abs(p_model - mp)))

    if edge_records:
        edge_records.sort(key=lambda r: r[3], reverse=True)
        cutoff = max(1, len(edge_records) // 5)
        top_edges = set((r[0], r[1], r[2]) for r in edge_records[:cutoff])

        filtered3 = _filter_obs_strikes(
            observations, top_edges, lambda o, k: (o.event_date, k, o.obs_minutes)
        )
        results["no_top_quintile"] = _run_filtered(filtered3)
    else:
        results["no_top_quintile"] = _EMPTY_ADVERSARIAL.copy()

    logger.info(
        f"对抗测试: no_last_10m={results['no_last_10m']['survived']}, "
        f"no_top5={results['no_top5_markets']['survived']}, "
        f"no_top_q={results['no_top_quintile']['survived']}"
    )
    return results


# ---------------------------------------------------------------------------
# Unit 2: 按价格区间分层校准 (ITM/ATM/OTM)
# ---------------------------------------------------------------------------


def compute_price_range_stats(observations: list, portfolio: dict) -> List[Dict]:
    """
    按 moneyness 分 3 组：ITM / ATM / OTM

    ITM: S0 > K * 1.01
    ATM: 0.99*K <= S0 <= 1.01*K
    OTM: S0 < K * 0.99

    返回每组的 n_obs, avg_pred, avg_label, brier, pnl, profit_factor
    """
    groups: Dict[str, List[Tuple[float, int]]] = {"ITM": [], "ATM": [], "OTM": []}

    for obs in observations:
        for k in obs.k_grid:
            p = obs.predictions.get(k)
            y = obs.labels.get(k)
            if p is None or y is None:
                continue
            if obs.s0 > k * 1.01:
                groups["ITM"].append((p, y))
            elif obs.s0 < k * 0.99:
                groups["OTM"].append((p, y))
            else:
                groups["ATM"].append((p, y))

    # 从 portfolio 中提取每个 (event_date, strike) 的 PnL
    market_pnl: Dict[Tuple[str, float], float] = {}
    for m in portfolio.get("markets", []):
        market_pnl[(m["event_date"], m["strike"])] = m["pnl"]

    # 每组按 obs 中 (event_date, strike) 归因 PnL
    group_pnl: Dict[str, float] = {"ITM": 0.0, "ATM": 0.0, "OTM": 0.0}
    group_profit: Dict[str, float] = {"ITM": 0.0, "ATM": 0.0, "OTM": 0.0}
    group_loss: Dict[str, float] = {"ITM": 0.0, "ATM": 0.0, "OTM": 0.0}
    counted_keys: Dict[str, set] = {"ITM": set(), "ATM": set(), "OTM": set()}

    for obs in observations:
        for k in obs.k_grid:
            if obs.s0 > k * 1.01:
                g = "ITM"
            elif obs.s0 < k * 0.99:
                g = "OTM"
            else:
                g = "ATM"
            key = (obs.event_date, k)
            if key not in counted_keys[g]:
                counted_keys[g].add(key)
                pnl = market_pnl.get(key, 0.0)
                group_pnl[g] += pnl
                if pnl > 0:
                    group_profit[g] += pnl
                elif pnl < 0:
                    group_loss[g] += abs(pnl)

    results = []
    for name in ["ITM", "ATM", "OTM"]:
        items = groups[name]
        if not items:
            results.append({
                "range": name, "n_obs": 0, "avg_pred": None,
                "avg_label": None, "brier": None, "pnl": 0.0,
                "profit_factor": None,
            })
            continue
        preds = np.array([p for p, _ in items])
        labels = np.array([y for _, y in items])
        gl = group_loss[name]
        pf = group_profit[name] / gl if gl > 0 else (
            float("inf") if group_profit[name] > 0 else None
        )
        results.append({
            "range": name,
            "n_obs": len(items),
            "avg_pred": float(np.mean(preds)),
            "avg_label": float(np.mean(labels)),
            "brier": float(brier_score(preds, labels)),
            "pnl": group_pnl[name],
            "profit_factor": pf,
        })
    return results


# ---------------------------------------------------------------------------
# Unit 3: 回撤详情
# ---------------------------------------------------------------------------


def compute_drawdown_details(equity_curve: list) -> Dict:
    """
    从权益曲线计算回撤详情

    Returns:
        {
            "max_dd_duration_events": 最大回撤持续事件数(峰→恢复),
            "max_consecutive_losses": 最大连续亏损事件数,
            "avg_dd_duration_events": 平均回撤持续事件数,
            "dd_periods": [{start_idx, end_idx, depth_pct, duration_events}],
        }
    """
    if len(equity_curve) < 2:
        return {
            "max_dd_duration_events": 0,
            "max_consecutive_losses": 0,
            "avg_dd_duration_events": 0.0,
            "dd_periods": [],
        }

    # 回撤期间识别
    dd_periods: List[Dict] = []
    peak = equity_curve[0]
    peak_idx = 0
    in_dd = False
    dd_start = 0
    max_depth = 0.0

    for i in range(1, len(equity_curve)):
        if equity_curve[i] >= peak:
            if in_dd:
                # 回撤结束
                dd_periods.append({
                    "start_idx": dd_start,
                    "end_idx": i,
                    "depth_pct": max_depth * 100,
                    "duration_events": i - dd_start,
                })
                in_dd = False
                max_depth = 0.0
            peak = equity_curve[i]
            peak_idx = i
        else:
            dd = (peak - equity_curve[i]) / peak if peak > 0 else 0.0
            if not in_dd:
                in_dd = True
                dd_start = peak_idx
            if dd > max_depth:
                max_depth = dd

    # 若结束时仍在回撤中
    if in_dd:
        dd_periods.append({
            "start_idx": dd_start,
            "end_idx": len(equity_curve) - 1,
            "depth_pct": max_depth * 100,
            "duration_events": len(equity_curve) - 1 - dd_start,
        })

    # 连续亏损事件数（权益曲线连续下降）
    max_consec = 0
    cur_consec = 0
    for i in range(1, len(equity_curve)):
        if equity_curve[i] < equity_curve[i - 1]:
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 0

    durations = [d["duration_events"] for d in dd_periods]
    return {
        "max_dd_duration_events": max(durations) if durations else 0,
        "max_consecutive_losses": max_consec,
        "avg_dd_duration_events": float(np.mean(durations)) if durations else 0.0,
        "dd_periods": dd_periods,
    }


# ---------------------------------------------------------------------------
# Unit 4: 手续费敏感性分析
# ---------------------------------------------------------------------------


def run_cost_sensitivity(
    observations: list,
    initial_capital: float = 100_000.0,
    shares_per_trade: int = 200,
    max_net_shares: int = 10_000,
    entry_threshold: float = 0.03,
    direction_filter: str = "both",
    cooldown_minutes: int = 0,
) -> List[Dict]:
    """
    不同 fee 乘数下的组合表现

    测试 fee × 0.5, fee × 1.0 (baseline), fee × 2.0
    由于 simulate_portfolio 不含手续费，这里用 post-hoc 方式扣除手续费。

    Returns:
        [{fee_mult, pnl, return_pct, profit_factor}, ...]
    """
    # 先跑基准组合
    portfolio = simulate_portfolio(
        observations,
        initial_capital=initial_capital,
        shares_per_trade=shares_per_trade,
        max_net_shares=max_net_shares,
        entry_threshold=entry_threshold,
        direction_filter=direction_filter,
        cooldown_minutes=cooldown_minutes,
    )

    results = []
    for fee_mult in [0.5, 1.0, 2.0]:
        # 按每笔交易事后扣除手续费
        total_fee = 0.0
        for mkt in portfolio.get("markets", []):
            for t in mkt.get("trades", []):
                mp = t["market_price"]
                fee = compute_opinion_fee(mp) * t["shares"] * fee_mult
                total_fee += fee

        adj_pnl = portfolio["total_pnl"] - total_fee
        adj_return = adj_pnl / initial_capital * 100

        # 调整 profit factor
        gp = sum(m["pnl"] for m in portfolio.get("markets", []) if m["pnl"] > 0)
        gl = abs(sum(m["pnl"] for m in portfolio.get("markets", []) if m["pnl"] < 0))
        # 手续费从 gross profit 中扣除（简化处理）
        adj_gp = max(0, gp - total_fee)
        adj_pf = adj_gp / gl if gl > 0 else (float("inf") if adj_gp > 0 else None)

        results.append({
            "fee_mult": fee_mult,
            "total_fee": total_fee,
            "pnl": adj_pnl,
            "return_pct": adj_return,
            "profit_factor": adj_pf,
        })

    return results


# ---------------------------------------------------------------------------
# Unit 6: 延迟敏感性分析
# ---------------------------------------------------------------------------


def run_latency_sensitivity(
    observations: list,
    initial_capital: float = 100_000.0,
    shares_per_trade: int = 200,
    max_net_shares: int = 10_000,
    entry_threshold: float = 0.03,
    direction_filter: str = "both",
    cooldown_minutes: int = 0,
) -> List[Dict]:
    """
    按 obs_minutes 分组评估不同观测延迟的信号质量

    分组: >60min, 30-60min, 10-30min, <10min
    每组: n_obs, brier, avg_edge, pnl
    """
    buckets = [
        (">60min", lambda m: m > 60),
        ("30-60min", lambda m: 30 <= m <= 60),
        ("10-30min", lambda m: 10 <= m < 30),
        ("<10min", lambda m: m < 10),
    ]

    results = []
    for name, filter_fn in buckets:
        filtered = [o for o in observations if filter_fn(o.obs_minutes)]
        if not filtered:
            results.append({
                "bucket": name, "n_obs": 0, "brier": None,
                "avg_edge": None, "pnl": None,
            })
            continue

        # Brier
        preds, labels = [], []
        edges = []
        for obs in filtered:
            for k in obs.k_grid:
                p = obs.predictions.get(k)
                y = obs.labels.get(k)
                mp = obs.market_prices.get(k)
                if p is not None and y is not None:
                    preds.append(p)
                    labels.append(y)
                if p is not None and mp is not None:
                    edges.append(abs(p - mp))

        b = float(brier_score(np.array(preds), np.array(labels))) if preds else None

        # PnL
        port = simulate_portfolio(
            filtered,
            initial_capital=initial_capital,
            shares_per_trade=shares_per_trade,
            max_net_shares=max_net_shares,
            entry_threshold=entry_threshold,
            direction_filter=direction_filter,
            cooldown_minutes=cooldown_minutes,
        )

        results.append({
            "bucket": name,
            "n_obs": len(filtered),
            "brier": b,
            "avg_edge": float(np.mean(edges)) if edges else None,
            "pnl": port["total_pnl"],
        })

    return results


# ---------------------------------------------------------------------------
# Unit 7: 容量扩展分析
# ---------------------------------------------------------------------------


def run_capacity_analysis(
    observations: list,
    initial_capital: float = 100_000.0,
    max_net_shares: int = 10_000,
    entry_threshold: float = 0.03,
    direction_filter: str = "both",
    cooldown_minutes: int = 0,
) -> List[Dict]:
    """
    不同 shares_per_trade 规模下的组合表现

    测试: 100, 200 (baseline), 500, 1000

    Returns:
        [{shares_per_trade, pnl, return_pct, max_dd_pct, profit_factor}, ...]
    """
    results = []
    for spt in [100, 200, 500, 1000]:
        port = simulate_portfolio(
            observations,
            initial_capital=initial_capital,
            shares_per_trade=spt,
            max_net_shares=max_net_shares,
            entry_threshold=entry_threshold,
            direction_filter=direction_filter,
            cooldown_minutes=cooldown_minutes,
        )
        results.append({
            "shares_per_trade": spt,
            "pnl": port["total_pnl"],
            "return_pct": port["total_return_pct"],
            "max_dd_pct": port["max_drawdown_pct"],
            "profit_factor": port["profit_factor"],
        })
    return results


def compute_direction_analysis(
    observations: list,
    entry_threshold: float = 0.03,
    n_bootstrap: int = 5000,
) -> Dict:
    """
    交易时点条件预测能力分析

    按 |p - mp| > entry_threshold 分为 traded / not_traded，
    再按方向拆分 BUY YES / BUY NO，计算条件 Brier/LogLoss、
    经济学指标、市场级别方向正确率、Bootstrap 方向价值检验。
    """
    traded_preds, traded_labels = [], []
    traded_market, traded_market_labels = [], []
    not_traded_preds, not_traded_labels = [], []
    not_traded_market, not_traded_market_labels = [], []

    # BUY YES / BUY NO 拆分
    buy_yes_costs, buy_yes_labels_list = [], []
    buy_yes_models = []
    buy_no_costs, buy_no_labels_list = [], []
    buy_no_models = []

    # 市场级别方向正确率（按 event_date + strike 去重）
    market_level_yes: Dict[str, List[int]] = {}  # key -> [label]
    market_level_no: Dict[str, List[int]] = {}

    # 所有 traded 的 PnL 明细（用于 bootstrap）
    traded_pnl_items: List[float] = []
    n_with_market_price = 0

    for obs in observations:
        for k in obs.k_grid:
            p = obs.predictions.get(k)
            mp = obs.market_prices.get(k)
            y = obs.labels.get(k)
            if p is None or y is None:
                continue

            if mp is None:
                continue
            n_with_market_price += 1

            edge = abs(p - mp)
            if edge > entry_threshold:
                # traded
                traded_preds.append(p)
                traded_labels.append(y)
                traded_market.append(mp)
                traded_market_labels.append(y)

                mkt_key = f"{obs.event_date}_{k}"

                if p > mp:
                    # BUY YES
                    buy_yes_costs.append(mp)
                    buy_yes_labels_list.append(y)
                    buy_yes_models.append(p)
                    pnl_item = y - mp  # label(1 or 0) - cost
                    traded_pnl_items.append(pnl_item)
                    if mkt_key not in market_level_yes:
                        market_level_yes[mkt_key] = []
                    market_level_yes[mkt_key].append(y)
                else:
                    # BUY NO
                    buy_no_costs.append(1 - mp)
                    buy_no_labels_list.append(y)
                    buy_no_models.append(1 - p)
                    pnl_item = mp - y  # (1 - label) - (1 - mp) = mp - label
                    traded_pnl_items.append(pnl_item)
                    if mkt_key not in market_level_no:
                        market_level_no[mkt_key] = []
                    market_level_no[mkt_key].append(y)
            else:
                # not_traded
                not_traded_preds.append(p)
                not_traded_labels.append(y)
                not_traded_market.append(mp)
                not_traded_market_labels.append(y)

    # 空结果
    if n_with_market_price == 0:
        return {"n_traded": 0, "n_not_traded": 0, "n_with_market_price": 0}

    result: Dict = {
        "n_traded": len(traded_preds),
        "n_not_traded": len(not_traded_preds),
        "n_with_market_price": n_with_market_price,
    }

    # --- 条件 Brier / LogLoss ---
    if traded_preds:
        tp = np.array(traded_preds)
        tl = np.array(traded_labels)
        tm = np.array(traded_market)
        tml = np.array(traded_market_labels)
        result["traded_model_brier"] = brier_score(tp, tl)
        result["traded_market_brier"] = brier_score(tm, tml)
        result["traded_model_logloss"] = log_loss(tp, tl)
        result["traded_market_logloss"] = log_loss(tm, tml)

    if not_traded_preds:
        ntp = np.array(not_traded_preds)
        ntl = np.array(not_traded_labels)
        ntm = np.array(not_traded_market)
        ntml = np.array(not_traded_market_labels)
        result["not_traded_model_brier"] = brier_score(ntp, ntl)
        result["not_traded_market_brier"] = brier_score(ntm, ntml)

    # --- BUY YES 经济学 ---
    if buy_yes_costs:
        costs_arr = np.array(buy_yes_costs)
        labels_arr = np.array(buy_yes_labels_list)
        models_arr = np.array(buy_yes_models)
        result["buy_yes"] = {
            "n": len(buy_yes_costs),
            "avg_cost": float(np.mean(costs_arr)),
            "avg_model": float(np.mean(models_arr)),
            "win_rate": float(np.mean(labels_arr)),  # label=1 即胜
            "breakeven": float(np.mean(costs_arr)),
            "pnl_per_share": float(np.mean(labels_arr - costs_arr)),
        }
    else:
        result["buy_yes"] = {"n": 0, "avg_cost": 0.0, "avg_model": 0.0,
                             "win_rate": 0.0, "breakeven": 0.0, "pnl_per_share": 0.0}

    # --- BUY NO 经济学 ---
    if buy_no_costs:
        costs_arr = np.array(buy_no_costs)
        labels_arr = np.array(buy_no_labels_list)
        models_arr = np.array(buy_no_models)
        win_rate = float(np.mean(labels_arr == 0))  # label=0 即 NO 赢
        # PnL per share = mean(mp - label) = mean((1 - cost) - label)
        pnl_items = np.array(buy_no_labels_list, dtype=float)
        mp_arr = 1.0 - costs_arr  # 还原 market_price
        result["buy_no"] = {
            "n": len(buy_no_costs),
            "avg_cost": float(np.mean(costs_arr)),
            "avg_model": float(np.mean(models_arr)),
            "win_rate": win_rate,
            "breakeven": float(np.mean(costs_arr)),
            "pnl_per_share": float(np.mean(mp_arr - pnl_items)),
        }
    else:
        result["buy_no"] = {"n": 0, "avg_cost": 0.0, "avg_model": 0.0,
                            "win_rate": 0.0, "breakeven": 0.0, "pnl_per_share": 0.0}

    # --- 市场级别方向正确率 ---
    # BUY YES 方向：市场级别看最终 label 是否 =1
    yes_total = len(market_level_yes)
    yes_correct = sum(1 for labels in market_level_yes.values()
                      if np.mean(labels) > 0.5)
    result["market_level_yes_total"] = yes_total
    result["market_level_yes_correct"] = yes_correct
    result["market_level_yes_accuracy"] = yes_correct / yes_total if yes_total > 0 else 0.0

    no_total = len(market_level_no)
    no_correct = sum(1 for labels in market_level_no.values()
                     if np.mean(labels) < 0.5)
    result["market_level_no_total"] = no_total
    result["market_level_no_correct"] = no_correct
    result["market_level_no_accuracy"] = no_correct / no_total if no_total > 0 else 0.0

    # --- Bootstrap 方向价值检验 ---
    if traded_pnl_items:
        real_pnl_sum = float(np.sum(traded_pnl_items))

        # 反向交易 PnL（翻转每笔交易方向）
        reverse_pnl = float(-np.sum(traded_pnl_items))

        # Bootstrap：在 traded 时点随机分配方向
        rng = np.random.default_rng(42)
        pnl_arr = np.array(traded_pnl_items)
        random_pnls = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            # 随机翻转每笔交易方向（50% 概率保持 / 50% 翻转）
            signs = rng.choice([-1, 1], size=len(pnl_arr))
            random_pnls[i] = float(np.sum(signs * pnl_arr))

        random_mean = float(np.mean(random_pnls))
        ci_lower = float(np.percentile(random_pnls, 2.5))
        ci_upper = float(np.percentile(random_pnls, 97.5))
        # p_value = 模型 PnL 优于随机的概率
        p_value = float(np.mean(random_pnls >= real_pnl_sum))

        result["bootstrap"] = {
            "real_pnl_per_share_sum": real_pnl_sum,
            "random_mean": random_mean,
            "random_ci_lower": ci_lower,
            "random_ci_upper": ci_upper,
            "p_value": p_value,
            "reverse_pnl": reverse_pnl,
        }
    else:
        result["bootstrap"] = {
            "real_pnl_per_share_sum": 0.0,
            "random_mean": 0.0,
            "random_ci_lower": 0.0,
            "random_ci_upper": 0.0,
            "p_value": 1.0,
            "reverse_pnl": 0.0,
        }

    return result


def compute_all_metrics(
    observations: list,
    initial_capital: float = 100_000.0,
    shares_per_trade: int = 200,
    max_net_shares: int = 10_000,
    entry_threshold: float = 0.03,
    direction_filter: str = "both",
    cooldown_minutes: int = 0,
) -> Dict:
    """
    汇总所有评估指标，按时间段分组

    Returns:
        {
            "overall": {brier, log_loss, calibration, pnl_summary},
            "by_time_bucket": {"T-24h~12h": {...}, "T-12h~6h": {...}, ...},
        }
    """
    # 收集所有 (pred, label) 对，同时收集市场价格
    all_preds: List[float] = []
    all_labels: List[int] = []
    all_market_preds: List[float] = []
    all_market_labels: List[int] = []
    by_bucket: Dict[str, Tuple[List[float], List[int]]] = {}
    market_by_bucket: Dict[str, Tuple[List[float], List[int]]] = {}

    for obs in observations:
        bucket = _bucket_obs_minutes(obs.obs_minutes)
        if bucket not in by_bucket:
            by_bucket[bucket] = ([], [])
        if bucket not in market_by_bucket:
            market_by_bucket[bucket] = ([], [])

        for k in obs.k_grid:
            p = obs.predictions.get(k)
            y = obs.labels.get(k)
            mp = obs.market_prices.get(k)
            if p is not None and y is not None:
                all_preds.append(p)
                all_labels.append(y)
                by_bucket[bucket][0].append(p)
                by_bucket[bucket][1].append(y)
            if mp is not None and y is not None:
                all_market_preds.append(mp)
                all_market_labels.append(y)
                market_by_bucket[bucket][0].append(mp)
                market_by_bucket[bucket][1].append(y)

    result: Dict = {"overall": {}, "by_time_bucket": {}}

    if all_preds:
        preds_arr = np.array(all_preds)
        labels_arr = np.array(all_labels)
        result["overall"]["brier_score"] = brier_score(preds_arr, labels_arr)
        result["overall"]["log_loss"] = log_loss(preds_arr, labels_arr)
        centers, freq, counts = calibration_curve(preds_arr, labels_arr)
        result["overall"]["calibration"] = {
            "bin_centers": centers.tolist(),
            "actual_freq": freq.tolist(),
            "counts": counts.tolist(),
        }
        result["overall"]["n_observations"] = len(observations)
        result["overall"]["n_predictions"] = len(all_preds)
        result["overall"]["ece"] = compute_ece(preds_arr, labels_arr)
        result["overall"]["auc"] = compute_auc(preds_arr, labels_arr)

    # 市场价格整体指标
    if all_market_preds:
        m_preds_arr = np.array(all_market_preds)
        m_labels_arr = np.array(all_market_labels)
        result["overall"]["market_brier_score"] = brier_score(m_preds_arr, m_labels_arr)
        result["overall"]["market_log_loss"] = log_loss(m_preds_arr, m_labels_arr)
        result["overall"]["market_ece"] = compute_ece(m_preds_arr, m_labels_arr)
        result["overall"]["market_auc"] = compute_auc(m_preds_arr, m_labels_arr)
        result["overall"]["market_n_predictions"] = len(all_market_preds)
        m_centers, m_freq, m_counts = calibration_curve(m_preds_arr, m_labels_arr)
        result["overall"]["market_calibration"] = {
            "bin_centers": m_centers.tolist(),
            "actual_freq": m_freq.tolist(),
            "counts": m_counts.tolist(),
        }

    # 按时间段分组
    bucket_order = ["T-24h~12h", "T-12h~6h", "T-6h~3h", "T-3h~1h", "T-1h~10m", "T-10m~0"]
    for bucket in bucket_order:
        bucket_result: Dict = {}

        # 模型指标
        if bucket in by_bucket:
            preds_list, labels_list = by_bucket[bucket]
            b_preds_arr = np.array(preds_list)
            b_labels_arr = np.array(labels_list)
            bucket_result["brier_score"] = brier_score(b_preds_arr, b_labels_arr)
            bucket_result["log_loss"] = log_loss(b_preds_arr, b_labels_arr)
            bucket_result["n_predictions"] = len(preds_list)

        # 市场指标
        if bucket in market_by_bucket:
            mp_list, ml_list = market_by_bucket[bucket]
            if mp_list:
                mb_preds_arr = np.array(mp_list)
                mb_labels_arr = np.array(ml_list)
                bucket_result["market_brier_score"] = brier_score(mb_preds_arr, mb_labels_arr)
                bucket_result["market_log_loss"] = log_loss(mb_preds_arr, mb_labels_arr)

        if bucket_result:
            result["by_time_bucket"][bucket] = bucket_result

    # 组合模拟
    portfolio = simulate_portfolio(
        observations,
        initial_capital=initial_capital,
        shares_per_trade=shares_per_trade,
        max_net_shares=max_net_shares,
        entry_threshold=entry_threshold,
        direction_filter=direction_filter,
        cooldown_minutes=cooldown_minutes,
    )
    result["overall"]["portfolio"] = portfolio

    # --- 风险指标 ---
    result["overall"]["risk_metrics"] = compute_risk_metrics(portfolio)

    # --- Edge 五分位 ---
    result["overall"]["edge_quintiles"] = compute_edge_quintile_stats(portfolio)

    # --- 对抗测试 ---
    result["overall"]["adversarial"] = run_adversarial_tests(
        observations,
        initial_capital=initial_capital,
        shares_per_trade=shares_per_trade,
        max_net_shares=max_net_shares,
        entry_threshold=entry_threshold,
        full_portfolio=portfolio,
        direction_filter=direction_filter,
        cooldown_minutes=cooldown_minutes,
    )

    # --- 交易时点条件分析 (§4.6) ---
    result["overall"]["direction_analysis"] = compute_direction_analysis(
        observations, entry_threshold=entry_threshold,
    )

    # --- 按价格区间分层 (§4.4) ---
    result["overall"]["price_range_stats"] = compute_price_range_stats(
        observations, portfolio
    )

    # --- 回撤详情 (§5.2) ---
    result["overall"]["drawdown_details"] = compute_drawdown_details(
        portfolio.get("equity_curve", [])
    )

    # --- 手续费敏感性 (§7.3) ---
    result["overall"]["cost_sensitivity"] = run_cost_sensitivity(
        observations,
        initial_capital=initial_capital,
        shares_per_trade=shares_per_trade,
        max_net_shares=max_net_shares,
        entry_threshold=entry_threshold,
        direction_filter=direction_filter,
        cooldown_minutes=cooldown_minutes,
    )

    # --- 延迟敏感性 (§7.2) ---
    result["overall"]["latency_sensitivity"] = run_latency_sensitivity(
        observations,
        initial_capital=initial_capital,
        shares_per_trade=shares_per_trade,
        max_net_shares=max_net_shares,
        entry_threshold=entry_threshold,
        direction_filter=direction_filter,
        cooldown_minutes=cooldown_minutes,
    )

    # --- 容量扩展 (§8.3) ---
    result["overall"]["capacity_analysis"] = run_capacity_analysis(
        observations,
        initial_capital=initial_capital,
        max_net_shares=max_net_shares,
        entry_threshold=entry_threshold,
        direction_filter=direction_filter,
        cooldown_minutes=cooldown_minutes,
    )

    return result
