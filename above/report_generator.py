"""
Above 合约回测报告生成器

从回测观测结果生成 Markdown 报告:
摘要、Brier/LogLoss、Murphy 分解、校准表、逐日 PnL、逐 K 表现、交易模拟
"""

import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from backtest.metrics import brier_score, log_loss
from pricing_core.execution import compute_opinion_fee, shrink_probability

from .models import AboveBacktestConfig, AboveObservation

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------


def _murphy_decomposition(
    preds: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> Dict[str, float]:
    """Murphy/Brier 分解: Brier = Reliability - Resolution + Uncertainty"""
    n = len(preds)
    if n == 0:
        return {"uncertainty": 0.0, "reliability": 0.0, "resolution": 0.0}
    base_rate = float(np.mean(labels))
    uncertainty = base_rate * (1 - base_rate)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    reliability = 0.0
    resolution = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (preds >= lo) & (preds <= hi)
        else:
            mask = (preds >= lo) & (preds < hi)
        nk = int(np.sum(mask))
        if nk == 0:
            continue
        pk = float(np.mean(preds[mask]))
        ok = float(np.mean(labels[mask]))
        reliability += nk * (pk - ok) ** 2
        resolution += nk * (ok - base_rate) ** 2

    reliability /= n
    resolution /= n

    return {
        "uncertainty": uncertainty,
        "reliability": reliability,
        "resolution": resolution,
    }


def _collect_predictions(
    observations: List[AboveObservation],
) -> Dict[str, np.ndarray]:
    """
    从观测中收集模型预测、市场价格、标签

    Returns:
        {"model_preds", "market_preds", "labels", "strikes", "timestamps", "dates"}
    """
    model_p, market_p, lbls = [], [], []
    strikes_list, ts_list, date_list = [], [], []

    for obs in observations:
        for K in obs.k_grid:
            if K not in obs.predictions or K not in obs.labels:
                continue
            model_p.append(obs.predictions[K])
            lbls.append(obs.labels[K])
            strikes_list.append(K)
            ts_list.append(obs.obs_utc_ms)
            date_list.append(obs.event_date)

            if K in obs.market_prices:
                market_p.append(obs.market_prices[K])
            else:
                market_p.append(float("nan"))

    return {
        "model_preds": np.array(model_p),
        "market_preds": np.array(market_p),
        "labels": np.array(lbls),
        "strikes": np.array(strikes_list),
        "timestamps": np.array(ts_list),
        "dates": date_list,
    }


def _simulate_trades(
    observations: List[AboveObservation],
    threshold: float,
    shrinkage: float = 0.6,
    shares: int = 200,
) -> Dict:
    """
    模拟交易: 每个 K 每小时最多交易1次

    Returns:
        {"n_trades", "n_yes", "n_no", "win_rate", "pnl_per_share",
         "total_pnl", "yes_pnl", "no_pnl", "trades": [...]}
    """
    trades = []
    last_trade_hour: Dict[str, int] = {}  # "date|strike" -> hour

    for obs in observations:
        for K in obs.k_grid:
            if K not in obs.market_prices or K not in obs.predictions:
                continue
            if K not in obs.labels:
                continue

            # 限频: 每 K 每小时最多 1 笔
            key = f"{obs.event_date}|{K}"
            hour = obs.obs_utc_ms // 3_600_000
            if last_trade_hour.get(key) == hour:
                continue

            mp = obs.market_prices[K]
            p_model = obs.predictions[K]
            p_trade = shrink_probability(p_model, mp, shrinkage)
            edge = p_trade - mp
            label = obs.labels[K]

            if abs(edge) <= threshold:
                continue

            last_trade_hour[key] = hour

            if edge > 0:
                # BUY YES
                fee = compute_opinion_fee(mp)
                pnl = (1.0 if label == 1 else 0.0) - mp - fee
                trades.append({
                    "date": obs.event_date, "strike": K,
                    "direction": "YES", "entry": mp,
                    "pnl": pnl, "edge": edge, "fee": fee,
                })
            else:
                # BUY NO
                entry_no = 1.0 - mp
                fee = compute_opinion_fee(entry_no)
                pnl = (1.0 if label == 0 else 0.0) - entry_no - fee
                trades.append({
                    "date": obs.event_date, "strike": K,
                    "direction": "NO", "entry": entry_no,
                    "pnl": pnl, "edge": -edge, "fee": fee,
                })

    if not trades:
        return {
            "n_trades": 0, "n_yes": 0, "n_no": 0,
            "win_rate": 0, "pnl_per_share": 0, "total_pnl": 0,
            "yes_pnl": 0, "no_pnl": 0, "trades": [],
        }

    n = len(trades)
    yes_trades = [t for t in trades if t["direction"] == "YES"]
    no_trades = [t for t in trades if t["direction"] == "NO"]
    wins = [t for t in trades if t["pnl"] > 0]

    total_pnl = sum(t["pnl"] for t in trades) * shares
    yes_pnl = sum(t["pnl"] for t in yes_trades) * shares
    no_pnl = sum(t["pnl"] for t in no_trades) * shares

    return {
        "n_trades": n,
        "n_yes": len(yes_trades),
        "n_no": len(no_trades),
        "win_rate": len(wins) / n,
        "pnl_per_share": sum(t["pnl"] for t in trades) / n,
        "total_pnl": total_pnl,
        "yes_pnl": yes_pnl,
        "no_pnl": no_pnl,
        "trades": trades,
    }


# ------------------------------------------------------------------
# 报告生成
# ------------------------------------------------------------------


def generate_report(
    observations: List[AboveObservation],
    config: AboveBacktestConfig,
    output_path: Optional[str] = None,
) -> str:
    """
    生成完整回测报告

    Args:
        observations: 回测观测结果
        config: 回测配置
        output_path: 输出路径（默认自动生成）

    Returns:
        生成的报告文件路径
    """
    if output_path is None:
        os.makedirs(config.output_dir, exist_ok=True)
        output_path = os.path.join(
            config.output_dir,
            f"report_{config.start_date}_{config.end_date}.md",
        )

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # 收集数据
    data = _collect_predictions(observations)
    model_preds = data["model_preds"]
    market_preds = data["market_preds"]
    labels = data["labels"]
    strikes_arr = data["strikes"]

    # 有市场价格的子集
    has_market = ~np.isnan(market_preds)
    mp_preds = market_preds[has_market]
    mp_labels = labels[has_market]
    mp_model = model_preds[has_market]

    # 基本统计
    all_dates = sorted(set(obs.event_date for obs in observations))
    all_strikes = sorted(set(strikes_arr))

    # 指标计算
    if len(mp_model) == 0:
        model_brier = market_brier = model_ll = market_ll = 0.0
        base_rate = naive_brier = model_bss = market_bss = 0.0
    else:
        model_brier = brier_score(mp_model, mp_labels)
        market_brier = brier_score(mp_preds, mp_labels)
        model_ll = log_loss(mp_model, mp_labels)
        market_ll = log_loss(mp_preds, mp_labels)
        base_rate = float(np.mean(mp_labels))
        naive_brier = base_rate * (1 - base_rate)
        model_bss = 1 - model_brier / naive_brier if naive_brier > 0 else 0
        market_bss = 1 - market_brier / naive_brier if naive_brier > 0 else 0

    model_murphy = _murphy_decomposition(mp_model, mp_labels)
    market_murphy = _murphy_decomposition(mp_preds, mp_labels)

    # DVOL 范围
    sigmas = [obs.sigma for obs in observations if obs.sigma > 0]
    sigma_min = min(sigmas) if sigmas else config.default_sigma
    sigma_max = max(sigmas) if sigmas else config.default_sigma

    # === 生成报告 ===
    lines: List[str] = []
    w = lines.append

    w("# Above 合约回测报告")
    w(f"**期间**: {config.start_date} ~ {config.end_date} | **生成时间**: {now_str}")
    w("")
    w("---")
    w("")

    # 1. 执行摘要
    w("## 1. 执行摘要")
    w("")
    w("| 指标 | 模型 | 市场 | 优势方 |")
    w("|------|------|------|--------|")

    def _winner(m: float, mk: float, lower_better: bool = True) -> str:
        if lower_better:
            return "模型" if m < mk else ("市场" if mk < m else "平")
        return "模型" if m > mk else ("市场" if mk > m else "平")

    brier_pct = (1 - model_brier / market_brier) * 100 if market_brier > 0 else 0
    w(f"| Brier Score | {model_brier:.6f} | {market_brier:.6f} "
      f"| {_winner(model_brier, market_brier)} ({brier_pct:.1f}% 优) |")
    w(f"| Log Loss | {model_ll:.6f} | {market_ll:.6f} "
      f"| {_winner(model_ll, market_ll)} |")
    w(f"| BSS | {model_bss:.4f} | {market_bss:.4f} "
      f"| {_winner(model_bss, market_bss, False)} |")
    w(f"| Reliability | {model_murphy['reliability']:.6f} "
      f"| {market_murphy['reliability']:.6f} "
      f"| {_winner(model_murphy['reliability'], market_murphy['reliability'])} |")
    w(f"| Resolution | {model_murphy['resolution']:.6f} "
      f"| {market_murphy['resolution']:.6f} "
      f"| {_winner(model_murphy['resolution'], market_murphy['resolution'], False)} |")
    w("")

    # 交易摘要
    trade_sim = _simulate_trades(
        observations, config.entry_threshold,
        config.shrinkage_lambda, config.shares_per_trade,
    )
    if trade_sim["n_trades"] > 0:
        w(f"**交易表现** (edge > {config.entry_threshold:.0%}): "
          f"{trade_sim['n_trades']} 笔交易, "
          f"胜率 {trade_sim['win_rate']:.1%}, "
          f"PnL/份 +${trade_sim['pnl_per_share']:.4f}, "
          f"总 PnL ${trade_sim['total_pnl']:,.0f} "
          f"({config.shares_per_trade}份/笔)")
    w("")

    # 2. 模型描述
    w("## 2. 模型描述")
    w("")
    w("### 2.1 定价模型")
    w("Above 合约 (BTC above K at ET noon) 解析公式，GBM 假设下:")
    w("```")
    w("P(S_T > K) = Phi(d2)")
    w("d2 = (ln(S0/K) + (mu - sigma^2/2)*T) / (sigma*sqrt(T))")
    w("```")
    w("")
    w("### 2.2 参数设置")
    w("| 参数 | 值 | 说明 |")
    w("|------|------|------|")
    w(f"| mu (漂移) | {config.mu} | 零漂移（保守） |")
    w(f"| sigma (波动率) | DVOL x vrp_k | |")
    w(f"| VRP k | {config.vrp_k} | Q->P 缩放 |")
    w(f"| 观测间隔 | {config.step_minutes} min | |")
    w(f"| lookback | {config.lookback_hours} h | |")
    w(f"| DVOL 范围 | {sigma_min:.4f} ~ {sigma_max:.4f} | 年化 |")
    w("")

    # 3. 数据概览
    w("## 3. 数据概览")
    w("")
    w("| 项目 | 值 |")
    w("|------|------|")
    w(f"| 回测期间 | {config.start_date} ~ {config.end_date} |")
    w(f"| 天数 | {len(all_dates)} |")
    w(f"| Strike 数量 | {len(all_strikes)} |")
    w(f"| 总观测数 | {len(observations):,} |")
    w(f"| 有市场价格的预测对 | {int(np.sum(has_market)):,} |")
    w(f"| YES 基准率 | {base_rate:.4f} ({base_rate*100:.1f}%) |")
    w("")

    # 3.1 每日结算结果
    w("### 3.1 每日结算结果")
    w("")
    w("| 日期 | 结算价 | Strike 数 | YES 数 |")
    w("|------|--------|----------|--------|")
    day_groups: Dict[str, List[AboveObservation]] = defaultdict(list)
    for obs in observations:
        day_groups[obs.event_date].append(obs)
    for date_str in all_dates:
        day_obs = day_groups[date_str]
        if not day_obs:
            continue
        settlement = day_obs[0].settlement_price
        day_labels = day_obs[0].labels
        n_strikes = len(day_labels)
        n_yes = sum(1 for v in day_labels.values() if v == 1)
        w(f"| {date_str} | ${settlement:,.0f} | {n_strikes} | {n_yes} |")
    w("")

    # 4. 预测表现
    w("## 4. 预测表现")
    w("")
    w("### 4.1 总体指标")
    w("")
    w("| 指标 | 模型 | 市场 | 差值 | 说明 |")
    w("|------|------|------|------|------|")
    w(f"| Brier Score | {model_brier:.6f} | {market_brier:.6f} "
      f"| {model_brier - market_brier:+.6f} | 越低越好 |")
    w(f"| Log Loss | {model_ll:.6f} | {market_ll:.6f} "
      f"| {model_ll - market_ll:+.6f} | 越低越好 |")
    w(f"| BSS | {model_bss:.4f} | {market_bss:.4f} "
      f"| {model_bss - market_bss:+.4f} | 越高越好 (1=完美) |")
    w("")

    # 4.2 Murphy 分解
    w("### 4.2 Murphy 分解")
    w("")
    w("Brier Score = Reliability - Resolution + Uncertainty")
    w("")
    w("| 分量 | 模型 | 市场 | 含义 |")
    w("|------|------|------|------|")
    w(f"| Uncertainty | {model_murphy['uncertainty']:.6f} "
      f"| {market_murphy['uncertainty']:.6f} | 固有不确定性 |")
    w(f"| Reliability | {model_murphy['reliability']:.6f} "
      f"| {market_murphy['reliability']:.6f} | 校准误差 (越低越好) |")
    w(f"| Resolution | {model_murphy['resolution']:.6f} "
      f"| {market_murphy['resolution']:.6f} | 区分能力 (越高越好) |")
    w("")

    # 4.3 校准曲线
    w("### 4.3 校准曲线")
    w("")
    w("| 预测区间 | N | 模型预测均值 | 实际频率 |")
    w("|----------|---|------------|----------|")

    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (model_preds >= lo) & (model_preds <= hi)
        else:
            mask = (model_preds >= lo) & (model_preds < hi)
        nk = int(np.sum(mask))
        if nk == 0:
            continue
        m_mean = float(np.mean(model_preds[mask]))
        m_actual = float(np.mean(labels[mask]))
        w(f"| [{lo:.1f}, {hi:.1f}{']' if i == n_bins - 1 else ')'} "
          f"| {nk:,} | {m_mean:.4f} | {m_actual:.4f} |")
    w("")

    # 5. 逐 Strike 表现
    w("## 5. 逐 Strike 表现")
    w("")
    w("| Strike | N_obs | 标签 YES% | 模型均值 | 市场均值 | Brier(模型) | Brier(市场) |")
    w("|--------|-------|----------|---------|---------|------------|------------|")

    for K in sorted(all_strikes):
        mask_k = strikes_arr == K
        k_labels = labels[mask_k]
        k_model = model_preds[mask_k]
        k_market = market_preds[mask_k]
        n_obs = len(k_labels)
        if n_obs == 0:
            continue
        yes_rate = float(np.mean(k_labels))
        m_mean = float(np.mean(k_model))
        mk_valid = k_market[~np.isnan(k_market)]
        mk_mean = float(np.mean(mk_valid)) if len(mk_valid) > 0 else 0.0
        m_brier = float(np.mean((k_model - k_labels) ** 2))
        mk_brier = (
            float(np.mean((mk_valid - k_labels[~np.isnan(k_market)]) ** 2))
            if len(mk_valid) > 0
            else 0.0
        )
        w(f"| ${K:,.0f} | {n_obs:,} | {yes_rate:.2%} "
          f"| {m_mean:.4f} | {mk_mean:.4f} "
          f"| {m_brier:.6f} | {mk_brier:.6f} |")
    w("")

    # 6. 交易模拟
    w("## 6. 交易模拟")
    w("")
    for thresh in [0.03, 0.05, 0.08]:
        t = _simulate_trades(
            observations, thresh, config.shrinkage_lambda,
            config.shares_per_trade,
        )
        w(f"### Edge > {thresh:.0%}")
        w(f"- 交易数: {t['n_trades']} (YES {t['n_yes']}, NO {t['n_no']})")
        if t["n_trades"] > 0:
            w(f"- 胜率: {t['win_rate']:.1%}")
            w(f"- PnL/份: ${t['pnl_per_share']:.4f}")
            w(f"- 总 PnL: ${t['total_pnl']:,.0f} ({config.shares_per_trade}份/笔)")
        w("")

    # 写入文件
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"报告生成: {output_path}")
    return output_path
