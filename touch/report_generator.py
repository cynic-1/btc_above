"""
触碰障碍期权回测报告生成器

从回测观测结果生成量化机构标准的 Markdown 报告
"""

import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from backtest.metrics import brier_score, calibration_curve, log_loss
from pricing_core.execution import compute_opinion_fee, shrink_probability

from .models import TouchBacktestConfig, TouchObservationResult

logger = logging.getLogger(__name__)


def _murphy_decomposition(
    preds: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> Dict[str, float]:
    """
    Murphy/Brier 分解: Brier = Reliability - Resolution + Uncertainty

    Returns:
        {"uncertainty": ..., "reliability": ..., "resolution": ...}
    """
    n = len(preds)
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
    observations: List[TouchObservationResult],
    exclude_touched: bool = True,
) -> Dict[str, np.ndarray]:
    """
    从观测中收集模型预测、市场价格、标签

    Returns:
        {"model_preds", "market_preds", "labels", "barriers", "timestamps"}
    """
    model_p, market_p, lbls = [], [], []
    barriers_list, ts_list = [], []

    for obs in observations:
        for barrier in obs.barriers:
            if barrier not in obs.predictions or barrier not in obs.labels:
                continue
            if exclude_touched and obs.already_touched.get(barrier, False):
                continue
            model_p.append(obs.predictions[barrier])
            lbls.append(obs.labels[barrier])
            barriers_list.append(barrier)
            ts_list.append(obs.obs_utc_ms)

            if barrier in obs.market_prices:
                market_p.append(obs.market_prices[barrier])
            else:
                market_p.append(float("nan"))

    return {
        "model_preds": np.array(model_p),
        "market_preds": np.array(market_p),
        "labels": np.array(lbls),
        "barriers": np.array(barriers_list),
        "timestamps": np.array(ts_list),
    }


def _simulate_trades(
    observations: List[TouchObservationResult],
    threshold: float,
    shrinkage: float = 0.6,
    shares: int = 200,
) -> Dict:
    """
    模拟交易: 每个 barrier 每小时最多交易1次

    Returns:
        {"n_trades", "n_yes", "n_no", "win_rate", "pnl_per_share",
         "total_pnl", "yes_pnl", "no_pnl", "trades": [...]}
    """
    trades = []
    last_trade_hour: Dict[float, int] = {}

    for obs in observations:
        for barrier in obs.barriers:
            if barrier not in obs.market_prices or barrier not in obs.predictions:
                continue
            if obs.already_touched.get(barrier, False):
                continue

            # 限频: 每 barrier 每小时最多1笔
            hour = obs.obs_utc_ms // 3_600_000
            if last_trade_hour.get(barrier) == hour:
                continue

            mp = obs.market_prices[barrier]
            p_model = obs.predictions[barrier]
            p_trade = shrink_probability(p_model, mp, shrinkage)
            edge = p_trade - mp
            label = obs.labels[barrier]

            if abs(edge) <= threshold:
                continue

            last_trade_hour[barrier] = hour

            if edge > 0:
                # BUY YES
                fee = compute_opinion_fee(mp)
                pnl = (1.0 if label == 1 else 0.0) - mp - fee
                trades.append({
                    "barrier": barrier, "direction": "YES",
                    "entry": mp, "pnl": pnl, "edge": edge, "fee": fee,
                })
            else:
                # BUY NO
                entry_no = 1.0 - mp
                fee = compute_opinion_fee(entry_no)
                pnl = (1.0 if label == 0 else 0.0) - entry_no - fee
                trades.append({
                    "barrier": barrier, "direction": "NO",
                    "entry": entry_no, "pnl": pnl, "edge": -edge, "fee": fee,
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


def generate_report(
    observations: List[TouchObservationResult],
    config: TouchBacktestConfig,
    output_path: Optional[str] = None,
) -> str:
    """
    生成完整回测报告

    Args:
        observations: 回测观测结果
        config: 回测配置
        output_path: 输出路径（默认 {output_dir}/report_{month}.md）

    Returns:
        生成的报告文件路径
    """
    if output_path is None:
        os.makedirs(config.output_dir, exist_ok=True)
        output_path = os.path.join(
            config.output_dir, f"report_{config.month}.md"
        )

    month = config.month
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # 收集数据
    data = _collect_predictions(observations, exclude_touched=True)
    model_preds = data["model_preds"]
    market_preds = data["market_preds"]
    labels = data["labels"]
    barriers_arr = data["barriers"]
    timestamps = data["timestamps"]

    # 有市场价格的子集
    has_market = ~np.isnan(market_preds)
    mp_preds = market_preds[has_market]
    mp_labels = labels[has_market]
    mp_model = model_preds[has_market]

    # === 基本统计 ===
    all_barriers = sorted(set(barriers_arr))
    first_obs = observations[0]
    last_obs = observations[-1]
    s0_first = first_obs.s0
    s0_last = last_obs.s0

    # 月内价格极值
    all_highs = [obs.running_high for obs in observations]
    all_lows = [obs.running_low for obs in observations]
    month_high = max(all_highs) if all_highs else s0_first
    month_low = min(obs.s0 for obs in observations)

    # 最终标签
    final_labels = observations[-1].labels

    # === 指标计算 ===
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

    # === 开始生成报告 ===
    lines = []
    w = lines.append

    # 标题
    w(f"# 一触碰障碍期权回测报告")
    w(f"**月份**: {month} | **生成时间**: {now_str}")
    w("")
    w("---")
    w("")

    # 1. 执行摘要
    w("## 1. 执行摘要")
    w("")
    w("| 指标 | 模型 | 市场 | 优势方 |")
    w("|------|------|------|--------|")

    def _winner(m, mk, lower_better=True):
        if lower_better:
            return "模型" if m < mk else ("市场" if mk < m else "平")
        return "模型" if m > mk else ("市场" if mk > m else "平")

    brier_pct = (1 - model_brier / market_brier) * 100 if market_brier > 0 else 0
    w(f"| Brier Score | {model_brier:.6f} | {market_brier:.6f} "
      f"| {_winner(model_brier, market_brier)} ({brier_pct:.1f}% 优) |")
    w(f"| Log Loss | {model_ll:.6f} | {market_ll:.6f} "
      f"| {_winner(model_ll, market_ll)} |")
    w(f"| Brier Skill Score | {model_bss:.4f} | {market_bss:.4f} "
      f"| {_winner(model_bss, market_bss, False)} |")
    w(f"| Reliability (校准) | {model_murphy['reliability']:.6f} "
      f"| {market_murphy['reliability']:.6f} "
      f"| {_winner(model_murphy['reliability'], market_murphy['reliability'])} |")
    w(f"| Resolution (区分力) | {model_murphy['resolution']:.6f} "
      f"| {market_murphy['resolution']:.6f} "
      f"| {_winner(model_murphy['resolution'], market_murphy['resolution'], False)} |")
    w("")

    # 交易摘要
    trade_5 = _simulate_trades(observations, 0.05, config.shrinkage_lambda,
                               config.shares_per_trade)
    if trade_5["n_trades"] > 0:
        w(f"**交易表现** (edge > 5%): {trade_5['n_trades']} 笔交易, "
          f"胜率 {trade_5['win_rate']:.1%}, "
          f"PnL/份 +${trade_5['pnl_per_share']:.4f}, "
          f"总 PnL ${trade_5['total_pnl']:,.0f} "
          f"({config.shares_per_trade}份/笔)")
    w("")

    # 2. 模型描述
    w("## 2. 模型描述")
    w("")
    w("### 2.1 定价模型")
    w("一触碰障碍期权 (One-Touch Barrier Option) 解析公式，GBM 假设下:")
    w("```")
    w("P(max S_t ≥ K) = Φ((νT−m)/(σ√T)) + exp(2νm/σ²)·Φ((−νT−m)/(σ√T))")
    w("其中 ν = μ−σ²/2, m = ln(K/S₀)")
    w("```")
    w("")
    w("### 2.2 参数设置")
    w("| 参数 | 值 | 说明 |")
    w("|------|------|------|")
    w(f"| μ (漂移) | {config.mu} | 零漂移（保守） |")
    w(f"| σ (波动率) | DVOL + 期限结构校正 "
      f"| `sigma_adj = dvol * (30/T_days)^{config.term_structure_alpha}` |")
    w(f"| VRP k | {config.vrp_k} | Q→P 缩放 |")
    w(f"| 观测间隔 | {config.step_minutes} min |  |")
    w(f"| DVOL 范围 | {sigma_min:.4f} ~ {sigma_max:.4f} | 年化 |")
    w("")

    # 3. 数据概览
    w("## 3. 数据概览")
    w("")
    w("| 项目 | 值 |")
    w("|------|------|")
    w(f"| 回测期间 | {month}-01 ~ {month}-末 |")
    w(f"| BTC 月初价格 | ${s0_first:,.0f} |")
    w(f"| BTC 月末价格 | ${s0_last:,.0f} "
      f"({(s0_last/s0_first-1)*100:+.1f}%) |")
    w(f"| 月内最高 | ${month_high:,.0f} |")
    w(f"| 月内最低 | ${month_low:,.0f} |")
    w(f"| Barrier 数量 | {len(all_barriers)} |")
    w(f"| 总观测数 | {len(observations):,} ({config.step_minutes}分钟级) |")
    w(f"| 有 orderbook 的预测对 | {int(np.sum(has_market)):,} |")
    w(f"| 排除已触碰后 | {len(mp_labels):,} |")
    w(f"| YES 基准率 | {base_rate:.4f} ({base_rate*100:.1f}%) |")
    w("")

    # 月末触碰结果
    w("### 3.1 月末触碰结果")
    w("")
    w("| Barrier | 方向 | 结果 | 说明 |")
    w("|---------|------|------|------|")
    for barrier in all_barriers:
        label = final_labels.get(barrier, 0)
        direction = "up" if barrier > s0_first else "down"
        result = "YES (触碰)" if label == 1 else "NO"
        note = ""
        if label == 1:
            if direction == "up":
                note = f"BTC 涨破 ${barrier:,.0f}"
            else:
                note = f"BTC 跌破 ${barrier:,.0f}"
        w(f"| ${barrier:,.0f} | {direction} | {result} | {note} |")
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

    # Murphy 分解
    w("### 4.2 Murphy 分解")
    w("")
    w("Brier Score = Reliability − Resolution + Uncertainty")
    w("")
    w("| 分量 | 模型 | 市场 | 含义 |")
    w("|------|------|------|------|")
    w(f"| Uncertainty | {model_murphy['uncertainty']:.6f} "
      f"| {market_murphy['uncertainty']:.6f} | 固有不确定性 (不可改善) |")
    w(f"| Reliability | {model_murphy['reliability']:.6f} "
      f"| {market_murphy['reliability']:.6f} | 校准误差 (越低越好) |")
    w(f"| Resolution | {model_murphy['resolution']:.6f} "
      f"| {market_murphy['resolution']:.6f} | 区分能力 (越高越好) |")
    w("")

    rel_winner = _winner(
        model_murphy["reliability"], market_murphy["reliability"]
    )
    res_winner = _winner(
        model_murphy["resolution"], market_murphy["resolution"], False
    )
    w(f"**结论**: 模型的优势主要来自更好的**"
      f"{'校准' if rel_winner == '模型' else '区分力'}** "
      f"(Reliability {model_murphy['reliability']:.6f} vs "
      f"{market_murphy['reliability']:.6f})，"
      f"区分能力 Resolution {model_murphy['resolution']:.6f} vs "
      f"{market_murphy['resolution']:.6f}。")
    w("")

    # 校准曲线
    w("### 4.3 校准曲线")
    w("")
    w("| 预测区间 | N | 模型预测均值 | 模型实际频率 | 市场预测均值 | 市场实际频率 |")
    w("|----------|---|------------|------------|------------|------------|")

    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (mp_model >= lo) & (mp_model <= hi)
        else:
            mask = (mp_model >= lo) & (mp_model < hi)
        nk = int(np.sum(mask))
        if nk == 0:
            continue
        m_mean = float(np.mean(mp_model[mask]))
        m_actual = float(np.mean(mp_labels[mask]))
        mk_mean = float(np.mean(mp_preds[mask]))
        mk_actual = float(np.mean(mp_labels[mask]))
        w(f"| [{lo:.1f}, {hi:.1f}{']' if i == n_bins - 1 else ')'} "
          f"| {nk:,} | {m_mean:.4f} | {m_actual:.4f} "
          f"| {mk_mean:.4f} | {mk_actual:.4f} |")
    w("")

    # 按概率区间分析
    w("### 4.4 按概率区间分析")
    w("")
    w("| 区间 | N | YES率 | 模型均值 | 市场均值 | 模型Brier | 市场Brier | 优势方 |")
    w("|------|---|-------|---------|---------|----------|----------|--------|")

    prob_bins = [
        ("p<1%", 0.0, 0.01),
        ("1-5%", 0.01, 0.05),
        ("5-15%", 0.05, 0.15),
        ("15-35%", 0.15, 0.35),
        ("35-65%", 0.35, 0.65),
        (">65%", 0.65, 1.01),
    ]
    for name, lo, hi in prob_bins:
        mask = (mp_model >= lo) & (mp_model < hi)
        nk = int(np.sum(mask))
        if nk < 5:
            continue
        yes_rate = float(np.mean(mp_labels[mask]))
        m_mean = float(np.mean(mp_model[mask]))
        mk_mean = float(np.mean(mp_preds[mask]))
        m_brier = brier_score(mp_model[mask], mp_labels[mask])
        mk_brier = brier_score(mp_preds[mask], mp_labels[mask])
        w(f"| {name} | {nk:,} | {yes_rate:.3f} | {m_mean:.4f} "
          f"| {mk_mean:.4f} | {m_brier:.4f} | {mk_brier:.4f} "
          f"| {_winner(m_brier, mk_brier)} |")
    w("")

    # 5. 交易表现
    w("## 5. 交易表现")
    w("")
    w("### 5.1 Edge 盈利分析")
    w("")
    w(f"假设: 模型概率 vs 市场 mid 的差值作为 edge，每 barrier 每小时最多1笔，"
      f"每笔 {config.shares_per_trade} 份")
    w("")
    w("| 阈值 | 交易数 | YES买 | NO买 | 胜率 | PnL/份 | "
      f"总PnL ({config.shares_per_trade}份) | YES_PnL | NO_PnL |")
    w("|------|--------|-------|------|------|--------|"
      "---------------|---------|--------|")

    for thr in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        t = _simulate_trades(
            observations, thr, config.shrinkage_lambda,
            config.shares_per_trade,
        )
        if t["n_trades"] == 0:
            continue
        w(f"| {thr:.0%} | {t['n_trades']:,} | {t['n_yes']:,} | {t['n_no']:,} "
          f"| {t['win_rate']:.1%} | ${t['pnl_per_share']:+.4f} "
          f"| ${t['total_pnl']:+,.0f} | ${t['yes_pnl']:+,.0f} "
          f"| ${t['no_pnl']:+,.0f} |")
    w("")

    # 5.2 Edge 方向拆分 (5% 阈值)
    w("### 5.2 Edge 方向拆分 (5% 阈值)")
    w("")
    if trade_5["trades"]:
        from collections import defaultdict
        dir_stats = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})
        for t in trade_5["trades"]:
            barrier = t["barrier"]
            direction = "up" if barrier > s0_first else "down"
            key = f"{direction} BUY {t['direction']}"
            dir_stats[key]["n"] += 1
            if t["pnl"] > 0:
                dir_stats[key]["wins"] += 1
            dir_stats[key]["pnl"] += t["pnl"]

        w("| 方向 | N | 胜率 | PnL/份 | 说明 |")
        w("|------|---|------|--------|------|")
        for key in sorted(dir_stats.keys()):
            s = dir_stats[key]
            wr = s["wins"] / s["n"] if s["n"] > 0 else 0
            avg_pnl = s["pnl"] / s["n"] if s["n"] > 0 else 0
            note = ""
            if "BUY YES" in key:
                note = "模型高估触碰" if avg_pnl < 0 else "模型正确高估"
            else:
                note = "市场高估触碰" if avg_pnl > 0 else "市场正确高估"
            w(f"| {key} | {s['n']} | {wr:.1%} | ${avg_pnl:+.4f} | {note} |")
    w("")

    # 6. 逐 Barrier 详情
    w("## 6. 逐 Barrier 详情")
    w("")
    w("| Barrier | 方向 | 结果 | N | 模型Brier | 市场Brier "
      "| 模型均值 | 市场均值 | 优势方 |")
    w("|---------|------|------|---|----------|----------"
      "|---------|---------|--------|")

    for barrier in all_barriers:
        mask = (barriers_arr == barrier) & has_market
        nk = int(np.sum(mask))
        if nk < 2:
            continue
        label = final_labels.get(barrier, 0)
        direction = "up" if barrier > s0_first else "down"
        result = "YES" if label == 1 else "NO"
        m_brier = brier_score(model_preds[mask], labels[mask])
        mk_brier = brier_score(market_preds[mask], labels[mask])
        m_mean = float(np.mean(model_preds[mask]))
        mk_mean = float(np.mean(market_preds[mask]))
        winner = _winner(m_brier, mk_brier)
        # 差距很小时标记为平
        if abs(m_brier - mk_brier) < 0.001:
            winner = "平"
        w(f"| ${barrier:,.0f} | {direction} | {result} | {nk:,} "
          f"| {m_brier:.4f} | {mk_brier:.4f} "
          f"| {m_mean:.4f} | {mk_mean:.4f} | {winner} |")
    w("")

    # 7. 时间维度
    w("## 7. 时间维度分析")
    w("")
    w("### 7.1 按周")
    w("")
    w("| 周 | N | YES率 | 模型Brier | 市场Brier | 优势方 |")
    w("|----|---|-------|----------|----------|--------|")

    # 按周分组
    if len(timestamps) > 0:
        min_ts = int(np.min(timestamps[has_market]))
        week_idx = ((timestamps[has_market] - min_ts) // (7 * 86400_000)).astype(int)
        for wk in sorted(set(week_idx)):
            wmask = week_idx == wk
            nk = int(np.sum(wmask))
            if nk < 5:
                continue
            yes_r = float(np.mean(mp_labels[wmask]))
            m_b = brier_score(mp_model[wmask], mp_labels[wmask])
            mk_b = brier_score(mp_preds[wmask], mp_labels[wmask])
            w(f"| W{wk+1} | {nk:,} | {yes_r:.3f} "
              f"| {m_b:.4f} | {mk_b:.4f} | {_winner(m_b, mk_b)} |")
    w("")

    # 8. 风险分析
    w("## 8. 风险分析")
    w("")
    if trade_5["trades"]:
        yes_pnls = [t["pnl"] for t in trade_5["trades"]
                    if t["direction"] == "YES"]
        no_pnls = [t["pnl"] for t in trade_5["trades"]
                   if t["direction"] == "NO"]

        w("| 风险指标 | 值 |")
        w("|----------|------|")
        if yes_pnls:
            w(f"| 最大单笔亏损/份 (YES) | ${min(yes_pnls):+.4f} |")
            w(f"| 最大单笔盈利/份 (YES) | ${max(yes_pnls):+.4f} |")
        if no_pnls:
            w(f"| 最大单笔亏损/份 (NO) | ${min(no_pnls):+.4f} |")
            w(f"| 最大单笔盈利/份 (NO) | ${max(no_pnls):+.4f} |")

        # 集中度: 前3大 barrier 占总 PnL 比例
        barrier_pnl: Dict[float, float] = {}
        for t in trade_5["trades"]:
            barrier_pnl[t["barrier"]] = (
                barrier_pnl.get(t["barrier"], 0) + t["pnl"]
            )
        sorted_pnl = sorted(barrier_pnl.values(), key=abs, reverse=True)
        total_abs = sum(abs(p) for p in sorted_pnl)
        top3_abs = sum(abs(p) for p in sorted_pnl[:3])
        concentration = top3_abs / total_abs if total_abs > 0 else 0
        w(f"| 前3大 barrier PnL 集中度 | {concentration:.1%} |")
    w("")

    # 9. 局限性
    w("## 9. 局限性与注意事项")
    w("")
    w(f"1. **单月样本**: 仅 {month} 一个月，统计显著性不足")
    w(f"2. **IV 来源**: 使用 DVOL (30天恒定到期) + 期限结构校正 "
      f"(α={config.term_structure_alpha})，非精确月底到期 IV")
    w("3. **GBM 假设**: 假设对数正态分布，不捕获跳跃/厚尾/波动率聚集")
    w(f"4. **零漂移**: μ={config.mu} 假设可能不适用于趋势性行情")
    w("5. **Orderbook 覆盖**: Dome API 数据可能有缺口，"
      "orderbook 快照间隔不均匀")
    w("6. **执行假设**: 假设可以按 best_ask/best_bid 成交，"
      "未考虑滑点和市场冲击")
    w("7. **已触碰排除**: 排除 already_touched 观测后，"
      "数据偏向未触碰 (NO) 结果")
    w("")
    w("---")
    w("*报告由 touch backtest system 自动生成*")

    # 写入文件
    content = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(content)

    logger.info(f"报告已生成: {output_path}")
    return output_path
