"""
回测报告输出
CSV 详情 + 汇总统计 + 按市场分组的交易报告
"""

import csv
import logging
import os
from typing import Dict

from .calibration import run_calibration_analysis
from .config import BacktestConfig
from .dd_report import generate_dd_report
from .metrics import compute_all_metrics
from .models import BacktestResult
from .strategy_report import generate_strategy_report
from .walk_forward import WalkForwardValidator

logger = logging.getLogger(__name__)


def write_detail_csv(result: BacktestResult, output_dir: str) -> str:
    """
    输出逐观测详细 CSV

    每行: event_date, obs_minutes, strike, s0, settlement, p_physical, label, ci_lower, ci_upper
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "detail.csv")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "event_date", "obs_minutes", "strike", "s0", "settlement",
            "p_physical", "label", "ci_lower", "ci_upper", "market_price",
            "market_bid", "market_ask",
        ])
        for obs in result.observations:
            for k in obs.k_grid:
                p = obs.predictions.get(k, "")
                label = obs.labels.get(k, "")
                ci = obs.confidence_intervals.get(k, ("", ""))
                mp = obs.market_prices.get(k)
                ba = obs.market_bid_ask.get(k)
                writer.writerow([
                    obs.event_date, obs.obs_minutes, k, f"{obs.s0:.2f}",
                    f"{obs.settlement_price:.2f}",
                    f"{p:.6f}" if isinstance(p, float) else "",
                    label,
                    f"{ci[0]:.6f}" if isinstance(ci[0], float) else "",
                    f"{ci[1]:.6f}" if isinstance(ci[1], float) else "",
                    f"{mp:.6f}" if isinstance(mp, float) else "",
                    f"{ba[0]:.6f}" if ba else "",
                    f"{ba[1]:.6f}" if ba else "",
                ])

    logger.info(f"详细结果 → {path}")
    return path


def write_trades_report(portfolio: Dict, output_dir: str) -> str:
    """输出按市场分组的人可读交易报告"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "trades_report.txt")

    lines = []
    lines.append("=" * 60)
    lines.append("交易详细报告")
    lines.append(f"期初资金: ${portfolio['initial_capital']:,.0f}")
    lines.append(f"总 PnL: ${portfolio['total_pnl']:,.2f}")
    lines.append(f"总投入成本: ${portfolio['total_cost']:,.2f}")
    lines.append(f"收益率: {portfolio['total_return_pct']:.2f}%")
    lines.append(f"市场数: {portfolio['n_markets']}  交易数: {portfolio['n_trades']}")
    lines.append(f"盈利市场: {portfolio['win_markets']}  亏损市场: {portfolio['lose_markets']}")
    lines.append(f"盈利因子: {portfolio['profit_factor']:.3f}")
    lines.append(f"最大回撤: {portfolio['max_drawdown_pct']:.2f}%")
    lines.append("=" * 60)

    for mkt in portfolio.get("markets", []):
        lines.append("")
        lines.append(f"--- {mkt['title']} (结算: {mkt['settlement']}) ---")
        if mkt["yes_shares"] > 0:
            lines.append(f"  买入 YES: {mkt['yes_shares']} 份, 均价 ${mkt['yes_avg_price']:.4f}")
        if mkt["no_shares"] > 0:
            lines.append(f"  买入 NO:  {mkt['no_shares']} 份, 均价 ${mkt['no_avg_price']:.4f}")
        lines.append(f"  PnL: ${mkt['pnl']:,.2f}")
        lines.append("")
        lines.append("  交易记录:")
        for t in mkt["trades"]:
            exec_p = t.get("exec_price", t["market_price"])
            lines.append(
                f"  T-{t['obs_minutes']:>4d}min  BUY {t['direction']:<3s}  "
                f"{t['shares']}份  模型:{t['model_price']:.4f}  "
                f"执行:{exec_p:.4f}  mid:{t['market_price']:.4f}  "
                f"成本:${t['cost']:.2f}"
            )

    lines.append("")
    lines.append("=" * 60)

    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text)

    logger.info(f"交易报告 → {path}")
    return path


def write_trades_csv(portfolio: Dict, output_dir: str) -> str:
    """输出机器可读的交易明细 CSV"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "trades.csv")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "event_date", "strike", "direction", "shares",
            "model_price", "market_price", "exec_price", "cost", "obs_minutes",
        ])
        for mkt in portfolio.get("markets", []):
            for t in mkt["trades"]:
                exec_p = t.get("exec_price", t["market_price"])
                writer.writerow([
                    mkt["event_date"],
                    mkt["strike"],
                    t["direction"],
                    t["shares"],
                    f"{t['model_price']:.6f}",
                    f"{t['market_price']:.6f}",
                    f"{exec_p:.6f}",
                    f"{t['cost']:.2f}",
                    t["obs_minutes"],
                ])

    n_trades = sum(len(m["trades"]) for m in portfolio.get("markets", []))
    logger.info(f"交易明细 → {path}  ({n_trades} 笔)")
    return path


def write_summary(result: BacktestResult, metrics: Dict, output_dir: str) -> str:
    """输出汇总统计文件"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "summary.txt")

    lines = []
    lines.append("=" * 60)
    lines.append("回测汇总报告")
    lines.append(f"日期范围: {result.start_date} ~ {result.end_date}")
    lines.append(f"事件天数: {len(result.event_outcomes)}")
    lines.append(f"观测总数: {len(result.observations)}")
    lines.append("=" * 60)

    overall = metrics.get("overall", {})
    if overall:
        lines.append("")
        lines.append("--- 整体指标 ---")
        lines.append(f"Brier Score:   {overall.get('brier_score', 'N/A'):.6f}"
                      if isinstance(overall.get('brier_score'), float)
                      else f"Brier Score:   N/A")
        lines.append(f"Log Loss:      {overall.get('log_loss', 'N/A'):.6f}"
                      if isinstance(overall.get('log_loss'), float)
                      else f"Log Loss:      N/A")
        lines.append(f"预测样本数:    {overall.get('n_predictions', 0)}")

        port = overall.get("portfolio", {})
        if port:
            lines.append("")
            lines.append("--- 组合模拟 ---")
            lines.append(f"期初资金:      ${port.get('initial_capital', 0):,.0f}")
            lines.append(f"总 PnL:        ${port.get('total_pnl', 0):,.2f}")
            lines.append(f"总投入成本:    ${port.get('total_cost', 0):,.2f}")
            lines.append(f"总收益率:      {port.get('total_return_pct', 0):.2f}%")
            lines.append(f"市场数:        {port.get('n_markets', 0)}")
            lines.append(f"交易笔数:      {port.get('n_trades', 0)}")
            lines.append(f"盈利市场:      {port.get('win_markets', 0)}")
            lines.append(f"亏损市场:      {port.get('lose_markets', 0)}")
            lines.append(f"盈利因子:      {port.get('profit_factor', 0):.3f}")
            lines.append(f"最大回撤:      {port.get('max_drawdown_pct', 0):.2f}%")

    # 按时间段分组
    by_bucket = metrics.get("by_time_bucket", {})
    if by_bucket:
        lines.append("")
        lines.append("--- 按时间段分组 ---")
        lines.append(f"{'时间段':>12} {'Brier':>10} {'LogLoss':>10} {'N':>8}")
        for bucket, m in by_bucket.items():
            lines.append(
                f"{bucket:>12} {m.get('brier_score', 0):>10.6f} "
                f"{m.get('log_loss', 0):>10.6f} {m.get('n_predictions', 0):>8}"
            )

    # 校准表
    cal = overall.get("calibration", {})
    if cal:
        lines.append("")
        lines.append("--- 校准曲线 ---")
        lines.append(f"{'Bin':>8} {'Predicted':>10} {'Actual':>10} {'Count':>8}")
        centers = cal.get("bin_centers", [])
        freqs = cal.get("actual_freq", [])
        counts = cal.get("counts", [])
        for c, f, n in zip(centers, freqs, counts):
            lines.append(f"{c:>8.2f} {c:>10.4f} {f:>10.4f} {n:>8}")

    lines.append("")
    lines.append("=" * 60)

    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text)

    logger.info(f"汇总报告 → {path}")
    return path


def generate_report(
    result: BacktestResult,
    config: BacktestConfig = None,
) -> Dict:
    """
    生成完整报告

    Returns:
        metrics dict
    """
    if config is None:
        config = BacktestConfig()

    # 计算指标
    metrics = compute_all_metrics(
        result.observations,
        initial_capital=config.initial_capital,
        shares_per_trade=config.shares_per_trade,
        max_net_shares=config.max_net_shares,
        entry_threshold=config.entry_threshold,
        direction_filter=config.direction_filter,
        cooldown_minutes=config.cooldown_minutes,
    )
    # 校准分析
    cal_result = run_calibration_analysis(
        result.observations,
        train_frac=config.calibration_train_frac,
    )
    if cal_result:
        metrics["overall"]["calibration_analysis"] = {
            "ece_before": cal_result.ece_before,
            "ece_after": cal_result.ece_after,
            "brier_before": cal_result.brier_before,
            "brier_after": cal_result.brier_after,
            "n_train": cal_result.n_train,
            "n_test": cal_result.n_test,
        }

    # Walk-forward 验证
    wf_validator = WalkForwardValidator(
        train_days=config.wf_train_days,
        test_days=config.wf_test_days,
        step_days=config.wf_step_days,
    )
    wf_result = wf_validator.run(
        result.observations,
        initial_capital=config.initial_capital,
        shares_per_trade=config.shares_per_trade,
        max_net_shares=config.max_net_shares,
        entry_threshold=config.entry_threshold,
    )
    if wf_result.windows:
        metrics["overall"]["walk_forward"] = {
            "windows": [
                {
                    "window_id": w.window_id,
                    "train_start": w.train_start,
                    "train_end": w.train_end,
                    "test_start": w.test_start,
                    "test_end": w.test_end,
                    "n_test_obs": w.n_test_obs,
                    "brier": w.brier,
                    "pnl": w.pnl,
                    "return_pct": w.return_pct,
                    "max_dd_pct": w.max_dd_pct,
                    "profit_factor": w.profit_factor,
                }
                for w in wf_result.windows
            ],
            "aggregate_brier": wf_result.aggregate_brier,
            "aggregate_pnl": wf_result.aggregate_pnl,
        }

    result.summary = metrics

    # 输出文件
    write_detail_csv(result, config.output_dir)

    portfolio = metrics.get("overall", {}).get("portfolio", {})
    if portfolio:
        write_trades_report(portfolio, config.output_dir)
        write_trades_csv(portfolio, config.output_dir)

    write_summary(result, metrics, config.output_dir)

    # 生成尽调清单和策略评审报告
    generate_dd_report(result, metrics, config)
    generate_strategy_report(result, metrics, config)

    # 打印到日志
    logger.info("\n" + open(os.path.join(config.output_dir, "summary.txt")).read())

    return metrics
