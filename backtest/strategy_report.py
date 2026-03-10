"""
策略评审报告自动生成

基于 report_template.md 模板结构，从回测数据自动填充可用字段。
不可用字段保持空白。
"""

import logging
import os
from datetime import datetime, timezone
from typing import Dict

from .config import BacktestConfig
from .dd_report import _compute_concentration
from .metrics import _bucket_obs_minutes
from .models import BacktestResult

logger = logging.getLogger(__name__)


def generate_strategy_report(
    result: BacktestResult,
    metrics: Dict,
    config: BacktestConfig,
) -> str:
    """
    生成策略评审报告 (report_template.md 模板)

    Args:
        result: 回测结果
        metrics: compute_all_metrics() 返回的指标字典
        config: 回测配置

    Returns:
        完整 markdown 字符串，同时写入文件
    """
    overall = metrics.get("overall", {})
    portfolio = overall.get("portfolio", {})
    by_bucket = metrics.get("by_time_bucket", {})
    cal = overall.get("calibration", {})

    n_events = len(result.event_outcomes)
    n_observations = len(result.observations)
    markets = portfolio.get("markets", [])

    # 额外计算
    concentration = _compute_concentration(markets, portfolio.get("total_pnl", 0))
    pnl_by_bucket = _compute_pnl_by_bucket(result.observations, markets)

    def f_num(v, fmt=".4f"):
        if v is None:
            return ""
        return f"{v:{fmt}}"

    def f_money(v, fmt=",.2f"):
        if v is None:
            return ""
        return f"${v:{fmt}}"

    def f_pf(v):
        if v is None:
            return ""
        if v == float('inf'):
            return "∞"
        return f"{v:.3f}"

    lines = []

    # === 头部 ===
    lines.append("# 策略评审报告")
    lines.append("")
    lines.append("- 策略名称：BTC 二元期权统计套利")
    lines.append("- 版本号：")
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"- 报告日期：{now_str}")
    lines.append("- 作者：自动生成")
    lines.append("- 审阅人：")
    lines.append("- 状态：")
    lines.append("  - [ ] 研究中")
    lines.append("  - [ ] 待评审")
    lines.append("  - [ ] 可仿真")
    lines.append("  - [ ] 可小规模上线")
    lines.append("  - [ ] 已上线")
    lines.append("")
    lines.append("---")
    lines.append("")

    # === §1 执行摘要 ===
    lines.append("# 1. 执行摘要")
    lines.append("")
    lines.append("## 1.1 一句话结论")
    lines.append("> ")
    lines.append("")
    lines.append("## 1.2 本次版本核心变化")
    lines.append("- 变更 1：")
    lines.append("- 变更 2：")
    lines.append("- 变更 3：")
    lines.append("")
    lines.append("## 1.3 关键结论")
    lines.append(f"- 预测能力：Brier={f_num(overall.get('brier_score'), '.6f')}, LogLoss={f_num(overall.get('log_loss'), '.6f')}")
    lines.append("- 交易可实现性：")
    lines.append(f"- 风险收益特征：PnL={f_money(portfolio.get('total_pnl'))}, Return={f_num(portfolio.get('total_return_pct'), '.2f')}%, MaxDD={f_num(portfolio.get('max_drawdown_pct'), '.2f')}%")
    lines.append("- 稳健性：")
    lines.append("- 容量：")
    lines.append("- 主要风险点：")
    lines.append("")
    lines.append("---")
    lines.append("")

    # === §2 策略说明 ===
    lines.append("# 2. 策略说明")
    lines.append("")
    lines.append("## 2.1 策略目标")
    lines.append("- 目标市场：BTC 二元期权 (Bitcoin above K on date?)")
    lines.append("- 目标 alpha：HAR-RV 波动率预测 + Student-t MC 定价优于市场")
    lines.append(f"- 主要持有周期：事件前 {config.lookback_hours} 小时至结算")
    lines.append("- 预期收益来源：市场定价偏离真实物理概率")
    lines.append("")
    lines.append("## 2.2 交易逻辑")
    lines.append(f"- 入场逻辑：|model_price - market_price| > {config.entry_threshold}")
    lines.append("- 出场逻辑：事件结算")
    lines.append(f"- 仓位逻辑：固定 {config.shares_per_trade} 份/笔，单市场上限 {config.max_net_shares} 份")
    lines.append("- 风控逻辑：")
    lines.append("")
    lines.append("## 2.3 数据与特征")
    lines.append("- 数据源：Binance BTC/USDT 1m klines, Polymarket 市场价格")
    lines.append("- 特征类别：HAR-RV, 日内季节性, Student-t 分布参数")
    lines.append("- 标签定义：settlement_price > K → 1, else 0")
    lines.append(f"- 预测频率：每 {config.step_minutes} 分钟")
    lines.append("")
    lines.append("---")
    lines.append("")

    # === §3 数据范围与实验设置 ===
    lines.append("# 3. 数据范围与实验设置")
    lines.append("")
    lines.append("## 3.1 数据区间")
    lines.append("| 项目 | 起始 | 结束 | 备注 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| 原始数据 | {result.start_date} | {result.end_date} |  |")
    lines.append("| 训练集 |  |  |  |")
    lines.append("| 验证集 |  |  |  |")
    lines.append("| 测试集 |  |  |  |")
    lines.append("")
    lines.append("## 3.2 样本统计")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---:|")
    lines.append(f"| 事件天数 | {n_events} |")
    lines.append(f"| 市场数 | {portfolio.get('n_markets', '')} |")
    lines.append(f"| 事件数 | {n_events} |")
    lines.append(f"| 观测总数 | {n_observations} |")
    lines.append(f"| 预测样本数 | {overall.get('n_predictions', '')} |")
    lines.append("")
    lines.append("## 3.3 实验设定")
    lines.append("- 模型：HAR-RV + Student-t Monte Carlo")
    lines.append(f"- 超参数：MC samples={config.mc_samples}, HAR train days={config.har_train_days}")
    lines.append("- 校准方法：")
    lines.append(f"- 回测成交假设：固定 {config.shares_per_trade} 份/笔")
    lines.append("- 手续费假设：")
    lines.append("- 滑点假设：")
    lines.append("- 延迟假设：")
    lines.append("")
    lines.append("---")
    lines.append("")

    # === §4 预测表现 ===
    lines.append("# 4. 预测表现")
    lines.append("")
    lines.append("## 4.1 整体指标")
    lines.append("| 指标 | 模型 | 市场 |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Brier Score | {f_num(overall.get('brier_score'), '.6f')} | {f_num(overall.get('market_brier_score'), '.6f')} |")
    lines.append(f"| Log Loss | {f_num(overall.get('log_loss'), '.6f')} | {f_num(overall.get('market_log_loss'), '.6f')} |")
    lines.append(f"| AUC | {f_num(overall.get('auc'), '.4f')} | {f_num(overall.get('market_auc'), '.4f')} |")
    lines.append(f"| PR-AUC |  |  |")
    lines.append(f"| ECE | {f_num(overall.get('ece'), '.6f')} | {f_num(overall.get('market_ece'), '.6f')} |")
    lines.append("")

    # 分时间段
    lines.append("## 4.2 分时间段表现")
    lines.append("| 时间段 | Brier(模型) | Brier(市场) | LogLoss(模型) | LogLoss(市场) | N | 备注 |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for bucket_name in ["T-24h~12h", "T-12h~6h", "T-6h~3h", "T-3h~1h", "T-1h~10m", "T-10m~0"]:
        b = by_bucket.get(bucket_name, {})
        lines.append(
            f"| {bucket_name} "
            f"| {f_num(b.get('brier_score'), '.6f')} "
            f"| {f_num(b.get('market_brier_score'), '.6f')} "
            f"| {f_num(b.get('log_loss'), '.6f')} "
            f"| {f_num(b.get('market_log_loss'), '.6f')} "
            f"| {b.get('n_predictions', '')} "
            f"|  |"
        )
    lines.append("")

    # 校准
    lines.append("## 4.3 校准结果")
    lines.append("")
    lines.append("### 模型校准")
    lines.append("| Bin | Predicted | Actual | Count |")
    lines.append("|---:|---:|---:|---:|")
    centers = cal.get("bin_centers", [])
    freqs = cal.get("actual_freq", [])
    counts = cal.get("counts", [])
    if centers:
        for c, freq, cnt in zip(centers, freqs, counts):
            lines.append(f"| {c:.2f} | {c:.4f} | {freq:.4f} | {cnt} |")
    else:
        for bin_center in [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]:
            lines.append(f"| {bin_center} |  |  |  |")
    lines.append("")

    # 市场校准
    market_cal = overall.get("market_calibration", {})
    lines.append("### 市场校准")
    lines.append("| Bin | Predicted | Actual | Count |")
    lines.append("|---:|---:|---:|---:|")
    m_centers = market_cal.get("bin_centers", [])
    m_freqs = market_cal.get("actual_freq", [])
    m_counts = market_cal.get("counts", [])
    if m_centers:
        for c, freq, cnt in zip(m_centers, m_freqs, m_counts):
            lines.append(f"| {c:.2f} | {c:.4f} | {freq:.4f} | {cnt} |")
    else:
        for bin_center in [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]:
            lines.append(f"| {bin_center} |  |  |  |")
    lines.append("")

    # 分层分析 (留空)
    lines.append("## 4.4 分层分析")
    lines.append("### 按流动性")
    lines.append("| 流动性分组 | Brier | LogLoss | PnL | 备注 |")
    lines.append("|---|---:|---:|---:|---|")
    lines.append("| 高 |  |  |  |  |")
    lines.append("| 中 |  |  |  |  |")
    lines.append("| 低 |  |  |  |  |")
    lines.append("")
    lines.append("### 按价格区间 (Moneyness)")
    price_range = overall.get("price_range_stats", [])
    lines.append("| 区间 | N | 平均预测 | 平均标签 | Brier | PnL | PF |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    if price_range:
        for pr in price_range:
            lines.append(
                f"| {pr['range']} "
                f"| {pr['n_obs']} "
                f"| {f_num(pr.get('avg_pred'), '.4f')} "
                f"| {f_num(pr.get('avg_label'), '.4f')} "
                f"| {f_num(pr.get('brier'), '.4f')} "
                f"| {f_money(pr.get('pnl'))} "
                f"| {f_pf(pr.get('profit_factor'))} |"
            )
    else:
        for name in ["ITM", "ATM", "OTM"]:
            lines.append(f"| {name} |  |  |  |  |  |  |")
    lines.append("")
    lines.append("### 按市场类型/主题")
    lines.append("| 类型 | Brier | LogLoss | PnL | 备注 |")
    lines.append("|---|---:|---:|---:|---|")
    lines.append("|  |  |  |  |  |")
    lines.append("")
    # 校准分析结论
    cal_analysis = overall.get("calibration_analysis", {})
    if cal_analysis:
        lines.append(f"- Isotonic 校准 (train={cal_analysis.get('n_train', '')}, test={cal_analysis.get('n_test', '')})")
        lines.append(f"  - ECE: {cal_analysis.get('ece_before', 0):.4f} → {cal_analysis.get('ece_after', 0):.4f}")
        lines.append(f"  - Brier: {cal_analysis.get('brier_before', 0):.4f} → {cal_analysis.get('brier_after', 0):.4f}")
    lines.append("")
    lines.append("## 4.5 预测层结论")
    lines.append("- 哪些时间段最强：")
    lines.append("- 哪些分组最强：")
    lines.append("- 校准是否存在系统偏差：")
    lines.append("- 是否存在明显失效区域：")
    lines.append("")
    # === §4.6 交易时点条件分析 ===
    da = overall.get("direction_analysis", {})
    lines.append("## 4.6 交易时点条件分析")
    lines.append("")
    lines.append("### 条件预测精度")
    lines.append("| 子集 | N | 模型 Brier | 市场 Brier | 差值 | 模型 LogLoss | 市场 LogLoss | 差值 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    n_traded = da.get("n_traded", 0)
    n_not_traded = da.get("n_not_traded", 0)
    if n_traded > 0:
        tmb = da.get("traded_model_brier", 0)
        tmkb = da.get("traded_market_brier", 0)
        tml = da.get("traded_model_logloss", 0)
        tmkl = da.get("traded_market_logloss", 0)
        lines.append(
            f"| Traded | {n_traded} "
            f"| {tmb:.6f} | {tmkb:.6f} | {tmb - tmkb:+.6f} "
            f"| {tml:.6f} | {tmkl:.6f} | {tml - tmkl:+.6f} |"
        )
    else:
        lines.append("| Traded | 0 |  |  |  |  |  |  |")
    if n_not_traded > 0:
        ntmb = da.get("not_traded_model_brier", 0)
        ntmkb = da.get("not_traded_market_brier", 0)
        lines.append(
            f"| Not Traded | {n_not_traded} "
            f"| {ntmb:.6f} | {ntmkb:.6f} | {ntmb - ntmkb:+.6f} "
            f"|  |  |  |"
        )
    else:
        lines.append("| Not Traded | 0 |  |  |  |  |  |  |")
    lines.append("")

    lines.append("### 交易方向经济学")
    lines.append("| 方向 | 笔数 | 平均买入价 | 平均模型 | 实际胜率 | 盈亏平衡 | 每份 PnL |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    by = da.get("buy_yes", {})
    bn = da.get("buy_no", {})
    if by.get("n", 0) > 0:
        lines.append(
            f"| BUY YES | {by['n']} "
            f"| {by['avg_cost']:.4f} | {by['avg_model']:.4f} "
            f"| {by['win_rate']:.1%} | {by['breakeven']:.4f} "
            f"| {by['pnl_per_share']:+.4f} |"
        )
    else:
        lines.append("| BUY YES | 0 |  |  |  |  |  |")
    if bn.get("n", 0) > 0:
        lines.append(
            f"| BUY NO | {bn['n']} "
            f"| {bn['avg_cost']:.4f} | {bn['avg_model']:.4f} "
            f"| {bn['win_rate']:.1%} | {bn['breakeven']:.4f} "
            f"| {bn['pnl_per_share']:+.4f} |"
        )
    else:
        lines.append("| BUY NO | 0 |  |  |  |  |  |")
    lines.append("")

    lines.append("### 市场级别方向正确率")
    lines.append("| 方向 | 市场数 | 正确数 | 正确率 |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| YES | {da.get('market_level_yes_total', 0)} "
        f"| {da.get('market_level_yes_correct', 0)} "
        f"| {da.get('market_level_yes_accuracy', 0):.1%} |"
    )
    lines.append(
        f"| NO | {da.get('market_level_no_total', 0)} "
        f"| {da.get('market_level_no_correct', 0)} "
        f"| {da.get('market_level_no_accuracy', 0):.1%} |"
    )
    lines.append("")

    lines.append("### 方向价值 Bootstrap 检验")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---:|")
    bs = da.get("bootstrap", {})
    lines.append(f"| 策略 PnL (per-share sum) | {bs.get('real_pnl_per_share_sum', 0):+.4f} |")
    lines.append(f"| 随机方向均值 | {bs.get('random_mean', 0):+.4f} |")
    lines.append(f"| 随机 95% CI | [{bs.get('random_ci_lower', 0):+.4f}, {bs.get('random_ci_upper', 0):+.4f}] |")
    lines.append(f"| p-value | {bs.get('p_value', 1.0):.4f} |")
    lines.append(f"| 反向交易 PnL | {bs.get('reverse_pnl', 0):+.4f} |")
    lines.append("")

    lines.append("---")
    lines.append("")

    # === §5 回测结果 ===
    lines.append("# 5. 回测结果")
    lines.append("")
    lines.append("## 5.1 基准回测表现")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---:|")
    lines.append(f"| 期初资金 | {f_money(portfolio.get('initial_capital'))} |")
    lines.append(f"| 总 PnL | {f_money(portfolio.get('total_pnl'))} |")
    lines.append(f"| 总收益率 | {f_num(portfolio.get('total_return_pct'), '.2f')}% |")
    lines.append(f"| 总投入成本 | {f_money(portfolio.get('total_cost'))} |")
    lines.append(f"| 交易笔数 | {portfolio.get('n_trades', '')} |")
    lines.append(f"| 盈利市场数 | {portfolio.get('win_markets', '')} |")
    lines.append(f"| 亏损市场数 | {portfolio.get('lose_markets', '')} |")
    lines.append(f"| Profit Factor | {f_pf(portfolio.get('profit_factor'))} |")
    lines.append(f"| Max Drawdown | {f_num(portfolio.get('max_drawdown_pct'), '.2f')}% |")
    risk = overall.get("risk_metrics", {})
    n_periods = risk.get("n_periods", 0)
    sharpe_label = "Sharpe (per-event)" if n_periods < 60 else "Sharpe"
    sortino_label = "Sortino (per-event)" if n_periods < 60 else "Sortino"
    calmar_suffix = " (短期)" if risk.get("calmar_short_period") else ""
    lines.append(f"| {sharpe_label} | {f_num(risk.get('sharpe'), '.3f')} |")
    lines.append(f"| {sortino_label} | {f_num(risk.get('sortino'), '.3f')} |")
    lines.append(f"| Calmar{calmar_suffix} | {f_num(risk.get('calmar'), '.3f')} |")
    if risk.get("annualized_sharpe") is not None:
        lines.append(f"| Sharpe (年化) | {f_num(risk.get('annualized_sharpe'), '.3f')} |")
    if risk.get("annualized_sortino") is not None:
        lines.append(f"| Sortino (年化) | {f_num(risk.get('annualized_sortino'), '.3f')} |")
    lines.append(f"| 事件天数 | {n_periods} |")
    lines.append("")

    # 权益曲线摘要
    dd_details = overall.get("drawdown_details", {})
    lines.append("## 5.2 权益曲线摘要")
    lines.append(f"- 最长回撤持续时间：{dd_details.get('max_dd_duration_events', '')} 事件")
    lines.append(f"- 最大单日亏损：{concentration['max_daily_loss']}")
    lines.append(f"- 最大单市场亏损：{concentration['max_market_loss']}")
    lines.append(f"- 连续亏损天数：{dd_details.get('max_consecutive_losses', '')}")
    lines.append(f"- 平均回撤持续时间：{f_num(dd_details.get('avg_dd_duration_events'), '.1f')} 事件")
    lines.append("")

    # PnL Attribution 按时间段
    lines.append("## 5.3 PnL Attribution")
    lines.append("")
    lines.append("### 按时间段")
    lines.append("| 时间段 | PnL | Return贡献 | 交易数 | PF | 备注 |")
    lines.append("|---|---:|---:|---:|---:|---|")
    total_pnl = portfolio.get("total_pnl", 0)
    for bucket_name in ["T-24h~12h", "T-12h~6h", "T-6h~3h", "T-3h~1h", "T-1h~10m", "T-10m~0"]:
        b = pnl_by_bucket.get(bucket_name, {})
        b_pnl = b.get("pnl", 0)
        b_trades = b.get("n_trades", 0)
        b_pf = b.get("profit_factor")
        contrib = f"{b_pnl / total_pnl * 100:.1f}%" if abs(total_pnl) > 1e-6 and b_pnl != 0 else ""
        lines.append(
            f"| {bucket_name} "
            f"| {f_money(b_pnl) if b_trades > 0 else ''} "
            f"| {contrib} "
            f"| {b_trades if b_trades > 0 else ''} "
            f"| {f_pf(b_pf) if b_trades > 0 else ''} "
            f"|  |"
        )
    lines.append("")

    # 按信号强度分位
    lines.append("### 按信号强度分位")
    lines.append("| 分位组 | PnL | PF | 胜率 | 平均每笔收益 | 交易数 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    edge_quintiles = overall.get("edge_quintiles", [])
    if edge_quintiles:
        for eq in edge_quintiles:
            lines.append(
                f"| Q{eq['quintile']} "
                f"| {f_money(eq['pnl'])} "
                f"| {f_pf(eq['profit_factor'])} "
                f"| {eq['win_rate']:.1%} "
                f"| {f_money(eq['avg_pnl'])} "
                f"| {eq['n_trades']} |"
            )
    else:
        for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            lines.append(f"| {q} |  |  |  |  |  |")
    lines.append("")

    # 按市场
    lines.append("### 按市场")
    lines.append("| 市场 | PnL | 交易数 | MaxDD | 备注 |")
    lines.append("|---|---:|---:|---:|---|")
    if markets:
        sorted_markets = sorted(markets, key=lambda m: m["pnl"], reverse=True)
        for m in sorted_markets:
            n_t = len(m.get("trades", []))
            lines.append(
                f"| {m['title']} "
                f"| {f_money(m['pnl'])} "
                f"| {n_t} "
                f"|  "
                f"| {m['settlement']} |"
            )
    else:
        lines.append("|  |  |  |  |  |")
    lines.append("")

    # 集中度
    lines.append("## 5.4 集中度分析")
    lines.append(f"- 前 5 个市场 PnL 占比：{concentration['top5_pct']}")
    lines.append(f"- 前 10 个市场 PnL 占比：{concentration['top10_pct']}")
    lines.append(f"- 去掉前 5 市场后总 PnL：{concentration['pnl_ex_top5']}")
    lines.append("- 去掉最强时间段后总 PnL：")
    adv = overall.get("adversarial", {})
    no_top_q = adv.get("no_top_quintile", {})
    lines.append(f"- 去掉最高信号分位后总 PnL：{f_money(no_top_q.get('pnl')) if no_top_q.get('pnl') is not None else ''}")
    lines.append("")
    lines.append("## 5.5 回测层结论")
    lines.append("- 收益主要来自：")
    lines.append("- 收益最不稳定部分：")
    lines.append("- 是否过度依赖局部 alpha：")
    lines.append("- 是否存在明显拖累分组：")
    lines.append("")
    lines.append("---")
    lines.append("")

    # === §6 样本外 (留空) ===
    lines.append("# 6. 样本外与滚动验证")
    lines.append("")
    lines.append("## 6.1 样本外结果")
    lines.append("| 窗口 | Brier | LogLoss | PnL | Return | MaxDD | PF | 备注 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for i in range(1, 4):
        lines.append(f"| OOS-{i} |  |  |  |  |  |  |  |")
    lines.append("")
    lines.append("## 6.2 Walk-forward 结果")
    wf_data = overall.get("walk_forward", {})
    wf_params = wf_data.get("actual_params")
    if wf_params:
        adapted = " (自适应)" if wf_params.get("adapted") else ""
        lines.append(f"- 窗口参数{adapted}: train={wf_params.get('train_days')}d, test={wf_params.get('test_days')}d, step={wf_params.get('step_days')}d")
        lines.append("")
    lines.append("| 窗口 | Train Period | Test Period | Return | MaxDD | Brier | PF | 结论 |")
    lines.append("|---|---|---|---:|---:|---:|---:|---|")
    wf_windows = wf_data.get("windows", [])
    if wf_windows:
        for w in wf_windows:
            lines.append(
                f"| {w['window_id']} "
                f"| {w['train_start']}~{w['train_end']} "
                f"| {w['test_start']}~{w['test_end']} "
                f"| {f_num(w.get('return_pct'), '.2f')}% "
                f"| {f_num(w.get('max_dd_pct'), '.2f')}% "
                f"| {f_num(w.get('brier'), '.4f')} "
                f"| {f_pf(w.get('profit_factor'))} "
                f"|  |"
            )
    else:
        for i in range(1, 4):
            lines.append(f"| {i} |  |  |  |  |  |  |  |")
    lines.append("")
    lines.append("## 6.3 参数冻结测试")
    lines.append("- 冻结参数版本：")
    lines.append("- 测试区间：")
    lines.append("- 结果摘要：")
    lines.append("- 与可调参版本差异：")
    lines.append("")
    lines.append("## 6.4 样本外结论")
    lines.append("- 样本外是否稳定：")
    lines.append("- 最差窗口是否可接受：")
    lines.append("- 是否存在 regime 依赖：")
    lines.append("- 是否需要分 regime 上线：")
    lines.append("")
    lines.append("---")
    lines.append("")

    # === §7 交易可实现性 (留空) ===
    lines.append("# 7. 交易可实现性与压力测试")
    lines.append("")
    lines.append("## 7.1 成交情景对比")
    lines.append("| 情景 | 假设 | PnL | Return | MaxDD | PF | 成交率 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for s in ["乐观", "基准", "保守", "极保守"]:
        lines.append(f"| {s} |  |  |  |  |  |  |")
    lines.append("")
    lines.append("## 7.2 延迟敏感性")
    latency = overall.get("latency_sensitivity", [])
    lines.append("| 观测时段 | N | Brier | 平均|Edge| | PnL | 备注 |")
    lines.append("|---|---:|---:|---:|---:|---|")
    if latency:
        for l in latency:
            lines.append(
                f"| {l['bucket']} "
                f"| {l['n_obs']} "
                f"| {f_num(l.get('brier'), '.4f')} "
                f"| {f_num(l.get('avg_edge'), '.4f')} "
                f"| {f_money(l.get('pnl'))} "
                f"|  |"
            )
    else:
        for d in [">60min", "30-60min", "10-30min", "<10min"]:
            lines.append(f"| {d} |  |  |  |  |  |")
    lines.append("")
    lines.append("## 7.3 成本敏感性")
    cost_sens = overall.get("cost_sensitivity", [])
    lines.append("| 场景 | Fee 乘数 | 手续费 | PnL | Return | PF |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    if cost_sens:
        for cs in cost_sens:
            label = {0.5: "Fee x0.5", 1.0: "Baseline", 2.0: "Fee x2.0"}.get(cs["fee_mult"], f"x{cs['fee_mult']}")
            lines.append(
                f"| {label} "
                f"| {cs['fee_mult']:.1f} "
                f"| {f_money(cs.get('total_fee'))} "
                f"| {f_money(cs.get('pnl'))} "
                f"| {f_num(cs.get('return_pct'), '.2f')}% "
                f"| {f_pf(cs.get('profit_factor'))} |"
            )
    else:
        for s in ["Fee x0.5", "Baseline", "Fee x2.0"]:
            lines.append(f"| {s} |  |  |  |  |  |")
    lines.append("")
    lines.append("## 7.4 对抗测试")
    lines.append("| 测试 | PnL | Return | MaxDD | 是否存活 | 备注 |")
    lines.append("|---|---:|---:|---:|---|---|")
    adv_tests = overall.get("adversarial", {})
    adv_labels = [
        ("去掉 T-10m~0", "no_last_10m"),
        ("去掉前 5 市场", "no_top5_markets"),
        ("去掉最高信号分位", "no_top_quintile"),
    ]
    for label, key in adv_labels:
        a = adv_tests.get(key, {})
        if a:
            survived_str = "是" if a.get("survived") else "否"
            lines.append(
                f"| {label} "
                f"| {f_money(a.get('pnl'))} "
                f"| {f_num(a.get('return_pct'), '.2f')}% "
                f"| {f_num(a.get('max_dd_pct'), '.2f')}% "
                f"| {survived_str} "
                f"|  |"
            )
        else:
            lines.append(f"| {label} |  |  |  |  |  |")
    lines.append("| 深度减半 |  |  |  |  |  |")
    lines.append("")
    lines.append("## 7.5 可实现性结论")
    lines.append("- 主要 execution 风险：")
    lines.append("- 策略对延迟敏感度：")
    lines.append("- 策略对成本敏感度：")
    lines.append("- 最接近真实可实现的情景：")
    lines.append("- 实盘预期相对回测折扣：")
    lines.append("")
    lines.append("---")
    lines.append("")

    # === §8 风险与容量 (留空) ===
    lines.append("# 8. 风险与容量评估")
    lines.append("")
    lines.append("## 8.1 风险概览")
    lines.append("| 风险类型 | 描述 | 当前暴露 | 缓解措施 |")
    lines.append("|---|---|---|---|")
    for r in ["市场风险", "流动性风险", "规则风险", "模型风险", "工程风险"]:
        lines.append(f"| {r} |  |  |  |")
    lines.append("")
    lines.append("## 8.2 集中度与尾部")
    lines.append(f"- 最大单市场亏损：{concentration['max_market_loss']}")
    lines.append("- 最大单主题亏损：")
    lines.append(f"- 最大单日亏损：{concentration['max_daily_loss']}")
    lines.append("- 最长回撤期：")
    lines.append("- 尾部事件案例：")
    lines.append("")
    lines.append("## 8.3 容量测试")
    capacity = overall.get("capacity_analysis", [])
    lines.append("| 份数/笔 | PnL | Return | MaxDD | PF | 备注 |")
    lines.append("|---|---:|---:|---:|---:|---|")
    if capacity:
        baseline_spt = config.shares_per_trade
        for ca in capacity:
            label = f"{ca['shares_per_trade']}"
            if ca['shares_per_trade'] == baseline_spt:
                label += " (baseline)"
            lines.append(
                f"| {label} "
                f"| {f_money(ca.get('pnl'))} "
                f"| {f_num(ca.get('return_pct'), '.2f')}% "
                f"| {f_num(ca.get('max_dd_pct'), '.2f')}% "
                f"| {f_pf(ca.get('profit_factor'))} "
                f"|  |"
            )
    else:
        for s in ["100", "200 (baseline)", "500", "1000"]:
            lines.append(f"| {s} |  |  |  |  |  |")
    lines.append("")
    lines.append("## 8.4 风险与容量结论")
    lines.append("- 当前适合资金规模：")
    lines.append("- 容量瓶颈：")
    lines.append("- 最主要尾部风险：")
    lines.append("- 是否适合放大资金：")
    lines.append("")
    lines.append("---")
    lines.append("")

    # === §9 失效机制 (留空) ===
    lines.append("# 9. 失效机制与监控")
    lines.append("")
    lines.append("## 9.1 失效机制")
    lines.append("- 失效机制 1：")
    lines.append("- 失效机制 2：")
    lines.append("- 失效机制 3：")
    lines.append("")
    lines.append("## 9.2 上线后监控指标")
    lines.append("| 指标 | 阈值 | 频率 | 触发动作 |")
    lines.append("|---|---:|---|---|")
    for m in ["实时命中率", "实时校准误差", "成交率", "单日亏损", "连续亏损市场数"]:
        lines.append(f"| {m} |  |  |  |")
    lines.append("")
    lines.append("## 9.3 Kill Switch / 降风控规则")
    lines.append("- 触发条件 1：")
    lines.append("- 触发条件 2：")
    lines.append("- 触发条件 3：")
    lines.append("- 恢复交易条件：")
    lines.append("")
    lines.append("---")
    lines.append("")

    # === §10 上线建议 (留空) ===
    lines.append("# 10. 上线建议")
    lines.append("")
    lines.append("## 10.1 建议结论")
    lines.append("- [ ] 不上线")
    lines.append("- [ ] 继续研究")
    lines.append("- [ ] 仅仿真")
    lines.append("- [ ] 小规模灰度上线")
    lines.append("- [ ] 正式上线")
    lines.append("")
    lines.append("## 10.2 上线方案")
    lines.append(f"- 初始资金：${config.initial_capital:,.0f}")
    lines.append(f"- 单市场限额：{config.max_net_shares} 份")
    lines.append("- 单日止损：")
    lines.append("- 限定市场范围：")
    lines.append("- 限定时间窗：")
    lines.append("- 是否只做高置信信号：")
    lines.append("")
    lines.append("## 10.3 上线前待办")
    lines.append("- [ ] 完成事项 1")
    lines.append("- [ ] 完成事项 2")
    lines.append("- [ ] 完成事项 3")
    lines.append("")
    lines.append("## 10.4 最终评语")
    lines.append("> 在当前版本、当前样本外结果、当前保守成交假设下，本策略的综合评价为：________。")

    content = "\n".join(lines)

    # 写入文件
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"report_{timestamp}.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"策略评审报告 → {filepath}")
    return content


def _compute_pnl_by_bucket(
    observations: list,
    markets: list,
) -> Dict[str, Dict]:
    """
    按时间段桶计算 PnL attribution

    通过将每个 trade 的 obs_minutes 映射到桶来归因。
    """
    bucket_data: Dict[str, Dict] = {}

    for mkt in markets:
        label = 1 if mkt["settlement"] == "YES" else 0
        for trade in mkt.get("trades", []):
            bucket = _bucket_obs_minutes(trade["obs_minutes"])
            if bucket not in bucket_data:
                bucket_data[bucket] = {"pnl": 0.0, "n_trades": 0, "gross_profit": 0.0, "gross_loss": 0.0}

            shares = trade["shares"]
            direction = trade["direction"]
            market_price = trade["market_price"]

            if direction == "YES":
                cost = shares * market_price
                payout = shares * (1.0 if label == 1 else 0.0)
            else:
                cost = shares * (1.0 - market_price)
                payout = shares * (1.0 if label == 0 else 0.0)

            profit = payout - cost
            bucket_data[bucket]["pnl"] += profit
            bucket_data[bucket]["n_trades"] += 1
            if profit > 0:
                bucket_data[bucket]["gross_profit"] += profit
            else:
                bucket_data[bucket]["gross_loss"] += abs(profit)

    # 计算 PF
    for b in bucket_data.values():
        gl = b["gross_loss"]
        b["profit_factor"] = b["gross_profit"] / gl if gl > 0 else (float("inf") if b["gross_profit"] > 0 else None)

    return bucket_data
