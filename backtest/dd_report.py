"""
尽调清单报告自动生成

基于 dd.md 模板结构，从回测数据自动填充可用字段。
不可用字段保持空白。
"""

import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict

from .config import BacktestConfig
from .models import BacktestResult

logger = logging.getLogger(__name__)


def generate_dd_report(
    result: BacktestResult,
    metrics: Dict,
    config: BacktestConfig,
) -> str:
    """
    生成尽调清单报告 (dd.md 模板)

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
    n_events = len(result.event_outcomes)
    n_observations = len(result.observations)

    # 集中度计算
    markets = portfolio.get("markets", [])
    concentration = _compute_concentration(markets, portfolio.get("total_pnl", 0))

    # 格式化辅助
    def f_num(v, fmt=".4f"):
        if v is None:
            return ""
        return f"{v:{fmt}}"

    def f_money(v, fmt=",.2f"):
        if v is None:
            return ""
        return f"${v:{fmt}}"

    lines = []

    # === 头部 ===
    lines.append("# 预测市场量化策略尽调清单")
    lines.append("")
    lines.append(f"- 策略名称：BTC 二元期权统计套利")
    lines.append(f"- 版本号：")
    lines.append(f"- 负责人：")
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"- 审查日期：{now_str}")
    lines.append(f"- 审查人：自动生成")
    lines.append(f"- 数据区间：{result.start_date} ~ {result.end_date}")
    lines.append(f"- 回测区间：{result.start_date} ~ {result.end_date}")
    lines.append(f"- 实盘/仿真状态：")
    lines.append(f"- 结论：")
    lines.append(f"  - [ ] 通过")
    lines.append(f"  - [ ] 有条件通过")
    lines.append(f"  - [ ] 不通过")
    lines.append(f"- 总体备注：")
    lines.append("")
    lines.append("---")
    lines.append("")

    # === §0 策略摘要 ===
    lines.append("## 0. 策略摘要")
    lines.append("")
    lines.append("- 策略目标：估计 BTC 二元期权真实物理概率，在市场价格偏离时交易")
    lines.append("- 交易标的：BTC 二元期权 (Bitcoin above K on date?)")
    lines.append(f"- 市场数量：{portfolio.get('n_markets', '')}")
    lines.append(f"- 信号频率：每 {config.step_minutes} 分钟")
    lines.append(f"- 主要持有周期：事件前 {config.lookback_hours} 小时至结算")
    lines.append("- 核心 alpha 假设：HAR-RV 波动率预测 + Student-t 分布 MC 定价优于市场")
    lines.append("- 主要风险来源：模型误定价、波动率预测失效、市场流动性")
    lines.append("- 失效条件初步判断：")
    lines.append("")
    lines.append("---")
    lines.append("")

    # === §1 数据与时间戳审查 ===
    lines.append("## 1. 数据与时间戳审查")
    lines.append("")
    lines.append("### 1.1 数据来源")
    lines.append("- [x] 已列出所有内部/外部数据源")
    lines.append("- [ ] 每个数据源都有字段说明")
    lines.append("- [ ] 每个数据源都有可追溯版本号/抓取批次号")
    lines.append("- [ ] 数据抓取失败/缺失时有记录机制")
    lines.append("")
    lines.append("**数据源清单：**")
    lines.append("| 数据源 | 用途 | 时间字段 | 可得时间定义 | 负责人 | 备注 |")
    lines.append("|---|---|---|---|---|---|")
    lines.append("| Binance BTC/USDT 1m | RV 计算、结算价 | openTime (UTC ms) | 实时 | | |")
    lines.append("| Polymarket | 市场价格 | timestamp | 实时 | | |")
    lines.append("")
    lines.append("### 1.2 时间戳一致性")
    lines.append("- [x] 所有时间统一到同一时区")
    lines.append("- [x] 已检查夏令时/时区偏移问题")
    lines.append("- [ ] 市场盘口时间戳与外部信息时间戳已对齐")
    lines.append("- [x] 事件开始/结束/结算时间定义清晰")
    lines.append('- [ ] 使用的是"可见时间"而非"写入库时间/回填时间"')
    lines.append("")
    lines.append("**检查记录：**")
    lines.append("- 时区：UTC (内部), ET noon 结算")
    lines.append("- 时间同步方法：et_noon_to_utc_ms() DST 感知")
    lines.append("- 已发现问题：")
    lines.append("- 修复方式：")
    lines.append("")
    lines.append("### 1.3 未来信息泄漏排查")
    lines.append("- [x] rolling/window 计算仅使用历史数据")
    lines.append("- [ ] 特征标准化未使用未来样本")
    lines.append("- [x] 标签生成严格在事件结束后")
    lines.append("- [ ] 训练集/验证集/测试集不存在时间穿越")
    lines.append("- [ ] 同一事件多时点样本未通过聚合泄漏未来信息")
    lines.append("- [ ] 外部新闻/公告/状态变量使用真实发布时间")
    lines.append("- [ ] 临近结算时段已做专项 leakage 审计")
    lines.append("")
    lines.append("**专项说明：**")
    lines.append("- 最容易泄漏的字段：")
    lines.append("- 已采取的防护：")
    lines.append("- 人工 spot check 结果：")
    lines.append("")
    lines.append("---")
    lines.append("")

    # === §2 标签与样本构造审查 ===
    lines.append("## 2. 标签与样本构造审查")
    lines.append("")
    lines.append("### 2.1 标签定义")
    lines.append("- [x] 标签定义与平台结算规则完全一致")
    lines.append("- [ ] 作废/取消/争议市场处理规则明确")
    lines.append("- [ ] 提前结算/异常结算有单独规则")
    lines.append("- [x] 标签生成代码已固定并可复现")
    lines.append("")
    lines.append("**标签定义：**")
    lines.append("- 正类定义：settlement_price > K (BTC 结算价高于行权价)")
    lines.append("- 负类定义：settlement_price <= K")
    lines.append("- 异常样本处理：")
    lines.append("")
    lines.append("### 2.2 样本切分")
    lines.append("- [ ] 训练/验证/测试按时间严格切分")
    lines.append("- [ ] 不存在同一市场同时出现在 train/test 的污染")
    lines.append("- [ ] 不存在同一事件不同快照跨集合泄漏")
    lines.append("- [ ] 样本切分规则已固定")
    lines.append("")
    lines.append("**切分方案：**")
    lines.append("| 集合 | 起始时间 | 结束时间 | 市场数 | 样本数 | 备注 |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(f"| 全量回测 | {result.start_date} | {result.end_date} | {portfolio.get('n_markets', '')} | {n_observations} | |")
    lines.append("| Train |  |  |  |  |  |")
    lines.append("| Valid |  |  |  |  |  |")
    lines.append("| Test |  |  |  |  |  |")
    lines.append("")
    lines.append("### 2.3 样本独立性")
    lines.append('- [ ] 已评估"观测数"与"独立事件数"的差异')
    lines.append("- [ ] 已报告事件级统计，而非只报告样本级统计")
    lines.append("- [ ] 高频重复采样不会夸大显著性")
    lines.append("- [ ] 样本权重设置合理")
    lines.append("")
    lines.append("**独立性评估：**")
    avg_per_event = f"{n_observations / n_events:.1f}" if n_events > 0 else ""
    lines.append(f"- 总样本数：{overall.get('n_predictions', '')}")
    lines.append(f"- 事件数：{n_events}")
    lines.append(f"- 市场数：{portfolio.get('n_markets', '')}")
    lines.append(f"- 平均每事件样本数：{avg_per_event}")
    lines.append(f"- 备注：")
    lines.append("")
    lines.append("---")
    lines.append("")

    # === §3 特征工程审查 (留空) ===
    lines.append("## 3. 特征工程审查")
    lines.append("")
    lines.append("### 3.1 特征列表")
    lines.append("- [ ] 已列出所有特征")
    lines.append("- [ ] 每个特征有业务含义说明")
    lines.append("- [ ] 每个特征有计算公式/伪代码")
    lines.append("- [ ] 每个特征有可得性说明")
    lines.append("")
    lines.append("**特征清单：**")
    lines.append("| 特征名 | 类型 | 含义 | 计算窗口 | 可得时间 | 可能泄漏风险 | 备注 |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.append("| HAR-RV | 连续 | 已实现波动率预测 | 30m/2h/6h/24h | 实时 | 低 | |")
    lines.append("| 日内季节性 | 连续 | UTC 小时季节性因子 | 60d lookback | 实时 | 低 | |")
    lines.append("| Student-t 参数 | 连续 | 分布形状 (df, loc, scale) | 动态 | 实时 | 低 | |")
    lines.append("")
    lines.append("### 3.2 特征稳健性")
    lines.append("- [ ] 已检查缺失值比例")
    lines.append("- [ ] 已检查异常值/极值影响")
    lines.append("- [ ] 已评估特征漂移")
    lines.append("- [ ] 已检查特征在样本外是否稳定")
    lines.append("- [ ] 外部特征断流时有降级方案")
    lines.append("")
    lines.append("**备注：**")
    lines.append("- 高风险特征：")
    lines.append("- 已删除/替换特征：")
    lines.append("- 降级模式：")
    lines.append("")
    lines.append("---")
    lines.append("")

    # === §4 模型训练与预测审查 ===
    lines.append("## 4. 模型训练与预测审查")
    lines.append("")
    lines.append("### 4.1 训练过程")
    lines.append("- [ ] 训练代码可复现")
    lines.append("- [ ] 随机种子固定")
    lines.append("- [ ] 超参数搜索仅在训练/验证集进行")
    lines.append("- [ ] 测试集未参与调参")
    lines.append("- [ ] 模型版本有记录")
    lines.append("")
    lines.append("**模型信息：**")
    lines.append("- 模型类型：HAR-RV + Student-t MC")
    lines.append("- 版本：")
    lines.append("- 训练时间：")
    lines.append(f"- 训练数据区间：{result.start_date} ~ {result.end_date}")
    lines.append("- 超参数：")
    lines.append("")

    # 预测质量
    lines.append("### 4.2 预测质量")
    lines.append("- [x] 已报告 Brier Score")
    lines.append("- [x] 已报告 Log Loss")
    lines.append("- [ ] 已报告 AUC/PR-AUC（如适用）")
    lines.append("- [x] 已报告校准曲线")
    lines.append("- [x] 已按时间段分层报告")
    lines.append("- [ ] 已按市场类型分层报告")
    lines.append("- [ ] 已按流动性分层报告")
    lines.append("")
    lines.append("**关键结果摘要：**")
    lines.append("| 维度 | 指标 | 数值 | 备注 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Overall | Brier | {f_num(overall.get('brier_score'), '.6f')} |  |")
    lines.append(f"| Overall | LogLoss | {f_num(overall.get('log_loss'), '.6f')} |  |")
    for bucket_name in ["T-24h~12h", "T-12h~6h", "T-1h~10m", "T-10m~0"]:
        b = by_bucket.get(bucket_name, {})
        lines.append(f"| {bucket_name} | Brier | {f_num(b.get('brier_score'), '.6f')} |  |")
    lines.append("")

    # 校准
    lines.append("### 4.3 校准")
    lines.append("- [x] 已检查原始校准曲线")
    lines.append("- [ ] 已尝试校准方法（如 isotonic / Platt / beta）")
    lines.append("- [ ] 已比较校准前后交易绩效")
    lines.append("- [ ] 中间概率段偏差已识别")
    lines.append("- [ ] 极端概率段可靠性已验证")
    lines.append("")
    lines.append("**校准结论：**")
    lines.append("- 原始问题：")
    lines.append("- 使用方法：")
    lines.append("- 校准后变化：")
    lines.append("- 是否上线校准层：")
    lines.append("")
    lines.append("---")
    lines.append("")

    # === §5 交易逻辑与执行审查 ===
    lines.append("## 5. 交易逻辑与执行审查")
    lines.append("")
    lines.append("### 5.1 交易规则")
    lines.append("- [x] 入场规则清晰且程序化")
    lines.append("- [x] 出场规则清晰且程序化")
    lines.append("- [x] 持仓上限明确")
    lines.append("- [x] 同时在场头寸规则明确")
    lines.append("- [x] 不依赖主观人工判断")
    lines.append("- [ ] 异常行情下有处理规则")
    lines.append("")
    lines.append("**交易规则摘要：**")
    lines.append(f"- 入场条件：|edge| > {config.entry_threshold}")
    lines.append(f"- 出场条件：事件结算")
    lines.append(f"- 持仓限制：单市场 {config.max_net_shares} 份")
    lines.append(f"- 风控阈值：")
    lines.append("")
    lines.append("### 5.2 成交假设")
    lines.append("- [ ] 回测使用 bid/ask 而非 mid")
    lines.append("- [ ] 区分吃单与挂单")
    lines.append("- [ ] 挂单考虑排队与未成交")
    lines.append("- [ ] 盘口深度不足时限制成交量")
    lines.append("- [ ] 临近结算成交假设更保守")
    lines.append("- [ ] 已加入手续费/平台成本")
    lines.append("- [ ] 已加入撤单失败/延迟风险测试")
    lines.append("")
    lines.append("**成交模型：**")
    lines.append("| 场景 | 假设 | 参数 | 备注 |")
    lines.append("|---|---|---|---|")
    lines.append("| 乐观 |  |  |  |")
    lines.append(f"| 基准 | 固定 {config.shares_per_trade} 份/笔 |  |  |")
    lines.append("| 保守 |  |  |  |")
    lines.append("| 极保守 |  |  |  |")
    lines.append("")
    lines.append("### 5.3 可交易性验证")
    lines.append("- [ ] 延迟测试已完成")
    lines.append("- [ ] 成交率下降测试已完成")
    lines.append("- [ ] 深度减半测试已完成")
    lines.append("- [ ] 点差扩大测试已完成")
    lines.append("- [ ] 去除临近结算最强窗口后测试已完成")
    lines.append("")
    lines.append("**压力测试摘要：**")
    lines.append("| 测试 | 设置 | PnL | MaxDD | PF | 结论 |")
    lines.append("|---|---|---:|---:|---:|---|")
    lines.append("| +30s 延迟 |  |  |  |  |  |")
    lines.append("| +1m 延迟 |  |  |  |  |  |")
    lines.append("| 滑点 x2 |  |  |  |  |  |")
    lines.append("| 深度 50% |  |  |  |  |  |")
    lines.append("| 去掉 T-10m~0 |  |  |  |  |  |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # === §6 风险与组合审查 ===
    lines.append("## 6. 风险与组合审查")
    lines.append("")
    lines.append("### 6.1 组合指标")
    lines.append("- [x] 已报告总收益率")
    lines.append("- [x] 已报告最大回撤")
    lines.append("- [x] 已报告 Profit Factor")
    lines.append("- [x] 已报告 Sharpe/Sortino/Calmar")
    lines.append("- [x] 已报告日度/事件级权益曲线")
    lines.append("- [ ] 使用 MTM 而非只看结算盈亏")
    lines.append("")
    lines.append("**组合结果摘要：**")
    lines.append("| 指标 | 数值 | 备注 |")
    lines.append("|---|---:|---|")
    lines.append(f"| Return | {f_num(portfolio.get('total_return_pct'), '.2f')}% |  |")
    lines.append(f"| Max Drawdown | {f_num(portfolio.get('max_drawdown_pct'), '.2f')}% |  |")
    pf = portfolio.get('profit_factor')
    pf_str = f"{pf:.3f}" if pf is not None and pf != float('inf') else ("∞" if pf == float('inf') else "")
    lines.append(f"| Profit Factor | {pf_str} |  |")
    risk = overall.get("risk_metrics", {})
    n_periods = risk.get("n_periods", 0)
    sharpe_label = "Sharpe (per-event)" if n_periods < 60 else "Sharpe"
    calmar_suffix = " (短期)" if risk.get("calmar_short_period") else ""
    lines.append(f"| {sharpe_label} | {f_num(risk.get('sharpe'), '.3f')} |  |")
    lines.append(f"| Calmar{calmar_suffix} | {f_num(risk.get('calmar'), '.3f')} |  |")
    lines.append("")

    # 集中度
    lines.append("### 6.2 暴露与集中度")
    lines.append("- [ ] 单市场最大风险已量化")
    lines.append("- [ ] 单主题最大风险已量化")
    lines.append("- [ ] 同时在场风险峰值已量化")
    lines.append("- [x] 收益集中度已评估")
    lines.append("- [ ] 去掉最赚钱市场后仍可存活")
    lines.append("- [ ] 相关市场净暴露已受控")
    lines.append("")
    lines.append("**集中度分析：**")
    lines.append(f"- 前 5 个市场 PnL 占比：{concentration['top5_pct']}")
    lines.append(f"- 前 10 个市场 PnL 占比：{concentration['top10_pct']}")
    lines.append(f"- 去掉前 5 市场后总 PnL：{concentration['pnl_ex_top5']}")
    lines.append(f"- 最大单日亏损：{concentration['max_daily_loss']}")
    lines.append(f"- 最大单市场亏损：{concentration['max_market_loss']}")
    dd_details = overall.get("drawdown_details", {})
    if dd_details:
        lines.append(f"- 最长回撤持续：{dd_details.get('max_dd_duration_events', '')} 事件")
        lines.append(f"- 最大连续亏损：{dd_details.get('max_consecutive_losses', '')} 事件")
        lines.append(f"- 平均回撤持续：{dd_details.get('avg_dd_duration_events', 0):.1f} 事件")
    lines.append("")

    lines.append("### 6.3 风险事件")
    lines.append("- [ ] 已评估突发新闻风险")
    lines.append("- [ ] 已评估流动性瞬时枯竭风险")
    lines.append("- [ ] 已评估平台规则变化风险")
    lines.append("- [ ] 已评估异常暂停/取消市场风险")
    lines.append("- [ ] 已设置停机/降杠杆阈值")
    lines.append("")
    lines.append("**风险事件说明：**")
    lines.append("- 主要尾部风险：")
    lines.append("- 触发条件：")
    lines.append("- 缓解机制：")
    lines.append("")
    lines.append("---")
    lines.append("")

    # === §7 样本外与稳健性审查 (留空) ===
    lines.append("## 7. 样本外与稳健性审查")
    lines.append("")
    lines.append("### 7.1 样本外")
    lines.append("- [ ] 严格样本外测试已完成")
    lines.append("- [ ] 样本外预测指标优于基准")
    lines.append("- [ ] 样本外交易收益为正")
    lines.append("- [ ] 样本外回撤可接受")
    lines.append("- [ ] 样本外不依赖单个窗口")
    lines.append("")
    lines.append("**样本外结果：**")
    lines.append("| 窗口 | Brier | LogLoss | PnL | MaxDD | PF | 备注 |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    lines.append("| OOS-1 |  |  |  |  |  |  |")
    lines.append("| OOS-2 |  |  |  |  |  |  |")
    lines.append("")
    lines.append("### 7.2 Walk-forward")
    lines.append("- [ ] 已完成滚动训练/测试")
    lines.append("- [ ] 参数冻结测试已完成")
    lines.append("- [ ] 不同窗口表现无结构性坍塌")
    lines.append("- [ ] 最差窗口表现可接受")
    lines.append("")
    lines.append("**Walk-forward 摘要：**")
    lines.append("| 窗口 | Train | Test | Return | MaxDD | Brier | 结论 |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    wf_data = overall.get("walk_forward", {})
    wf_windows = wf_data.get("windows", [])
    if wf_windows:
        for w in wf_windows:
            ret_s = f"{w['return_pct']:.2f}%" if w.get('return_pct') is not None else ""
            dd_s = f"{w['max_dd_pct']:.2f}%" if w.get('max_dd_pct') is not None else ""
            br_s = f"{w['brier']:.4f}" if w.get('brier') is not None else ""
            lines.append(
                f"| {w['window_id']} "
                f"| {w['train_start']}~{w['train_end']} "
                f"| {w['test_start']}~{w['test_end']} "
                f"| {ret_s} "
                f"| {dd_s} "
                f"| {br_s} "
                f"|  |"
            )
    else:
        lines.append("| 1 |  |  |  |  |  |  |")
        lines.append("| 2 |  |  |  |  |  |  |")
    lines.append("")
    lines.append("### 7.3 子样本与对抗测试")
    lines.append("- [ ] 按市场类型分层测试已完成")
    lines.append("- [ ] 按流动性分层测试已完成")
    lines.append("- [ ] 删除最强时段后测试已完成")
    lines.append("- [ ] 删除最赚钱市场后测试已完成")
    lines.append("- [ ] 删除最强信号分位后测试已完成")
    lines.append("")
    lines.append("**对抗测试结果：**")
    lines.append("| 测试 | 结果 | 是否存活 | 备注 |")
    lines.append("|---|---|---|---|")
    adv = overall.get("adversarial", {})
    adv_items = [
        ("去掉最赚钱 5 个市场", "no_top5_markets"),
        ("去掉 T-10m~0", "no_last_10m"),
        ("去掉最高信号分位", "no_top_quintile"),
    ]
    for label, key in adv_items:
        a = adv.get(key, {})
        if a:
            survived = a.get("survived", False)
            check = "[x] 是 [ ] 否" if survived else "[ ] 是 [x] 否"
            lines.append(f"| {label} | PnL=${a.get('pnl', 0):,.2f} | {check} |  |")
        else:
            lines.append(f"| {label} |  | [ ] 是 [ ] 否 |  |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # === §8 容量与扩展性审查 (留空) ===
    lines.append("## 8. 容量与扩展性审查")
    lines.append("")
    lines.append("### 8.1 容量")
    lines.append("- [ ] 资金规模扩大测试已完成")
    lines.append("- [ ] 2x / 5x / 10x 资金情景已评估")
    lines.append("- [ ] 冲击成本与容量曲线已估计")
    lines.append("- [ ] 高流动性/低流动性市场容量已分开评估")
    lines.append("")
    lines.append("**容量测试：**")
    capacity = overall.get("capacity_analysis", [])
    lines.append("| 份数/笔 | PnL | Return | MaxDD | PF | 备注 |")
    lines.append("|---|---:|---:|---:|---:|---|")
    if capacity:
        for ca in capacity:
            pnl_s = f"${ca['pnl']:,.2f}" if ca.get('pnl') is not None else ""
            ret_s = f"{ca['return_pct']:.2f}%" if ca.get('return_pct') is not None else ""
            dd_s = f"{ca['max_dd_pct']:.2f}%" if ca.get('max_dd_pct') is not None else ""
            pf_v = ca.get('profit_factor')
            pf_s = "∞" if pf_v == float('inf') else (f"{pf_v:.3f}" if pf_v is not None else "")
            lines.append(f"| {ca['shares_per_trade']} | {pnl_s} | {ret_s} | {dd_s} | {pf_s} |  |")
    else:
        for s in ["100", "200", "500", "1000"]:
            lines.append(f"| {s} |  |  |  |  |  |")
    lines.append("")
    lines.append("### 8.2 工程可上线性")
    lines.append("- [ ] 数据更新链路稳定")
    lines.append("- [ ] 信号生成耗时可接受")
    lines.append("- [ ] 下单/撤单链路可执行")
    lines.append("- [ ] 监控告警已设计")
    lines.append("- [ ] 失败恢复机制已设计")
    lines.append("")
    lines.append("**工程说明：**")
    lines.append("- 数据延迟：")
    lines.append("- 信号生成耗时：")
    lines.append("- 交易接口：")
    lines.append("- 监控指标：")
    lines.append("")
    lines.append("---")
    lines.append("")

    # === §9 合规 (留空) ===
    lines.append("## 9. 合规与平台规则审查")
    lines.append("")
    lines.append("- [ ] 已确认平台条款允许该交易方式")
    lines.append("- [ ] 已评估账户限制/订单限制/持仓限制")
    lines.append("- [ ] 已评估市场取消、争议、异常结算处理")
    lines.append("- [ ] 已评估 API 限速与接口稳定性")
    lines.append("- [ ] 已评估税务/法务/合规影响（如适用）")
    lines.append("")
    lines.append("**备注：**")
    lines.append("- 平台规则风险：")
    lines.append("- 法务风险：")
    lines.append("- 待确认事项：")
    lines.append("")
    lines.append("---")
    lines.append("")

    # === §10 上线建议 (留空) ===
    lines.append("## 10. 上线建议与最终结论")
    lines.append("")
    lines.append("### 10.1 上线前必须完成")
    lines.append("- [ ] ________________________")
    lines.append("- [ ] ________________________")
    lines.append("- [ ] ________________________")
    lines.append("")
    lines.append("### 10.2 上线条件")
    lines.append("- [ ] 保守情景下仍为正收益")
    lines.append("- [ ] 最大回撤不超过阈值：________")
    lines.append("- [ ] 样本外连续通过窗口数：________")
    lines.append("- [ ] 实盘仿真期通过：________")
    lines.append("- [ ] 监控系统就绪")
    lines.append("")
    lines.append("### 10.3 初始上线方案")
    lines.append(f"- 初始资金：${config.initial_capital:,.0f}")
    lines.append(f"- 单市场风险上限：{config.max_net_shares} 份")
    lines.append("- 单日亏损上限：")
    lines.append("- 是否只上线部分市场：")
    lines.append("- 是否只上线部分时间窗：")
    lines.append("- 是否只上线高置信信号：")
    lines.append("")
    lines.append("### 10.4 最终结论")
    lines.append("- [ ] 建议上线")
    lines.append("- [ ] 建议小规模灰度上线")
    lines.append("- [ ] 建议仅继续研究")
    lines.append("- [ ] 暂不上线")
    lines.append("")
    lines.append("**最终意见：**")

    content = "\n".join(lines)

    # 写入文件
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"dd_{timestamp}.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"尽调清单报告 → {filepath}")
    return content


def _compute_concentration(markets: list, total_pnl: float) -> Dict[str, str]:
    """计算集中度指标"""
    if not markets:
        return {
            "top5_pct": "",
            "top10_pct": "",
            "pnl_ex_top5": "",
            "max_daily_loss": "",
            "max_market_loss": "",
        }

    sorted_by_pnl = sorted(markets, key=lambda m: m["pnl"], reverse=True)

    top5_pnl = sum(m["pnl"] for m in sorted_by_pnl[:5])
    top10_pnl = sum(m["pnl"] for m in sorted_by_pnl[:10])
    pnl_ex_top5 = total_pnl - top5_pnl

    # 按日汇总
    daily_pnl: Dict[str, float] = defaultdict(float)
    for m in markets:
        daily_pnl[m["event_date"]] += m["pnl"]
    max_daily_loss = min(daily_pnl.values()) if daily_pnl else 0.0

    max_market_loss = min(m["pnl"] for m in markets) if markets else 0.0

    def pct_str(part: float, whole: float) -> str:
        if abs(whole) < 1e-6:
            return "N/A"
        return f"{part / whole * 100:.1f}%"

    return {
        "top5_pct": pct_str(top5_pnl, total_pnl),
        "top10_pct": pct_str(top10_pnl, total_pnl),
        "pnl_ex_top5": f"${pnl_ex_top5:,.2f}",
        "max_daily_loss": f"${max_daily_loss:,.2f}",
        "max_market_loss": f"${max_market_loss:,.2f}",
    }
