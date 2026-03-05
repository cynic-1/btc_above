下面是一份**完整、明确、可执行、可迭代**的定价/下注优势（stat arb）文档模板，专门针对你这个二元预测市场：

> **合约：**“Bitcoin above K on March 5?”
> **结算：**若 *Binance* 上 BTC/USDT 在标题日期 **ET 12:00** 那根 **1m candle** 的 **Close** 严格大于 K，则 YES，否则 NO。
> **目标：**估计真实概率 (p_P=\mathbb P(S_T>K))，在市场价格偏离时下注获取正期望。

---

# 1. 定义与符号（Contract Spec）

## 1.1 结算点（必须精确到毫秒）

* 事件时刻：标题日期的 **12:00:00 ET**，取该分钟 **1m kline**（openTime=12:00:00）对应的 **close**。
* 标的：**Binance BTC/USDT**（不是指数、不是其他交易所）。
* 判定：严格不等式 **Close > K**。

> 工程上：你需要一个函数把“标题日期 + 12:00 ET”转成**UTC 毫秒时间戳**，并且能处理夏令时（DST）。

## 1.2 数学形式

* (S_T)：结算 close 价（Binance）
* (K)：阈值
* payoff：(\mathbf 1(S_T>K))
* YES 份额“公平价”（忽略利率）：(p \approx \mathbb P(S_T>K))

---

# 2. 数据源与数据层（Data Layer）

你可以拿到：

1. **Deribit**：期权链（IV/smile）、永续/期货（用于 forward、风险溢价、市场状态）
2. **Binance**：现货与合约（分钟K线、盘口/成交可选）

## 2.1 必备字段清单

### Binance

* 1m Kline：openTime, open, high, low, close, volume（现货 BTCUSDT）
* （可选）盘口：best bid/ask，mid，深度（用于滑点与执行质量）

### Deribit

* 期权链：expiry、strike、call/put mid、IV（或标价用于反推IV）
* 永续：资金费率、mark/index、基差（可选）
* （建议）Deribit index（用于“Deribit体系价格”与 Binance 价格基差建模）

## 2.2 时间对齐标准

* 内部统一用 UTC 毫秒。
* 事件时刻由 ET 计算得到 UTC（DST 规则要正确）。
* 拉 Binance Kline 时，用 **startTime/endTime（UTC）** 精确定位那一分钟 openTime。

---

# 3. 总体定价框架（Model Overview）

你要的是真实概率 (p_P)。推荐用“三段式”：

1. **分布形状（shape）**：来自 Deribit 期权 smile（偏度、胖尾结构）
2. **波动水平（level）**：来自 Binance 高频实现波动预测（HAR-RV/EWMA + 日内季节性）
3. **映射与修正（P↔Q、基差、微结构）**：把 Deribit 的风险中性信息转成更接近真实分布

最终用 Monte Carlo（推荐）或封闭近似，输出：

* (p_P(K))（对多个 K 一次性出概率曲线）
* 不确定性区间（用于下注折扣）
* edge = (p_P - \text{mkt_price})

---

# 4. 事件时刻的价格分布建模（Core）

## 4.1 先做最关键的“水平”：预测未来 <12h 的实现方差

设从现在到事件时刻剩余时间为 (\tau)（年化或以天计都行，保持一致）。

### 4.1.1 实现方差（RV）

用 Binance 1m log return：
[
r_t=\ln\frac{S_t}{S_{t-1}}
]
[
RV(\Delta)=\sum r_t^2
]

### 4.1.2 HAR-RV（推荐默认）

用多尺度 RV 预测未来 (\tau) 的方差水平：

* 特征：RV_{30m}, RV_{2h}, RV_{6h}, RV_{24h}
* 目标：未来 horizon 的 RV（例如从现在到事件时刻的 RV）
* 形式（线性/岭回归都行）：
  [
  \widehat{RV}*{0\to T} = \beta_0+\beta_1 RV*{30m}+\beta_2 RV_{2h}+\beta_3 RV_{6h}+\beta_4 RV_{24h}
  ]

### 4.1.3 日内季节性校正（强烈建议）

BTC 日内波动结构显著。做一个“按 UTC 小时的波动因子”：

* 回看 N 天（30–90 天），统计每个 UTC 小时的平均 RV 密度
* 计算未来从 now 到 event 覆盖的小时段的季节性倍率 (g(\text{path}))
* 校正：(\widehat{RV}\leftarrow \widehat{RV}\cdot g)

> 这是很多预测市场/散户模型没做的点，通常能稳定贡献 edge。

---

## 4.2 用 Deribit smile 做“形状”：给尾部/偏度一个合理先验

到期 <12h 时，Deribit 期权 expiry 往往与事件时刻不完全一致（Deribit 常有固定的结算时刻/结算窗）。工程策略：

### 4.2.1 选取用于 shape 的 expiry

* 选择 **最接近且在事件时刻之后** 的 expiry（避免用“事件之后不存在”的信息）
* 若只有更远 expiry：用 variance scaling 缩放到 (\tau)

### 4.2.2 Total variance 缩放（时间缩放）

对每个 strike/delta 的隐含波动 (\sigma_{\text{imp}}(K, T_{\text{opt}}))，换成 total variance：
[
w(K)=\sigma_{\text{imp}}^2(K)\cdot T_{\text{opt}}
]
缩放到事件 horizon：
[
w_{\text{event}}(K)=w(K)\cdot \frac{\tau}{T_{\text{opt}}}
\quad\Rightarrow\quad
\sigma_{\text{imp,event}}(K)=\sqrt{\frac{w_{\text{event}}(K)}{\tau}}
]

### 4.2.3 smile 拟合（两种实现难度）

* **MVP：**只取 10/25 delta 的 RR、BF + ATM，做一个简化 smile（比如对 log-moneyness 二次插值）
* **完整版：**SVI / piecewise linear in strike，对 (\sigma(K)) 平滑拟合，然后用于 MC 采样时的“偏度控制”（或直接用风险中性密度做形状先验）

---

## 4.3 把 Q 的 level 调成 P：VRP 标定（关键迭代点）

期权给的是风险中性（含波动风险溢价），你下注要更接近真实（physical）。最稳的做法是**经验标定一个缩放/回归**：

### 4.3.1 标定数据集（滚动生成）

对过去 M 天（建议 30–60 天），每隔固定间隔（例如每小时）取一个样本：

* 当时刻 (t)：记录 Deribit 缩放到 (\tau) 的 (\sigma_{\text{imp,event}}(t))
* 未来实现：从 (t) 到 (t+\tau) 的 realized (\sigma_{\text{real}}(t))（用 Binance RV）

### 4.3.2 映射形式（从简单到复杂）

* **最简单稳健：**(k = \text{median}(\sigma_{\text{real}}/\sigma_{\text{imp,event}}))，然后 (\sigma_P = k\cdot \sigma_{\text{imp,event}})
* **线性回归：**(\sigma_{\text{real}} = a + b \sigma_{\text{imp,event}})
* **状态依赖（迭代项）：**按 regime 分桶（高波动/低波动、资金费率极端/正常、趋势/震荡）分别估 k

> 你会发现短周期里 k 不是常数；把 k 做成“状态条件化”通常能显著提升稳定性。

---

## 4.4 Deribit vs Binance：基差模型（数字合约必须做）

你的结算是 Binance spot close，而 Deribit 的 smile/forward 常围绕 Deribit index 或其生态价格。做基差：
[
b_t = S^{\text{Binance}}_t - S^{\text{DeribitIndex}}_t
]

* 估计 (\mu_b=\mathbb E[b_T])、(\sigma_b^2=\text{Var}(b_T))（用最近 1–7 天分钟数据）
* 实务修正（两种）：

  1. **行权价平移：**用 (K' = K - \mu_b)
  2. **总方差加和：**在 MC 里采样 (b_T\sim \mathcal N(\mu_b,\sigma_b^2))（或经验分布）

---

# 5. 概率计算：Monte Carlo（推荐默认实现）

## 5.1 生成 (S_T) 的路径

你需要一个可采样的 return 生成器，建议从“标准化残差经验分布/Student-t”入手，兼顾胖尾。

### 5.1.1 标准化残差

用历史窗口里的同 horizon 样本构造：
[
z = \frac{\ln(S_{t+\tau}/S_t)}{\sqrt{\widehat{RV}_{t\to t+\tau}}}
]
拟合 (z) 的分布：

* MVP：Student-t（自由度 (\nu) + scale）
* 更稳：直接用经验分布（bootstrap z）

### 5.1.2 合成事件时刻 return

[
R = z\cdot \sqrt{\widehat{RV}_{0\to T}}
]
[
S_T^{(\text{binance})} = S_0\cdot e^{R} + b_T \quad (\text{如果你用“加性基差”})
]
或用乘性基差（更贴近比例）也可以，但要与数据一致。

## 5.2 输出 (p_P(K))

对 N 条模拟：
[
p_P(K)\approx \frac{1}{N}\sum_{i=1}^N \mathbf 1(S_T^{(i)} > K)
]
一次性对一组 K（K grid）做向量化统计，得到整条“概率曲线”。

## 5.3 不确定性（给下注折扣用）

* 用 bootstrap（对模拟样本重采样）估计 (p) 的标准误/置信区间
* 或对关键输入（(\widehat{RV})、k、(\mu_b)）做扰动敏感性分析，得到 conservative (p^{-})

---

# 6. 下注与执行（Trading Rules）

## 6.1 定义市场价格与边际优势

假设 YES 的市场价格为 (m\in[0,1])（或以 cents 表示）。

* 期望优势：(\text{edge}=p_{\text{trade}}-m)

### 6.1.1 Shrinkage（防过拟合/防模型自信）

建议交易概率用折扣：
[
p_{\text{trade}} = \lambda p_P + (1-\lambda)m
]
默认 (\lambda=0.6)，回测后再调。

## 6.2 入场门槛（最重要的风控）

* 费用与滑点缓冲：设 (c)（例如 1–3% 视流动性）
* 模型不确定性缓冲：设 (u)（例如 (p) 的下置信界偏移）
* 买 YES 条件：(p_{\text{trade}}-m > c+u)
* 卖 YES（或买 NO）条件：(m-p_{\text{trade}} > c+u)

## 6.3 仓位：fractional Kelly（推荐）

对二元 payout=1 的情形，简化可用：
[
f = \eta\cdot \frac{\text{edge}}{p_{\text{trade}}(1-p_{\text{trade}})}
]

* (\eta)：0.1–0.3 起步（强烈建议小），逐步放大
* 并设置硬上限（单笔/单日最大亏损）

## 6.4 微结构细节（你这个合约特有）

1. **严格 “>K”**：当 K 恰好落在 tick 网格上，且临近到期，(S_T=K) 的概率不再可忽略（尤其 ATM）。可在 MC 中对“最后一分钟价格离散化”做一个 tick rounding 模型（例如向最近 tick 取整）。
2. **最后 1m 的操纵/跳跃风险**：预测市场价格常在最后几十分钟失真。建议把“最后 5–10 分钟”单独建模（更高 jump 概率、更高方差）。
3. **跨市场套利约束**：你能对冲的是 BTC 本身（Binance/Deribit），但合约结算点是“单一分钟 close”。对冲无法完全消掉“取样风险”，所以 edge 门槛要更保守。

---

# 7. 回测与评估（Backtest & Metrics）

## 7.1 回测单位：按“事件合约”回测

对历史每个日期（或每小时滚动的“伪事件”）构造：

* 事件时刻 T（对应 ET noon -> UTC）
* 当时刻 t0（例如事件前 12h 内的多个时点）
* 对每个 t0 运行定价管线，得到 (p_P(K))

## 7.2 标签（ground truth）

用 Binance Kline 的规则复刻结算：取 T 对应 1m openTime 的 close，判定 (S_T>K)。

## 7.3 评分指标（别只看准确率）

* **Brier score**（概率预测质量）
* **Log loss**
* **Calibration curve**（分桶后预测概率 vs 实现频率）
* **PnL 模拟**：按你的实际交易规则（含费用/滑点/仓位）跑策略回测

> 目标：不是让 (p) 更“准”，而是让 “在你会下注的区间” 更准、更稳定。

---

# 8. 迭代路线图（Iteration Plan）

## 阶段 A：MVP（1–3 天能跑通）

1. Binance：实现 RV + HAR-RV + 日内季节性
2. 基差：Deribit index 与 Binance spot 的 (\mu_b,\sigma_b)
3. 分布：Student-t（或经验分布） + Monte Carlo
4. 输出：对一组 K 给 (p)、给 edge、给交易信号

**验收标准：**

* 回测 Brier score 明显优于“正态+常数波动”
* 校准曲线不过度偏斜
* PnL 在保守费用下仍有正期望（哪怕不大）

## 阶段 B：引入 Deribit smile shape（提升尾部与 ATM 精度）

1. 期权链清洗（mid、剔除离谱点、插值）
2. variance scaling 到事件 horizon
3. 用 smile 控制分布偏度（SVI 或 delta 插值）
4. 输出 (p_P(K)) 曲线更平滑、对远 OTM 更可信

## 阶段 C：VRP/状态条件化（把稳定性做出来）

1. 标定 k 的滚动回归/分桶
2. 资金费率/趋势因子作为 regime 特征
3. 动态融合 (\sigma_{\text{HAR}}) 与 (k\sigma_{\text{imp}})

## 阶段 D：最后一分钟微结构（专打你这个合约的“取样点风险”）

1. 最后 10 分钟单独估 jump 概率/方差放大
2. tick rounding 模型
3. 执行模型（盘口滑点、成交概率）

---

# 9. 工程实现参考（可直接照着写代码的模块边界）

## 9.1 模块划分

1. `time_utils`

* `et_noon_to_utc_ms(date)`（含 DST）
* `utc_ms_to_binance_kline_open(utc_ms, interval=1m)`

2. `binance_data`

* `get_klines(symbol, start_ms, end_ms, interval=1m)`
* `get_close_at_event(symbol, event_open_ms)`（返回该 1m 的 close）

3. `deribit_data`

* `get_option_chain(expiry)`
* `get_index_price()` / `get_perp_mark()`（用于基差）

4. `vol_forecast`

* `compute_rv(returns, windows=[30m,2h,6h,24h])`
* `har_predict(features) -> RV_hat`
* `intraday_seasonality_factor(path_hours)`

5. `vrp_calibration`

* `fit_k_mapping(samples) -> k or (a,b)`
* `regime_classifier(features)`

6. `distribution`

* `fit_student_t(z_samples)` 或 `build_empirical_cdf(z_samples)`
* `sample_return(RV_hat, dist_params, n)`

7. `pricing`

* `simulate_ST(S0, RV_hat, dist, basis_model, n)`
* `prob_above_K(ST_samples, K_list)`

8. `execution`

* `compute_edge(p_trade, market_price, costs, uncertainty)`
* `position_size(edge, p_trade, eta, limits)`

## 9.2 每次运行输出（必须可复盘）

* 时间：now_utc、event_utc、距离到期分钟数
* 输入快照：S0、RV_hat、季节性倍率、k、基差参数、dist 参数
* 输出：每个 K 的 (p_P)、置信区间、p_trade、edge、建议仓位
* 执行：下单价格、成交、滑点、最终盈亏

---

# 10. 你现在就能用的默认参数（建议起步值）

* RV 频率：1m returns
* HAR 窗口：30m/2h/6h/24h
* 季节性：过去 60 天按 UTC 小时统计
* VRP 标定：过去 45 天滚动，k 用 median 比率
* 基差：过去 3 天分钟数据估 (\mu_b,\sigma_b)
* MC 样本数：50k（MVP 可 10k）
* shrinkage：(\lambda=0.6)
* Kelly 系数：(\eta=0.2)
* 入场门槛：`edge > 0.03 + uncertainty_buffer`

---
