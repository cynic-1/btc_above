# Progress

## Completed
- **Unit 1: Foundation** — `pricing_core/__init__.py`, `config.py`, `models.py`, `time_utils.py`, `utils/`, `.env.example`, `.gitignore`, `requirements.txt`
- **Unit 2: binance_data** — `BinanceClient` with `get_klines()`, `get_close_at_event()`, `get_current_price()`, token bucket rate limiter
- **Unit 3: deribit_data** — `DeribitClient` with `get_option_chain()`, `get_index_price()`, `get_perp_mark()`
- **Unit 4: vol_forecast** — `compute_rv()`, `har_features()`, `har_predict()`, `har_fit()`, `intraday_seasonality_factor()`
- **Unit 5: distribution** — `fit_student_t()`, `sample_return()`, `build_empirical_cdf()`
- **Unit 6: pricing** — `simulate_ST()`, `prob_above_K()`, `confidence_interval()`, `price_strikes()`
- **Unit 7: execution** — `compute_opinion_fee()`, `shrink_probability()`, `compute_edge()`, `should_trade()`, `kelly_position()`, `generate_signal()`
- **Unit 8: Integration** — `PricingPipeline`, `run_pricing.py` CLI entry point, all modules wired

- **Backtest Framework** — Complete backtesting system per PRD §7:
  - `backtest/config.py` — `BacktestConfig` 回测配置（日期范围、观测时间、HAR 训练参数、K 网格偏移）
  - `backtest/models.py` — `EventOutcome`, `ObservationResult`, `BacktestResult` 数据模型
  - `backtest/data_cache.py` — `KlineCache` 按天缓存 Binance 1m K线为 gzip CSV
  - `backtest/historical_client.py` — `HistoricalBinanceClient` 防前视偏差的历史数据客户端（`set_now()` 硬截断）
  - `backtest/har_trainer.py` — Walk-forward HAR 系数训练（按 retrain_interval 缓存）
  - `backtest/metrics.py` — Brier score, log loss, calibration curve, PnL 模拟
  - `backtest/engine.py` — `BacktestEngine` 主编排器（遍历事件日 × 观测时间 × K 网格）
  - `backtest/report.py` — CSV 详情 + 汇总统计输出
  - `run_backtest.py` — CLI 入口（`--start`, `--end`, `--download-only`）
  - Pipeline 改造: `PricingPipeline.run()` 新增 `now_utc_ms` 参数支持回测

- **Polymarket 真实市场价格接入** — 回测使用 Polymarket 历史交易价替代固定 market_price=0.5:
  - `backtest/polymarket_discovery.py` — Gamma API 市场发现 + 问题解析（strike/日期 regex）
  - `backtest/polymarket_trades.py` — CLOB API 交易数据缓存（gzip CSV）+ bisect_right 防前瞻偏差价格查询
  - `backtest/config.py` — 新增 `use_market_prices`, `polymarket_cache_dir` 等配置
  - `backtest/models.py` — `ObservationResult` 新增 `market_prices` 字段
  - `backtest/engine.py` — 集成 discovery + price_lookup 到观测循环
  - `backtest/metrics.py` — `simulate_pnl()` 使用 per-strike 真实价格（`default_market_price` 回退）
  - `backtest/report.py` — CSV 新增 `market_price` 列
  - `run_backtest.py` — CLI 新增 `--no-market-prices`, `--polymarket-cache-dir` 选项

- **模型定价 vs Polymarket 对比图表** — 直观展示模型定价与真实市场价格差异:
  - `backtest/chart_engine.py` — `ChartConfig` + `FastPricingEngine`（轻量定价引擎，缓存分布参数、低 MC 采样）+ `ChartGenerator`（按日期批处理、matplotlib 对比图）
  - `run_charts.py` — CLI 入口（`--event-date`, `--strike`, `--step-minutes`, `--lookback-hours`）
  - 优化策略: 跳过 bootstrap CI、Student-t 每 30min 重拟合、MC 采样 2000、同日期多 strike 共享 ST 样本
  - 输出: `charts/{date}/BTC_above_{K}.png` + `.csv`（timestamp, minutes_to_event, model_p, market_price, edge, btc_price）
  - 图表含双 Y 轴: 左轴=概率（模型蓝线 + Polymarket 橙线 + edge 填充），右轴=BTC 价格（灰线 + strike 红色虚线）
  - `requirements.txt` 新增 `matplotlib>=3.7`

- **Bug fix: 方差时间缩放** — 修复模型概率不随到期收敛的严重 bug:
  - 原因: HAR 预测 360min 前向 RV，但 `simulate_ST` 直接使用不缩放，导致 T-0 时仍有大方差
  - 修复: `rv_remaining = rv_hat * min(tau, 360) / 360`，在 `pipeline.py` 和 `chart_engine.py` 同步修复
  - 效果: 略高于当前价 strike, tau=1min 时 P 从 ~0.44 降到 ~0.01; tau=0.1min 时趋近 0

- **回测引擎重构** — 三项核心改动:
  1. **去掉概率收缩**: `shrinkage_lambda=1.0`，p_trade 直接用模型概率 p_P
  2. **手续费为 0**: `fee_rate=0.0`，PnL 模拟不扣手续费
  3. **逐分钟观测**: 用 `step_minutes=1` + `lookback_hours=12` 替代固定 4 个观测点
  - 引擎改用 `FastPricingEngine`（缓存 dist_params、低 MC）替代 `PricingPipeline` 以支持 720× 更多观测
  - 指标按时间段桶分组（T-12h~6h / T-6h~3h / T-3h~1h / T-1h~10m / T-10m~0）替代逐 obs_minutes 分组
  - 164/164 测试全部通过

- **组合模拟器 v2 (固定份额 + 净仓位限制)** — 重写 `simulate_portfolio`，简化为固定份额交易模型:
  - 期初资本 $100k, 每次交易固定 200 shares, 净仓位限制 10,000 shares/市场
  - 入场: `|model_price - market_price| > 0.03`, model > market → BUY YES, 反之 BUY NO
  - YES/NO 分别跟踪 shares + cost, 互相抵消计算净仓位
  - 结算: `yes_payout = yes_shares * (1 if YES else 0)`, `pnl = (payout - cost)`
  - 删除 `_enter_or_add()` 和 `kelly_eta`/`max_position_pct` 等旧逻辑
  - `backtest/config.py` — 新增 `initial_capital`, `shares_per_trade`, `max_net_shares`
  - `backtest/report.py` — 新增 `write_trades_report()` 按市场分组人可读报告 + `write_trades_csv()` 机器可读
  - `run_backtest.py` — 更新打印输出（市场数/交易笔数/盈亏市场数）
  - `tests/test_metrics.py` — 5 个测试: no_trade, buy_yes_wins, buy_no_wins, net_position_limit, market_summary

- **自动化尽调 & 策略评审报告** — 每次回测自动生成两份标准化 markdown 报告:
  - `backtest/dd_report.py` — `generate_dd_report()` 按 dd.md 模板 10 个章节输出尽调清单
  - `backtest/strategy_report.py` — `generate_strategy_report()` 按 report_template.md 模板 10 个章节输出策略评审
  - 自动填写: Brier/LogLoss、校准曲线、分时间段指标、PnL/PF/MaxDD、按市场明细、集中度分析
  - 额外计算: PnL attribution by time bucket、前5/10市场集中度、最大单日/单市场亏损
  - 不可用字段 (OOS/容量等) 保留空白占位符
  - 输出: `backtest_results/dd_YYYYMMDD_HHMMSS.md` + `report_YYYYMMDD_HHMMSS.md`
  - `backtest/report.py` — `generate_report()` 集成两个新生成器
  - `run_backtest.py` — 完成后打印报告文件路径
  - `tests/conftest.py` — 共享测试数据工厂 `make_test_backtest_data()`
  - 19 个新测试全部通过

- **metrics.py 扩展: 风险指标 + ECE/AUC + Edge 五分位** — 新增 7 个函数:
  - `compute_sharpe()` — 基于事件级 PnL 的年化 Sharpe ratio
  - `compute_sortino()` — 使用下行偏差的 Sortino ratio
  - `compute_calmar()` — 年化收益 / 最大回撤
  - `compute_risk_metrics()` — 便捷封装，一次返回 sharpe/sortino/calmar
  - `compute_ece()` — Expected Calibration Error，按桶加权 |pred_mean - actual_freq|
  - `compute_auc()` — AUC-ROC（依赖 sklearn，graceful fallback）
  - `compute_edge_quintile_stats()` — 按 |edge| 等频分 5 桶，统计 PnL/PF/win_rate/avg_pnl
  - `compute_all_metrics()` 集成调用，结果存入 `result["overall"]`

- **回测报告第二轮完善** — 修复已知 Bug + 填充剩余空白 (7 units):
  1. **年化指标修复**: `compute_sharpe/sortino` 改为 per-event 不年化，新增 `compute_annualized_sharpe/sortino` 仅 n≥60 时计算；`compute_risk_metrics` 返回 `n_periods` + `calmar_short_period` 标记；报告层标注 "(per-event)"/"(短期)"
  2. **按价格区间分层 (§4.4)**: `compute_price_range_stats()` 按 moneyness 分 ITM/ATM/OTM，每组计算 n_obs/avg_pred/avg_label/brier/pnl/PF
  3. **回撤详情 (§5.2)**: `compute_drawdown_details()` 从权益曲线计算最大回撤持续、连续亏损、回撤期间列表
  4. **手续费敏感性 (§7.3)**: `run_cost_sensitivity()` 测试 fee×0.5/1.0/2.0 三场景
  5. **Walk-forward 窗口调优 (§6.2)**: 默认改为 train=7/test=5/step=3；自适应逻辑在数据不足时自动缩小窗口；结果中记录实际参数
  6. **延迟敏感性 (§7.2)**: `run_latency_sensitivity()` 按 obs_minutes 分 4 组评估
  7. **容量扩展 (§8.3)**: `run_capacity_analysis()` 测试 100/200/500/1000 份/笔
  - 报告生成器 (`strategy_report.py`, `dd_report.py`) 对应章节已更新
  - 229/229 tests passing (5 skipped for sklearn)

- **时间窗口实验** — 从现有回测数据中找出策略最优启动/停止时间:
  - `backtest/timing_experiment.py` — 核心逻辑: `load_observations_from_csv()` 从 detail.csv 重建 ObservationResult; `run_timing_grid()` 遍历 7×8 (start, stop) 网格; `find_optimal_windows()` 综合评分排序; `compute_incremental_value()` 逐 30m 桶增量 PnL 分析
  - `backtest/timing_plots.py` — 4 张热力图 (PnL/PF/Sharpe/ROI) + 2 张边际效应柱状图 + 1 张增量贡献图
  - `run_timing_experiment.py` — CLI 入口 (`--detail-csv`, `--output-dir`, `--entry-threshold` 等)，输出 ASCII 排行表 + grid_results.csv + summary.txt + 7 张 PNG
  - `tests/test_timing_experiment.py` — 15 个测试: CSV 加载、窗口过滤、边界条件、综合评分、网格运行、排序逻辑
  - 关键发现: 最优窗口 T-30m~T-0m (PnL=$8,455, PF=14.47, Sharpe=0.55, ROI=20.3%); 最后 30 分钟信号质量最高; T-660m 以上桶为负贡献

- **Strike 过滤实验** — 验证只交易离 BTC 价格最近的 N 个 strike 是否优于全部 strike:
  - `backtest/strike_filter_experiment.py` — 核心逻辑: `select_nearest_strikes()` 按 T-720m S0 选最近 N 个 strike; `filter_observations_by_strikes()` 过滤观测; `run_strike_filter_experiment()` 对比 baseline vs nearest-N
  - `run_strike_filter_experiment.py` — CLI 入口 (`--detail-csv`, `--n-nearest`, `--output-dir`)，输出 ASCII 对比表 + comparison.csv + summary.txt
  - `tests/test_strike_filter_experiment.py` — 15 个测试: strike 选择、过滤逻辑、边界条件、完整 pipeline

- **周末效应实验** — 检验策略表现是否存在周末效应（周六/日 vs 工作日）:
  - `backtest/weekend_experiment.py` — 核心逻辑: `is_weekend()`/`weekday_name()` 日期分类; `filter_by_dates()` 按日期集合过滤; `compute_group_result()` 计算信号质量+交易表现+风险+市场微观指标; `run_weekend_experiment()` 生成 weekend/weekday/逐日 7 组结果
  - `run_weekend_experiment.py` — CLI 入口 (`--detail-csv`, `--output-dir`, `--entry-threshold` 等)，输出 ASCII 对比表 + 逐日明细表 + comparison.csv + summary.txt
  - `tests/test_weekend_experiment.py` — 26 个测试: 日期分类、过滤逻辑、指标计算、spread 数据、完整 pipeline
  - 关键发现: 日均 PnL Weekend $1,468 vs Weekday $2,505; 周六亏损(-$1,863/日) 但周日盈利($4,800/日); Brier 周末更优(0.1085 vs 0.2073); 市场价格覆盖率接近(21% vs 24%); 周末 spread 更窄(0.01 vs 0.02)

## Test Results
- 296/296 tests passing (5 skipped for sklearn; `test_orderbook_preprocessor` 收集错误，pre-existing)

- **Polymarket 历史订单簿数据下载** — 从 archive.pmxt.dev 提取 BTC above 市场数据:
  - `download_polymarket_data.py` — 流水线架构下载脚本（8 线程下载 + 1 线程处理 + 信号量限磁盘）
  - `data/btc_above_events.json` — 15 个事件日期（Feb 21 - Mar 8）的市场元数据（Gamma API）
  - `data/btc_above_condition_ids.txt` — 165 个 conditionId（用于 parquet 过滤）
  - `data/polymarket_btc_above/` — 213 个小时快照文件（3.9GB），覆盖 2026-02-21T17 至 2026-03-02T15
  - Schema: `timestamp_received`, `timestamp_created_at`, `market_id`, `update_type`, `data`（JSON 订单簿）
  - 数据源: `https://r2.pmxt.dev/polymarket_orderbook_{YYYY-MM-DDTHH}.parquet`（每文件 ~600MB，含所有市场）
  - 限制: 存档从 Mar 2 T16 起更换 market_id 体系，不再包含 BTC above 市场；缺失 Feb 24 T14/T15

- **订单簿数据增强回测** — 用 tick 级 bid/ask 数据替代稀疏 CLOB 价格历史:
  - `backtest/orderbook_preprocessor.py` — Parquet → npz 预处理（向量化 pandas + 正则提取，165 市场 37.6M 条报价，88MB 缓存）
  - `backtest/orderbook_reader.py` — `OrderbookPriceLookup`（与 `PolymarketPriceLookup` 接口兼容）+ `load_markets_from_events_json()`
  - `backtest/config.py` — 新增 `use_orderbook`, `orderbook_cache_dir`, `orderbook_events_json`
  - `backtest/models.py` — `ObservationResult` 新增 `market_bid_ask` 字段
  - `backtest/engine.py` — `_init_orderbook_lookup()` + `download_polymarket_data()` 自动回退
  - `backtest/chart_engine.py` — `_load_polymarket_data()` + `_get_first_trade_time()` 适配
  - `backtest/report.py` — detail.csv 新增 `market_bid`, `market_ask` 列
  - `run_backtest.py` — 新增 `--preprocess-orderbook`, `--no-orderbook` 参数
  - `requirements.txt` — 新增 `pyarrow>=14.0`
  - `tests/test_orderbook_preprocessor.py` — 4 个测试（基础、去重、过滤、跨文件排序）
  - `tests/test_orderbook_reader.py` — 11 个测试（bisect 正确性、无前瞻偏差、接口兼容性、bid-only 回退）
  - `backtest/metrics.py` — `simulate_portfolio` 改为 ask/bid 执行价（BUY YES 用 ask, BUY NO 用 1-bid），无 bid/ask 时回退 mid
  - `backtest/report.py` — trades 报告/CSV 新增 `exec_price` 列

- **实盘量化交易系统** — 生产级实盘交易引擎，连接 Polymarket BTC "above $K" 二元期权市场:
  - `live/__init__.py` — 包导出
  - `live/config.py` — `LiveTradingConfig` dataclass（Polymarket/Binance/交易/定价/HAR 全参数配置，环境变量加载）
  - `live/models.py` — 领域模型: `MarketInfo`, `OrderBookLevel`, `OrderBookState`, `Position`, `Signal`, `TradeRecord`
  - `live/market_discovery.py` — Gamma API 市场发现（复用 polymarket_discovery 解析模式，新增 tick_size/neg_risk 查询）
  - `live/polymarket_ws.py` — `PolymarketOrderbookWS` WebSocket 客户端（book snapshot + price_change 增量更新，PING 心跳，指数退避重连，线程安全 orderbook 缓存）
  - `live/binance_feed.py` — `BinanceKlineFeed` 双通道 K线流（WebSocket 实时推送 + REST 24h 回填，24h 滚动缓冲区）
  - `live/order_client.py` — `PolymarketOrderClient`（封装 py-clob-client ClobClient L2 模式，GTC/FOK 下单，指数退避重试）
  - `live/position_manager.py` — `PositionManager`（按 condition_id 跟踪 YES/NO 仓位，净仓位限制 + 总成本限制检查）
  - `live/pricing_engine.py` — `LivePricingEngine`（改编 FastPricingEngine，去掉 HistoricalBinanceClient 依赖，直接消费实时 K线，HAR-RV → 季节性 → VRP → Student-t 缓存 → MC 定价）
  - `live/engine.py` — `LiveTradingEngine` 主编排引擎（市场发现 → WS 连接 → 定价循环 → edge 检测 → 下单执行 → 日志，支持 dry-run，SIGINT/SIGTERM 优雅关闭）
  - `run_live.py` — CLI 入口（`--event-date`, `--dry-run`, `--shares-per-trade`, `--order-type` 等参数）
  - `.env.example` — 新增 Polymarket 凭证模板（PM_PRIVATE_KEY, PM_API_KEY 等）
  - `requirements.txt` — 新增 `websocket-client>=1.6`
  - 8 个测试文件共 63 个测试全部通过

- **Polymarket WS 协议修复** — 根据官方 API 文档修复三个协议错误:
  1. **订阅消息格式**: 改为 `{"assets_ids": [...], "type": "market"}`（一条消息包含所有 asset，去掉 `"channel"` 和 `"subscribe"`）
  2. **PING 格式**: 改为纯文本 `PING`（非 JSON `{"type": "ping"}`），PONG 回复也是纯文本
  3. **price_change 结构**: 字段名改为 `price_changes`（非 `changes`），`asset_id` 在每个 change 项内而非外层消息
  - `_apply_price_change(ob, data)` → `_apply_single_change(ob, change)` 方法重构
  - `_handle_price_change()` 按 asset_id 分组批量更新
  - 测试更新: 14/14 通过

## Test Results
- 63/63 live trading tests passing (14 polymarket_ws tests updated)

- **CLOB 价格历史下载 + 混合查询集成** — Orderbook + CLOB 回退的混合价格源:
  - `download_clob_prices.py` — 独立 CLI 脚本，从 events.json 批量下载/更新 CLOB 价格历史（`--check` 仅检查、`--force` 强制重下载）
  - `backtest/hybrid_lookup.py` — `HybridPriceLookup` 包装 OrderbookPriceLookup + PolymarketPriceLookup，per-query 检测 orderbook 新鲜度（默认 2h 阈值），过期自动回退 CLOB
  - `backtest/config.py` — `BacktestConfig` 新增 `orderbook_staleness_hours` 字段
  - `backtest/chart_engine.py` — `ChartConfig` 新增 `orderbook_staleness_hours` 字段；`_load_polymarket_data()` 重写为混合初始化逻辑
  - `backtest/engine.py` — `download_polymarket_data()` 重写: 始终加载 events.json → 同时初始化 orderbook + CLOB → 构建 HybridPriceLookup
  - `tests/test_hybrid_lookup.py` — 16 个测试: 新鲜 orderbook 优先、过期回退 CLOB、仅 CLOB/仅 orderbook/都无、staleness 边界、market_prices/bid_ask 集成
  - 删除 `_init_orderbook_lookup()` 方法（engine.py + chart_engine.py）
  - `data/btc_above_events.json` 扩展至 73 天 (2026-01-01 ~ 2026-03-15), 792 个市场（从 discovery_cache.json 重建）
  - 792/792 CLOB 价格缓存完整
  - 16/16 hybrid 测试 + 31/31 相关旧测试通过

- **数据增量更新脚本** — 自动发现新事件 + 更新价格 + 更新 K线:
  - `update_data.py` — 统一 CLI 入口，三步流水线:
    1. Gamma API 事件发现 → 增量更新 `btc_above_events.json`（仅搜索现有最后日期之后的新日期）
    2. CLOB 价格历史下载 → 增量更新 `data/polymarket/trades_*.csv.gz`
    3. Binance K线下载 → 增量更新 `data/klines/BTCUSDT_1m_*.csv.gz`
  - 支持模式: `--discover-only`（仅发现）、`--prices-only`（仅价格）、`--klines-only`（仅K线）、`--force-prices`（强制重下载）
  - `--until` 参数控制搜索截止日期（默认今天 +7 天）
  - 典型用法: `python3 update_data.py` 一键全量更新; cron 每日自动运行

- **预测表现增加市场价格 Brier/LogLoss 对比** — §4 报告中模型 vs 市场定价能力对比:
  - `backtest/metrics.py` — `compute_all_metrics()` 同步收集 `obs.market_prices` 计算 market_brier_score/market_log_loss/market_ece/market_auc/market_calibration；按时间桶也计算 market_brier_score/market_log_loss
  - `backtest/strategy_report.py` — §4.1 表格改为 3 列（指标/模型/市场）；§4.2 增加 Brier(市场)/LogLoss(市场) 列；§4.3 校准拆分为模型校准 + 市场校准两个子表
  - `report_template.md` — 同步更新 §4 模板格式
  - 385/385 tests passing (5 skipped)

- **可选仓位上限开关** — `run_backtest.py` 新增 `--no-position-limit` flag:
  - 设置时将 `max_net_shares` 覆盖为 `10^9`，使单市场净仓位限制实质失效
  - 用法: `python run_backtest.py --no-position-limit`
  - 仅改动 `run_backtest.py`，`BacktestConfig`/`simulate_portfolio` 无需修改
  - 385/385 tests passing (5 skipped)

- **交易时点条件预测能力分析 (§4.6)** — 量化模型"方向选择 alpha"而非"精度 alpha":
  - `backtest/metrics.py` — 新增 `compute_direction_analysis()`: traded/not_traded 分组条件 Brier/LogLoss; BUY YES/NO 经济学 (avg_cost, win_rate, pnl_per_share); 市场级别方向正确率; Bootstrap 方向价值检验 (5000 次随机方向重排, p-value)
  - `backtest/metrics.py` — `compute_all_metrics()` 集成调用，结果存入 `result["overall"]["direction_analysis"]`
  - `backtest/strategy_report.py` — §4.6 交易时点条件分析: 条件预测精度表、交易方向经济学表、市场级别方向正确率表、Bootstrap 检验表
  - `tests/test_metrics.py` — 11 个新测试: traded/not_traded 分组、条件 Brier、BUY YES/NO 经济学、混合方向、市场级别正确率、Bootstrap 结构/p_value/反向PnL、无市场价格回退

- **方向过滤对照实验** — 支持只买 YES 或只买 NO 的对照实验，拆解各方向 PnL 贡献:
  - `backtest/config.py` — `BacktestConfig` 新增 `direction_filter: str = "both"` 字段
  - `backtest/metrics.py` — `simulate_portfolio()` 新增 `direction_filter` 参数，用 `allow_yes`/`allow_no` 门控交易分支；`compute_all_metrics()`、`run_adversarial_tests()`、`run_cost_sensitivity()`、`run_latency_sensitivity()`、`run_capacity_analysis()` 全链路透传
  - `backtest/report.py` — `generate_report()` 从 config 读取 `direction_filter` 透传
  - `run_backtest.py` — 新增 `--direction` CLI 参数（choices: both/yes_only/no_only）
  - 用法: `python3 run_backtest.py --direction yes_only` / `--direction no_only`

- **交易冷却期 (cooldown)** — 防止相邻分钟重复信号累加仓位，降低回撤:
  - `backtest/config.py` — `BacktestConfig` 新增 `cooldown_minutes: int = 0` 字段
  - `backtest/metrics.py` — `simulate_portfolio()` 新增 `cooldown_minutes` 参数，per-(date,strike) 跟踪 `last_trade_minutes`，交易间距不足则跳过；全链路透传（同 direction_filter）
  - `backtest/report.py` — `generate_report()` 从 config 读取 `cooldown_minutes` 透传
  - `run_backtest.py` — 新增 `--cooldown` CLI 参数
  - 用法: `python3 run_backtest.py --direction no_only --cooldown 60`
  - NO-only 冷却期对照实验结果 (2026-02-07 ~ 2026-03-09, --no-position-limit):

    | cooldown | PnL | 收益率 | MaxDD | PF | 交易数 | 盈/亏市场 |
    |----------|-----|--------|-------|----|--------|-----------|
    | 0 (原始) | $358,215 | 358.2% | 79.4% | 1.625 | 20,903 | 48/33 |
    | 2min | $196,314 | 196.3% | 46.3% | 1.621 | 11,463 | 48/34 |
    | 4min | $107,973 | 108.0% | 25.9% | 1.608 | 6,380 | 48/33 |
    | 8min | $59,616 | 59.6% | 17.0% | 1.581 | 3,628 | 48/34 |
    | 15min | $38,695 | 38.7% | 11.0% | 1.630 | 2,201 | 48/34 |
    | 30min | $21,666 | 21.7% | 8.2% | 1.591 | 1,291 | 48/33 |
    | 60min | $13,801 | 13.8% | 4.9% | 1.657 | 757 | 48/34 |
    | 120min | $7,423 | 7.4% | 3.5% | 1.580 | 448 | 48/35 |

  - 关键发现: PF 恒定 ~1.6（cooldown 只去除重复信号，不损失 edge 质量）；盈利市场数恒定 48；收益与交易数近似线性（每笔 ~$17）；MaxDD 与交易数正相关。MaxDD<20% 选 cd=8min (59.6%/17.0%)；MaxDD<10% 选 cd=15~30min
  - 392/392 tests passing (5 skipped)

- **核心盈利机制分析** — 深度拆解 both+max_net=10000 为何是最优配置:
  - **Alpha 来源**: 模型低估波动率（HAR-RV + tau_scale 线性缩放偏差），高估 P(YES) 约 11-15%；但 Polymarket 市场高估更多（18-23%）→ 差值 = BUY NO alpha（win rate 63.4%，+$0.08/share）
  - **机制 1 — NET 隐性杠杆**: 仓位限制用 NET（yes-no）而非 GROSS → YES 仓位扩大 NO 可用额度（max 3× 总持仓）→ both 模式 13,637 笔交易 vs no_only 3,515 笔
  - **机制 2 — 方向对冲**: 跨市场 YES/NO 组合在 YES 结算时部分抵消 → MaxDD 从 16.56%(no_only) 降至 4.17%(both)
  - **机制 3 — YES 作保险费**: YES 单独亏（-$0.04/share），但提供额度扩容 + 损失对冲 → PF 1.585→3.326, Sharpe 0.243→0.872
  - **利润集中度**: 前 10 市场贡献 65%；T-12h~6h 窗口贡献 37%；OTM 是主要利润来源
  - **已知风险**: VRP 未校准(k=1.0)、回测期仅 22 天、市场冲击未建模、Alpha 可能暂时
  - 详见分析文件: `/home/ubuntu/.claude/plans/wise-growing-kurzweil.md`

- **CLOB 价格缓存过期修复 + Dome API 订单簿历史下载**:
  - `update_data.py` — 修复已结算事件的 CLOB 价格缓存不刷新 bug（`has_trades` 仅检查文件存在，不检查数据完整性）；新增 `event_date+1d` 阈值，缓存文件修改时间早于此则重新下载
  - `backtest/dome_orderbook.py` — `DomeOrderbookFetcher` 从 Dome API (`api.domeapi.io`) 分页获取历史订单簿快照，提取 best_bid/best_ask 保存为 npz 缓存（与现有 `OrderbookPriceLookup` 完全兼容）
  - `update_data.py` — 新增 Step 4 订单簿更新: `--orderbooks-only`, `--force-orderbooks`, `--ob-lookback-hours` 参数；默认下载结算前 24h 订单簿
  - 覆盖所有 69 个已结算事件日 768 个合约

- **回测-实盘差异修复 (3 units)** — 修复导致回测高估盈利、实盘低估的三个独立问题:
  1. **Stale orderbook → 跳过交易**: `HybridPriceLookup.get_quote_at()` 过期时返回 `None`（不再构造零价差合成 quote）；`simulate_portfolio()` 无真实 bid/ask 或 bid==ask 时 `continue` 跳过
  2. **固定 strike 网格**: `BacktestConfig` 新增 `use_fixed_strikes: bool = True`；`BacktestEngine.run()` 从 `_poly_markets` 预计算 `fixed_strikes_by_date`，传入 `_run_single_observation()`；有固定 strikes 时使用，否则回退动态网格
  3. **HAR 训练缓冲区扩容**: `BinanceKlineFeed.__init__()` 新增 `_buffer_minutes = config.har_train_days * 24 * 60`；`_load_initial_klines()` 和 `_update_kline()` 使用 `_buffer_minutes` 替代硬编码 `_24H_MINUTES`
  - 404/404 tests passing (5 skipped)

- **订单簿缓存完整性检查** — 自动检测并重新下载不完整的订单簿缓存:
  - `backtest/dome_orderbook.py` — `DomeOrderbookFetcher.is_cache_complete(cid, settlement_ms, tolerance_ms=1h)`: 检查 npz 最后快照是否覆盖到结算前 1h 以内
  - `update_data.py` — `update_orderbooks()` 改用 `is_cache_complete()` 替代 `has_cache()`; `date_str >= today` 跳过（严格小于，确保结算已完成再下载）; 日志标注"缺失"/"不完整"原因
  - `tests/test_dome_orderbook.py` — 8 个测试: 完整/不完整/无文件/空文件/边界/自定义 tolerance/结算后数据
  - 412/412 tests passing (5 skipped)

- **实盘定价与回测对齐** — 消除实盘 vs 回测的 3 个输入级差异:
  1. **HAR 训练排除事件日**: `LivePricingEngine.train_har()` 新增 `event_date` 参数，过滤 `open_time < midnight_utc(event_date)` 的 K线参与训练（与 `har_trainer.py` 的 `end_ms = _date_to_utc_ms(end_date) - 1` 一致）
  2. **定价 K线窗口限制 24h**: `compute_prices()` 内部过滤 `klines` 到最近 24h（`open_time >= now_utc_ms - 24h`），与 `chart_engine.py` 的 `lookback_ms = 24 * 60 * 60 * 1000` 一致
  3. **engine.py 传递 event_date**: `LiveTradingEngine.start()` 调用 `train_har(klines, event_date)` 而非 `train_har(klines)`
  - 新增 2 个测试: `test_train_excludes_event_day`（验证事件日数据被排除）、`test_24h_kline_filter`（验证 24h 窗口过滤）
  - **Off-by-one 修复**: lookback 从 24h 改为 24h+1min（`chart_engine.py` + `pricing_engine.py`），确保 1441 条 K线差分后 returns=1440 ≥ HAR 最大窗口 1440，消除 `returns 长度 1439 < window 1440` 警告
  - 414/414 tests passing (5 skipped)

- **价格收敛分析 — 中途卖出的胜率与赔率**:
  - `backtest/convergence.py` — 核心逻辑: `load_price_lookup()` 从 detail.csv 构建 (date,strike,obs_min)→(bid,ask) 查询表; `find_trade_signals()` 逐行判断入场信号（同 simulate_portfolio 的 YES/NO elif 逻辑）; `compute_exit_pnl()` 查找退出时 bid/ask 计算中途卖出 PnL; `compute_settlement_pnl()` 持有到结算对照; `aggregate_holding_period()` 聚合胜率/赔率/PnL; `run_convergence()` 串联全流程
  - `run_convergence.py` — CLI 入口 (`--detail-csv`, `--output-dir`, `--entry-threshold`, `--shares-per-trade`)，输出 2 张 ASCII 表（总体 + 方向拆分）+ convergence.csv + convergence_summary.txt
  - `tests/test_convergence.py` — 33 个测试: 价格加载、信号识别、YES/NO PnL 计算、边界条件（无数据/极端价格/bid==ask）、聚合统计、完整 pipeline
  - 持有期: 30m / 1h / 2h / 3h / 6h / 结算
  - 关键发现 (threshold=0.03, 73 天 178,401 signals, 全部市场组合):
    - 组合 6h 退出即正 EV: TotalPnL=+$98,585，接近结算的 +$101,009
    - NO 方向是 alpha 来源: 3h 起正 EV(+$70,322), 6h +$234,926, 结算 +$365,565
    - YES 方向全亏: 所有持有期均负，结算 -$264,556
    - 收敛速度: NO 信号 WinR 从 30m 的 32.9% 升至 6h 的 66.2%，赔率从 1.07 降至 0.70
    - 结论: NO 信号确有中途收敛价值(3h+即可获利)；YES 信号无论持有多久均亏损；"快进快出"对 NO-only 策略可行

- **一触碰障碍期权定价模型 & 回测系统** — Polymarket "What price will Bitcoin hit?" 月度触碰合约的定价与回测:
  - **核心定价** (`touch/barrier_pricing.py`):
    - GBM 下一触碰概率解析公式: `one_touch_up()` (上触碰)、`one_touch_down()` (下触碰)、`one_touch()` (自动判断方向)
    - `price_touch_barriers()` 批量定价，自动处理已触碰 (running_high/low) → P=1.0
    - 边界处理: K=S0→1.0, T≤0→0/1, σ=0→确定性, σ√T极小→clamp 防溢出
  - **数据模型** (`touch/models.py`):
    - `TouchMarketInfo` — Polymarket 触碰合约信息 (month, barrier, direction, condition_id, token_ids)
    - `TouchStrikeResult` — 单 barrier 定价结果 (p_touch, already_touched, p_trade, edge)
    - `TouchObservationResult` — 时间步观测 (s0, running_high/low, T_remaining, sigma, predictions, labels, market_prices)
    - `TouchBacktestConfig` — 回测配置 (month, step_minutes=60, iv_source, vrp_k, default_sigma, mu, shrinkage, orderbook_cache_dir)
  - **IV 数据管道** (`touch/iv_source.py`):
    - `DeribitIVSource` — 实时 ATM IV (期权链) + DVOL 获取
    - `DeribitIVCache` — DVOL 历史缓存 (gzip CSV)，bisect 查询防前瞻偏差，自动百分比/小数转换
  - **市场发现** (`touch/market_discovery.py`):
    - `discover_touch_markets()` — Gamma API 按 slug 查询 + 缓存
    - 解析 "$90,000 or above" → barrier=90000, direction="up"
    - `parse_month_from_slug()` — 从 slug 提取月份
  - **回测引擎** (`touch/backtest_engine.py`):
    - `TouchBacktestEngine` — 月度循环回测
    - O(N) 预计算 running_high/running_low (使用 kline.high/low 捕获分钟内极值)
    - IV 回退链: DVOL 缓存 → VRP 缩放 → 默认 sigma
    - 复用 `pricing_core/execution.py` (shrinkage, edge)、`backtest/data_cache.py` (K线缓存)
    - 市场价格来源: 优先使用 Dome 订单簿 (`OrderbookPriceLookup`)，回退到 CLOB 交易数据 (`PolymarketPriceLookup`)
    - `compute_metrics()` — Brier score + per-barrier 详情
  - **图表生成** (`touch/chart_engine.py`):
    - `TouchChartGenerator` — 双轴图表 (概率 + BTC 价格)
    - 模型 p_touch vs Polymarket 价格 + edge 区域填充
    - 标注 "TOUCHED" 时刻 + barrier 虚线
    - 输出 PNG + CSV
  - **报告生成** (`touch/report_generator.py`):
    - `generate_report()` — 从观测结果自动生成量化机构标准 Markdown 报告
    - 9 节结构: 执行摘要、模型描述、数据概览、预测表现 (Brier/BSS/Murphy/校准曲线)、交易表现 (多阈值 PnL/方向拆分)、逐 barrier 详情、时间维度、风险分析、局限性
    - 交易模拟: 每 barrier 每小时限频、shrinkage + fee 计算、YES/NO 双向
  - **CLI 入口** (`run_touch_backtest.py`):
    - `--month`, `--step-minutes`, `--iv-source`, `--default-sigma`, `--mu`, `--vrp-k`
    - `--download-only`, `--no-market-prices`, `--no-charts`, `--no-report`
    - 自动生成报告 + 图表 + 控制台指标输出
  - **基础设施扩展**:
    - `pricing_core/time_utils.py` — 新增 `month_boundaries_utc_ms()` 月份边界计算
    - `pricing_core/deribit_data.py` — `DeribitClient` 新增: `get_volatility_index_data()` (DVOL OHLC 历史), `get_dvol()` (当前 DVOL 指数), `get_historical_volatility()` (已实现波动率，供 VRP 校准用)
  - **IV 来源修复**: 原错误调用 `get_historical_volatility` (已实现波动率 HV)，改为 `get_volatility_index_data` (DVOL 隐含波动率指数); `get_dvol()` 改用 `get_index_price(btc_dvol)` 替代错误的 `ticker(BTC-DVOL)`
  - **IV 回退链**: option_chain (月底交割期权 ATM IV, 最精确但仅实时可用) → dvol (30天 ATM IV 指数 + 期限结构校正, 回测代理) → default_sigma (固定默认值); 所有 IV 乘以 vrp_k 做 Q→P 缩放; option_chain 模式每小时刷新一次 ATM IV 缓存
  - **DVOL 期限结构校正**: `sigma_adj = dvol * (30 / T_days_remaining)^alpha`; alpha=0.05 (BTC 温和反向期限结构); 月初校正≈0%, 7天剩余+7.5%, 3天剩余+12%; clamp T_days≥0.5 防极端值; `--term-alpha` CLI 参数
  - **ATM IV 定期采集器** (`touch/iv_collector.py`): cron 每小时调用 `get_option_chain_by_expiry(month_end_ms)` 获取月底 ATM IV → 追加到 `data/deribit_iv/atm_iv_YYYY-MM.csv.gz`; 30分钟去重; 与 DeribitIVCache 格式兼容; ATM IV 数据自动合并覆盖同期 DVOL（更精确）
  - **市场发现解析修复**: `parse_direction_from_question()` 新增 "dip/drop/fall/crash" → down, "reach" → up 模式（原只匹配 "or above/below"）
  - **2026-01 回测报告**: Brier 模型 0.074 vs 市场 0.086 (14.4% 优); BSS 0.37 vs 0.27; 所有概率区间模型均优; 5% edge 阈值 1764 笔交易胜率 64.6% PnL/份 +$0.28; Murphy 分解显示优势来自更好的校准 (Reliability 0.009 vs 0.020)
  - **测试**: 70 个新测试全部通过; 517/517 全量通过
    - `test_barrier_pricing.py` — 29 个: 已知值、边界条件、单调性 (strike/time/sigma)、概率范围、BTC 实际参数
    - `test_touch_iv_source.py` — 7 个: 数据查询、防前瞻偏差、百分比转换、缓存持久化
    - `test_touch_market_discovery.py` — 13 个: barrier/direction/month 解析、缓存加载
    - `test_touch_backtest_engine.py` — 11 个: running extremes、bisect 查找、IV 回退、VRP 缩放、合成数据完整回测、指标计算、月份边界
  - 507/507 全量测试通过 (5 skipped)

- **市场价格漂移分析 — 买入后 mid-price 方向性漂移** — 扩展收敛分析，增加不含 spread 成本的方向性指标:
  - `backtest/convergence.py` — HOLDINGS 新增 4h(240min); `HoldingPeriodResult` 新增 `n_favorable`/`favorable_rate`/`avg_mid_drift`/`median_mid_drift` 四字段（默认值=0 向后兼容）; 新增 `compute_exit_mid_drift()` 和 `compute_settlement_mid_drift()` mid-price 漂移计算; `aggregate_holding_period()` 新增 `mid_drifts` 参数; `run_convergence()` 内部同时计算 PnL 和 mid drift
  - `backtest/convergence_chart.py` — 新建收敛可视化模块: 上半图折线(favorable_rate vs win_rate, ALL/YES/NO 三色), 下半图分组柱状(avg_mid_drift), 50% 基线参考线
  - `run_convergence.py` — 总体表新增 `FavR%`/`AvgDrift` 列; 方向拆分表新增 `FavR%` 列; CSV 新增 4 列; 新增 `--chart` 参数触发图表生成; import `plot_convergence`
  - `tests/test_convergence.py` — 新增 15 个测试: `TestComputeExitMidDrift`(6), `TestComputeSettlementMidDrift`(4), `TestAggregateWithMidDrifts`(3), `TestConvergenceChart`(1), `TestRunConvergence.test_mid_drift_populated`(1)
  - 48/48 tests passing
  - 关键发现 (threshold=0.03, 327,383 signals, 73 天):
    - **Alpha 本质是结算预测，非价格收敛**: 所有中途退出均亏损（6h ALL 仍 -$300k），只有持有到结算才盈利（+$311k）。模型的 alpha = 71.5% 的二元结算方向预测准确率，不存在可交易的中途收敛机会
    - **FavR%-WinR% 差距 = spread 成本**: 30m 时 FavR%=52.7% 但 WinR%=29.0%（差 23.7pp），说明 mid 方向对了但 spread 完全吃掉利润；差距随持有期缩小（6h 差 6.7pp），结算时归零
    - **NO 有真实方向 alpha**: NO FavR% 从 53.2%(30m) 单调升至 75.8%(结算)，avg_mid_drift 全程为正且递增；YES FavR% 也 >50% 但 drift 极低，settlement 转负（-0.04），所有持有期 PnL 均负
    - **Polymarket spread 定价高效**: mid 方向对了 53-65% 但 PnL 为负，说明做市商精确定价了信息优势的价值
    - **策略启示**: 中途退出永远是错的；没有"快进快出"可能性；全部价值 = 持有到结算的预测准确率

## Next Steps (Phase B+)
- Deribit smile shape integration (SVI fitting, variance scaling)
- VRP calibration (rolling k estimation, regime classification)
- Last-minute microstructure (jump model, tick rounding)
