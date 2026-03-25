# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Statistical arbitrage / pricing-edge system for binary prediction markets. The contract type is "Bitcoin above K on March 5?" settling on the **Binance BTC/USDT 1-minute candle close at ET 12:00**. The system estimates the true physical probability `p_P = P(S_T > K)` and bets when market price deviates.

The full PRD is in `prd.md` (Chinese). A reference implementation for cross-platform arbitrage lives at `/home/ubuntu/pm_arb` — follow its patterns for code style and structure.

## Architecture

Three-part pricing model:
1. **Distribution shape** — from Deribit option smile (skew, fat tails)
2. **Volatility level** — HAR-RV from Binance 1m klines + intraday seasonality
3. **Q→P mapping + basis correction** — VRP calibration (risk-neutral → physical)

Output via Monte Carlo: `p_P(K)` probability curve, edge, trade signal for a grid of K values.

### Module Boundaries (from PRD §9.1)

| Module | Purpose |
|--------|---------|
| `time_utils` | `et_noon_to_utc_ms(date)` (DST-aware), `utc_ms_to_binance_kline_open()` |
| `binance_data` | `get_klines()`, `get_close_at_event()` |
| `deribit_data` | `get_option_chain()`, `get_index_price()`, `get_perp_mark()` |
| `vol_forecast` | `compute_rv()`, `har_predict()`, `intraday_seasonality_factor()` |
| `vrp_calibration` | `fit_k_mapping()`, `regime_classifier()` |
| `distribution` | `fit_student_t()`, `build_empirical_cdf()`, `sample_return()` |
| `pricing` | `simulate_ST()`, `prob_above_K()` |
| `execution` | `compute_edge()`, `position_size()` |

### Data Sources

- **Binance**: BTC/USDT 1m klines (spot). Settlement = `close` of 1m candle whose `openTime` equals ET 12:00 converted to UTC ms.
- **Deribit**: Option chain (expiry, strike, call/put mid, IV), perpetual (funding rate, mark/index, basis), index price.

### Critical Time Handling

All internal timestamps use **UTC milliseconds**. ET→UTC must handle DST correctly. The function `et_noon_to_utc_ms(date_str)` is foundational — many modules depend on it.

## Key Formulas

- **HAR-RV**: `RV_hat = b0 + b1*RV_30m + b2*RV_2h + b3*RV_6h + b4*RV_24h`
- **VRP scaling**: `sigma_P = k * sigma_imp_event` where `k = median(sigma_real / sigma_imp)`
- **Variance scaling**: `sigma_imp_event = sqrt(sigma_imp² * T_opt / tau)`
- **Basis**: `b_t = S_binance - S_deribit_index`, estimate `mu_b`, `sigma_b` from recent 3d of 1m data
- **Shrinkage**: `p_trade = 0.6 * p_P + 0.4 * market_price`
- **Kelly position**: `f = eta * edge / (p_trade * (1 - p_trade))`
- **Entry threshold**: `edge > 0.03 + uncertainty_buffer`
- **Opinion fee**: `fee_rate = 0.06 * price * (1 - price) + 0.0025` (min $0.50)

## Default Parameters

| Parameter | Value |
|-----------|-------|
| RV frequency | 1m returns |
| HAR windows | 30m / 2h / 6h / 24h |
| Intraday seasonality lookback | 60 days, by UTC hour |
| VRP calibration window | 45 days rolling, median ratio k |
| Basis estimation window | 3 days of 1m data |
| MC sample count | 50k (MVP: 10k) |
| Shrinkage lambda | 0.6 |
| Kelly fraction eta | 0.2 |

## Iteration Plan

- **Phase A (MVP)**: Binance RV + HAR-RV + seasonality, basis model, Student-t + MC, output p_P/edge/signal
- **Phase B**: Deribit smile shape (option chain cleaning, variance scaling, SVI)
- **Phase C**: VRP/regime conditioning (rolling k calibration, funding rate features)
- **Phase D**: Last-minute microstructure (jump model, tick rounding, execution model)

## Run Output Requirements

Every pricing run must log: `now_utc`, `event_utc`, minutes-to-expiry, input snapshot (S0, RV_hat, seasonality multiplier, k, basis params, dist params), output (p_P per K, confidence interval, p_trade, edge, position size), and execution details when live.

## 资源限制（必须遵守）

本机内存和 CPU 有限，代码必须控制资源占用，避免 OOM 或卡死：

- **禁止一次性加载全部数据到内存**。K线、订单簿、交易数据等必须按月/按天分批加载、处理完即释放
- **禁止在内存中累积大量中间结果**。回测观测数据按月序列化到磁盘（pkl），不要跨月保留在内存
- **Polygon 链上数据下载**必须使用分段扫描（segment-based），每段完成后写入磁盘。禁止全部收集到内存再写入
- **并发下载控制**：Polygon RPC workers ≤ 3，Binance/Deribit API 遵守 token bucket 限速
- **numpy 数组**：避免对整月 1m K线（~44000 行）做不必要的全量复制。用 slice 视图替代 copy
- **回测主循环**：逐月运行，每月结束后生成报告并释放该月数据，再进入下月
- **图表生成**：每张图生成后 `plt.close()` 释放内存，不要累积 Figure 对象

## Conventions

Follow the patterns established in `/home/ubuntu/pm_arb`:
- Python with type hints and dataclasses for domain objects (no raw dicts)
- Dataclass-based config loaded from environment variables via `python-dotenv`
- `logging.getLogger(__name__)` in each module; `setup_logger()` at startup replaces `print()`
- Comments and documentation primarily in Chinese
- Token bucket rate limiting for API calls
- `pytest` for tests
- `.env` for secrets (gitignored), `.env.example` as template
