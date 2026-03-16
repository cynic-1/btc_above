# BTC Above K — 统计套利系统

Polymarket "Bitcoin above $K on date?" 二元期权的统计套利定价与自动交易系统。

## 快速部署（新服务器）

### 1. 克隆代码

```bash
git clone <repo-url> ~/arbitrage
cd ~/arbitrage
```

### 2. Python 环境

需要 Python 3.12+。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. 安装 py-clob-client（Polymarket 下单 SDK）

实盘下单依赖 [py-clob-client](https://github.com/Polymarket/py-clob-client)，需手动安装：

```bash
# 方式 A: pip 安装（推荐）
pip install py-clob-client

# 方式 B: 从源码安装到指定路径
git clone https://github.com/Polymarket/py-clob-client.git ~/libs/py-clob-client
cd ~/libs/py-clob-client && pip install -e .
```

如果使用方式 B 且不做 pip install，需在 `.env` 中设置路径：

```bash
PY_CLOB_CLIENT_PATH=/home/ubuntu/libs/py-clob-client
```

### 4. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入：

| 变量 | 必填 | 说明 |
|------|------|------|
| `PM_PRIVATE_KEY` | 实盘必填 | Polygon 钱包私钥，API 凭证自动派生 |
| `PY_CLOB_CLIENT_PATH` | 视安装方式 | py-clob-client 库路径（pip 安装则不需要） |
| `BINANCE_BASE_URL` | 可选 | 默认 `https://api.binance.com`，可改为镜像 |
| `LOG_DIR` | 可选 | 日志目录，默认 `logs` |

### 5. 下载数据

```bash
# 一键下载全部数据（事件发现 + CLOB 价格 + K线 + 订单簿）
python3 update_data.py

# 或分步下载
python3 update_data.py --discover-only   # 仅发现新事件
python3 update_data.py --klines-only     # 仅更新 K线
python3 update_data.py --prices-only     # 仅更新 CLOB 价格
python3 update_data.py --orderbooks-only # 仅更新订单簿
```

### 6. 运行测试

```bash
pytest tests/ -v
```

---

## 实盘交易

### 启动实盘（dry-run 模式先测试）

```bash
# dry-run 模式：不下单，仅输出信号
python3 run_live.py --event-date 2026-03-18 --dry-run

# 实盘：真实下单
python3 run_live.py --event-date 2026-03-18
```

### 常用参数

```bash
python3 run_live.py --event-date 2026-03-18 \
    --shares-per-trade 200 \     # 单次下单份数（默认 200）
    --max-net-shares 10000 \     # 单市场净仓位上限（默认 10000）
    --max-total-cost 50000 \     # 总金额限制（默认 $50,000）
    --entry-threshold 0.03 \     # edge 入场阈值（默认 0.03）
    --order-type GTC \           # 订单类型 GTC/FOK（默认 GTC）
    --order-cooldown 20 \        # 同市场下单冷却秒数（默认 20）
    --pricing-interval 10 \      # 定价间隔秒（默认 10）
    --mc-samples 2000            # Monte Carlo 采样数（默认 2000）
```

### 实盘运行流程

1. **市场发现** — 从 Gamma API 查询当日所有 "BTC above $K" 市场
2. **数据流** — WebSocket 连接 Binance（BTC 1m K线）+ Polymarket（orderbook）
3. **HAR 训练** — 基于历史 K线训练 HAR-RV 波动率预测模型
4. **定价循环**（每 10s）:
   - Binance 获取最新 BTC 价格和 K线数据
   - HAR-RV → 季节性调整 → VRP 缩放 → Student-t → Monte Carlo → P(BTC > K)
   - 对比模型概率 vs Polymarket best bid/ask → 计算 edge
   - edge > 阈值 → 下单（YES 或 NO 方向）
5. **风控** — 净仓位限制 + 总成本限制 + 下单冷却期
6. **优雅退出** — Ctrl+C 或 SIGTERM 安全关闭

### 后台运行

```bash
# 使用 nohup
nohup python3 run_live.py --event-date 2026-03-18 > logs/live.out 2>&1 &

# 或使用 screen/tmux
screen -S arb
python3 run_live.py --event-date 2026-03-18
# Ctrl+A D 分离
```

### 日志

运行日志保存在 `logs/` 目录，包含完整的定价扫描表、信号、下单结果和健康检查。

---

## 回测

```bash
# 基础回测
python3 run_backtest.py --start 2026-02-07 --end 2026-03-09

# NO-only + 冷却期
python3 run_backtest.py --start 2026-02-07 --end 2026-03-09 \
    --direction no_only --cooldown 15

# 不限仓位
python3 run_backtest.py --start 2026-02-07 --end 2026-03-09 --no-position-limit
```

回测结果输出到 `backtest_results/`。

## Touch 合约回测

"What price will Bitcoin hit?" 月度一触碰合约：

```bash
python3 run_touch_backtest.py --month 2026-01
```

## 数据更新（建议 cron 每日运行）

```bash
# crontab -e
0 */6 * * * cd ~/arbitrage && .venv/bin/python3 update_data.py >> logs/update.log 2>&1
```

---

## 项目结构

```
├── pricing_core/        # 核心定价引擎（HAR-RV, Student-t, MC, execution）
├── live/                # 实盘交易系统
│   ├── engine.py        #   主编排引擎
│   ├── binance_feed.py  #   Binance WebSocket K线流
│   ├── polymarket_ws.py #   Polymarket WebSocket orderbook
│   ├── pricing_engine.py#   实时定价引擎
│   ├── order_client.py  #   Polymarket 下单客户端
│   ├── position_manager.py  # 仓位管理
│   └── market_discovery.py  # Gamma API 市场发现
├── backtest/            # 回测框架
├── touch/               # 一触碰合约定价与回测
├── tests/               # 测试
├── run_live.py          # 实盘 CLI 入口
├── run_backtest.py      # 回测 CLI 入口
├── run_touch_backtest.py# Touch 回测 CLI 入口
├── update_data.py       # 数据更新脚本
├── .env.example         # 环境变量模板
└── requirements.txt     # Python 依赖
```
