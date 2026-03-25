"""
参数探索：系统性搜索最优交易过滤器参数

分阶段网格搜索，每阶段最优参数传递到下一阶段。
包含前半/后半时间拆分验证，防止过拟合。

用法:
    python3 explore_params.py --obs backtest_results/observations_2025-11-01_2026-03-24.pkl
"""

import argparse
import logging
import time
from typing import Dict, List, Optional

import numpy as np

from backtest.observation_cache import load_observations
from backtest.metrics import simulate_portfolio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def split_observations(observations, ratio=0.5):
    """按 event_date 时间顺序拆分前半/后半"""
    dates = sorted(set(o.event_date for o in observations))
    split_idx = int(len(dates) * ratio)
    first_dates = set(dates[:split_idx])
    second_dates = set(dates[split_idx:])
    first = [o for o in observations if o.event_date in first_dates]
    second = [o for o in observations if o.event_date in second_dates]
    return first, second


def run_single(obs_full, obs_first, obs_second, **kwargs):
    """运行单组参数，返回结果 dict"""
    p = simulate_portfolio(obs_full, **kwargs)
    p1 = simulate_portfolio(obs_first, **kwargs)
    p2 = simulate_portfolio(obs_second, **kwargs)

    ep = np.array([e["pnl"] for e in p["event_pnls"]])
    sharpe = float(np.mean(ep) / np.std(ep)) if len(ep) > 1 and np.std(ep) > 0 else 0.0

    return {
        "pnl": p["total_pnl"],
        "ret": p["total_return_pct"],
        "pf": p["profit_factor"],
        "dd": p["max_drawdown_pct"],
        "trades": p["n_trades"],
        "markets": p["n_markets"],
        "win_rate": p["win_markets"] / p["n_markets"] if p["n_markets"] > 0 else 0,
        "sharpe": sharpe,
        "pnl_1h": p1["total_pnl"],
        "pnl_2h": p2["total_pnl"],
        "pf_1h": p1["profit_factor"],
        "pf_2h": p2["profit_factor"],
    }


def fmt_pf(pf):
    return f"{pf:.3f}" if pf != float("inf") else "inf"


def print_table(rows: List[Dict]):
    """打印对比表"""
    header = (
        f"{'Label':<40s} {'PnL':>9s} {'Ret%':>6s} {'PF':>6s} "
        f"{'MaxDD':>6s} {'Trades':>6s} {'WinR':>5s} {'Shrp':>6s} "
        f"{'PnL_1H':>9s} {'PnL_2H':>9s} {'PF_1H':>6s} {'PF_2H':>6s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['label']:<40s} "
            f"{r['pnl']:>9,.0f} "
            f"{r['ret']:>6.1f} "
            f"{fmt_pf(r['pf']):>6s} "
            f"{r['dd']:>5.1f}% "
            f"{r['trades']:>6d} "
            f"{r['win_rate']:>5.1%} "
            f"{r['sharpe']:>6.3f} "
            f"{r['pnl_1h']:>9,.0f} "
            f"{r['pnl_2h']:>9,.0f} "
            f"{fmt_pf(r['pf_1h']):>6s} "
            f"{fmt_pf(r['pf_2h']):>6s}"
        )
    print()


def select_best(rows, min_trades=2000):
    """选 PF 最高且前后半都盈利的配置"""
    valid = [r for r in rows if r["trades"] >= min_trades and r["pnl_1h"] > 0 and r["pnl_2h"] > 0]
    if not valid:
        valid = [r for r in rows if r["trades"] >= min_trades]
    if not valid:
        valid = rows
    return max(valid, key=lambda r: r["pf"] if r["pf"] != float("inf") else 0)


def main():
    parser = argparse.ArgumentParser(description="参数探索")
    parser.add_argument("--obs", required=True, help="观测缓存 pkl 路径")
    args = parser.parse_args()

    logger.info(f"加载观测缓存: {args.obs}")
    result = load_observations(args.obs)
    obs = result.observations
    obs_1h, obs_2h = split_observations(obs)
    logger.info(f"总观测: {len(obs)}, 前半: {len(obs_1h)}, 后半: {len(obs_2h)}")

    t0 = time.time()

    # =====================================================================
    # Phase 1: 仓位限制
    # =====================================================================
    print("\n" + "=" * 80)
    print("Phase 1: 仓位限制 (entry_threshold=0.03, direction=both)")
    print("=" * 80)
    base_kwargs = dict(entry_threshold=0.03, direction_filter="both")
    rows_p1 = []
    for ms in [500, 1000, 2000, 3000, 5000, 10000]:
        kw = {**base_kwargs, "max_net_shares": ms}
        r = run_single(obs, obs_1h, obs_2h, **kw)
        r["label"] = f"max_shares={ms}"
        r["params"] = kw
        rows_p1.append(r)
    print_table(rows_p1)
    best_p1 = select_best(rows_p1)
    best_ms = best_p1["params"]["max_net_shares"]
    print(f">>> Phase 1 最优: max_shares={best_ms} (PF={fmt_pf(best_p1['pf'])}, DD={best_p1['dd']:.1f}%)")

    # =====================================================================
    # Phase 2: 时间窗口
    # =====================================================================
    print("\n" + "=" * 80)
    print(f"Phase 2: 时间窗口 (max_shares={best_ms})")
    print("=" * 80)
    time_configs = [
        (99999, 0, "全时段"),
        (720, 0, "T-12h起"),
        (360, 0, "T-6h起"),
        (180, 0, "T-3h起"),
        (99999, 5, "停5min"),
        (99999, 30, "停30min"),
        (720, 5, "T-12h,停5min"),
        (360, 5, "T-6h,停5min"),
        (360, 30, "T-6h,停30min"),
        (180, 5, "T-3h,停5min"),
    ]
    rows_p2 = []
    for max_om, min_om, desc in time_configs:
        kw = {**base_kwargs, "max_net_shares": best_ms,
              "max_obs_minutes": max_om, "min_obs_minutes": min_om}
        r = run_single(obs, obs_1h, obs_2h, **kw)
        r["label"] = desc
        r["params"] = kw
        rows_p2.append(r)
    print_table(rows_p2)
    best_p2 = select_best(rows_p2)
    best_max_om = best_p2["params"]["max_obs_minutes"]
    best_min_om = best_p2["params"]["min_obs_minutes"]
    print(f">>> Phase 2 最优: max_obs={best_max_om}, min_obs={best_min_om} (PF={fmt_pf(best_p2['pf'])}, DD={best_p2['dd']:.1f}%)")

    # =====================================================================
    # Phase 3: 非对称阈值
    # =====================================================================
    print("\n" + "=" * 80)
    print(f"Phase 3: 非对称阈值 (max_shares={best_ms}, obs=[{best_min_om},{best_max_om}])")
    print("=" * 80)
    asym_configs = [
        (0.03, 0.03, "both", "对称 0.03/0.03"),
        (0.05, 0.02, "both", "Y=0.05 N=0.02"),
        (0.05, 0.03, "both", "Y=0.05 N=0.03"),
        (0.07, 0.02, "both", "Y=0.07 N=0.02"),
        (0.07, 0.03, "both", "Y=0.07 N=0.03"),
        (0.10, 0.02, "both", "Y=0.10 N=0.02"),
        (0.10, 0.03, "both", "Y=0.10 N=0.03"),
        (None, 0.03, "no_only", "NO only t=0.03"),
        (None, 0.05, "no_only", "NO only t=0.05"),
    ]
    rows_p3 = []
    for yt, nt, dirn, desc in asym_configs:
        kw = dict(max_net_shares=best_ms,
                  max_obs_minutes=best_max_om, min_obs_minutes=best_min_om,
                  direction_filter=dirn, entry_threshold=nt)
        if dirn == "both":
            kw["yes_threshold"] = yt
            kw["no_threshold"] = nt
        r = run_single(obs, obs_1h, obs_2h, **kw)
        r["label"] = desc
        r["params"] = kw
        rows_p3.append(r)
    print_table(rows_p3)
    best_p3 = select_best(rows_p3)
    print(f">>> Phase 3 最优: {best_p3['label']} (PF={fmt_pf(best_p3['pf'])}, DD={best_p3['dd']:.1f}%)")

    # =====================================================================
    # Phase 4: Spread 过滤
    # =====================================================================
    print("\n" + "=" * 80)
    print(f"Phase 4: Spread 过滤")
    print("=" * 80)
    rows_p4 = []
    for ms_val in [None, 0.05, 0.03, 0.02, 0.01]:
        kw = {**best_p3["params"], "max_spread": ms_val}
        r = run_single(obs, obs_1h, obs_2h, **kw)
        r["label"] = f"spread<={ms_val}" if ms_val else "无限制"
        r["params"] = kw
        rows_p4.append(r)
    print_table(rows_p4)
    best_p4 = select_best(rows_p4)
    print(f">>> Phase 4 最优: {best_p4['label']} (PF={fmt_pf(best_p4['pf'])}, DD={best_p4['dd']:.1f}%)")

    # =====================================================================
    # Phase 5: 最优组合 + 对照
    # =====================================================================
    print("\n" + "=" * 80)
    print("Phase 5: 最优组合 vs 对照")
    print("=" * 80)
    rows_p5 = []

    # 最优组合
    kw = {**best_p4["params"]}
    r = run_single(obs, obs_1h, obs_2h, **kw)
    r["label"] = "★ 最优组合"
    r["params"] = kw
    rows_p5.append(r)

    # 对照: no_only
    kw_no = {**best_p4["params"], "direction_filter": "no_only",
             "yes_threshold": None, "entry_threshold": best_p4["params"].get("no_threshold", best_p4["params"].get("entry_threshold", 0.03))}
    r = run_single(obs, obs_1h, obs_2h, **kw_no)
    r["label"] = "对照: NO only"
    r["params"] = kw_no
    rows_p5.append(r)

    # 对照: 去掉时间窗口
    kw_nt = {**best_p4["params"], "max_obs_minutes": 99999, "min_obs_minutes": 0}
    r = run_single(obs, obs_1h, obs_2h, **kw_nt)
    r["label"] = "对照: 无时间窗口"
    r["params"] = kw_nt
    rows_p5.append(r)

    # 对照: 去掉 spread 过滤
    kw_ns = {**best_p4["params"], "max_spread": None}
    r = run_single(obs, obs_1h, obs_2h, **kw_ns)
    r["label"] = "对照: 无 spread 过滤"
    r["params"] = kw_ns
    rows_p5.append(r)

    # 对照: 去掉非对称阈值
    kw_sym = {**best_p4["params"], "yes_threshold": None, "no_threshold": None}
    r = run_single(obs, obs_1h, obs_2h, **kw_sym)
    r["label"] = "对照: 对称阈值"
    r["params"] = kw_sym
    rows_p5.append(r)

    # 基准: 原始 both t=0.03 max=10000
    kw_base = dict(entry_threshold=0.03, direction_filter="both", max_net_shares=10000)
    r = run_single(obs, obs_1h, obs_2h, **kw_base)
    r["label"] = "基准: both t=0.03 max=10K"
    r["params"] = kw_base
    rows_p5.append(r)

    # 基准: no_only t=0.05 max=10000
    kw_base2 = dict(entry_threshold=0.05, direction_filter="no_only", max_net_shares=10000)
    r = run_single(obs, obs_1h, obs_2h, **kw_base2)
    r["label"] = "基准: no_only t=0.05 max=10K"
    r["params"] = kw_base2
    rows_p5.append(r)

    print_table(rows_p5)

    elapsed = time.time() - t0
    logger.info(f"探索完成, 总耗时: {elapsed:.0f}s")

    # 打印最优参数
    print("\n" + "=" * 80)
    print("最优参数:")
    print("=" * 80)
    for k, v in best_p4["params"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
