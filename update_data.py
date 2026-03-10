#!/usr/bin/env python3
"""
数据增量更新脚本

自动发现新的 BTC above 事件 → 更新 events.json → 下载 CLOB 价格 → 下载 K线

用法:
    python update_data.py                        # 更新到今天 +7 天
    python update_data.py --until 2026-04-01     # 更新到指定日期
    python update_data.py --discover-only        # 仅发现新事件，不下载价格
    python update_data.py --prices-only          # 仅更新价格（不发现新事件）
    python update_data.py --klines-only          # 仅更新 K线缓存
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests

from backtest.polymarket_discovery import (
    PolymarketMarketInfo,
    parse_date_from_question,
    parse_strike_from_question,
)
from backtest.polymarket_trades import PolymarketTradeCache
from backtest.data_cache import KlineCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

EVENTS_JSON = "data/btc_above_events.json"
GAMMA_API = "https://gamma-api.polymarket.com"


# ── 1. 事件发现 ──────────────────────────────────────────────


def load_events(path: str = EVENTS_JSON) -> dict:
    """加载现有 events.json"""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_events(events: dict, path: str = EVENTS_JSON) -> None:
    """保存 events.json（按日期排序）"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sorted_events = dict(sorted(events.items()))
    with open(path, "w") as f:
        json.dump(sorted_events, f, indent=2, ensure_ascii=False)
    logger.info(f"保存 {len(sorted_events)} 个事件日 → {path}")


def _search_gamma_events(
    date_str: str,
    year: int = 2026,
) -> Optional[dict]:
    """
    通过 Gamma API 搜索指定日期的 BTC above 事件

    返回 events.json 格式的单日事件 dict，或 None
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    month_name = dt.strftime("%B")
    day = dt.day

    query = f"Bitcoin above {month_name} {day}"
    markets_found = []

    for status_param in [None, "active"]:
        params = {"q": query}
        if status_param:
            params["events_status"] = status_param

        try:
            resp = requests.get(
                f"{GAMMA_API}/public-search",
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Gamma API 搜索失败 (q={query}, status={status_param}): {e}")
            continue

        for event in data.get("events", []):
            event_title = event.get("title", "")
            event_slug = event.get("slug", "")

            for market in event.get("markets", []):
                question = market.get("question", "")
                if "bitcoin" not in question.lower() or "above" not in question.lower():
                    continue

                strike = parse_strike_from_question(question)
                evt_date = parse_date_from_question(question, year=year)
                if strike is None or evt_date is None or evt_date != date_str:
                    continue

                condition_id = market.get("conditionId", "")
                token_ids_raw = market.get("clobTokenIds", "[]")
                token_ids = (
                    json.loads(token_ids_raw)
                    if isinstance(token_ids_raw, str)
                    else token_ids_raw
                )
                if not condition_id or len(token_ids) < 2:
                    continue

                # 去重
                if any(m["conditionId"] == condition_id for m in markets_found):
                    continue

                slug = market.get("slug", "")
                group_title = market.get("groupItemTitle", f"{strike:,.0f}")

                markets_found.append({
                    "question": question,
                    "conditionId": condition_id,
                    "clobTokenIds": token_ids,
                    "groupItemTitle": group_title,
                    "slug": slug,
                })

            # 记录事件级别信息
            if markets_found and not hasattr(_search_gamma_events, "_title"):
                _search_gamma_events._title = event_title
                _search_gamma_events._slug = event_slug

    if not markets_found:
        return None

    title = getattr(_search_gamma_events, "_title", f"Bitcoin above ___ on {month_name} {day}?")
    slug = getattr(_search_gamma_events, "_slug", f"bitcoin-above-on-{month_name.lower()}-{day}")

    # 清理临时属性
    for attr in ("_title", "_slug"):
        if hasattr(_search_gamma_events, attr):
            delattr(_search_gamma_events, attr)

    # 按 strike 排序
    markets_found.sort(key=lambda m: parse_strike_from_question(m["question"]) or 0)

    return {
        "title": title,
        "slug": slug,
        "date": date_str,
        "markets": markets_found,
    }


def discover_new_events(
    until_date: str,
    events: dict,
) -> Tuple[dict, int]:
    """
    发现 events.json 中缺失的事件日

    Args:
        until_date: 搜索截止日期 (含)
        events: 现有事件字典

    Returns:
        (更新后的 events, 新发现天数)
    """
    existing_dates = set(events.keys())

    # 确定搜索范围: 从最后已有日期的下一天 → until_date
    if existing_dates:
        last_date = max(existing_dates)
        start_dt = datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
    else:
        start_dt = datetime(2026, 1, 1)

    end_dt = datetime.strptime(until_date, "%Y-%m-%d")

    if start_dt > end_dt:
        logger.info(f"events.json 已覆盖到 {last_date}，无需发现新事件")
        return events, 0

    # 逐日搜索
    new_count = 0
    current = start_dt
    while current <= end_dt:
        date_str = current.strftime("%Y-%m-%d")
        if date_str not in existing_dates:
            logger.info(f"搜索 {date_str}...")
            event_data = _search_gamma_events(date_str)
            if event_data and event_data["markets"]:
                events[date_str] = event_data
                n = len(event_data["markets"])
                logger.info(f"  发现 {n} 个市场")
                new_count += 1
            else:
                logger.info(f"  未找到市场（可能尚未上线）")
            time.sleep(0.5)  # 限速

        current += timedelta(days=1)

    return events, new_count


# ── 2. CLOB 价格更新 ─────────────────────────────────────────


def update_clob_prices(
    events: dict,
    cache_dir: str = "data/polymarket",
    force: bool = False,
) -> int:
    """
    更新所有合约的 CLOB 价格历史

    Returns:
        新下载的合约数
    """
    cache = PolymarketTradeCache(cache_dir=cache_dir)

    # 收集所有 (condition_id, yes_token_id)
    cid_map: Dict[str, Tuple[str, str, float]] = {}
    for date_str, event in events.items():
        for mkt in event.get("markets", []):
            cid = mkt.get("conditionId", "")
            token_ids = mkt.get("clobTokenIds", [])
            if cid and len(token_ids) >= 2:
                strike = parse_strike_from_question(mkt.get("question", "")) or 0
                cid_map[cid] = (token_ids[0], date_str, strike)

    # 筛选需要下载的
    if force:
        to_download = list(cid_map.items())
    else:
        to_download = [(cid, info) for cid, info in cid_map.items()
                       if not cache.has_trades(cid)]

    if not to_download:
        logger.info(f"CLOB 价格缓存完整 ({len(cid_map)} 个合约)")
        return 0

    logger.info(f"下载 CLOB 价格: {len(to_download)}/{len(cid_map)} 个合约")

    success = 0
    for i, (cid, (token_id, date_str, strike)) in enumerate(to_download):
        logger.info(f"  [{i + 1}/{len(to_download)}] {date_str} K={strike:.0f}")
        cache.download_prices(cid, token_id)
        if cache.has_trades(cid):
            success += 1
        if i < len(to_download) - 1:
            time.sleep(0.3)

    logger.info(f"CLOB 价格下载完成: {success}/{len(to_download)} 成功")
    return success


# ── 3. K线更新 ────────────────────────────────────────────────


def update_klines(
    events: dict,
    cache_dir: str = "data/klines",
    har_train_days: int = 30,
) -> int:
    """
    确保所有事件日 + HAR 训练所需的 K线数据已缓存

    Returns:
        新下载的天数
    """
    if not events:
        return 0

    dates = sorted(events.keys())
    earliest_event = dates[0]
    latest_event = dates[-1]

    # HAR 需要事件日前 har_train_days 天的数据
    start_dt = datetime.strptime(earliest_event, "%Y-%m-%d") - timedelta(days=har_train_days + 1)
    start_date = start_dt.strftime("%Y-%m-%d")

    # 结束日期: 最晚事件日 + 1 天（包含结算 K线）
    end_dt = datetime.strptime(latest_event, "%Y-%m-%d") + timedelta(days=1)
    # 不超过今天
    today = datetime.now()
    if end_dt > today:
        end_dt = today
    end_date = end_dt.strftime("%Y-%m-%d")

    logger.info(f"K线缓存范围: {start_date} ~ {end_date}")
    cache = KlineCache(cache_dir=cache_dir)
    downloaded = cache.ensure_range(start_date, end_date)
    return len(downloaded)


# ── CLI ───────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="数据增量更新")
    parser.add_argument(
        "--until",
        help="事件发现截止日期 (YYYY-MM-DD)，默认今天 +7 天",
    )
    parser.add_argument(
        "--events-json", default=EVENTS_JSON,
        help=f"事件 JSON 路径 (默认: {EVENTS_JSON})",
    )
    parser.add_argument(
        "--discover-only", action="store_true",
        help="仅发现新事件，不下载价格/K线",
    )
    parser.add_argument(
        "--prices-only", action="store_true",
        help="仅更新 CLOB 价格（含已有空缓存的重试）",
    )
    parser.add_argument(
        "--klines-only", action="store_true",
        help="仅更新 K线缓存",
    )
    parser.add_argument(
        "--force-prices", action="store_true",
        help="强制重新下载所有 CLOB 价格",
    )
    parser.add_argument(
        "--polymarket-cache-dir", default="data/polymarket",
        help="Polymarket 缓存目录",
    )
    parser.add_argument(
        "--klines-cache-dir", default="data/klines",
        help="K线缓存目录",
    )
    args = parser.parse_args()

    # 默认截止日期: 今天 + 7 天
    if args.until:
        until_date = args.until
    else:
        until_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")

    events = load_events(args.events_json)
    logger.info(f"现有事件: {len(events)} 天 ({min(events.keys()) if events else 'N/A'} ~ {max(events.keys()) if events else 'N/A'})")

    # 选择性执行
    only_mode = args.discover_only or args.prices_only or args.klines_only

    # Step 1: 发现新事件
    if not args.prices_only and not args.klines_only:
        events, new_events = discover_new_events(until_date, events)
        if new_events > 0:
            save_events(events, args.events_json)
            logger.info(f"新发现 {new_events} 个事件日")
        else:
            logger.info("无新事件")

        if args.discover_only:
            return

    # Step 2: 更新 CLOB 价格
    if not args.discover_only and not args.klines_only:
        # 如果 --prices-only + --force-prices，重新下载之前空的缓存
        new_prices = update_clob_prices(
            events,
            cache_dir=args.polymarket_cache_dir,
            force=args.force_prices,
        )
        logger.info(f"CLOB 价格更新: {new_prices} 个新下载")

        if args.prices_only:
            return

    # Step 3: 更新 K线
    if not args.discover_only and not args.prices_only:
        new_klines = update_klines(
            events,
            cache_dir=args.klines_cache_dir,
        )
        logger.info(f"K线更新: {new_klines} 天新下载")

    logger.info("全部更新完成!")


if __name__ == "__main__":
    main()
