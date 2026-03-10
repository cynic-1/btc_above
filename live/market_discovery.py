"""
Gamma API 市场发现

发现指定日期的所有 BTC above 事件的 strike/condition_id，
复用 backtest/polymarket_discovery.py 中的解析模式
"""

import json
import logging
import re
import time
from datetime import datetime
from typing import List, Optional

import requests

from .config import LiveTradingConfig
from .models import MarketInfo

logger = logging.getLogger(__name__)

# 月份名称 → 数字
_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

# 匹配 "$85,000" 或 "$85000" 格式的价格
_STRIKE_RE = re.compile(r"\$\s*([\d,]+)")
# 匹配 "on March 5" 格式的日期
_DATE_RE = re.compile(
    r"on\s+(January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+(\d{1,2})",
    re.IGNORECASE,
)


def parse_strike_from_question(question: str) -> Optional[float]:
    """
    从问题中解析行权价
    'Will Bitcoin be above $85,000 on March 5?' → 85000.0
    """
    m = _STRIKE_RE.search(question)
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def parse_date_from_question(question: str, year: int = 2026) -> Optional[str]:
    """
    从问题中解析事件日期
    '... on March 5?' → '2026-03-05'
    """
    m = _DATE_RE.search(question)
    if not m:
        return None
    month_name = m.group(1).lower()
    day = int(m.group(2))
    month = _MONTH_MAP.get(month_name)
    if month is None:
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def _fetch_tick_size_and_neg_risk(
    token_id: str,
    clob_host: str,
) -> tuple:
    """通过 CLOB REST API 查询 tick_size 和 neg_risk"""
    tick_size = "0.01"
    neg_risk = False
    try:
        resp = requests.get(
            f"{clob_host}/tick-size",
            params={"token_id": token_id},
            timeout=10,
        )
        if resp.ok:
            data = resp.json()
            tick_size = str(data.get("minimum_tick_size", "0.01"))
    except Exception as e:
        logger.debug(f"获取 tick_size 失败: {e}")

    try:
        resp = requests.get(
            f"{clob_host}/neg-risk",
            params={"token_id": token_id},
            timeout=10,
        )
        if resp.ok:
            data = resp.json()
            neg_risk = data.get("neg_risk", False)
    except Exception as e:
        logger.debug(f"获取 neg_risk 失败: {e}")

    return tick_size, neg_risk


def discover_today_markets(
    event_date: str,
    gamma_api: str = "https://gamma-api.polymarket.com",
    clob_host: str = "https://clob.polymarket.com",
    year: int = 2026,
) -> List[MarketInfo]:
    """
    发现指定日期的所有 BTC above 市场

    1. 构造搜索词 "Bitcoin above {Month} {Day}"
    2. 调用 Gamma API public-search
    3. 解析 question → strike, date
    4. 提取 conditionId, clobTokenIds
    5. 查询 tick_size 和 neg_risk

    Args:
        event_date: "YYYY-MM-DD"
        gamma_api: Gamma API 地址
        clob_host: CLOB REST API 地址
        year: 年份（用于日期解析）

    Returns:
        MarketInfo 列表
    """
    dt = datetime.strptime(event_date, "%Y-%m-%d")
    month_name = dt.strftime("%B")
    day = dt.day
    query = f"Bitcoin above {month_name} {day}"

    logger.info(f"市场发现: 搜索 '{query}' (event_date={event_date})")

    results: List[MarketInfo] = []
    seen_conditions: set = set()

    for status_param in [None, "active"]:
        params = {"q": query}
        if status_param:
            params["events_status"] = status_param

        try:
            resp = requests.get(
                f"{gamma_api}/public-search",
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Gamma API 搜索失败 (q={query}, status={status_param}): {e}")
            continue

        events = data.get("events", [])
        for event in events:
            markets = event.get("markets", [])
            for market in markets:
                question = market.get("question", "")
                # 只处理 Bitcoin above 类型
                if "bitcoin" not in question.lower() or "above" not in question.lower():
                    continue

                strike = parse_strike_from_question(question)
                parsed_date = parse_date_from_question(question, year=year)
                if strike is None or parsed_date is None:
                    continue

                # 只保留目标日期
                if parsed_date != event_date:
                    continue

                # 解析 token IDs
                token_ids_raw = market.get("clobTokenIds", "[]")
                token_ids = (
                    json.loads(token_ids_raw)
                    if isinstance(token_ids_raw, str)
                    else token_ids_raw
                )
                if len(token_ids) < 2:
                    continue

                condition_id = market.get("conditionId", "")
                if not condition_id or condition_id in seen_conditions:
                    continue
                seen_conditions.add(condition_id)

                # 查询 tick_size 和 neg_risk
                tick_size, neg_risk = _fetch_tick_size_and_neg_risk(
                    token_ids[0], clob_host,
                )

                info = MarketInfo(
                    event_date=parsed_date,
                    strike=strike,
                    condition_id=condition_id,
                    yes_token_id=token_ids[0],
                    no_token_id=token_ids[1],
                    question=question,
                    tick_size=tick_size,
                    neg_risk=neg_risk,
                )
                results.append(info)
                logger.info(
                    f"  发现市场: K={strike:.0f}, cid={condition_id[:12]}..., "
                    f"tick={tick_size}, neg_risk={neg_risk}"
                )

        # 限速
        time.sleep(0.3)

    logger.info(f"市场发现完成: {len(results)} 个市场 (event_date={event_date})")
    return results
