"""
Deribit 数据模块
提供期权链、指数价格、永续标记价格等数据获取
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import requests

from .utils.helpers import TokenBucket

logger = logging.getLogger(__name__)


@dataclass
class OptionQuote:
    """期权报价"""
    instrument_name: str
    strike: float
    option_type: str          # "call" 或 "put"
    bid_price: Optional[float]
    ask_price: Optional[float]
    mid_price: Optional[float]
    mark_iv: float            # 标记隐含波动率 (%)
    underlying_price: float
    expiry_timestamp: int     # 到期时间 (UTC ms)


@dataclass
class PerpInfo:
    """永续合约信息"""
    mark_price: float
    index_price: float
    funding_rate: float
    basis: float              # mark - index


class DeribitClient:
    """Deribit REST API 客户端"""

    def __init__(self, base_url: str = "https://www.deribit.com/api/v2", max_rps: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self._limiter = TokenBucket(rate=max_rps)

    def _get(self, method: str, params: Optional[dict] = None) -> dict:
        """发送 GET 请求"""
        self._limiter.acquire()
        url = f"{self.base_url}/public/{method}"
        resp = self.session.get(url, params=params or {}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if "result" not in data:
            raise ValueError(f"Deribit API 异常响应: {data}")
        return data["result"]

    def get_index_price(self, currency: str = "BTC") -> float:
        """
        获取 Deribit 指数价格

        Args:
            currency: 币种（默认 BTC）

        Returns:
            指数价格
        """
        result = self._get("get_index_price", {"index_name": f"btc_usd"})
        price = result["index_price"]
        logger.debug(f"Deribit index price: {price}")
        return float(price)

    def get_perp_mark(self, instrument: str = "BTC-PERPETUAL") -> PerpInfo:
        """
        获取永续合约标记价格和资金费率

        Args:
            instrument: 合约名称

        Returns:
            PerpInfo 数据
        """
        ticker = self._get("ticker", {"instrument_name": instrument})

        mark_price = float(ticker["mark_price"])
        index_price = float(ticker["index_price"])
        funding_rate = float(ticker.get("current_funding", 0.0))

        info = PerpInfo(
            mark_price=mark_price,
            index_price=index_price,
            funding_rate=funding_rate,
            basis=mark_price - index_price,
        )
        logger.debug(f"Deribit perp: mark={mark_price}, index={index_price}, basis={info.basis}")
        return info

    def get_option_chain(self, currency: str = "BTC", expired: bool = False) -> List[OptionQuote]:
        """
        获取期权链

        Args:
            currency: 币种
            expired: 是否包含已到期期权

        Returns:
            OptionQuote 列表
        """
        # 获取所有期权合约
        instruments = self._get("get_instruments", {
            "currency": currency,
            "kind": "option",
            "expired": str(expired).lower(),
        })

        quotes: List[OptionQuote] = []
        # 批量获取报价
        for inst in instruments:
            inst_name = inst["instrument_name"]
            try:
                ticker = self._get("ticker", {"instrument_name": inst_name})

                bid = ticker.get("best_bid_price")
                ask = ticker.get("best_ask_price")
                mid = None
                if bid is not None and ask is not None and bid > 0 and ask > 0:
                    mid = (float(bid) + float(ask)) / 2

                quote = OptionQuote(
                    instrument_name=inst_name,
                    strike=float(inst["strike"]),
                    option_type=inst["option_type"],
                    bid_price=float(bid) if bid and float(bid) > 0 else None,
                    ask_price=float(ask) if ask and float(ask) > 0 else None,
                    mid_price=mid,
                    mark_iv=float(ticker.get("mark_iv", 0)),
                    underlying_price=float(ticker.get("underlying_price", 0)),
                    expiry_timestamp=int(inst["expiration_timestamp"]),
                )
                quotes.append(quote)
            except Exception as e:
                logger.warning(f"获取 {inst_name} 报价失败: {e}")

        logger.info(f"获取期权链: {currency}, 共 {len(quotes)} 条报价")
        return quotes

    def get_option_chain_by_expiry(
        self,
        currency: str = "BTC",
        expiry_timestamp: Optional[int] = None,
    ) -> List[OptionQuote]:
        """
        获取指定到期日的期权链

        Args:
            currency: 币种
            expiry_timestamp: 目标到期时间 (UTC ms)，选最接近且晚于此时间的到期日

        Returns:
            OptionQuote 列表（已按 strike 排序）
        """
        all_quotes = self.get_option_chain(currency)

        if expiry_timestamp is None:
            return sorted(all_quotes, key=lambda q: q.strike)

        # 找出所有到期时间
        expiries = sorted(set(q.expiry_timestamp for q in all_quotes))

        # 选择最接近且 >= expiry_timestamp 的到期日
        target_expiry = None
        for exp in expiries:
            if exp >= expiry_timestamp:
                target_expiry = exp
                break

        if target_expiry is None and expiries:
            target_expiry = expiries[-1]  # 回退到最远到期日

        if target_expiry is None:
            return []

        filtered = [q for q in all_quotes if q.expiry_timestamp == target_expiry]
        logger.info(f"筛选到期日 {target_expiry}: {len(filtered)} 条报价")
        return sorted(filtered, key=lambda q: q.strike)
