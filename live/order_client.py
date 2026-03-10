"""
Polymarket 订单客户端

封装 py-clob-client ClobClient:
- 初始化: 私钥 + API 凭证 → L2 认证
- 下单: create_order + post_order
- 支持 GTC/FOK 订单类型
- 重试逻辑（指数退避）
"""

import logging
import sys
import time
from typing import Optional

from .config import LiveTradingConfig

logger = logging.getLogger(__name__)

# 添加 py-clob-client 到路径
_CLOB_CLIENT_PATH = "/home/ubuntu/pm_arb/libs/py-clob-client"
if _CLOB_CLIENT_PATH not in sys.path:
    sys.path.insert(0, _CLOB_CLIENT_PATH)

from py_clob_client import (
    ClobClient,
    ApiCreds,
    OrderArgs,
    OrderType,
    PartialCreateOrderOptions,
)

# 重试参数
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 0.5


class PolymarketOrderClient:
    """
    Polymarket 订单客户端

    封装 ClobClient 的 L2 模式:
    - create_order → 签名
    - post_order → 提交
    - cancel/cancel_all → 撤单
    """

    def __init__(self, config: LiveTradingConfig):
        self._config = config

        # L1 初始化（仅需 host + chain_id + private_key）
        self._client = ClobClient(
            host=config.polymarket_host,
            chain_id=config.polymarket_chain_id,
            key=config.polymarket_private_key,
            signature_type=config.polymarket_signature_type,
            funder=config.polymarket_funder or None,
        )

        # 自动派生 API 凭证 → 升级到 L2
        logger.info("正在派生 Polymarket API 凭证...")
        creds = self._client.create_or_derive_api_creds()
        if creds is None:
            raise RuntimeError("无法派生 Polymarket API 凭证，请检查私钥")
        self._client.set_api_creds(creds)

        logger.info(
            f"Polymarket 订单客户端已初始化 (L2): "
            f"host={config.polymarket_host}, "
            f"address={self._client.get_address()}"
        )

    def place_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
        order_type: str = "GTC",
        tick_size: str = "0.01",
        neg_risk: bool = False,
    ) -> dict:
        """
        签名并下单

        Args:
            token_id: YES/NO token ID
            side: "BUY" or "SELL"
            price: 下单价格 (0, 1)
            size: 份数
            order_type: "GTC" / "FOK"
            tick_size: 最小价格变动
            neg_risk: 是否为 neg_risk 市场

        Returns:
            API 响应 dict
        """
        ot = OrderType.FOK if order_type == "FOK" else OrderType.GTC

        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=size,
            side=side,
        )

        options = PartialCreateOrderOptions(
            tick_size=tick_size,
            neg_risk=neg_risk,
        )

        last_error = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                # 签名
                signed_order = self._client.create_order(order_args, options)
                # 提交
                result = self._client.post_order(signed_order, ot)
                logger.info(
                    f"下单成功: token={token_id[:12]}..., "
                    f"side={side}, price={price:.4f}, size={size}, "
                    f"type={order_type}"
                )
                return result
            except Exception as e:
                last_error = e
                err_str = str(e).lower()

                # 余额/授权不足：无需重试
                if "balance" in err_str or "allowance" in err_str:
                    logger.error(
                        f"余额/授权不足，跳过重试: {e}\n"
                        f"  请确认: 1) Polymarket 账户有足够 USDC  "
                        f"2) 已批准 CTF Exchange 合约的 USDC 授权"
                    )
                    return {"error": str(e), "success": False, "reason": "insufficient_balance"}

                logger.warning(
                    f"下单失败 (尝试 {attempt}/{_MAX_RETRIES}): {e}"
                )
                if attempt < _MAX_RETRIES:
                    delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    time.sleep(delay)

        logger.error(f"下单最终失败: {last_error}")
        return {"error": str(last_error), "success": False}

    def cancel_order(self, order_id: str) -> dict:
        """撤销单个订单"""
        try:
            result = self._client.cancel(order_id)
            logger.info(f"撤单成功: order_id={order_id}")
            return result
        except Exception as e:
            logger.error(f"撤单失败: order_id={order_id}, error={e}")
            return {"error": str(e)}

    def cancel_all(self) -> dict:
        """撤销所有订单"""
        try:
            result = self._client.cancel_all()
            logger.info("已撤销所有订单")
            return result
        except Exception as e:
            logger.error(f"撤销所有订单失败: {e}")
            return {"error": str(e)}

    def get_open_orders(self) -> list:
        """获取当前挂单"""
        try:
            return self._client.get_orders()
        except Exception as e:
            logger.error(f"获取挂单失败: {e}")
            return []
