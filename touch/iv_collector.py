"""
月底交割期权 ATM IV 定期采集器（方案 2）

通过 cron 或定时任务每小时运行，调用 Deribit get_option_chain_by_expiry()
获取最接近月底到期的 ATM IV，追加到 CSV 缓存文件。

缓存格式与 DeribitIVCache 兼容:
  data/deribit_iv/atm_iv_YYYY-MM.csv.gz
  columns: timestamp_ms, volatility

用法:
    # 手动运行（采集一次当前 ATM IV）
    python -m touch.iv_collector --month 2026-03

    # cron 每小时运行
    0 * * * * cd /home/ubuntu/arbitrage && python -m touch.iv_collector --month 2026-03
"""

import argparse
import csv
import gzip
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def collect_atm_iv(
    month: str,
    cache_dir: str = "data/deribit_iv",
    currency: str = "BTC",
) -> Optional[float]:
    """
    采集一次月底交割期权 ATM IV 并追加到缓存

    Args:
        month: 月份 "YYYY-MM"
        cache_dir: IV 缓存目录
        currency: 币种

    Returns:
        ATM IV (小数形式)，采集失败返回 None
    """
    from pricing_core.deribit_data import DeribitClient
    from pricing_core.time_utils import month_boundaries_utc_ms

    from .iv_source import DeribitIVSource

    os.makedirs(cache_dir, exist_ok=True)

    # 获取月底时间戳
    _, month_end_ms = month_boundaries_utc_ms(month)

    # 调用期权链获取 ATM IV
    client = DeribitClient()
    iv_source = DeribitIVSource(deribit_client=client)
    atm_iv = iv_source.get_atm_iv(target_expiry_ms=month_end_ms, currency=currency)

    if atm_iv is None:
        logger.warning(f"ATM IV 采集失败 ({month})")
        return None

    # 当前时间戳
    now_ms = int(time.time() * 1000)

    # 追加到缓存文件
    path = os.path.join(cache_dir, f"atm_iv_{month}.csv.gz")

    # 读取已有数据（如果有）
    existing_rows = []
    if os.path.exists(path):
        try:
            with gzip.open(path, "rt") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_rows.append((int(row["timestamp_ms"]), float(row["volatility"])))
        except Exception:
            pass

    # 去重：如果最后一条在 30 分钟内，跳过
    if existing_rows:
        last_ts = existing_rows[-1][0]
        if now_ms - last_ts < 30 * 60 * 1000:
            logger.info(f"距上次采集 <30min，跳过 (last={last_ts})")
            return atm_iv

    # 追加新数据
    # atm_iv 是小数形式 (0.50)，转为百分比存储 (50.0) 以与 DVOL 缓存格式一致
    existing_rows.append((now_ms, atm_iv * 100.0))

    # 重写文件（gzip 不支持追加）
    with gzip.open(path, "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ms", "volatility"])
        for ts, vol in existing_rows:
            writer.writerow([ts, vol])

    dt_str = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    logger.info(f"ATM IV 采集: {atm_iv:.4f} ({atm_iv*100:.2f}%) at {dt_str} → {path}")
    return atm_iv


def main():
    parser = argparse.ArgumentParser(description="采集月底交割期权 ATM IV")
    parser.add_argument("--month", required=True, help="目标月份 (YYYY-MM)")
    parser.add_argument("--cache-dir", default="data/deribit_iv", help="缓存目录")
    parser.add_argument("--currency", default="BTC", help="币种")
    args = parser.parse_args()

    from pricing_core.utils.logger import setup_logger
    setup_logger()

    iv = collect_atm_iv(
        month=args.month,
        cache_dir=args.cache_dir,
        currency=args.currency,
    )

    if iv is not None:
        print(f"ATM IV ({args.month}): {iv:.4f} ({iv*100:.2f}%)")
    else:
        print(f"ATM IV 采集失败")


if __name__ == "__main__":
    main()
