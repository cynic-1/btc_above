"""
观测缓存：序列化/反序列化回测观测结果
支持保存和加载，避免重复计算定价
"""

import logging
import os
import pickle
from typing import Optional

from .models import BacktestResult

logger = logging.getLogger(__name__)

CACHE_VERSION = 2


def save_observations(result: BacktestResult, output_dir: str, tag: Optional[str] = None) -> str:
    """序列化 BacktestResult 到 pkl 文件"""
    os.makedirs(output_dir, exist_ok=True)
    if tag is None:
        tag = f"{result.start_date}_{result.end_date}"
    path = os.path.join(output_dir, f"observations_{tag}.pkl")
    payload = {
        "version": CACHE_VERSION,
        "tag": tag,
        "start_date": result.start_date,
        "end_date": result.end_date,
        "n_observations": len(result.observations),
        "n_events": len(result.event_outcomes),
        "observations": result.observations,
        "event_outcomes": result.event_outcomes,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    logger.info(f"观测缓存已保存: {len(result.observations)} 观测, "
                f"{len(result.event_outcomes)} 事件 -> {path} ({size_mb:.1f}MB)")
    return path


def load_observations(path: str) -> BacktestResult:
    """从 pkl 文件加载 BacktestResult"""
    with open(path, "rb") as f:
        data = pickle.load(f)
    version = data.get("version")
    if version != CACHE_VERSION:
        raise ValueError(f"缓存版本不匹配: 文件={version}, 当前={CACHE_VERSION}。请重新运行完整回测以生成新缓存")
    observations = data.get("observations", [])
    event_outcomes = data.get("event_outcomes", [])
    result = BacktestResult(
        start_date=data.get("start_date", ""),
        end_date=data.get("end_date", ""),
        observations=observations,
        event_outcomes=event_outcomes,
    )
    size_mb = os.path.getsize(path) / (1024 * 1024)
    logger.info(f"观测缓存已加载: {len(observations)} 观测, {len(event_outcomes)} 事件 <- {path} ({size_mb:.1f}MB)")
    return result
