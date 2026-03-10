#!/usr/bin/env python3
"""
从 archive.pmxt.dev 下载 Polymarket 订单簿快照，
提取 "Bitcoin above ___" 相关市场数据，保存后删除原始文件。

数据源: https://r2.pmxt.dev/polymarket_orderbook_{timestamp}.parquet
时间范围: 2026-02-21T16 UTC 至最新

流水线架构:
  下载线程池(8) → 队列 → 处理线程(1): 读parquet+过滤+保存+删除
  下载是I/O密集不占内存，读parquet是内存密集所以串行处理
"""

import gc
import logging
import subprocess
import threading
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

import pyarrow.parquet as pq
import pyarrow as pa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "polymarket_btc_above"
TEMP_DIR = DATA_DIR / "polymarket_raw"
CONDITION_IDS_FILE = DATA_DIR / "btc_above_condition_ids.txt"

R2_BASE = "https://r2.pmxt.dev"

# 下载并发数（I/O密集，不占内存）
DOWNLOAD_WORKERS = 8
# 磁盘上同时存在的原始文件上限（每个~600MB，3个≈1.8GB）
MAX_PENDING_FILES = 3


def load_condition_ids() -> set[str]:
    """加载目标市场的 conditionId 集合"""
    with open(CONDITION_IDS_FILE) as f:
        return {line.strip() for line in f if line.strip()}


def generate_file_urls(start: datetime, end: datetime) -> list[tuple[str, str]]:
    """生成时间范围内所有小时快照的 (url, filename) 列表"""
    urls = []
    current = start
    while current <= end:
        fname = f"polymarket_orderbook_{current.strftime('%Y-%m-%dT%H')}.parquet"
        url = f"{R2_BASE}/{fname}"
        urls.append((url, fname))
        current += timedelta(hours=1)
    return urls


def download_file(url: str, dest: Path) -> bool:
    """用 wget 下载文件，返回是否成功"""
    try:
        result = subprocess.run(
            ["wget", "-q", "--timeout=120", "--tries=3", url, "-O", str(dest)],
            capture_output=True,
            timeout=600,
        )
        if result.returncode != 0:
            logger.warning(f"下载失败: {url}")
            dest.unlink(missing_ok=True)
            return False
        if dest.stat().st_size < 1_000_000:
            logger.warning(f"文件太小，可能无效: {dest.name} ({dest.stat().st_size} bytes)")
            dest.unlink(missing_ok=True)
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.warning(f"下载超时: {url}")
        dest.unlink(missing_ok=True)
        return False


def extract_and_save(src: Path, out_path: Path, condition_ids: set[str]) -> int:
    """从 parquet 文件中提取目标市场数据，保存并删除原始文件。返回行数。"""
    try:
        table = pq.read_table(
            src,
            filters=[("market_id", "in", condition_ids)],
        )
        rows = table.num_rows
        if rows > 0:
            pq.write_table(table, out_path)
        del table
        gc.collect()
        return rows
    except Exception as e:
        logger.error(f"无法读取 parquet: {src.name} - {e}")
        return 0
    finally:
        src.unlink(missing_ok=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    condition_ids = load_condition_ids()
    logger.info(f"目标市场数: {len(condition_ids)}")

    start = datetime(2026, 2, 21, 16)
    end = datetime(2026, 3, 7, 8)
    urls = generate_file_urls(start, end)
    logger.info(f"需要处理 {len(urls)} 个小时快照文件")

    done_files = {f.stem for f in OUTPUT_DIR.glob("*.parquet")}
    logger.info(f"已完成: {len(done_files)} 个")

    # 过滤已完成的
    pending = []
    skipped = 0
    for url, fname in urls:
        out_stem = fname.replace("polymarket_orderbook_", "btc_above_").replace(".parquet", "")
        if out_stem in done_files:
            skipped += 1
            continue
        pending.append((url, fname, out_stem))

    logger.info(f"跳过: {skipped}, 待处理: {len(pending)}")

    if not pending:
        logger.info("全部完成!")
        return

    # 信号量控制磁盘上待处理文件数（限制磁盘占用）
    disk_semaphore = threading.Semaphore(MAX_PENDING_FILES)
    # 处理队列: 下载完成后放入队列，处理线程逐个读取
    process_queue: Queue[tuple[str, Path, Path] | None] = Queue()

    total_rows = 0
    processed = 0
    failed = 0
    lock = threading.Lock()

    def download_one(url: str, fname: str, out_stem: str):
        """下载线程: 获取信号量 → 下载 → 放入队列"""
        tmp_path = TEMP_DIR / fname
        out_path = OUTPUT_DIR / f"{out_stem}.parquet"

        # 等待磁盘空间（限制同时存在的大文件数）
        disk_semaphore.acquire()

        if not download_file(url, tmp_path):
            disk_semaphore.release()
            with lock:
                nonlocal failed
                failed += 1
            return

        # 下载成功，放入处理队列
        process_queue.put((fname, tmp_path, out_path))

    def process_worker():
        """处理线程: 从队列取文件 → 读parquet+过滤+保存 → 删除原始文件 → 释放信号量"""
        nonlocal total_rows, processed, failed
        while True:
            item = process_queue.get()
            if item is None:  # 退出信号
                break

            fname, tmp_path, out_path = item
            rows = extract_and_save(tmp_path, out_path, condition_ids)

            # 原始文件已删除，释放磁盘信号量
            disk_semaphore.release()

            with lock:
                if rows > 0:
                    total_rows += rows
                    processed += 1
                    logger.info(f"  {fname}: 提取 {rows} 行 (累计: {processed} 成功, {total_rows} 行)")
                else:
                    failed += 1
                    logger.info(f"  {fname}: 无匹配数据")

    # 启动处理线程（单线程，避免内存爆炸）
    processor = threading.Thread(target=process_worker, daemon=True)
    processor.start()

    # 启动下载线程池
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        futures = []
        for url, fname, out_stem in pending:
            fut = executor.submit(download_one, url, fname, out_stem)
            futures.append(fut)

        # 等待所有下载完成
        for fut in as_completed(futures):
            fut.result()  # 触发异常（如有）

    # 发送退出信号给处理线程
    process_queue.put(None)
    processor.join()

    logger.info(f"完成! 提取成功 {processed}, 跳过 {skipped}, 无数据/失败 {failed}, 共 {total_rows} 行")

    # 清理
    if TEMP_DIR.exists() and not any(TEMP_DIR.iterdir()):
        TEMP_DIR.rmdir()


if __name__ == "__main__":
    main()
