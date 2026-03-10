"""
日志配置模块
提供统一的日志配置和 print 函数替换
"""

import os
import logging
import builtins as _builtins
from datetime import datetime
from typing import Optional


def setup_logger(log_dir: str = "logs", log_pointer_env: Optional[str] = None) -> None:
    """
    配置日志系统，将 print 替换为基于 logger 的函数

    Args:
        log_dir: 日志文件目录
        log_pointer_env: 指向当前日志文件的指针文件路径
    """
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        pass

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.abspath(os.path.join(log_dir, f"pricing_{ts}.log"))

    # 移除现有处理器
    for h in list(logging.root.handlers):
        logging.root.removeHandler(h)

    fmt = logging.Formatter(
        '%(asctime)s.%(msecs)03d %(filename)s:%(lineno)d %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    fh = logging.FileHandler(logfile, encoding='utf-8')
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(fh)
    logging.root.addHandler(sh)

    # 写入指针文件
    if log_pointer_env:
        pointer_file = os.path.abspath(log_pointer_env)
    else:
        pointer_file = os.path.abspath(os.path.join(log_dir, "CURRENT_LOG"))

    try:
        with open(pointer_file, "w", encoding="utf-8") as pf:
            pf.write(logfile)
    except Exception:
        pass

    _replace_print_with_logger()

    logger = logging.getLogger(__name__)
    logger.info(f"日志系统已初始化: {logfile}")


def _replace_print_with_logger() -> None:
    """将内置 print 替换为 logger 版本"""
    _logger = logging.getLogger("print_replacement")

    def _print(*args, sep=' ', end='\n', file=None, flush=False, level=logging.INFO):
        try:
            msg = sep.join(str(a) for a in args)
        except Exception:
            msg = ' '.join([repr(a) for a in args])
        try:
            _logger.log(level, msg, stacklevel=3)
        except TypeError:
            _logger.log(level, msg)

    _builtins.print = _print
