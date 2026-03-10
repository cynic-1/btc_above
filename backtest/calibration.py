"""
校准分析模块
Isotonic regression 校准 + ECE/Brier before/after 比较
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """校准分析结果"""
    ece_before: float
    ece_after: float
    brier_before: float
    brier_after: float
    n_train: int
    n_test: int


class IsotonicCalibrator:
    """Isotonic regression 校准器包装"""

    def __init__(self):
        self._model = None

    def fit(self, preds: np.ndarray, labels: np.ndarray) -> bool:
        """拟合校准器。sklearn 不可用时返回 False"""
        try:
            from sklearn.isotonic import IsotonicRegression
            self._model = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            self._model.fit(preds, labels)
            return True
        except ImportError:
            logger.warning("sklearn 未安装，跳过 isotonic 校准")
            return False

    def transform(self, preds: np.ndarray) -> Optional[np.ndarray]:
        """校准预测值"""
        if self._model is None:
            return None
        return self._model.transform(preds)


def run_calibration_analysis(
    observations: list,
    train_frac: float = 0.6,
) -> Optional[CalibrationResult]:
    """
    校准分析：前 train_frac 训练 isotonic，后 1-train_frac 测试

    按 now_utc_ms 排序观测，提取所有 (pred, label) 对，
    前 60% 训练，后 40% 测试，计算 ECE/Brier before/after。

    sklearn 不可用时返回 None。
    """
    # Import from metrics (same package)
    from .metrics import brier_score, compute_ece

    # 收集所有 (pred, label) 对，按时间排序
    sorted_obs = sorted(observations, key=lambda o: o.now_utc_ms)
    pairs = []
    for obs in sorted_obs:
        for k in obs.k_grid:
            p = obs.predictions.get(k)
            y = obs.labels.get(k)
            if p is not None and y is not None:
                pairs.append((obs.now_utc_ms, p, y))

    if len(pairs) < 10:
        logger.warning(f"样本不足 ({len(pairs)})，跳过校准分析")
        return None

    # 按时间排序（已排序）
    split = int(len(pairs) * train_frac)
    train_pairs = pairs[:split]
    test_pairs = pairs[split:]

    if len(train_pairs) < 5 or len(test_pairs) < 5:
        logger.warning("训练/测试样本不足，跳过校准分析")
        return None

    train_preds = np.array([p for _, p, _ in train_pairs])
    train_labels = np.array([y for _, _, y in train_pairs])
    test_preds = np.array([p for _, p, _ in test_pairs])
    test_labels = np.array([y for _, _, y in test_pairs])

    # 拟合 isotonic
    calibrator = IsotonicCalibrator()
    if not calibrator.fit(train_preds, train_labels):
        return None

    # 校准后预测
    calibrated_preds = calibrator.transform(test_preds)
    if calibrated_preds is None:
        return None

    # 计算指标
    ece_before = compute_ece(test_preds, test_labels)
    ece_after = compute_ece(calibrated_preds, test_labels)
    brier_before = brier_score(test_preds, test_labels)
    brier_after = brier_score(calibrated_preds, test_labels)

    result = CalibrationResult(
        ece_before=ece_before,
        ece_after=ece_after,
        brier_before=brier_before,
        brier_after=brier_after,
        n_train=len(train_pairs),
        n_test=len(test_pairs),
    )

    logger.info(
        f"校准分析: ECE {ece_before:.4f} → {ece_after:.4f}, "
        f"Brier {brier_before:.4f} → {brier_after:.4f}"
    )
    return result
