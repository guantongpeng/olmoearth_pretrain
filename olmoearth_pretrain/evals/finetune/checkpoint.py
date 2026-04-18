"""微调训练检查点保存/加载工具。

本模块提供微调训练的检查点管理功能，支持训练中断后的恢复。

主要函数：
- save_training_checkpoint: 原子性地保存可恢复的训练检查点
- load_training_checkpoint: 加载训练检查点并验证完整性

设计要点：
- 使用临时文件+重命名实现原子写入，防止写入过程中崩溃导致文件损坏
- 检查点包含完整训练状态：模型、优化器、调度器、最佳模型状态等
- 加载时验证所有必要键的存在
"""

from __future__ import annotations

import os
import shutil
import tempfile
from logging import getLogger

import torch

logger = getLogger(__name__)


def save_training_checkpoint(
    path: str,
    epoch: int,
    model_state: dict[str, torch.Tensor],
    optimizer_state: dict,
    scheduler_state: dict,
    best_state: dict[str, torch.Tensor],
    best_val_metric: float,
    backbone_unfrozen: bool,
) -> None:
    """原子性地保存可恢复的训练检查点。

    使用"先写临时文件，再重命名"的策略确保原子性（POSIX 系统上重命名是原子操作）。

    Args:
        path: 检查点保存路径
        epoch: 当前 epoch 编号
        model_state: 模型状态字典
        optimizer_state: 优化器状态字典
        scheduler_state: 学习率调度器状态字典
        best_state: 最佳模型的状态字典
        best_val_metric: 最佳验证指标值
        backbone_unfrozen: 骨干网络是否已解冻
    """
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "best_state": best_state,
        "best_val_metric": best_val_metric,
        "backbone_unfrozen": backbone_unfrozen,
    }

    # Write to temp file first, then atomic rename
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=dir_path if dir_path else ".",
    )
    try:
        torch.save(checkpoint, tmp_path)
        os.close(tmp_fd)
        shutil.move(tmp_path, path)  # Atomic on POSIX
        logger.info(f"Saved training checkpoint to {path} at epoch {epoch}")
    except Exception:
        os.close(tmp_fd)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_training_checkpoint(path: str, device: torch.device) -> dict:
    """加载训练检查点并验证完整性。

    Args:
        path: 检查点文件路径
        device: 目标设备

    Returns:
        dict: 包含完整训练状态的检查点字典

    Raises:
        RuntimeError: 如果加载失败
        ValueError: 如果检查点缺少必要的键
    """
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {path}: {e}") from e
    required = [
        "epoch",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "best_state",
        "best_val_metric",
        "backbone_unfrozen",
    ]
    missing = [k for k in required if k not in ckpt]
    if missing:
        raise ValueError(f"Checkpoint {path} missing keys: {missing}")
    logger.info(f"Loaded training checkpoint from {path}")
    return ckpt
