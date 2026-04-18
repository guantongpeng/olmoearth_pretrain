"""评估流程工具函数模块。

本模块提供评估流程中使用的辅助工具函数。

核心函数：
- adjust_learning_rate: 学习率调整函数，实现余弦退火调度
  先线性预热 (warmup)，然后半周期余弦衰减到最小学习率

使用场景：
  在线性探针训练过程中，每个训练步调用此函数调整学习率。
"""

import math

import torch


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: float,
    warmup_epochs: int,
    total_epochs: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """调整学习率：预热阶段线性增长，之后半周期余弦衰减。

    调度策略：
    - epoch < warmup_epochs: 线性预热，lr = max_lr * epoch / warmup_epochs
    - epoch >= warmup_epochs: 余弦退火，lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))

    Args:
        optimizer: 优化器实例
        epoch: 当前 epoch（支持浮点数用于步级调度）
        warmup_epochs: 预热 epoch 数
        total_epochs: 总 epoch 数
        max_lr: 最大学习率
        min_lr: 最小学习率

    Returns:
        float: 调整后的学习率
    """
    if epoch < warmup_epochs:
        # 线性预热阶段
        lr = max_lr * epoch / warmup_epochs
    else:
        # 余弦退火阶段
        lr = min_lr + (max_lr - min_lr) * 0.5 * (
            1.0
            + math.cos(
                math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            )
        )
    # 更新优化器中所有参数组的学习率
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr
