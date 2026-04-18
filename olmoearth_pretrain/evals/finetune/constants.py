"""微调训练超参数常量。

本模块定义了微调训练流程中使用的默认超参数常量，
包括骨干网络冻结策略、学习率调度器参数等。

常量说明：
- FREEZE_EPOCH_FRACTION: 骨干网络冻结的比例（前 20% 的 epoch 冻结骨干网络）
- UNFREEZE_LR_FACTOR: 解冻骨干网络时的学习率缩减因子（降为原来的 0.1）
- SCHEDULER_FACTOR: 验证指标停滞时的学习率缩减因子
- SCHEDULER_PATIENCE: 验证指标停滞多少 epoch 后缩减学习率
- SCHEDULER_MIN_LR: 学习率下限
- SCHEDULER_COOLDOWN: 学习率缩减后的冷却 epoch 数
"""

# Fraction of total epochs to keep backbone frozen before unfreezing.
FREEZE_EPOCH_FRACTION = 0.2

# Factor to multiply learning rate by when unfreezing backbone (e.g., 0.1 = reduce LR by 10x).
UNFREEZE_LR_FACTOR = 0.1

# Factor by which to reduce learning rate when validation metric plateaus.
SCHEDULER_FACTOR = 0.2

# Number of epochs with no improvement before reducing learning rate.
SCHEDULER_PATIENCE = 2

# Minimum learning rate the scheduler can reduce to.
SCHEDULER_MIN_LR = 0.0

# Number of epochs to wait after a LR reduction before resuming normal operation.
SCHEDULER_COOLDOWN = 10
