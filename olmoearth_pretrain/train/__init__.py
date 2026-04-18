"""
训练模块初始化文件。

本模块提供 OlmoEarth Pretrain 的训练功能，需要安装 olmo-core 依赖。
使用方式：pip install olmoearth-pretrain[training]

主要功能：
- 训练模块基类 (OlmoEarthTrainModule) 及其子类（MAE、LatentMIM、Galileo 等）
- 掩码策略（随机掩码、空间掩码、时间掩码、跨模态掩码等）
- 损失函数（Patch判别损失、L1/L2损失、MAE重建损失等）
- 训练回调（下游评估、速度监控、W&B日志等）
"""

"""Training module for OlmoEarth Pretrain.

This module requires olmo-core. Install with: pip install olmoearth-pretrain[training]
"""

# 导入 olmo-core 依赖检查函数
from olmoearth_pretrain.config import require_olmo_core

# 确保 olmo-core 已安装，否则抛出错误
require_olmo_core("Training")
