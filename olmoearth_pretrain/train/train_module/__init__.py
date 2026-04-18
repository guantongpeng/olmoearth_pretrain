"""
训练模块子包初始化文件。

本子包包含 OlmoEarth Pretrain 的各种训练模块实现：
- train_module.py: 基类 OlmoEarthTrainModule，定义训练的基本框架
  （优化器、EMA目标编码器更新、梯度裁剪、状态字典管理等）
- mae.py: MAE（掩码自编码器）训练模块，用于重建预训练
- latent_mim.py: Latent MIM 训练模块，基于 latent 空间的掩码图像建模
- contrastive_latentmim.py: 对比学习 + Latent MIM 联合训练模块
- galileo.py: Galileo 训练模块，双视角掩码策略 + 对比学习
"""

"""OlmoEarth Pretrain train modules."""
