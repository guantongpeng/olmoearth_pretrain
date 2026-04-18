"""
OlmoEarth Pretrain 数据包（data package）。

本包提供了 OlmoEarth 预训练任务所需的所有数据加载与处理组件，包括：
- constants: 模态规格、BandSet、分辨率常量等定义
- dataset: 基于 H5 文件的 OlmoEarthDataset 数据集类
- dataloader: OlmoEarthDataLoader 数据加载器，支持 DDP 和多进程
- collate: 批次整理（collate）函数，支持单/双掩码视图
- normalize: 归一化器，支持预定义和计算两种策略
- transform: 数据增强（翻转、旋转、Mixup 等）
- concat: 多数据集拼接
- utils: 坐标转换、流式统计等工具函数
- visualize: 数据样本可视化
"""

