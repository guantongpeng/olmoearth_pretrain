"""微调模型组件模块。

本模块定义了微调训练所需的模型组件和工具函数。

主要组件：
- BackboneWithHead: 骨干网络 + 分类/分割头 的组合模型
- to_device: 将 MaskedOlmoEarthSample 移动到指定设备
- snapshot_state_dict: 克隆模型状态字典到 CPU
- set_backbone_trainable: 切换骨干网络的可训练状态
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from olmoearth_pretrain.evals.datasets.configs import TaskType
from olmoearth_pretrain.evals.eval_wrapper import get_eval_wrapper
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample


class BackboneWithHead(nn.Module):
    """骨干网络 + 分类/分割头 的组合模型，用于微调训练。

    在预训练骨干网络上添加任务特定的线性头：
    - 分类任务: Linear(emb_dim, num_classes)
    - 分割任务: Linear(emb_dim, num_classes * patch_size^2)

    头的输入维度 (emb_dim) 在第一次前向传播时自动推断，
    无需手动指定。

    关键属性：
        backbone: 预训练骨干网络
        wrapper: EvalWrapper 评估包装器
        task_type: 任务类型
        patch_size: patch 大小
        num_classes: 类别数
        _head: 线性头（首次前向传播时初始化）
        _inited: 头是否已初始化
    """

    def __init__(
        self,
        model: nn.Module,
        task_type: TaskType,
        patch_size: int,
        pooling_type: str,
        num_classes: int,
        use_pooled_tokens: bool = False,
    ) -> None:
        """Initialize the backbone with head."""
        super().__init__()
        self.backbone = model
        self.wrapper = get_eval_wrapper(
            model,
            task_type=task_type,
            patch_size=patch_size,
            pooling_type=pooling_type,
            concat_features=False,
            use_pooled_tokens=use_pooled_tokens,
        )
        self.task_type = task_type
        self.patch_size = patch_size
        self.num_classes = num_classes
        # placeholder head; real in_dim discovered on first forward
        self._head = nn.Linear(1, 1, bias=True)
        self._inited = False

    def _init_head(self, emb_dim: int, device: torch.device) -> None:
        """根据嵌入维度初始化线性头。

        分类任务: 输出维度 = num_classes
        分割任务: 输出维度 = num_classes * patch_size^2 (每个 patch 预测多个像素)

        Args:
            emb_dim: 嵌入维度
            device: 目标设备
        """
        if self.task_type == TaskType.CLASSIFICATION:
            self._head = nn.Linear(emb_dim, self.num_classes, bias=True)
        else:
            logits_per_patch = int(self.num_classes * self.patch_size * self.patch_size)
            self._head = nn.Linear(emb_dim, logits_per_patch, bias=True)

        self._head = self._head.to(device=device)
        self._inited = True

    def forward(
        self, batch: MaskedOlmoEarthSample, labels: torch.Tensor, is_train: bool = True
    ) -> torch.Tensor:
        """前向传播：通过包装器获取嵌入，然后通过线性头得到预测。

        Args:
            batch: 掩码后的 OlmoEarth 样本
            labels: 标签张量
            is_train: 是否为训练模式

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (logits, labels)
        """
        dev = next(self.wrapper.parameters()).device
        emb, labels = self.wrapper(batch, labels, is_train=is_train)
        emb = cast(torch.Tensor, emb)
        emb_dim = emb.shape[-1]
        if not self._inited:
            self._init_head(emb_dim, dev)
        if emb.device != dev:
            emb = emb.to(dev, non_blocking=True)
        return self._head(emb), labels


def to_device(
    masked: MaskedOlmoEarthSample, device: torch.device
) -> MaskedOlmoEarthSample:
    """将 MaskedOlmoEarthSample 移动到指定设备。

    timestamps 保持原始精度，其他数据转换为 bfloat16。

    Args:
        masked: 掩码后的 OlmoEarth 样本
        device: 目标设备

    Returns:
        MaskedOlmoEarthSample: 移动到目标设备的样本
    """
    d = masked.as_dict()
    for k, v in d.items():
        if k == "timestamps":
            d[k] = v.to(device=device)
        else:
            d[k] = v.to(device=device, dtype=torch.bfloat16)
    return MaskedOlmoEarthSample.from_dict(d)


def snapshot_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """克隆模块的状态字典到 CPU，用于后续恢复。

    Args:
        module: PyTorch 模块

    Returns:
        dict: 深度克隆的状态字典（在 CPU 上）
    """
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def set_backbone_trainable(backbone: nn.Module, requires_grad: bool) -> None:
    """切换骨干网络参数的梯度计算状态。

    Args:
        backbone: 骨干网络模块
        requires_grad: True 启用梯度，False 冻结参数
    """
    for param in backbone.parameters():
        param.requires_grad = requires_grad
