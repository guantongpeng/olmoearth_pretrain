"""微调模型评估函数模块。

本模块提供微调后模型的评估功能，支持分类和分割两大任务类型。

主要函数：
- eval_cls: 评估分类指标（支持单标签和多标签）
- eval_seg: 评估分割指标（mIoU, overall_acc 等）

核心逻辑：
- 使用 torch.no_grad 和 bfloat16 自动混合精度
- 对分割任务，将 patch 级别的 logits 重塑为像素级预测
- 使用双线性插值对齐预测和标签的空间尺寸
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader

from olmoearth_pretrain.evals.finetune.model import BackboneWithHead, to_device
from olmoearth_pretrain.evals.metrics import (
    EvalMetric,
    EvalResult,
    classification_metrics,
    segmentation_metrics,
)


@torch.no_grad()
def eval_cls(
    module: BackboneWithHead,
    loader: DataLoader,
    device: torch.device,
    is_multilabel: bool,
    primary_metric: EvalMetric | None = None,
    primary_metric_class: int | None = None,
) -> EvalResult:
    """评估分类指标。

    遍历数据加载器，获取模型 logits，然后计算分类指标。

    Args:
        module: 带有分类头的骨干模型
        loader: 数据加载器
        device: 计算设备
        is_multilabel: 是否为多标签分类
        primary_metric: 主指标（用于模型选择）
        primary_metric_class: 主指标的类别索引

    Returns:
        EvalResult: 分类评估结果
    """
    module.eval()
    logits_all, labels_all = [], []
    for masked, label in loader:
        label = label.to(device=device)
        masked = to_device(masked, device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, _ = module(masked, label, is_train=False)  # (B, C)
        logits_all.append(logits.float().cpu())
        labels_all.append(label.cpu())
    logits = torch.cat(logits_all, 0)
    labels = torch.cat(labels_all, 0)
    if is_multilabel:
        preds = torch.sigmoid(logits).gt(0.5).int()
    else:
        preds = torch.argmax(logits, dim=-1)
    return classification_metrics(
        preds,
        labels,
        is_multilabel=is_multilabel,
        primary_metric=primary_metric,
        primary_metric_class=primary_metric_class,
    )


@torch.no_grad()
def eval_seg(
    module: BackboneWithHead,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    patch_size: int,
    primary_metric: EvalMetric | None = None,
    primary_metric_class: int | None = None,
) -> EvalResult:
    """评估分割指标。

    遍历数据加载器，获取模型 logits，将 patch 级别 logits 重塑为像素级预测，
    然后计算分割指标。

    核心逻辑：
    1. 模型输出 logits 形状为 (B, H, W, C*p*p)，其中 p 为 patch_size
    2. 使用 einops rearrange 将其重塑为 (B, C, H*p, W*p)
    3. 如果预测和标签空间尺寸不匹配，使用双线性插值对齐

    Args:
        module: 带有分割头的骨干模型
        loader: 数据加载器
        device: 计算设备
        num_classes: 类别数
        patch_size: patch 大小
        primary_metric: 主指标
        primary_metric_class: 主指标的类别索引

    Returns:
        EvalResult: 分割评估结果
    """
    module.eval()
    preds_all, labels_all = [], []
    for masked, label in loader:
        label = label.to(device=device)
        masked = to_device(masked, device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, _ = module(masked, label, is_train=False)  # (B, H, W, C*p*p)
            H, W = logits.shape[1], logits.shape[2]
            logits = rearrange(
                logits,
                "b h w (c i j) -> b c (h i) (w j)",
                h=H,
                w=W,
                c=num_classes,
                i=patch_size,
                j=patch_size,
            )
            if logits.shape[-2:] != label.shape[-2:]:
                logits = F.interpolate(
                    logits.float(),
                    size=label.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
        preds_all.append(torch.argmax(logits, dim=1).cpu())
        labels_all.append(label.cpu())
    preds = torch.cat(preds_all, 0)
    labels = torch.cat(labels_all, 0)
    return segmentation_metrics(
        preds,
        labels,
        num_classes=num_classes,
        ignore_label=-1,
        primary_metric=primary_metric,
        primary_metric_class=primary_metric_class,
    )
