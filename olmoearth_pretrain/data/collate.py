"""
OlmoEarth Pretrain 数据集的批次整理（collate）函数模块。

本模块提供了将多个样本整理为一个批次的函数，支持：
- 自动处理样本中存在的任意模态
- 单掩码视图整理（collate_single_masked_batched）：先整理为批次张量，再应用增强和掩码
- 双掩码视图整理（collate_double_masked_batched）：支持 Galileo 风格的双掩码训练

核心设计：在批次级别应用增强和掩码以实现向量化操作，提高效率。
"""

from __future__ import annotations

import torch

from olmoearth_pretrain.data.transform import Transform
from olmoearth_pretrain.datatypes import (
    MaskedOlmoEarthSample,
    OlmoEarthSample,
)
from olmoearth_pretrain.train.masking import MaskingStrategy


def collate_olmoearth_pretrain(
    batch: list[tuple[int, OlmoEarthSample]],
) -> tuple[int, OlmoEarthSample]:
    """OlmoEarth 样本的批次整理函数，自动处理样本中存在的任意模态。

    将列表中的多个 OlmoEarthSample 堆叠为批次张量。
    对于值为 None 的模态字段，保持 None 不变。

    Args:
        batch: (patch_size, OlmoEarthSample) 元组的列表。

    Returns:
        元组 (patch_size, 批次化的 OlmoEarthSample)。
    """

    def stack_or_none(attr: str) -> torch.Tensor | None:
        """堆叠张量，同时处理 None 值。

        对于部分缺失的样本使用 MISSING_VALUE，因此仅检查第一个样本即可判断。

        Args:
            attr: 属性名称。

        Returns:
            堆叠后的批次张量，或 None（如果该属性为 None）。
        """
        if getattr(batch[0][1], attr) is None:
            return None
        stacked_tensor = torch.stack(
            [torch.from_numpy(getattr(sample, attr)) for _, sample in batch], dim=0
        )
        return stacked_tensor

    patch_size, batch_zero = batch[0]
    # 获取所有字段（包括 timestamps）
    sample_fields = batch_zero.modalities_with_timestamps

    # 为每个字段创建堆叠后的张量字典
    collated_dict = {field: stack_or_none(field) for field in sample_fields}
    return patch_size, OlmoEarthSample(**collated_dict)


def collate_single_masked_batched(
    batch: list[tuple[int, OlmoEarthSample]],
    transform: Transform | None,
    masking_strategy: MaskingStrategy,
) -> tuple[int, MaskedOlmoEarthSample]:
    """单掩码视图的批次整理函数。

    先将原始 OlmoEarthSample 整理为批次张量，
    然后对整个批次一次性应用增强和掩码，实现向量化操作。

    Args:
        batch: (patch_size, OlmoEarthSample) 元组的列表。
        transform: 可选的数据增强变换。
        masking_strategy: 掩码策略。

    Returns:
        元组 (patch_size, MaskedOlmoEarthSample)。
    """
    # 先将原始样本整理为批次化的 OlmoEarthSample
    patch_size, stacked_sample = collate_olmoearth_pretrain(batch)

    # 对批次应用增强（如果已配置）
    if transform is not None:
        stacked_sample = transform.apply(stacked_sample)

    # 对批次应用掩码
    masked_sample = masking_strategy.apply_mask(stacked_sample, patch_size)

    return patch_size, masked_sample


def collate_double_masked_batched(
    batch: list[tuple[int, OlmoEarthSample]],
    transform: Transform | None,
    masking_strategy: MaskingStrategy,
    masking_strategy_b: MaskingStrategy | None,
) -> tuple[int, MaskedOlmoEarthSample, MaskedOlmoEarthSample]:
    """双掩码视图的批次整理函数（用于 Galileo 风格训练）。

    先将原始 OlmoEarthSample 整理为批次张量，
    然后对整个批次一次性应用增强和两个独立的掩码策略，实现向量化操作。

    Args:
        batch: (patch_size, OlmoEarthSample) 元组的列表。
        transform: 可选的数据增强变换。
        masking_strategy: 第一个掩码策略。
        masking_strategy_b: 第二个掩码策略。若为 None，则使用 masking_strategy。

    Returns:
        元组 (patch_size, MaskedOlmoEarthSample_a, MaskedOlmoEarthSample_b)。
    """
    # 先将原始样本整理为批次化的 OlmoEarthSample
    patch_size, stacked_sample = collate_olmoearth_pretrain(batch)

    # 对批次应用增强（如果已配置）
    if transform is not None:
        stacked_sample = transform.apply(stacked_sample)

    # 对批次应用两个掩码策略
    masked_sample_a = masking_strategy.apply_mask(stacked_sample, patch_size)
    # 如果未提供第二掩码策略，则使用第一掩码策略
    strategy_b = (
        masking_strategy_b if masking_strategy_b is not None else masking_strategy
    )
    masked_sample_b = strategy_b.apply_mask(stacked_sample, patch_size)

    return patch_size, masked_sample_a, masked_sample_b
