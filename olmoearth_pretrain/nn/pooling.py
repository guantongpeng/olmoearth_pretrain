"""Token 池化（Pooling）操作模块。

本模块提供多种池化策略，将 TokensAndMasks 中的 token 信息
聚合为更紧凑的表示，用于对比学习、下游任务等。

主要功能：
- 空间池化：在时间维度上聚合，保留空间结构
- 实例级池化：将所有 token 聚合为单个向量
- 特征拼接：将不同模态的空间特征拼接
- 支持最大池化和平均池化两种策略
"""

from __future__ import annotations

import logging
from enum import StrEnum

import torch
from torch import Tensor

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskValue, TokensAndMasks

logger = logging.getLogger(__name__)


class PoolingType(StrEnum):
    """池化策略枚举。

    - MAX: 最大池化，取所有 token 中每个维度的最大值
    - MEAN: 平均池化，取所有 token 的均值
    """

    MAX = "max"
    MEAN = "mean"


def pool_spatially_and_concat_modalities(tokens_and_masks: TokensAndMasks) -> Tensor:
    """在时间维度上池化各模态的空间特征，并拼接所有模态。

    仅处理空间模态（is_spatial=True）且被在线编码器处理的 token。
    对每个模态在时间维度上取均值，然后在模态维度上拼接。

    Args:
        tokens_and_masks: 包含各模态 token 和掩码的数据结构

    Returns:
        拼接后的空间特征，形状 [B, H*W, num_modalities, D]
    """
    spatial_stacked_features = []
    for attr_name in tokens_and_masks.modalities:
        if Modality.get(attr_name).is_spatial:
            mask_attr_name = tokens_and_masks.get_masked_modality_name(attr_name)
            masked_attr = getattr(tokens_and_masks, mask_attr_name)
            if masked_attr is None:
                continue
            if (masked_attr == MaskValue.ONLINE_ENCODER.value).all():
                attr = getattr(tokens_and_masks, attr_name)
                pooled_attr = torch.mean(attr, dim=(-3))
                spatial_stacked_features.append(pooled_attr)
    if len(spatial_stacked_features) == 0:
        raise ValueError("Missing unmasked spatial modalities for spatial pooling.")
    spatial_stacked_features = torch.cat(spatial_stacked_features, dim=-2)
    return spatial_stacked_features


def pool_spatially(
    tokens_and_masks: TokensAndMasks, pooling_type: PoolingType
) -> Tensor:
    """在时间维度上池化各模态，保留空间结构。

    仅处理空间模态且被在线编码器处理的 token。
    根据掩码信息，仅对在线编码器可见的 token 进行池化。

    Args:
        tokens_and_masks: 包含各模态 token 和掩码的数据结构
        pooling_type: 池化策略（MEAN 或 MAX）

    Returns:
        池化后的空间特征
    """
    spatial_average = []
    for attr_name in tokens_and_masks.modalities:
        if Modality.get(attr_name).is_spatial:
            mask_attr_name = tokens_and_masks.get_masked_modality_name(attr_name)
            masked_attr = getattr(tokens_and_masks, mask_attr_name)
            if masked_attr is None:
                continue
            online_mask = masked_attr == MaskValue.ONLINE_ENCODER.value
            has_any_online = online_mask.any()
            if not has_any_online:
                continue
            attr = getattr(tokens_and_masks, attr_name)
            # Collapse mask trailing dims to a single bool per position,
            # then pad with size-1 dims to broadcast against token tensor.
            while online_mask.ndim > attr.ndim:
                online_mask = online_mask.any(dim=-1)
            token_mask = online_mask
            while token_mask.ndim < attr.ndim:
                token_mask = token_mask.unsqueeze(-1)
            masked_tokens = attr * token_mask.float()
            if pooling_type == PoolingType.MEAN:
                count = token_mask.float().sum(dim=(-2, -3)).clamp(min=1)
                spatial_average.append(masked_tokens.sum(dim=(-2, -3)) / count)
            else:
                masked_tokens = masked_tokens.masked_fill(~token_mask, float("-inf"))
                spatial_average.append(
                    torch.max(torch.max(masked_tokens, dim=-2).values, dim=-2).values
                )
    if len(spatial_average) == 0:
        raise ValueError(
            "Missing unmasked spatial modalities for spatial pooling. "
            f"Available modalities: {tokens_and_masks.modalities}."
        )
    spatial_average_t = torch.stack(spatial_average, dim=-1)
    if pooling_type == PoolingType.MEAN:
        return spatial_average_t.mean(dim=-1)
    else:
        return spatial_average_t.max(dim=-1).values


def pool_instance_wise(
    tokens_and_masks: TokensAndMasks, pooling_type: PoolingType
) -> Tensor:
    """实例级池化，将样本中所有未掩码的 token 聚合为单个向量。

    首先将所有模态的 token 展平，然后仅对在线编码器可见的 token
    进行最大或平均池化。

    Args:
        tokens_and_masks: 包含各模态 token 和掩码的数据结构
        pooling_type: 池化策略（MEAN 或 MAX）

    Returns:
        池化后的实例特征，形状 [B, D]

    Raises:
        ValueError: 当没有任何 token 被编码时（num_encoded_tokens == 0）
    """
    x, mask = tokens_and_masks.flatten_all_tokens_and_masks()
    assert isinstance(x, Tensor) and isinstance(mask, Tensor)
    mask = (mask == MaskValue.ONLINE_ENCODER.value).long()
    x_for_pooling = x * mask.unsqueeze(-1)
    if pooling_type == PoolingType.MAX:
        x_for_pooling = x_for_pooling.masked_fill(
            ~mask.bool().unsqueeze(-1), -float("inf")
        )
        return x_for_pooling.max(dim=1).values
    elif pooling_type == PoolingType.MEAN:
        num_encoded_tokens = torch.sum(mask, -1, keepdim=True)
        logger.debug(f"num_encoded_tokens: {num_encoded_tokens}")
        if (num_encoded_tokens == 0).any():
            raise ValueError(
                f"num_encoded_tokens is 0 for some samples {num_encoded_tokens}"
            )
        return x_for_pooling.sum(dim=1) / num_encoded_tokens
    else:
        raise ValueError(f"Invalid pooling type: {pooling_type}")


def pool_unmasked_tokens(
    tokens_and_masks: TokensAndMasks,
    pooling_type: PoolingType | None = None,
    spatial_pooling: bool = False,
    concat_features: bool = False,
) -> Tensor:
    """池化未掩码的 token，根据参数选择不同的池化策略。

    这是池化操作的统一入口，支持以下组合：
    - 实例级池化（spatial_pooling=False）：聚合所有 token 为单个向量
    - 空间池化（spatial_pooling=True）：在时间维度聚合，保留空间结构
    - 特征拼接（concat_features=True）：拼接各模态的空间特征

    Args:
        tokens_and_masks: 包含各模态 token 和掩码的数据结构
        pooling_type: 池化策略，默认为 MAX
        spatial_pooling: 是否保留空间维度（True=空间池化，False=实例级池化）
        concat_features: 是否拼接特征而非平均（仅在 spatial_pooling=True 时有效）

    Returns:
        池化后的特征张量
    """
    if pooling_type is None:
        pooling_type = PoolingType.MAX

    if concat_features and spatial_pooling:
        return pool_spatially_and_concat_modalities(tokens_and_masks)
    if concat_features:
        raise ValueError("concat_features is not supported for non-spatial pooling")
    if not spatial_pooling:
        return pool_instance_wise(tokens_and_masks, pooling_type)
    else:
        return pool_spatially(tokens_and_masks, pooling_type)
