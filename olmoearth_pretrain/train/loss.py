"""
损失函数模块。

本模块实现了 OlmoEarth Pretrain 训练中使用的各种损失函数，包括：

对比学习损失：
- AllDiscriminationLoss: 全批次判别损失，跨所有样本对比 patch
- PatchDiscriminationLoss: 逐样本判别损失，内存高效的对比损失
- ModalityPatchDiscriminationLoss: 逐模态逐样本判别损失
- ModalityPatchDiscriminationMaskedNegatives: 带掩码负样本的判别损失
- ModalityPatchDiscriminationLossVec: 全向量化实现的逐模态判别损失（无 for 循环）
- AdjustedPatchDiscriminationLoss: 调整后的判别损失（NeurIPS 2023）
- InfoNCELoss: InfoNCE 对比损失

重建损失：
- L1Loss: L1 范数损失（平均绝对误差）
- L2Loss: L2 范数损失（均方误差）
- MAELoss: 掩码自编码重建损失
- CrossEntropyLoss: 交叉熵损失

正则化损失：
- KoLeoLoss: KoLeo 正则化项（基于 Kozachenko-Leonenko 微分熵估计器，
  鼓励特征均匀分布，源自 DINOv2）

所有损失函数通过 LOSS_REGISTRY 注册，可通过 LossConfig 配置并构建。
"""

"""Loss functions for training."""

import logging
import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from class_registry import ClassRegistry
from einops import rearrange, repeat
from torch import Tensor

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.nn.flexi_vit import TokensAndMasks
from olmoearth_pretrain.nn.pooling import PoolingType, pool_unmasked_tokens
from olmoearth_pretrain.nn.tokenization import TokenizationConfig
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, MaskValue

logger = logging.getLogger(__name__)


class Loss(ABC):
    """损失函数抽象基类。

    所有损失函数的基类，定义了 compute 接口。
    每个子类必须实现 compute 方法，并设置 name 属性用于日志记录。

    关键属性:
        name: 损失函数名称，用于训练日志中标识该损失
    """

    name: str  # 损失函数名称

    @abstractmethod
    def compute(self, predictions: Any, targets: Any, **kwargs: Any) -> Tensor:
        """计算预测值和目标值之间的损失。

        Args:
            predictions: 模型预测输出（通常是 TokensAndMasks 对象）
            targets: 目标值（通常是 TokensAndMasks 或 MaskedOlmoEarthSample 对象）
            **kwargs: 额外参数

        Returns:
            Tensor: 标量损失值
        """
        pass

    @staticmethod
    def _expand_and_reciprocate(t: Tensor) -> Tensor:
        """对张量取倒数并按原值重复展开。

        用于在判别损失中对不同大小的样本进行批次平均时，
        计算每个 token 的权重倒数。

        示例:
            >>> _expand_and_reciprocate(torch.tensor([1, 2, 3]))
            tensor([1.0000, 0.5000, 0.5000, 0.3333, 0.3333, 0.3333])

        Args:
            t: 输入整数张量

        Returns:
            Tensor: 取倒数后按原值重复展开的结果
        """
        reciprocals = torch.reciprocal(t.float())  # 取倒数
        return torch.repeat_interleave(reciprocals, t)  # 按原值重复


# 损失函数注册表，用于通过字符串名称创建对应的损失函数实例
LOSS_REGISTRY = ClassRegistry[Loss]()


@LOSS_REGISTRY.register("all_discrimination")
class AllDiscriminationLoss(Loss):
    """全批次判别损失。

    跨整个批次的所有样本进行 patch 对比，是 Galileo 论文中提出的损失函数。
    对每个解码 token，计算它与批次中所有目标 token 的相似度，
    以交叉熵损失训练模型将对应的预测-目标对匹配。

    适用场景：需要跨样本对比的大批次训练。

    关键属性:
        tau: softmax 温度参数，控制对比损失的集中度
        pred2unit: 是否对预测进行批次统计标准化
    """

    name = "AllDisc"

    def __init__(self, tau: float = 0.1, pred2unit: bool = False):
        """Initialize all patch discrimination loss.

        Args:
            tau: the softmax temperature
            pred2unit: whether to standardize the predictions using batch statistics
        """
        self.tau = tau
        self.pred2unit = pred2unit

    def compute(
        self, predictions: TokensAndMasks, targets: TokensAndMasks, **kwargs: Any
    ) -> Tensor:
        """Compute all patch discrimination loss between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        all_preds, all_masks = predictions.flatten_all_tokens_and_masks()
        all_targets = targets.flatten_all_tokens_and_masks()[0]

        pred = all_preds[all_masks == MaskValue.DECODER.value].unsqueeze(dim=0)
        target = all_targets[all_masks == MaskValue.DECODER.value].unsqueeze(dim=0)
        bs, nt, _ = pred.shape
        if self.pred2unit:
            pred_mu = pred.mean(1, keepdims=True)
            pred_std = pred.std(1, keepdims=True)
            pred = (pred - pred_mu) / (pred_std + 1e-4)

        pred = F.normalize(pred, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)

        scores = torch.einsum("npd,nqd->npq", pred, target) / self.tau
        count = (all_masks == MaskValue.DECODER.value).sum(dim=-1)

        labels = torch.arange(nt, dtype=torch.long, device=pred.device)[None].repeat(
            bs, 1
        )
        loss = F.cross_entropy(
            scores.flatten(0, 1), labels.flatten(0, 1), reduction="none"
        ) * (self.tau * 2)

        # emulate averaging across the batch dimension
        loss_multiplier = self._expand_and_reciprocate(count)
        # can't use bs here since this is after the unsqueezing, so bs == 1
        loss = (loss * loss_multiplier).sum() / all_preds.shape[0]
        return loss


@LOSS_REGISTRY.register("modality_all_discrimination")
class ModalityAllDiscriminationLoss(Loss):
    """Loss function for all discrimination task.

    Discriminates across patches using all samples in a batch.
    """

    name = "ModalityAllDisc"

    def __init__(self, tau: float = 0.1, pred2unit: bool = False):
        """Initialize all patch discrimination loss.

        Args:
            tau: the softmax temperature
            pred2unit: whether to standardize the predictions using batch statistics
        """
        self.tau = tau
        self.pred2unit = pred2unit

    def compute(
        self, predictions: TokensAndMasks, targets: TokensAndMasks, **kwargs: Any
    ) -> Tensor:
        """Compute all patch discrimination loss between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        modality_preds, modality_masks = (
            predictions.flatten_tokens_and_masks_per_modality()
        )
        modality_targets = targets.flatten_tokens_and_masks_per_modality()[0]

        total_loss = 0
        for all_preds, all_masks, all_targets in zip(
            modality_preds, modality_masks, modality_targets
        ):
            pred = all_preds[all_masks == MaskValue.DECODER.value].unsqueeze(dim=0)
            target = all_targets[all_masks == MaskValue.DECODER.value].unsqueeze(dim=0)
            bs, nt, _ = pred.shape
            if nt == 0:
                # If no decoded values, skip this modality
                logger.warning("No decoded values for this modality")
                continue
            if self.pred2unit:
                pred_mu = pred.mean(1, keepdims=True)
                pred_std = pred.std(1, keepdims=True)
                pred = (pred - pred_mu) / (pred_std + 1e-4)

            pred = F.normalize(pred, p=2, dim=-1)
            target = F.normalize(target, p=2, dim=-1)

            scores = torch.einsum("npd,nqd->npq", pred, target) / self.tau
            count = (all_masks == MaskValue.DECODER.value).sum(dim=-1)

            labels = torch.arange(nt, dtype=torch.long, device=pred.device)[
                None
            ].repeat(bs, 1)
            loss = F.cross_entropy(
                scores.flatten(0, 1), labels.flatten(0, 1), reduction="none"
            ) * (self.tau * 2)

            # emulate averaging across the batch dimension
            loss_multiplier = self._expand_and_reciprocate(count)
            # can't use bs here since this is after the unsqueezing, so bs == 1
            loss = (loss * loss_multiplier).sum() / all_preds.shape[0]
            total_loss += loss

        return total_loss


@LOSS_REGISTRY.register("patch_discrimination")
class PatchDiscriminationLoss(Loss):
    """逐样本判别损失（内存高效）。

    对每个样本独立计算对比损失，而非跨整个批次。
    这是 Latent MIM 论文中使用的核心损失函数。
    对每个样本的解码 token，计算它与同样本内所有目标 token 的相似度，
    以交叉熵损失训练匹配。

    优势：内存使用与单样本 token 数成正比，而非批次大小 × token 数。

    关键属性:
        tau: softmax 温度参数
        pred2unit: 是否对预测进行批次统计标准化
        weight: 损失权重
    """

    name = "PatchDisc"

    def __init__(self, tau: float = 0.1, pred2unit: bool = False, weight: float = 1.0):
        """Initialize patch discrimination loss.

        Args:
            tau: the softmax temperature
            pred2unit: whether to standardize the predictions using batch statistics
            mask_other_samples: whether to apply the contrastive loss drawing samples
                from within a sample (True) or using all other instances in a batch (False).
                If this is False, then this is the AllDisc loss from the Galileo paper
            weight: the weight to apply to this loss
        """
        self.tau = tau
        self.pred2unit = pred2unit
        self.weight = weight

    def compute(
        self, predictions: TokensAndMasks, targets: TokensAndMasks, **kwargs: Any
    ) -> Tensor:
        """Compute patch discrimination loss between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        all_preds, all_masks = predictions.flatten_all_tokens_and_masks()
        all_targets = targets.flatten_all_tokens_and_masks()[0]

        # Samples may have different number of tokens
        # TODO: Skip unqueeze and the for loop when mask_other_samples is True
        pred = all_preds[all_masks == MaskValue.DECODER.value].unsqueeze(dim=0)
        target = all_targets[all_masks == MaskValue.DECODER.value].unsqueeze(dim=0)
        bs, nt, _ = pred.shape

        if self.pred2unit:
            pred_mu = pred.mean(1, keepdims=True)
            pred_std = pred.std(1, keepdims=True)
            pred = (pred - pred_mu) / (pred_std + 1e-4)

        pred = F.normalize(pred, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)

        count = (all_masks == MaskValue.DECODER.value).sum(dim=-1)
        losses = []
        start = 0
        for c in count:
            end = start + c
            if c == 0:
                # we will occasionally get a sample with no decoded values due to missing data this will let us skip it
                logger.warning("No decoded values for this sample")
                continue
            pred_sample = pred[:, start:end, :]
            target_sample = target[:, start:end, :]
            score_sample = (
                torch.einsum("npd,nqd->npq", pred_sample, target_sample) / self.tau
            )
            labels = torch.arange(c, dtype=torch.long, device=pred.device)[None]
            loss = F.cross_entropy(
                score_sample.flatten(0, 1),
                labels.flatten(0, 1),
                reduction="none",
            ) * (self.tau * 2)
            loss = loss.mean()
            losses.append(loss)
            start = end
        loss = torch.stack(losses).mean()
        return self.weight * loss


@LOSS_REGISTRY.register("modality_patch_discrimination")
class ModalityPatchDiscriminationLoss(Loss):
    """逐模态逐样本判别损失。

    对每个模态独立计算 PatchDiscriminationLoss，然后累加。
    允许为不同模态设置不同权重。

    关键属性:
        tau: softmax 温度参数
        pred2unit: 是否对预测进行批次统计标准化
        weight: 损失权重
        modality_weights: 各模态的权重字典
    """

    name = "ModalityPatchDisc"

    def __init__(
        self,
        tau: float = 0.1,
        pred2unit: bool = False,
        weight: float = 1.0,
        modality_weights: dict[str, float] | None = None,
    ):
        """Initialize patch discrimination loss.

        Args:
            tau: the softmax temperature
            pred2unit: whether to standardize the predictions using batch statistics
            mask_other_samples: whether to apply the contrastive loss drawing samples
                from within a sample (True) or using all other instances in a batch (False).
                If this is False, then this is the AllDisc loss from the Galileo paper
            weight: the weight to apply to this loss
            modality_weights: the weights to apply to each modality
        """
        self.tau = tau
        self.pred2unit = pred2unit
        self.weight = weight
        self.modality_weights = modality_weights

    def compute(
        self, predictions: TokensAndMasks, targets: TokensAndMasks, **kwargs: Any
    ) -> Tensor:
        """Compute patch discrimination loss between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        modality_preds, modality_masks = (
            predictions.flatten_tokens_and_masks_per_modality()
        )
        modality_targets = targets.flatten_tokens_and_masks_per_modality()[0]

        # Accumulate to the total loss
        total_loss = 0
        for all_preds, all_masks, all_targets, modality in zip(
            modality_preds, modality_masks, modality_targets, targets.modalities
        ):
            # Samples may have different number of tokens
            # TODO: Skip unqueeze and the for loop when mask_other_samples is True
            pred = all_preds[all_masks == MaskValue.DECODER.value].unsqueeze(dim=0)
            target = all_targets[all_masks == MaskValue.DECODER.value].unsqueeze(dim=0)
            bs, nt, _ = pred.shape

            if self.pred2unit:
                pred_mu = pred.mean(1, keepdims=True)
                pred_std = pred.std(1, keepdims=True)
                pred = (pred - pred_mu) / (pred_std + 1e-4)

            pred = F.normalize(pred, p=2, dim=-1)
            target = F.normalize(target, p=2, dim=-1)

            count = (all_masks == MaskValue.DECODER.value).sum(dim=-1)
            losses = []
            start = 0
            for c in count:
                end = start + c
                if c == 0:
                    # we will occasionally get a sample with no decoded values due to missing data this will let us skip it
                    # logger.warning("No decoded values for this sample")
                    continue
                pred_sample = pred[:, start:end, :]
                target_sample = target[:, start:end, :]
                score_sample = (
                    torch.einsum("npd,nqd->npq", pred_sample, target_sample) / self.tau
                )
                labels = torch.arange(c, dtype=torch.long, device=pred.device)[None]
                loss = F.cross_entropy(
                    score_sample.flatten(0, 1),
                    labels.flatten(0, 1),
                    reduction="none",
                ) * (self.tau * 2)
                loss = loss.mean()
                losses.append(loss)
                start = end
            if len(losses) == 0:
                # If no losses were computed, skip this modality
                # logger.warning("No decoded values for this modality")
                continue
            loss = torch.stack(losses).mean()
            if self.modality_weights is not None:
                loss = loss * self.modality_weights[modality]
            total_loss += loss

        return self.weight * total_loss


@LOSS_REGISTRY.register("modality_patch_discrimination_masked_negatives")
class ModalityPatchDiscriminationMaskedNegatives(Loss):
    """带掩码负样本的逐模态判别损失。

    在计算对比损失时，若目标 token 的嵌入向量过于相似（余弦相似度 > 阈值），
    则将这些"同目标"负样本从分母中屏蔽掉（设为 -inf）。
    这对地图类模态特别有用，因为许多 token 可能具有相同的目标嵌入。

    关键属性:
        tau: softmax 温度参数
        same_target_threshold: 判定目标相同的余弦相似度阈值
        mask_negatives_for_modalities: 需要屏蔽负样本的模态列表
    """

    name = "ModalityPatchDiscMasked"

    def __init__(
        self,
        tau: float = 0.1,
        pred2unit: bool = False,
        weight: float = 1.0,
        modality_weights: dict[str, float] | None = None,
        same_target_threshold: float = 0.999,
        mask_negatives_for_modalities: list[str] | None = None,
    ):
        """Initialize masked negatives patch discrimination loss.

        Args:
            tau: the softmax temperature
            pred2unit: whether to standardize the predictions using batch statistics
            weight: the weight to apply to this loss
            modality_weights: the weights to apply to each modality
            same_target_threshold: cosine similarity threshold to consider targets as same
            mask_negatives_for_modalities: list of modality names to apply masking for.
                If None, applies to all modalities.
        """
        self.tau = tau
        self.pred2unit = pred2unit
        self.weight = weight
        self.modality_weights = modality_weights
        self.same_target_threshold = same_target_threshold
        self.mask_negatives_for_modalities = mask_negatives_for_modalities

    def compute(
        self, predictions: TokensAndMasks, targets: TokensAndMasks, **kwargs: Any
    ) -> Tensor:
        """Compute patch discrimination loss with masked same-target negatives."""
        modality_preds, modality_masks = (
            predictions.flatten_tokens_and_masks_per_modality()
        )
        modality_targets = targets.flatten_tokens_and_masks_per_modality()[0]

        total_loss = 0
        for all_preds, all_masks, all_targets, modality in zip(
            modality_preds, modality_masks, modality_targets, targets.modalities
        ):
            pred = all_preds[all_masks == MaskValue.DECODER.value].unsqueeze(dim=0)
            target = all_targets[all_masks == MaskValue.DECODER.value].unsqueeze(dim=0)
            pred = pred.float()
            target = target.float()
            bs, nt, _ = pred.shape
            if nt == 0:
                continue

            if self.pred2unit:
                pred_mu = pred.mean(1, keepdims=True)
                pred_std = pred.std(1, keepdims=True)
                pred = (pred - pred_mu) / (pred_std + 1e-4)

            pred = F.normalize(pred, p=2, dim=-1)
            target = F.normalize(target, p=2, dim=-1)

            # Check if we should mask negatives for this modality
            should_mask = (
                self.mask_negatives_for_modalities is None
                or modality in self.mask_negatives_for_modalities
            )

            count = (all_masks == MaskValue.DECODER.value).sum(dim=-1)
            losses = []
            start = 0

            for c in count:
                c_val = c.item() if hasattr(c, "item") else int(c)
                end = start + c_val
                if c_val == 0:
                    continue

                pred_sample = pred[:, start:end, :]
                target_sample = target[:, start:end, :]

                # Compute similarity scores
                score_sample = (
                    torch.einsum("npd,nqd->npq", pred_sample, target_sample) / self.tau
                )

                # Apply same-target masking if enabled for this modality
                if should_mask and c_val > 1:
                    target_flat = target_sample.squeeze(0)  # [c, dim]
                    target_sim = target_flat @ target_flat.T  # [c, c]
                    same_target = target_sim > self.same_target_threshold

                    # Mask: same target but not self (diagonal)
                    diagonal = torch.eye(
                        c_val, dtype=torch.bool, device=target_flat.device
                    )
                    invalid_negatives = same_target & ~diagonal

                    # Check if any token has valid negatives
                    valid_neg_count = (~same_target).sum(dim=-1)
                    if valid_neg_count.min() == 0:
                        # Some tokens have no valid negatives - skip this sample
                        start = end
                        continue

                    # Apply mask to scores: set invalid negatives to -inf
                    score_sample = score_sample.masked_fill(
                        invalid_negatives[None, :, :], float("-inf")
                    )

                # Standard cross-entropy
                labels = torch.arange(c_val, dtype=torch.long, device=pred.device)[None]
                loss = F.cross_entropy(
                    score_sample.flatten(0, 1),
                    labels.flatten(0, 1),
                    reduction="none",
                ) * (self.tau * 2)

                loss = loss.mean()
                losses.append(loss)
                start = end

            if len(losses) == 0:
                continue

            loss = torch.stack(losses).mean()
            if self.modality_weights is not None:
                loss = loss * self.modality_weights.get(modality, 1.0)

            total_loss += loss

        return self.weight * total_loss


@LOSS_REGISTRY.register("modality_patch_discrimination_masked_negatives_vec")
class ModalityPatchDiscriminationMaskedNegativesVec(Loss):
    """Vectorized patch discrimination with same-target negative masking.

    Equivalent to ModalityPatchDiscriminationMaskedNegatives but fully batched:
    no per-sample Python loops, no .item() syncs, no repeated torch.eye allocations.
    """

    name = "ModalityPatchDiscMaskedVec"

    def __init__(
        self,
        tau: float = 0.1,
        pred2unit: bool = False,
        weight: float = 1.0,
        modality_weights: dict[str, float] | None = None,
        same_target_threshold: float = 0.999,
        mask_negatives_for_modalities: list[str] | None = None,
    ) -> None:
        """Initialize with same params as ModalityPatchDiscriminationMaskedNegatives."""
        self.tau = tau
        self.pred2unit = pred2unit
        self.weight = weight
        self.modality_weights = modality_weights
        self.same_target_threshold = same_target_threshold
        self.mask_negatives_for_modalities = mask_negatives_for_modalities

    def _compute_modality_loss_parallel(
        self,
        all_preds: Tensor,
        all_masks: Tensor,
        all_targets: Tensor,
        modality: str,
    ) -> Tensor:
        batch_size, num_tokens, dim = all_preds.shape
        decoder_mask = all_masks == MaskValue.DECODER.value
        count = decoder_mask.sum(dim=-1)  # (batch,)

        # Sort so decoder tokens come first per sample
        _, sort_indices = decoder_mask.long().sort(dim=1, descending=True, stable=True)
        sort_expanded = sort_indices.unsqueeze(-1).expand(-1, -1, dim)
        sorted_preds = all_preds.gather(1, sort_expanded).float()
        sorted_targets = all_targets.gather(1, sort_expanded).float()

        # valid_mask[b, i] = True iff position i is a decoder token for sample b
        range_tensor = torch.arange(num_tokens, device=count.device)
        valid_mask = range_tensor.unsqueeze(0) < count.unsqueeze(1)  # (batch, T)

        if self.pred2unit:
            mask_float = valid_mask.unsqueeze(-1).float()
            total_decoder = mask_float.sum().clamp(min=1)
            pred_mu = (sorted_preds * mask_float).sum(
                dim=(0, 1), keepdim=True
            ) / total_decoder
            centered = sorted_preds - pred_mu
            pred_var = (centered**2 * mask_float).sum(dim=(0, 1), keepdim=True) / (
                total_decoder - 1
            ).clamp(min=1)
            sorted_preds = (sorted_preds - pred_mu) / (pred_var.sqrt() + 1e-4)

        sorted_preds = F.normalize(sorted_preds, p=2, dim=-1)
        sorted_targets = F.normalize(sorted_targets, p=2, dim=-1)

        # Score matrix: (batch, T, T) — each sample independent
        scores = torch.bmm(sorted_preds, sorted_targets.transpose(1, 2)) / self.tau

        should_mask = (
            self.mask_negatives_for_modalities is None
            or modality in self.mask_negatives_for_modalities
        )

        # Track which samples to skip (default: none)
        sample_skip = torch.zeros(batch_size, dtype=torch.bool, device=scores.device)

        if should_mask:
            # Target self-similarity per sample: (batch, T, T)
            target_sim = torch.bmm(sorted_targets, sorted_targets.transpose(1, 2))
            same_target = target_sim > self.same_target_threshold

            # Only consider valid token pairs
            valid_2d = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(
                2
            )  # (batch, T, T)

            # Diagonal (self) is never an invalid negative
            diag = torch.eye(num_tokens, dtype=torch.bool, device=scores.device)
            invalid_negatives = same_target & ~diag.unsqueeze(0) & valid_2d

            # The original only applies masking when c_val > 1, so restrict
            # invalid_negatives and skip-detection to samples with count > 1.
            multi_token = (count > 1).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1)
            invalid_negatives = invalid_negatives & multi_token

            # Skip samples where any valid token has zero valid negatives
            valid_neg_count = (~same_target & valid_2d).sum(dim=-1)  # (batch, T)
            has_zero_neg = (
                (valid_neg_count == 0) & valid_mask & (count > 1).unsqueeze(1)
            )
            sample_skip = has_zero_neg.any(dim=1)

            scores = scores.masked_fill(invalid_negatives, float("-inf"))

        # Mask out non-decoder columns
        col_mask = valid_mask.unsqueeze(1).expand_as(scores)
        scores = scores.masked_fill(~col_mask, -torch.finfo(scores.dtype).max)

        # Mask rows for zero-count samples to prevent NaN
        row_mask = valid_mask.unsqueeze(2).expand_as(scores)
        scores = scores.masked_fill(~row_mask, 0.0)

        # Labels: diagonal (token i matches target i)
        labels = range_tensor.unsqueeze(0).expand(batch_size, -1)

        loss_per_pos = F.cross_entropy(
            scores.reshape(-1, num_tokens),
            labels.reshape(-1),
            reduction="none",
        ) * (self.tau * 2)
        loss_per_pos = loss_per_pos.reshape(batch_size, num_tokens)

        # Zero out invalid positions and skipped samples
        sample_contributes = (count > 0) & ~sample_skip
        effective_valid = valid_mask.float() * sample_contributes.unsqueeze(1).float()
        effective_count = count.float() * sample_contributes.float()
        num_contributing = sample_contributes.sum()

        loss_per_sample = (loss_per_pos * effective_valid).sum(
            dim=1
        ) / effective_count.clamp(min=1)
        loss = loss_per_sample.sum() / num_contributing.float().clamp(min=1)

        return loss

    def compute(
        self, predictions: TokensAndMasks, targets: TokensAndMasks, **kwargs: Any
    ) -> Tensor:
        """Compute patch discrimination loss with masked same-target negatives (vectorized)."""
        modality_preds, modality_masks = (
            predictions.flatten_tokens_and_masks_per_modality()
        )
        modality_targets = targets.flatten_tokens_and_masks_per_modality()[0]

        total_loss = 0
        for all_preds, all_masks, all_targets, modality in zip(
            modality_preds, modality_masks, modality_targets, targets.modalities
        ):
            loss = self._compute_modality_loss_parallel(
                all_preds, all_masks, all_targets, modality
            )
            if self.modality_weights is not None:
                loss = loss * self.modality_weights.get(modality, 1.0)
            total_loss += loss

        return self.weight * total_loss


@LOSS_REGISTRY.register("modality_patch_discrimination_vec")
class ModalityPatchDiscriminationLossVec(Loss):
    """全向量化逐模态判别损失（无 for 循环）。

    使用排序和掩码替代布尔索引和 GPU→CPU 同步点，
    完全消除了样本级别的 for 循环，提高 GPU 并行度。
    不支持全批次判别损失（仅支持逐样本）。

    关键属性:
        tau: softmax 温度参数
        pred2unit: 是否对预测进行批次统计标准化
        weight: 损失权重
        modality_weights: 各模态的权重字典
    """

    name = "ModalityPatchDisc"

    def __init__(
        self,
        tau: float = 0.1,
        pred2unit: bool = False,
        weight: float = 1.0,
        modality_weights: dict[str, float] | None = None,
    ):
        """Initialize patch discrimination loss.

        Args:
            tau: the softmax temperature
            pred2unit: whether to standardize the predictions using batch statistics
            weight: the weight to apply to this loss
            modality_weights: the weights to apply to each modality
        """
        self.tau = tau
        self.pred2unit = pred2unit
        self.weight = weight
        self.modality_weights = modality_weights

    def _compute_modality_loss_parallel(
        self,
        all_preds: Tensor,
        all_masks: Tensor,
        all_targets: Tensor,
    ) -> Tensor:
        """Compute patch discrimination loss for a single modality in parallel.

        Uses sort-based token reordering and pure masking to avoid all
        boolean indexing (nonzero) and GPU→CPU sync points.

        Args:
            all_preds: Predictions tensor of shape (batch, tokens, dim)
            all_masks: Mask tensor of shape (batch, tokens)
            all_targets: Targets tensor of shape (batch, tokens, dim)

        Returns:
            The computed loss value for this modality. Zero if no decoder tokens are present.
        """
        batch_size, num_tokens, dim = all_preds.shape
        decoder_mask = all_masks == MaskValue.DECODER.value
        count = decoder_mask.sum(dim=-1)  # (batch,)
        num_valid = (count > 0).sum()  # stays as tensor, no sync

        # Sort tokens so decoder tokens come first, preserving relative order.
        _, sort_indices = decoder_mask.long().sort(dim=1, descending=True, stable=True)
        sort_expanded = sort_indices.unsqueeze(-1).expand(-1, -1, dim)
        sorted_preds = all_preds.gather(1, sort_expanded)
        sorted_targets = all_targets.gather(1, sort_expanded)

        # Validity mask: first count[b] positions per sample are decoder tokens.
        range_tensor = torch.arange(num_tokens, device=count.device)
        valid_mask = range_tensor.unsqueeze(0) < count.unsqueeze(
            1
        )  # (batch, num_tokens)

        if self.pred2unit:
            # Global mean/std across all decoder tokens (matches original flat behavior)
            mask_float = valid_mask.unsqueeze(-1).float()  # (batch, tokens, 1)
            total_decoder = mask_float.sum().clamp(min=1)
            pred_mu = (sorted_preds * mask_float).sum(
                dim=(0, 1), keepdim=True
            ) / total_decoder
            centered = sorted_preds - pred_mu
            pred_var = (centered**2 * mask_float).sum(dim=(0, 1), keepdim=True) / (
                total_decoder - 1
            ).clamp(min=1)
            pred_std = pred_var.sqrt()
            sorted_preds = (sorted_preds - pred_mu) / (pred_std + 1e-4)

        sorted_preds = F.normalize(sorted_preds, p=2, dim=-1)
        sorted_targets = F.normalize(sorted_targets, p=2, dim=-1)

        # Compute scores: (batch, num_tokens, num_tokens)
        scores = torch.bmm(sorted_preds, sorted_targets.transpose(1, 2)) / self.tau

        # Mask out non-decoder columns with -inf so they don't affect softmax
        col_mask = valid_mask.unsqueeze(1).expand_as(scores)
        scores = scores.masked_fill(~col_mask, -torch.finfo(scores.dtype).max)

        # Also mask rows for zero-count samples to prevent NaN from all-inf softmax
        row_mask = valid_mask.unsqueeze(2).expand_as(scores)
        scores = scores.masked_fill(~row_mask, 0.0)

        # Labels: diagonal (decoder token i should match decoder token i)
        labels = range_tensor.unsqueeze(0).expand(batch_size, -1)

        # Cross entropy per position
        loss_per_pos = F.cross_entropy(
            scores.reshape(-1, num_tokens),
            labels.reshape(-1),
            reduction="none",
        ) * (self.tau * 2)
        loss_per_pos = loss_per_pos.reshape(batch_size, num_tokens)

        # Zero out invalid positions, average per sample, then over valid samples only
        valid_mask_float = valid_mask.float()
        loss_per_sample = (loss_per_pos * valid_mask_float).sum(
            dim=1
        ) / count.float().clamp(min=1)
        loss = loss_per_sample.sum() / num_valid.float().clamp(min=1)

        return loss

    def compute(
        self, predictions: TokensAndMasks, targets: TokensAndMasks, **kwargs: Any
    ) -> Tensor:
        """Compute patch discrimination loss between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        modality_preds, modality_masks = (
            predictions.flatten_tokens_and_masks_per_modality()
        )
        modality_targets = targets.flatten_tokens_and_masks_per_modality()[0]

        # Accumulate to the total loss
        total_loss = 0
        for all_preds, all_masks, all_targets, modality in zip(
            modality_preds, modality_masks, modality_targets, targets.modalities
        ):
            loss = self._compute_modality_loss_parallel(
                all_preds, all_masks, all_targets
            )
            if self.modality_weights is not None:
                loss = loss * self.modality_weights[modality]
            total_loss += loss

        return self.weight * total_loss


# --- Deprecated aliases for backward compatibility ---


class _DeprecatedPatchDiscriminationLossNew(PatchDiscriminationLoss):
    def __init__(self, *args: Any, **kwargs: Any):
        warnings.warn(
            '"patch_discrimination_new" is deprecated, use "patch_discrimination" instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


LOSS_REGISTRY.register("patch_discrimination_new")(
    _DeprecatedPatchDiscriminationLossNew
)


class _DeprecatedModalityPatchDiscriminationLossNew(ModalityPatchDiscriminationLoss):
    def __init__(self, *args: Any, **kwargs: Any):
        warnings.warn(
            '"modality_patch_discrimination_new" is deprecated, '
            'use "modality_patch_discrimination" instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


LOSS_REGISTRY.register("modality_patch_discrimination_new")(
    _DeprecatedModalityPatchDiscriminationLossNew
)

# Backward-compat class name aliases
PatchDiscriminationLossNew = PatchDiscriminationLoss
ModalityPatchDiscriminationLossNew = ModalityPatchDiscriminationLoss


@LOSS_REGISTRY.register("adjusted_patch_discrimination")
class AdjustedPatchDiscriminationLoss(Loss):
    """调整后的判别损失。

    基于 NeurIPS 2023 论文，对负样本的相似度分数施加高斯权重，
    使得接近正样本的负样本获得更高权重，远离的获得更低权重。
    这有助于模型关注"困难"负样本。

    参考: https://proceedings.neurips.cc/paper_files/paper/2023/file/48aaa5ea741ae8430bd58e25917d267d-Paper-Conference.pdf

    关键属性:
        tau: softmax 温度参数
        mu: 高斯分布的均值
        sigma: 高斯分布的标准差
        pred2unit: 是否对预测进行批次统计标准化
    """

    name = "AdjustedPatchDisc"

    def __init__(
        self,
        tau: float = 0.1,
        mu: float = 0.7,
        sigma: float = 1.0,
        pred2unit: bool = False,
    ):
        """Initialize adjusted patch discrimination loss.

        Args:
            tau: the softmax temperature
            mu: the mean of the Gaussian distribution
            sigma: the standard deviation of the Gaussian distribution
            pred2unit: whether to standardize the predictions using batch statistics
        """
        self.tau = tau
        self.mu = mu
        self.sigma = sigma
        self.pred2unit = pred2unit

    def compute(
        self, predictions: TokensAndMasks, targets: TokensAndMasks, **kwargs: Any
    ) -> Tensor:
        """Compute patch discrimination loss between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        all_preds, all_masks = predictions.flatten_all_tokens_and_masks()
        all_targets = targets.flatten_all_tokens_and_masks()[0]

        pred = all_preds[all_masks == MaskValue.DECODER.value].unsqueeze(dim=0)
        target = all_targets[all_masks == MaskValue.DECODER.value].unsqueeze(dim=0)
        bs, nt, _ = pred.shape

        if self.pred2unit:
            pred_mu = pred.mean(1, keepdims=True)
            pred_std = pred.std(1, keepdims=True)
            pred = (pred - pred_mu) / (pred_std + 1e-4)

        pred = F.normalize(pred, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)

        count = (all_masks == MaskValue.DECODER.value).sum(dim=-1)

        losses = []
        start = 0
        for c in count:
            end = start + c
            pred_sample = pred[:, start:end, :]  # (1, c, d)
            target_sample = target[:, start:end, :]  # (1, c, d)

            sim_matrix = torch.einsum(
                "npd,nqd->npq", pred_sample, target_sample
            )  # (1, c, c)

            pos_scores = torch.diagonal(sim_matrix, dim1=-2, dim2=-1)  # (1, c)
            pos_scores = pos_scores / self.tau

            # Mask out diagonal (positives) to get negatives
            mask = ~torch.eye(c, dtype=torch.bool, device=pred.device)
            neg_scores = sim_matrix.masked_select(mask).view(1, c, c - 1)  # (1, c, c-1)
            neg_scores = neg_scores / self.tau

            # Apply Gaussian-based weights to negatives
            # Weight is computed based on the neg_scores from a sample
            weight = (
                1.0
                / (self.sigma * math.sqrt(2 * math.pi))
                * torch.exp(
                    -((neg_scores * self.tau - self.mu) ** 2)
                    / (2 * math.pow(self.sigma, 2))
                )
            )  # (1, c, c-1)
            # Normalize the weights per query
            weight = weight / weight.mean(dim=-1, keepdim=True)
            neg_scores = neg_scores * weight.detach()

            # Reconstruct the sim_matrix
            sim_matrix = torch.zeros(
                1, c, c, device=pred.device, dtype=neg_scores.dtype
            )
            sim_matrix.diagonal(dim1=-2, dim2=-1).copy_(pos_scores)
            sim_matrix.masked_scatter_(mask, neg_scores)

            labels = torch.arange(c, dtype=torch.long, device=pred.device)[None]
            loss = F.cross_entropy(
                sim_matrix.flatten(0, 1),
                labels.flatten(0, 1),
                reduction="none",
            ) * (self.tau * 2)
            loss = loss.mean()
            losses.append(loss)
            start = end

        loss = torch.stack(losses).mean()
        return loss


@LOSS_REGISTRY.register("l1")
class L1Loss(Loss):
    """L1 损失函数（平均绝对误差）。

    仅对解码 token 计算预测和目标之间的 L1 距离。
    """

    name = "L1"

    def compute(
        self, predictions: TokensAndMasks, targets: TokensAndMasks, **kwargs: Any
    ) -> Tensor:
        """Compute L1 loss between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        all_preds, all_masks = predictions.flatten_all_tokens_and_masks()
        all_targets = targets.flatten_all_tokens_and_masks()[0]
        pred = all_preds[all_masks == MaskValue.DECODER.value]
        target = all_targets[all_masks == MaskValue.DECODER.value]

        return F.l1_loss(pred, target)


@LOSS_REGISTRY.register("l2")
class L2Loss(Loss):
    """L2 损失函数（均方误差）。

    仅对解码 token 计算预测和目标之间的 L2 距离。
    """

    name = "L2"

    def compute(
        self, predictions: TokensAndMasks, targets: TokensAndMasks, **kwargs: Any
    ) -> Tensor:
        """Compute L2 loss between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        all_preds, all_masks = predictions.flatten_all_tokens_and_masks()
        all_targets = targets.flatten_all_tokens_and_masks()[0]
        pred = all_preds[all_masks == MaskValue.DECODER.value]
        target = all_targets[all_masks == MaskValue.DECODER.value]
        return F.mse_loss(pred, target)


@LOSS_REGISTRY.register("mae")
class MAELoss(Loss):
    """掩码自编码重建损失。

    计算模型重建值与原始值之间的像素级损失（如 MSE、L1、SmoothL1）。
    支持仅对解码 token 计算损失，或对所有非缺失 token 计算。

    关键属性:
        only_decode: 是否仅对 DECODER 掩码的 token 计算损失
        loss: PyTorch 损失函数实例（如 MSELoss、SmoothL1Loss）
        weight: 损失权重
        tokenization_config: 分组配置，用于自定义波段分组
    """

    name = "MAE"

    def __init__(
        self,
        loss_function: str = "MSELoss",
        only_decode: bool = True,
        weight: float = 1.0,
        tokenization_config: TokenizationConfig | None = None,
        **kwargs: Any,
    ):
        """Initialize MAE loss.

        Args:
            loss_function: pytorch loss to use
            only_decode: only calculate loss on DECODER masked tokens, otherwise all
            weight: the weight to apply to this loss
            tokenization_config: Optional config for custom band groupings
            **kwargs: arguments for pytorch loss constructor
        """
        self.only_decode = only_decode
        self.loss = getattr(torch.nn, loss_function)(reduction="sum", **kwargs)
        self.weight = weight
        self.tokenization_config = tokenization_config or TokenizationConfig()

    # data: [B, H, W, T, C]
    def _flatten_spatiotemporal_data(
        self, data: TokensAndMasks
    ) -> tuple[Tensor, Tensor]:
        masks = []
        datas = []
        for modality in data.modalities:
            pred = getattr(data, modality)
            if pred is not None:
                mask = getattr(data, data.get_masked_modality_name(modality))
                for idx, channel_set_idxs in enumerate(
                    self.tokenization_config.get_bandset_indices(modality)
                ):
                    bs_mask = mask[..., idx]
                    bs_mask = repeat(
                        bs_mask, "b h w t -> b h w t c", c=len(channel_set_idxs)
                    )
                    bs_mask = rearrange(bs_mask, "b h w t c -> b (h w t c)")
                    masks.append(bs_mask)
                    bs_data = pred[..., channel_set_idxs]
                    bs_data = rearrange(bs_data, "b h w t c -> b (h w t c)")
                    datas.append(bs_data)
        return torch.cat(datas, dim=1), torch.cat(masks, dim=1)

    def compute(
        self, predictions: TokensAndMasks, targets: MaskedOlmoEarthSample, **kwargs: Any
    ) -> Tensor:
        """Compute MAE loss between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        data, masks = self._flatten_spatiotemporal_data(predictions)
        valid_dict = {}
        for modality in predictions.modalities:
            if getattr(predictions, modality) is not None:
                masked_name = predictions.get_masked_modality_name(modality)
                valid_dict[modality] = getattr(targets, modality)
                valid_dict[masked_name] = getattr(targets, masked_name)
        valid_targets = TokensAndMasks(**valid_dict)
        labels, label_masks = self._flatten_spatiotemporal_data(valid_targets)
        if self.only_decode:
            decode = label_masks == MaskValue.DECODER.value
        else:
            decode = label_masks != MaskValue.MISSING.value
        data = data * decode
        labels = labels * decode
        return self.weight * self.loss(data, labels) / torch.count_nonzero(decode)


@LOSS_REGISTRY.register("cross_entropy")
class CrossEntropyLoss(Loss):
    """Loss function for cross entropy."""

    name = "CrossEntropy"

    def compute(
        self, predictions: TokensAndMasks, targets: TokensAndMasks, **kwargs: Any
    ) -> Tensor:
        """Compute cross entropy between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        all_preds, all_masks = predictions.flatten_all_tokens_and_masks()
        all_targets = targets.flatten_all_tokens_and_masks()[0]
        pred = all_preds[all_masks == MaskValue.DECODER.value]
        target = all_targets[all_masks == MaskValue.DECODER.value]

        return F.cross_entropy(pred, target.squeeze())


@LOSS_REGISTRY.register("InfoNCE")
class InfoNCELoss(Loss):
    """InfoNCE 对比损失。

    标准的 InfoNCE 损失，对 L2 归一化后的预测和目标计算相似度矩阵，
    对角线为正样本对，其余为负样本对。

    关键属性:
        tau: softmax 温度参数
        weight: 损失权重
    """

    name = "InfoNCE"

    def __init__(self, tau: float = 0.1, weight: float = 1):
        """Initialize InfoNCE loss.

        Args:
            tau: the softmax temperature
            weight: the weight to apply to this loss
        """
        self.tau = tau
        self.weight = weight

    def compute(
        self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs: Any
    ) -> Tensor:
        """Compute InfoNCE between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        predictions = F.normalize(predictions, p=2, dim=-1)
        targets = F.normalize(targets, p=2, dim=-1)
        logits = predictions @ targets.transpose(-2, -1)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(predictions), device=predictions.device)
        return self.weight * F.cross_entropy(logits / self.tau, labels)


@LOSS_REGISTRY.register("KoLeo")
class KoLeoLoss(Loss):
    """KoLeo 正则化损失。

    基于 Kozachenko-Leonenko 微分熵估计器的正则化项，
    鼓励批次内特征在超球面上均匀分布，防止特征坍缩。
    源自 DINOv2 的实现。

    参考: https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py

    关键属性:
        weight: 正则化权重（默认 0.1，与 DINOv2 一致）
        mode: 计算模式，"instance" 在实例级计算最近邻，"patch" 在 patch 级计算
        eps: 避免除零的小常数
    """

    name = "KoLeo"

    def __init__(
        self,
        weight: float = 0.1,
        mode: str = "instance",
        eps: float = 1e-8,
    ) -> None:
        """Initialize KoLeo regularizer.

        Args:
            weight: a weight to apply to the regularization value. Default value follows Dinov2
            eps: small value to avoid division by zero.
            mode: one of "instance" or "patch" - whether to compute
                nearest neighbourst at the instance or patch level
        """
        self.eps = eps
        self.pdist = torch.nn.PairwiseDistance(2, eps=eps)
        if mode not in ["instance", "patch"]:
            raise ValueError(f"Unsupported mode {mode}")
        self.mode = mode
        self.weight = weight

    @staticmethod
    def pairwise_nearest_neighbours(x: torch.Tensor) -> torch.Tensor:
        """Pairwise nearest neighbors for L2-normalized vectors.

        Uses Torch rather than Faiss to remain on GPU.

        Args:
            x: embeddings against which we want to compute nearest neighbours.

        Returns:
            indices: indices of nearest neighbour (i.e. indices[i] will return
            the index for the nearest neighbour of the ith embedding).
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, indices = torch.max(dots, dim=1)
        return indices

    def compute(
        self, predictions: TokensAndMasks, targets: None, **kwargs: Any
    ) -> Tensor:
        """Compute the KoLeo regularization term.

        Args:
            predictions: Model predictions. Unlike other losses, these are
                _online encoder outputs_, not decoder outputs.
            targets: Unused, and only kept for consistency.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.
        """
        if isinstance(predictions, TokensAndMasks):
            if self.mode == "patch":
                if not isinstance(predictions, TokensAndMasks):
                    raise ValueError(
                        "predictions must be TokensAndMasks for patch mode"
                    )
                all_preds, all_masks = predictions.flatten_all_tokens_and_masks()
                online_encodings = all_preds[
                    all_masks == MaskValue.ONLINE_ENCODER.value
                ]
            else:
                online_encodings = pool_unmasked_tokens(
                    predictions, PoolingType.MEAN, spatial_pooling=False
                )
        else:
            online_encodings = predictions

        # apply l2 norm
        online_encodings = F.normalize(online_encodings, eps=self.eps, p=2, dim=-1)
        idx_of_nn = self.pairwise_nearest_neighbours(online_encodings)
        distances_to_nn = self.pdist(online_encodings, online_encodings[idx_of_nn])
        return self.weight * -torch.log(distances_to_nn + self.eps).mean()


@dataclass
class LossConfig(Config):
    """损失函数配置类。

    通过 loss_config 字典指定损失函数类型和参数，
    使用 LOSS_REGISTRY 注册表创建对应的损失函数实例。

    Args:
        loss_config: 损失函数配置字典，格式为：
            {
                "type": "patch_discrimination",  # 注册表中的损失名称
                # 其余为损失函数的 __init__ 参数
            }
    """

    loss_config: dict[str, Any]  # 损失配置字典

    def build(self) -> Loss:
        """从配置构建损失函数实例。

        从 loss_config 中提取 "type" 键作为损失名称，
        其余键值作为损失函数的初始化参数。

        Returns:
            Loss: 构建好的损失函数实例
        """
        loss_key = self.loss_config.pop("type")  # 提取损失名称
        return LOSS_REGISTRY.get_class(loss_key)(**self.loss_config)  # 从注册表创建实例
