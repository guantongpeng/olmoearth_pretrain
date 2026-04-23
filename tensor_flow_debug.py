#!/usr/bin/env python3
"""
OlmoEarth Tensor 流程调试脚本

本脚本提供工具函数来监控和追踪训练过程中的tensor形状、值域和梯度信息。
可以集成到训练循环中进行实时监控。

使用:
    python tensor_flow_debug.py --config config.yaml --checkpoint checkpoint.pt
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TensorStatistics:
    """追踪单个tensor的统计信息"""
    shape: tuple
    dtype: torch.dtype
    device: str
    mean: float
    std: float
    min: float
    max: float
    has_nan: bool
    has_inf: bool
    grad_norm: Optional[float] = None


class TensorFlowMonitor:
    """监控训练过程中的tensor流"""

    def __init__(self, log_level: int = logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.stats_history = {}

    def check_tensor(self, tensor: torch.Tensor, name: str) -> TensorStatistics:
        """
        检查单个tensor的统计信息

        Args:
            tensor: 要检查的tensor
            name: tensor的名称

        Returns:
            TensorStatistics: tensor的统计信息
        """
        if not isinstance(tensor, torch.Tensor):
            self.logger.warning(f"{name} is not a tensor: {type(tensor)}")
            return None

        # 计算统计信息
        with torch.no_grad():
            stats = TensorStatistics(
                shape=tuple(tensor.shape),
                dtype=tensor.dtype,
                device=str(tensor.device),
                mean=tensor.float().mean().item() if tensor.numel() > 0 else 0.0,
                std=tensor.float().std().item() if tensor.numel() > 0 else 0.0,
                min=tensor.float().min().item() if tensor.numel() > 0 else 0.0,
                max=tensor.float().max().item() if tensor.numel() > 0 else 0.0,
                has_nan=torch.isnan(tensor.float()).any().item(),
                has_inf=torch.isinf(tensor.float()).any().item(),
                grad_norm=None
            )

        # 如果tensor有梯度，计算梯度范数
        if tensor.requires_grad and tensor.grad is not None:
            with torch.no_grad():
                stats.grad_norm = tensor.grad.norm().item()

        # 记录统计信息
        self._log_stats(name, stats)
        self.stats_history[name] = stats

        return stats

    def _log_stats(self, name: str, stats: TensorStatistics):
        """记录tensor的统计信息"""
        log_msg = (
            f"{name:30s} | "
            f"shape: {str(stats.shape):20s} | "
            f"dtype: {str(stats.dtype):10s} | "
            f"mean: {stats.mean:8.4f} | "
            f"std: {stats.std:8.4f} | "
            f"min: {stats.min:8.4f} | "
            f"max: {stats.max:8.4f}"
        )

        if stats.grad_norm is not None:
            log_msg += f" | grad_norm: {stats.grad_norm:8.4f}"

        if stats.has_nan or stats.has_inf:
            log_msg += " | ⚠️  WARNING: "
            if stats.has_nan:
                log_msg += "NaN "
            if stats.has_inf:
                log_msg += "Inf"
            self.logger.error(log_msg)
        else:
            self.logger.info(log_msg)

    def trace_forward_pass(self,
                          model: nn.Module,
                          batch: Any,
                          patch_size: int) -> dict[str, TensorStatistics]:
        """
        追踪前向传播中的tensor流

        Args:
            model: LatentMIM 模型
            batch: MaskedOlmoEarthSample
            patch_size: patch 大小

        Returns:
            dict: 各阶段的tensor统计信息
        """
        trace_stats = {}

        self.logger.info("=" * 100)
        self.logger.info("FORWARD PASS TRACING")
        self.logger.info("=" * 100)

        # Step 1: 检查输入数据
        self.logger.info("\n>>> Input Data:")
        if hasattr(batch, 'data'):
            for modality_name, modality_data in batch.data.items():
                if isinstance(modality_data, torch.Tensor):
                    stats = self.check_tensor(modality_data, f"batch.data[{modality_name}]")
                    trace_stats[f"input/{modality_name}"] = stats

        # Step 2: 检查掩码
        self.logger.info("\n>>> Masking Information:")
        if hasattr(batch, 'masks'):
            for mask_name, mask_data in batch.masks.items():
                if isinstance(mask_data, torch.Tensor):
                    num_true = mask_data.sum().item() if mask_data.dtype == torch.bool else 0
                    self.logger.info(
                        f"Mask {mask_name:30s} | "
                        f"shape: {str(tuple(mask_data.shape)):20s} | "
                        f"dtype: {str(mask_data.dtype):10s} | "
                        f"num_true: {num_true}"
                    )

        # Step 3: 在线编码器前向传播
        self.logger.info("\n>>> Online Encoder Forward Pass:")
        with torch.enable_grad():
            latent, decoded, _, reconstructed, extra_metrics = model(batch, patch_size)

            # 检查编码器输出
            if hasattr(latent, 'tokens'):
                stats = self.check_tensor(latent.tokens, "encoder_output.tokens")
                trace_stats["encoder_output/tokens"] = stats

            if hasattr(decoded, 'tokens'):
                stats = self.check_tensor(decoded.tokens, "decoder_output.tokens")
                trace_stats["decoder_output/tokens"] = stats

        # Step 4: 目标编码器前向传播 (无梯度)
        self.logger.info("\n>>> Target Encoder Forward Pass (no grad):")
        with torch.no_grad():
            from olmoearth_pretrain.nn.utils import unpack_encoder_output

            output_dict = model.target_encoder.forward(
                batch.unmask(),
                patch_size=patch_size,
                token_exit_cfg={}
            )
            target_output, _, _ = unpack_encoder_output(output_dict)

            if hasattr(target_output, 'tokens'):
                stats = self.check_tensor(target_output.tokens, "target_output.tokens")
                trace_stats["target_encoder_output/tokens"] = stats

        self.logger.info("\n" + "=" * 100)
        return trace_stats

    def trace_loss_computation(self,
                              loss: torch.Tensor,
                              loss_name: str = "loss") -> TensorStatistics:
        """
        追踪损失的计算

        Args:
            loss: 损失tensor
            loss_name: 损失的名称

        Returns:
            TensorStatistics: 损失的统计信息
        """
        self.logger.info("=" * 100)
        self.logger.info("LOSS COMPUTATION")
        self.logger.info("=" * 100)

        stats = self.check_tensor(loss, loss_name)
        trace_stats = {f"loss/{loss_name}": stats}

        self.logger.info("=" * 100)
        return trace_stats

    def trace_backward_pass(self, loss: torch.Tensor, model: nn.Module):
        """
        追踪反向传播中的梯度流

        Args:
            loss: 损失tensor
            model: 要追踪梯度的模型
        """
        self.logger.info("=" * 100)
        self.logger.info("BACKWARD PASS - GRADIENT COMPUTATION")
        self.logger.info("=" * 100)

        # 执行反向传播
        loss.backward()

        # 检查各层的梯度
        self.logger.info("\n>>> Gradient Information by Layer:")
        grad_stats = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                with torch.no_grad():
                    grad_norm = param.grad.norm().item()
                    param_norm = param.data.norm().item()
                    grad_ratio = grad_norm / (param_norm + 1e-8)

                    log_msg = (
                        f"  {name:50s} | "
                        f"param_norm: {param_norm:8.4f} | "
                        f"grad_norm: {grad_norm:8.4f} | "
                        f"ratio: {grad_ratio:8.4f}"
                    )

                    if torch.isnan(param.grad).any():
                        log_msg += " | ⚠️  NaN"
                        self.logger.error(log_msg)
                    elif torch.isinf(param.grad).any():
                        log_msg += " | ⚠️  Inf"
                        self.logger.error(log_msg)
                    elif grad_ratio > 0.1:
                        log_msg += " | ⚠️  Large ratio"
                        self.logger.warning(log_msg)
                    else:
                        self.logger.info(log_msg)

                    grad_stats[name] = {
                        'grad_norm': grad_norm,
                        'param_norm': param_norm,
                        'ratio': grad_ratio
                    }

        self.logger.info("=" * 100)
        return grad_stats

    def trace_optimizer_step(self, optimizer, model: nn.Module, lr_scheduler=None):
        """
        追踪优化器步骤

        Args:
            optimizer: PyTorch 优化器
            model: 模型
            lr_scheduler: 学习率调度器 (可选)
        """
        self.logger.info("=" * 100)
        self.logger.info("OPTIMIZER STEP")
        self.logger.info("=" * 100)

        # 记录更新前的参数范数
        with torch.no_grad():
            param_norms_before = {
                name: param.data.norm().item()
                for name, param in model.named_parameters()
            }

        # 优化器步骤
        optimizer.step()

        # 记录更新后的参数范数
        with torch.no_grad():
            param_norms_after = {
                name: param.data.norm().item()
                for name, param in model.named_parameters()
            }

        # 记录参数变化
        self.logger.info("\n>>> Parameter Update Information:")
        for name in param_norms_before:
            before = param_norms_before[name]
            after = param_norms_after[name]
            change = abs(after - before)
            change_ratio = change / (before + 1e-8)

            log_msg = (
                f"  {name:50s} | "
                f"before: {before:8.4f} | "
                f"after: {after:8.4f} | "
                f"change: {change:8.4f} ({change_ratio:.2%})"
            )
            self.logger.info(log_msg)

        # 学习率调度
        if lr_scheduler is not None:
            old_lr = optimizer.param_groups[0]['lr']
            lr_scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            self.logger.info(f"\n>>> Learning Rate Update: {old_lr:.6f} -> {new_lr:.6f}")

        self.logger.info("=" * 100)

    def summarize(self):
        """打印统计信息摘要"""
        self.logger.info("\n" + "=" * 100)
        self.logger.info("TENSOR FLOW SUMMARY")
        self.logger.info("=" * 100)

        if not self.stats_history:
            self.logger.info("No tensor statistics recorded.")
            return

        # 按类别分组
        by_category = {}
        for name, stats in self.stats_history.items():
            category = name.split('/')[0] if '/' in name else 'other'
            if category not in by_category:
                by_category[category] = []
            by_category[category].append((name, stats))

        for category, items in sorted(by_category.items()):
            self.logger.info(f"\n{category.upper()}:")
            for name, stats in items:
                self.logger.info(
                    f"  {name:40s} | shape: {str(stats.shape):20s} | "
                    f"range: [{stats.min:6.2f}, {stats.max:6.2f}]"
                )


# 使用示例
def example_trace_training_step():
    """
    示例: 如何在训练循环中使用 TensorFlowMonitor
    """
    from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModule

    # 初始化监控器
    monitor = TensorFlowMonitor()

    # 假设已有模型、batch、损失函数等
    # train_module: LatentMIMTrainModule
    # batch: MaskedOlmoEarthSample

    # 在训练循环中使用
    # for batch_idx, batch in enumerate(dataloader):
    #     if batch_idx == 0:  # 仅在第一个batch调试
    #         # 追踪前向传播
    #         forward_stats = monitor.trace_forward_pass(
    #             train_module.model, batch, patch_size=16
    #         )
    #
    #         # 计算损失
    #         loss, latent, decoded, target = train_module.model_forward(
    #             batch, patch_size=16, token_exit_cfg={}
    #         )
    #
    #         # 追踪损失
    #         loss_stats = monitor.trace_loss_computation(loss)
    #
    #         # 反向传播
    #         loss.backward()
    #         backward_stats = monitor.backward_pass(loss, train_module.model)
    #
    #         # 优化器步骤
    #         monitor.trace_optimizer_step(
    #             train_module.optimizer,
    #             train_module.model,
    #             train_module.scheduler
    #         )
    #
    #         # 摘要
    #         monitor.summarize()
    #
    #     # 正常训练步骤
    #     train_module.train_batch(batch)


if __name__ == "__main__":
    logger.info("Tensor Flow Debug Module Loaded")
    logger.info("Use TensorFlowMonitor class to trace tensor operations during training")
