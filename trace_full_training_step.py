#!/usr/bin/env python3
"""
OlmoEarth 完整的Tensor流程追踪示例

本脚本演示如何在真实的训练步骤中使用 TensorFlowMonitor 追踪tensor的流动。

运行方式:
    python trace_full_training_step.py --config path/to/config.yaml
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

# 导入 OlmoEarth 模块
try:
    from olmoearth_pretrain.train.train_module.latent_mim import (
        LatentMIMTrainModule, LatentMIMTrainModuleConfig)
    from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoader
    from tensor_flow_debug import TensorFlowMonitor
except ImportError as e:
    print(f"Warning: Some modules could not be imported: {e}")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingStepTracer:
    """在训练步骤中集成tensor追踪"""

    def __init__(self, trace_interval: int = 1, log_dir: Optional[Path] = None):
        """
        Args:
            trace_interval: 每隔多少步进行一次详细追踪 (0=禁用)
            log_dir: 日志输出目录
        """
        self.trace_interval = trace_interval
        self.step_count = 0
        self.monitor = TensorFlowMonitor()

        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            # 配置文件日志
            handler = logging.FileHandler(log_dir / f"tensor_trace.log")
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    def should_trace(self) -> bool:
        """判断是否应该进行详细追踪"""
        if self.trace_interval == 0:
            return False
        return self.step_count % self.trace_interval == 0

    def trace_training_step(self,
                           train_module: LatentMIMTrainModule,
                           batch: tuple,
                           batch_idx: int = 0):
        """
        完整地追踪一个训练步骤的所有阶段

        Args:
            train_module: LatentMIMTrainModule 实例
            batch: 从数据加载器获得的batch (patch_size, MaskedOlmoEarthSample)
            batch_idx: batch 索引
        """
        self.step_count += 1

        if not self.should_trace():
            logger.info(f"Skipping detailed trace for step {self.step_count}")
            return

        logger.info(f"\n{'='*120}")
        logger.info(f"FULL TRAINING STEP TRACE - Step {self.step_count}, Batch {batch_idx}")
        logger.info(f"{'='*120}\n")

        patch_size, batch_data = batch

        logger.info(f">>> Batch Information:")
        logger.info(f"    Patch Size: {patch_size}")
        logger.info(f"    Batch Size: {batch_data.batch_size}")

        # === Phase 1: Input Data Inspection ===
        self._trace_input_data(batch_data)

        # === Phase 2: Masking Strategy ===
        self._trace_masking_strategy(batch_data)

        # === Phase 3: Forward Pass ===
        loss, latent, decoded, target_output = self._trace_forward_pass(
            train_module, batch_data, patch_size
        )

        # === Phase 4: Loss Computation ===
        self._trace_loss(loss)

        # === Phase 5: Regularization (if applicable) ===
        self._trace_regularization(train_module, latent)

        # === Phase 6: Backward Pass ===
        self._trace_backward_pass(loss, train_module)

        # === Phase 7: Gradient Clipping ===
        self._trace_gradient_clipping(train_module)

        # === Phase 8: Optimizer Step ===
        self._trace_optimizer_step(train_module)

        # === Phase 9: EMA Update ===
        self._trace_ema_update(train_module)

        # Summary
        self.monitor.summarize()
        logger.info(f"\n{'='*120}\n")

    def _trace_input_data(self, batch_data):
        """追踪输入数据信息"""
        logger.info(f"\n>>> PHASE 1: Input Data")
        logger.info(f"{'─'*120}")

        if hasattr(batch_data, 'data'):
            for modality_name, data in batch_data.data.items():
                if isinstance(data, torch.Tensor):
                    self.monitor.check_tensor(data, f"input.{modality_name}")

    def _trace_masking_strategy(self, batch_data):
        """追踪掩码策略"""
        logger.info(f"\n>>> PHASE 2: Masking Strategy")
        logger.info(f"{'─'*120}")

        if hasattr(batch_data, 'masks'):
            for mask_name, mask in batch_data.masks.items():
                if isinstance(mask, torch.Tensor):
                    if mask.dtype == torch.bool:
                        num_true = mask.sum().item()
                        num_false = (~mask).sum().item()
                        logger.info(
                            f"  {mask_name:40s} | "
                            f"shape: {str(tuple(mask.shape)):20s} | "
                            f"True: {num_true:6d} ({100*num_true/(num_true+num_false+1e-8):.1f}%) | "
                            f"False: {num_false:6d}"
                        )
                    else:
                        self.monitor.check_tensor(mask, f"mask.{mask_name}")

    def _trace_forward_pass(self, train_module, batch_data, patch_size):
        """追踪前向传播"""
        logger.info(f"\n>>> PHASE 3: Forward Pass (Online Encoder + Decoder)")
        logger.info(f"{'─'*120}")

        # 在线编码器
        logger.info("\n  Online Encoder:")
        with torch.enable_grad():
            latent, decoded, _, reconstructed, extra_metrics = train_module.model(
                batch_data, patch_size
            )

            if hasattr(latent, 'tokens'):
                self.monitor.check_tensor(latent.tokens, "    encoder_latent")

            if hasattr(decoded, 'tokens'):
                self.monitor.check_tensor(decoded.tokens, "    decoder_output")

        # 目标编码器 (无梯度)
        logger.info("\n  Target Encoder (no_grad):")
        with torch.no_grad():
            from olmoearth_pretrain.nn.utils import unpack_encoder_output

            output_dict = train_module.model.target_encoder.forward(
                batch_data.unmask(),
                patch_size=patch_size,
                token_exit_cfg=train_module.token_exit_cfg
            )
            target_output, _, _ = unpack_encoder_output(output_dict)

            if hasattr(target_output, 'tokens'):
                self.monitor.check_tensor(target_output.tokens, "    target_encoder_output")

        return None, latent, decoded, target_output

    def _trace_loss(self, loss):
        """追踪损失计算"""
        logger.info(f"\n>>> PHASE 4: Loss Computation")
        logger.info(f"{'─'*120}")

        if loss is not None:
            self.monitor.check_tensor(loss, "  loss")

    def _trace_regularization(self, train_module, latent):
        """追踪正则化项"""
        logger.info(f"\n>>> PHASE 5: Regularization")
        logger.info(f"{'─'*120}")

        if train_module.regularizer is not None:
            reg_loss = train_module.regularizer.compute(latent)
            if reg_loss is not None:
                self.monitor.check_tensor(reg_loss, "  regularization_loss")
        else:
            logger.info("  No regularization applied")

    def _trace_backward_pass(self, loss, train_module):
        """追踪反向传播"""
        logger.info(f"\n>>> PHASE 6: Backward Pass")
        logger.info(f"{'─'*120}")

        if loss is not None:
            logger.info("  Computing gradients with loss.backward()...")
            loss.backward()

            # 检查梯度
            logger.info("\n  Gradient Information (sample of parameters):")
            param_count = 0
            for name, param in train_module.model.named_parameters():
                if param.grad is not None and param_count < 5:  # 仅显示前5个参数
                    grad_norm = param.grad.norm().item()
                    logger.info(f"    {name:50s} | grad_norm: {grad_norm:.6f}")
                    param_count += 1

    def _trace_gradient_clipping(self, train_module):
        """追踪梯度剪裁"""
        logger.info(f"\n>>> PHASE 7: Gradient Clipping")
        logger.info(f"{'─'*120}")

        if train_module.max_grad_norm is not None:
            logger.info(f"  Clipping gradients with max_grad_norm={train_module.max_grad_norm}")

            # 计算总梯度范数
            total_grad_norm = 0.0
            for param in train_module.model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm() ** 2
            total_grad_norm = total_grad_norm ** 0.5

            logger.info(f"  Total gradient norm before clipping: {total_grad_norm:.6f}")

            # 执行剪裁
            torch.nn.utils.clip_grad_norm_(
                train_module.model.parameters(),
                train_module.max_grad_norm
            )

            # 计算剪裁后的梯度范数
            total_grad_norm_clipped = 0.0
            for param in train_module.model.parameters():
                if param.grad is not None:
                    total_grad_norm_clipped += param.grad.norm() ** 2
            total_grad_norm_clipped = total_grad_norm_clipped ** 0.5

            logger.info(f"  Total gradient norm after clipping:  {total_grad_norm_clipped:.6f}")
            if total_grad_norm > train_module.max_grad_norm:
                logger.warning(f"  ⚠️  Gradients were clipped!")
        else:
            logger.info("  No gradient clipping applied")

    def _trace_optimizer_step(self, train_module):
        """追踪优化器步骤"""
        logger.info(f"\n>>> PHASE 8: Optimizer Step")
        logger.info(f"{'─'*120}")

        current_lr = train_module.optimizer.param_groups[0]['lr']
        logger.info(f"  Current learning rate: {current_lr:.6f}")

        # 记录参数范数 (before step)
        param_norms_before = {}
        for name, param in train_module.model.named_parameters():
            param_norms_before[name] = param.data.norm().item()

        # 优化器步骤
        train_module.optimizer.step()
        train_module.optimizer.zero_grad()

        logger.info(f"  Optimizer step completed")
        logger.info(f"  Gradients zeroed")

        # 记录参数范数 (after step) - 仅显示样本
        logger.info("\n  Parameter Updates (sample):")
        param_count = 0
        for name, param in train_module.model.named_parameters():
            if param_count < 5:  # 仅显示前5个参数
                before = param_norms_before.get(name, 0.0)
                after = param.data.norm().item()
                change = abs(after - before)
                logger.info(
                    f"    {name:50s} | before: {before:.6f} | "
                    f"after: {after:.6f} | change: {change:.6f}"
                )
                param_count += 1

    def _trace_ema_update(self, train_module):
        """追踪EMA更新"""
        logger.info(f"\n>>> PHASE 9: EMA Target Encoder Update")
        logger.info(f"{'─'*120}")

        if hasattr(train_module, 'start_ema'):
            logger.info(f"  EMA decay range: ({train_module.start_ema}, {train_module.end_ema})")
            logger.info(f"  ✓ Target encoder will be updated in next iteration")
        else:
            logger.info("  No EMA update configured")


def create_example_trace_script():
    """
    创建一个可运行的示例脚本模板
    """
    example_script = '''
#!/usr/bin/env python3
"""
实际训练中的tensor追踪使用示例
"""

import torch
from pathlib import Path
from trace_full_training_step import TrainingStepTracer
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModule

def main():
    # 初始化追踪器 (每10步进行一次详细追踪)
    tracer = TrainingStepTracer(
        trace_interval=10,
        log_dir=Path("./tensor_traces")
    )

    # 假设已初始化:
    # - train_module: LatentMIMTrainModule
    # - data_loader: 数据加载器

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(data_loader):
            # 使用tracer追踪训练步骤
            tracer.trace_training_step(train_module, batch, batch_idx)

            # 注意: trace_training_step 只在满足 trace_interval 条件时执行详细追踪
            # 其他步骤可以正常训练 (推荐用于调试和验证)

if __name__ == "__main__":
    main()
'''
    return example_script


if __name__ == "__main__":
    logger.info("TrainingStepTracer Module Loaded")
    logger.info("Use TrainingStepTracer to trace complete training steps")

    # 打印示例脚本
    logger.info("\n=== Example Usage ===")
    logger.info(create_example_trace_script())
