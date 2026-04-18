"""
Latent MIM 训练模块。

本模块实现 Latent Masked Image Modeling (Latent MIM) 的训练逻辑，
是 OlmoEarth Pretrain 的核心训练方法之一。

训练流程：
1. 对输入数据应用掩码策略，将 token 分为编码/解码/目标编码器三类
2. 在线编码器处理编码 token，解码器尝试从编码表示重建解码 token 的潜在表示
3. 目标编码器（EMA 更新）处理未掩码数据，提供训练目标
4. 使用判别损失（如 PatchDiscrimination）训练解码器匹配目标编码器输出
5. 可选的 MAE 重建损失和正则化损失

参考：Latent MIM 论文（掩码图像建模在潜在空间进行对比学习）
"""

"""Training and optimizer abstraction for OlmoEarth Pretrain."""

from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from olmo_core.distributed.parallel import DataParallelConfig
from olmo_core.distributed.utils import get_local_rank, get_local_tensor
from olmo_core.optim import OptimConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.common import ReduceType

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.transform import TransformConfig
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.nn.flexi_vit import TokensAndMasks
from olmoearth_pretrain.nn.latent_mim import LatentMIM
from olmoearth_pretrain.nn.utils import unpack_encoder_output
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.train_module import (
    OlmoEarthTrainModule,
    OlmoEarthTrainModuleConfig,
)
from olmoearth_pretrain.train.utils import split_masked_batch

logger = getLogger(__name__)


@dataclass
class LatentMIMTrainModuleConfig(OlmoEarthTrainModuleConfig):
    """A configuration class for building :class:`LatentMIMTrainModule` instances.

    Args:
        loss_config: The loss configuration for the model.
        masking_config: The masking configuration for the model.
        ema_decay: EMA decay rate for target encoder (default: 0.99).
    """

    loss_config: LossConfig = field(
        default_factory=lambda: LossConfig(loss_config={"type": "patch_discrimination"})
    )
    mae_loss_config: LossConfig | None = None
    masking_config: MaskingConfig = field(
        default_factory=lambda: MaskingConfig(strategy_config={"type": "random"})
    )
    token_exit_cfg: dict[str, int] = field(
        default_factory=lambda: {modality: 0 for modality in Modality.names()}
    )
    ema_decay: tuple[float, float] = (0.996, 1.0)
    max_grad_norm: float = 1.0

    def build(
        self,
        model: LatentMIM,
        device: torch.device | None = None,
    ) -> "LatentMIMTrainModule":
        """Build the corresponding :class:`LatentMIMTrainModule`.

        Args:
            model: The model to train.
            device: The device to train on.
        """
        kwargs = self.prepare_kwargs()
        return LatentMIMTrainModule(
            model=model,
            device=device,
            **kwargs,
        )


class LatentMIMTrainModule(OlmoEarthTrainModule):
    """Latent MIM 训练模块。

    实现 Latent Masked Image Modeling 的训练逻辑：
    - 在线编码器编码可见 token，解码器重建掩码 token 的潜在表示
    - 目标编码器（EMA 更新）提供训练目标
    - 判别损失训练预测匹配目标
    - 可选的 MAE 重建损失和正则化损失
    - 支持微批次训练（梯度累积）

    关键属性:
        base_loss: 基础损失函数（通常为 PatchDiscrimination）
        mae_loss: 可选的 MAE 重建损失
        masking_strategy: 掩码策略
        regularizer: 可选的正则化项
        start_ema / end_ema: EMA 衰减率的起止值
        token_exit_cfg: 各模态的 token 退出层配置
    """

    def __init__(
        self,
        model: LatentMIM,
        optim_config: OptimConfig,
        transform_config: TransformConfig,
        masking_config: MaskingConfig,
        loss_config: LossConfig,
        rank_microbatch_size: int,
        token_exit_cfg: dict[str, int],
        mae_loss_config: LossConfig | None = None,
        compile_model: bool = False,
        dp_config: DataParallelConfig | None = None,
        compile_loss: bool = False,
        autocast_precision: torch.dtype | None = None,
        max_grad_norm: float | None = None,
        scheduler: Scheduler | None = None,
        device: torch.device | None = None,
        state_dict_save_opts: dist_cp_sd.StateDictOptions | None = None,
        state_dict_load_opts: dist_cp_sd.StateDictOptions | None = None,
        ema_decay: tuple[float, float] = (0.996, 1.0),
        regularizer_config: LossConfig | None = None,
        find_unused_parameters: bool = True,
    ):
        """Initialize the training module.

        Args:
            model: The transformer model to train.
            optim_config: The corresponding optimizer config.
            transform_config: The transform configuration for the model.
            masking_config: The masking configuration for the model.
            loss_config: The loss configuration for the model.
            rank_microbatch_size: The rank microbatch size in instances.
            compile_model: Whether to compile to the model.
            dp_config: Data parallel configuration for the model.
            loss_fn: Loss function to use.
            compile_loss: Whether to compile the loss function.
            autocast_precision: Enable AMP with this data type.
            max_grad_norm: Clip gradient norms to this value.
            scheduler: Optional learning rate scheduler.
            device: The device to train on.
            state_dict_save_opts: Override state dict options for saving.
            state_dict_load_opts: Override state dict options for loading.
            ema_decay: EMA decay rate for target encoder, as a tuple of (start_ema_decay, end_ema_decay)
            token_exit_cfg: The token exit configuration for the model.
            mae_loss_config: Optional loss config for masked auto-encoding.
            regularizer_config: An optional regularizer configuration for the model.
            find_unused_parameters: Whether to find unused parameters in the model, only used for DDP.
        """
        super().__init__(
            model=model,
            optim_config=optim_config,
            transform_config=transform_config,
            rank_microbatch_size=rank_microbatch_size,
            compile_model=compile_model,
            dp_config=dp_config,
            compile_loss=compile_loss,
            autocast_precision=autocast_precision,
            max_grad_norm=max_grad_norm,
            scheduler=scheduler,
            device=device,
            state_dict_save_opts=state_dict_save_opts,
            state_dict_load_opts=state_dict_load_opts,
            find_unused_parameters=find_unused_parameters,
        )
        self.start_ema, self.end_ema = ema_decay
        self.token_exit_cfg = token_exit_cfg
        self.base_loss = loss_config.build()
        self.masking_strategy = masking_config.build()
        self.regularizer = (
            regularizer_config.build() if regularizer_config is not None else None
        )

        self.total_loss_name = self.base_loss.name
        if self.regularizer is not None:
            self.total_loss_name = f"{self.base_loss.name}+{self.regularizer.name}"

        self.mae_loss = mae_loss_config.build() if mae_loss_config is not None else None
        if self.mae_loss is not None:
            self.total_loss_name = f"{self.total_loss_name}+{self.mae_loss.name}"

    def loss_fn(self, pred: Any, targets: Any) -> torch.Tensor:
        """Compute the loss between the predicted and target tensors."""
        return self.base_loss.compute(pred, targets)

    def train_batch(
        self,
        batch: tuple[int, MaskedOlmoEarthSample],
        dry_run: bool = False,
    ) -> None:
        """Train a batch.

        NOTE: Gradient accumulation/microbatching is not invariant for all losses across the same global batch size.

        - All Disc loss with same global batch size but different micro-batch sizes result in different gradients,
        though this matches the implementation in gallileo.
        - If the min hw is too low when subsampling, we may get micro-batches with uneven
        numbers of tokens making the loss for token averaged losses
        like l1 and l2 weight microbatches with less tokens relatively more.

        NOTE: For non contrastive losses, the loss is invariant to the global batch size across GPUS as well

        Args:
            batch: A (patch_size, MaskedOlmoEarthSample) tuple from the dataloader.
            dry_run: If True, skip metric recording and just run forward/backward.
        """
        if not dry_run:
            self.update_target_encoder()
        # Set the model to train mode
        self.model.train()
        total_batch_loss = torch.zeros([], device=self.device)
        total_batch_reg = torch.zeros([], device=self.device)
        patch_size = batch[0]
        batch_data = batch[1]

        # Split batch into microbatches
        masked_microbatches = split_masked_batch(batch_data, self.rank_microbatch_size)
        num_microbatches = len(masked_microbatches)

        for microbatch_idx in range(num_microbatches):
            with self._train_microbatch_context(microbatch_idx, num_microbatches):
                microbatch_masked = masked_microbatches[microbatch_idx]
                logger.info(
                    f"Training microbatch {microbatch_idx} of {num_microbatches} "
                    f"with batch size {microbatch_masked.batch_size} on rank {get_local_rank()}"
                )
                masked_batch = microbatch_masked.to_device(self.device)

                # Run Encoder and decoder on the augmented input
                loss, latent, decoded, target_output = self.model_forward(
                    masked_batch, patch_size, self.token_exit_cfg
                )
                reg_term = self.compute_regularization(latent)
                if reg_term is not None:
                    loss = loss + reg_term
                    total_batch_reg += (
                        get_local_tensor(reg_term.detach()) / num_microbatches
                    )
                # Scale loss by number of microbatches
                loss = loss / num_microbatches

                loss_val = get_local_tensor(loss.detach())
                total_batch_loss += loss_val

                # Skip bad batches
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    logger.warning(
                        f"NaN or Inf detected in loss at microbatch {microbatch_idx}, stopping training for this batch."
                    )
                    print(f"rank {get_local_rank()} has nan or inf")

                loss.backward()

        if dry_run:
            return

        self.trainer.record_metric(
            f"train/{self.total_loss_name}",
            total_batch_loss,
            ReduceType.mean,
        )
        self.log_regularization(total_batch_reg)

        del batch, batch_data  # In case this helps with memory utilization.
        del masked_batch
        del latent, decoded, target_output

    def model_forward(
        self,
        batch: MaskedOlmoEarthSample,
        patch_size: int,
        token_exit_cfg: dict[str, int],
    ) -> tuple[torch.Tensor, TokensAndMasks, TokensAndMasks, TokensAndMasks]:
        """Run a forward pass."""
        with self._model_forward_context():
            latent, decoded, _, reconstructed, extra_metrics = self.model(
                batch, patch_size
            )

            if extra_metrics is not None:
                self.log_extra_metrics(extra_metrics)
            with torch.no_grad():
                logger.info("Target Encoder forward pass...")
                output_dict = self.model.target_encoder.forward(
                    batch.unmask(),
                    patch_size=patch_size,
                    token_exit_cfg=token_exit_cfg,
                )
                target_output, _, _ = unpack_encoder_output(output_dict)
            loss = self.loss_fn(decoded, target_output)
            if self.mae_loss is not None:
                loss += self.mae_loss.compute(reconstructed, batch)
            return loss, latent, decoded, target_output
