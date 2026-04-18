"""Latent MIM（Masked Image Modeling）模型模块。

本模块实现 Latent MIM 风格的自监督预训练模型，核心思想是：
- 在线编码器（encoder）：处理被掩码的输入，提取潜在表示
- 目标编码器（target_encoder）：编码器的 EMA 副本，处理完整输入
- 解码器（decoder）：从在线编码器的潜在表示预测目标编码器的输出
- 重建器（reconstructor）：可选的 MAE 风格像素重建头

该架构借鉴了 BYOL/I-JEPA 的思想，使用 stop-gradient 的目标编码器
提供训练目标，避免了对比学习需要负样本的问题。
"""

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
    register_fsdp_forward_method,
)

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.nn.flexi_vit import TokensAndMasks
from olmoearth_pretrain.nn.utils import DistributedMixins, unpack_encoder_output

logger = logging.getLogger(__name__)


class LatentMIM(nn.Module, DistributedMixins):
    """Latent MIM 模型，基于 BYOL/I-JEPA 风格的自监督学习。

    核心组件：
        encoder: 在线编码器，处理被掩码的输入
        target_encoder: 目标编码器（encoder 的深拷贝，参数冻结），
            在训练中通过 EMA 更新，用于提供回归目标
        decoder: 解码器，从在线编码器的潜在表示预测目标编码器的输出
        reconstructor: 可选的重建器，用于 MAE 风格的像素级重建

    关键设计：
        - target_encoder 是 encoder 的深拷贝且 requires_grad=False
        - target_encoder 禁用 band dropout，始终看到完整光谱信息
        - 支持 FSDP 分布式训练和 torch.compile 加速

    使用场景：
        适用于遥感多模态数据的自监督预训练，学习跨模态的通用表示。
    """

    supports_multiple_modalities_at_once = True  # 标记：支持同时处理多个模态

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        reconstructor: torch.nn.Module | None = None,
    ):
        """初始化 Latent MIM 模型。

        Args:
            encoder: 在线编码器模块
            decoder: 解码器模块
            reconstructor: 可选的重建器模块（用于 MAE 风格像素重建）
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reconstructor = reconstructor
        self.target_encoder = deepcopy(self.encoder)  # 深拷贝编码器作为目标编码器
        for p in self.target_encoder.parameters():
            p.requires_grad = False  # 冻结目标编码器参数（通过 EMA 更新）
        # 禁用目标编码器的 band dropout，确保其始终看到完整光谱信息
        if hasattr(self.target_encoder, "disable_band_dropout"):
            self.target_encoder.disable_band_dropout()

    def forward(
        self, x: MaskedOlmoEarthSample, patch_size: int
    ) -> tuple[
        TokensAndMasks,
        TokensAndMasks,
        torch.Tensor,
        TokensAndMasks | None,
        dict[str, Any],
    ]:
        """Latent MIM 前向传播。

        流程：
        1. 编码器处理被掩码的输入，提取潜在表示
        2. 可选：重建器从潜在表示重建像素值
        3. 解码器从潜在表示预测目标编码器的输出

        Args:
            x: 被掩码的 OlmoEarth 样本
            patch_size: Patch 大小

        Returns:
            latent: 编码器的嵌入输出
            decoded: 解码器对掩码 token 的预测
            latent_projected_and_pooled: 投影和池化后的 token（用于对比损失）
            reconstructed: MAE 预测结果（若启用重建器）
            extra_metrics: 额外的度量信息（如 token 范数统计）
        """
        # 步骤1：编码器处理被掩码的输入
        output_dict = self.encoder(x, patch_size=patch_size)
        token_norm_stats = output_dict.pop("token_norm_stats", None)  # 提取 token 范数统计
        # 解包编码器输出：分离潜在表示、投影池化结果和解码器参数
        latent, latent_projected_and_pooled, decoder_kwargs = unpack_encoder_output(
            output_dict
        )
        extra_metrics = {}
        if token_norm_stats is not None:
            extra_metrics["token_norm_stats"] = token_norm_stats
        # 步骤2：可选 — 重建器从潜在表示重建像素值
        reconstructed = None
        if self.reconstructor:
            reconstructed = self.reconstructor(latent, x.timestamps, patch_size)
        # 步骤3：解码器从潜在表示预测目标编码器的输出
        decoded = self.decoder(
            latent, timestamps=x.timestamps, patch_size=patch_size, **decoder_kwargs
        )
        return (
            latent,
            decoded,
            latent_projected_and_pooled,
            reconstructed,
            extra_metrics,
        )

    def apply_fsdp(
        self,
        dp_mesh: DeviceMesh | None = None,
        param_dtype: torch.dtype | None = None,
        reduce_dtype: torch.dtype = torch.float32,
        prefetch_factor: int = 0,
    ) -> None:
        """Apply FSDP to the model."""
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype
        )
        fsdp_config = dict(mesh=dp_mesh, mp_policy=mp_policy)

        self.encoder.apply_fsdp(**fsdp_config)
        self.decoder.apply_fsdp(**fsdp_config)
        self.target_encoder.apply_fsdp(**fsdp_config)
        if self.reconstructor:
            self.reconstructor.apply_fsdp(**fsdp_config)
        # TODO: More finegrained wrapping of the encoder transformer layers next time
        fully_shard(self, **fsdp_config)
        register_fsdp_forward_method(self.target_encoder, "forward")

    def apply_compile(self) -> None:
        """Apply torch.compile to the model."""
        logger.info("Applying torch.compile to the model")
        self.encoder.apply_compile()
        logger.info("Applied torch.compile to the encoder")
        self.decoder.apply_compile()
        logger.info("Applied torch.compile to the decoder")
        self.target_encoder.apply_compile()
        logger.info("Applied torch.compile to the target encoder")


@dataclass
class LatentMIMConfig(Config):
    """Latent MIM 模型的配置类。

    验证规则：
    - 编码器和解码器必须支持相同的模态
    - 编码器和解码器的最大序列长度必须一致
    - 编码器的输出嵌入大小必须与解码器的编码器嵌入大小一致

    Attributes:
        encoder_config: 编码器配置
        decoder_config: 解码器配置
        reconstructor_config: 可选的重建器配置
    """

    encoder_config: Config
    decoder_config: Config
    reconstructor_config: Config | None = None

    def validate(self) -> None:
        """Validate the configuration."""
        if (
            self.encoder_config.supported_modalities
            != self.decoder_config.supported_modalities
        ):
            raise ValueError("Encoder and decoder must support the same modalities")
        if (
            self.encoder_config.max_sequence_length
            != self.decoder_config.max_sequence_length
        ):
            raise ValueError(
                "Encoder and decoder must have the same max sequence length"
            )
        encoder_output_size = (
            self.encoder_config.output_embedding_size
            or self.encoder_config.embedding_size
        )
        if encoder_output_size != self.decoder_config.encoder_embedding_size:
            raise ValueError("Encoder embedding size must be consistent!")

    def build(self) -> "LatentMIM":
        """Build the Latent Predictor."""
        self.validate()
        encoder = self.encoder_config.build()
        decoder = self.decoder_config.build()
        reconstructor = (
            self.reconstructor_config.build()
            if self.reconstructor_config is not None
            else None
        )
        return LatentMIM(
            encoder=encoder,
            decoder=decoder,
            reconstructor=reconstructor,
        )
