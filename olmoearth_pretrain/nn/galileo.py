"""Galileo 模型模块，基于双预测器的自监督学习。

本模块实现 Galileo 风格的自监督预训练模型，核心思想是：
- 使用两个独立的解码器（decoder_a 和 decoder_b）
- 编码器同时为两个解码器提供潜在表示
- 目标编码器（target_encoder）提供回归目标
- 两个解码器分别从不同的增强视图进行预测

这种双预测器设计可以提高模型的鲁棒性和泛化能力，
借鉴了 Galileo 地球基础模型的思想。
"""

import logging
from copy import deepcopy
from dataclasses import dataclass

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


class Galileo(nn.Module, DistributedMixins):
    """Galileo 风格的自监督学习模型，使用双预测器。

    核心组件：
        encoder: 编码器，处理被掩码的输入
        decoder_a: 第一个解码器
        decoder_b: 第二个解码器（decoder_a 的深拷贝）
        target_encoder: 目标编码器（encoder 的深拷贝，参数冻结）
        reconstructor: 可选的重建器

    关键设计：
        - 两个解码器分别处理不同的增强视图
        - target_encoder 参数冻结，通过 EMA 更新
        - 支持 FSDP 分布式训练

    使用场景：
        需要双视图自监督训练的遥感基础模型。
    """

    supports_multiple_modalities_at_once = True  # 标记：支持同时处理多个模态

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        reconstructor: torch.nn.Module | None = None,
    ):
        """初始化 Galileo 模型。

        Args:
            encoder: 在线编码器模块
            decoder: 解码器模块（将被深拷贝为 decoder_a 和 decoder_b）
            reconstructor: 可选的重建器模块
        """
        super().__init__()
        self.encoder = encoder
        self.decoder_a = decoder  # 第一个解码器
        self.decoder_b = deepcopy(decoder)  # 第二个解码器（深拷贝）
        self.target_encoder = deepcopy(self.encoder)  # 目标编码器（深拷贝，参数冻结）
        self.reconstructor = reconstructor
        for p in self.target_encoder.parameters():
            p.requires_grad = False  # 冻结目标编码器参数

    def forward_a(
        self, x: MaskedOlmoEarthSample, patch_size: int
    ) -> tuple[TokensAndMasks, TokensAndMasks, torch.Tensor, TokensAndMasks | None]:
        """使用解码器 A 的前向传播。

        流程：编码 → 可选重建 → 解码器 A 预测

        Args:
            x: 被掩码的 OlmoEarth 样本
            patch_size: Patch 大小

        Returns:
            latent: 编码器嵌入
            decoded: 解码器 A 的预测
            latent_projected_and_pooled: 投影池化后的 token
            reconstructed: 重建器输出（若启用）
        """
        # TODO: Input And outputs here are not consistent between encoder and decoder need a tokensandmaks++
        output_dict = self.encoder(x, patch_size=patch_size)
        latent, latent_projected_and_pooled, decoder_kwargs = unpack_encoder_output(
            output_dict
        )
        reconstructed = None
        if self.reconstructor:
            reconstructed = self.reconstructor(latent, x.timestamps, patch_size)
        decoded = self.decoder_a(
            latent, timestamps=x.timestamps, patch_size=patch_size, **decoder_kwargs
        )
        return latent, decoded, latent_projected_and_pooled, reconstructed

    def forward_b(
        self, x: MaskedOlmoEarthSample, patch_size: int
    ) -> tuple[TokensAndMasks, TokensAndMasks, torch.Tensor, TokensAndMasks | None]:
        """使用解码器 B 的前向传播。

        流程：编码 → 可选重建 → 解码器 B 预测

        Args:
            x: 被掩码的 OlmoEarth 样本
            patch_size: Patch 大小

        Returns:
            latent: 编码器嵌入
            decoded: 解码器 B 的预测
            latent_projected_and_pooled: 投影池化后的 token
            reconstructed: 重建器输出（若启用）
        """
        # TODO: Input And outputs here are not consistent between encoder and decoder need a tokensandmaks++
        output_dict = self.encoder(x, patch_size=patch_size)
        latent, latent_projected_and_pooled, decoder_kwargs = unpack_encoder_output(
            output_dict
        )
        reconstructed = None
        if self.reconstructor:
            reconstructed = self.reconstructor(latent, x.timestamps, patch_size)
        decoded = self.decoder_b(
            latent, timestamps=x.timestamps, patch_size=patch_size, **decoder_kwargs
        )
        return latent, decoded, latent_projected_and_pooled, reconstructed

    def forward(
        self,
        input_a: MaskedOlmoEarthSample,
        input_b: MaskedOlmoEarthSample,
        patch_size: int,
    ) -> dict[
        str, tuple[TokensAndMasks, TokensAndMasks, torch.Tensor, TokensAndMasks | None]
    ]:
        """Galileo 前向传播，分别使用两个解码器处理两个输入。

        Args:
            input_a: 第一个增强视图的掩码样本
            input_b: 第二个增强视图的掩码样本
            patch_size: Patch 大小

        Returns:
            包含 'a' 和 'b' 两个键的字典，分别对应两个解码器的输出
        """
        return {
            "a": self.forward_a(input_a, patch_size),
            "b": self.forward_b(input_b, patch_size),
        }

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
        self.decoder_a.apply_fsdp(**fsdp_config)
        self.decoder_b.apply_fsdp(**fsdp_config)
        self.target_encoder.apply_fsdp(**fsdp_config)
        if self.reconstructor:
            self.reconstructor.apply_fsdp(**fsdp_config)
        # TODO: More finegrained wrapping of the encoder transformer layers next time
        fully_shard(self, **fsdp_config)
        register_fsdp_forward_method(self.target_encoder, "forward")

    def apply_compile(self) -> None:
        """Apply torch.compile to the model."""
        self.encoder.apply_compile()
        self.decoder_a.apply_compile()
        self.decoder_b.apply_compile()
        self.target_encoder.apply_compile()
        if self.reconstructor is not None:
            self.reconstructor.apply_compile()


@dataclass
class GalileoConfig(Config):
    """Galileo 模型的配置类。

    验证规则：
    - 编码器和解码器必须支持相同的模态
    - 编码器和解码器的最大序列长度必须一致
    - 编码器输出嵌入大小与解码器的编码器嵌入大小必须一致

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

    def build(self) -> "Galileo":
        """Build the Galileo model."""
        self.validate()
        encoder = self.encoder_config.build()
        decoder = self.decoder_config.build()
        reconstructor = (
            self.reconstructor_config.build()
            if self.reconstructor_config is not None
            else None
        )
        return Galileo(
            encoder=encoder,
            decoder=decoder,
            reconstructor=reconstructor,
        )
