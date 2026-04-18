"""MAE（Masked Auto-Encoder）模型模块。

本模块实现经典的 MAE 自监督预训练架构：
- 编码器（encoder）：仅处理未被掩码的 token
- 解码器（decoder）：从编码器输出预测被掩码的 token
- 重建器（reconstructor）：可选的像素级重建头

MAE 的核心思想是通过掩码大部分输入（如 75%），迫使模型学习
数据的内在结构表示，仅对可见部分进行编码以大幅提升训练效率。
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
)

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.nn.flexi_vit import (
    EncoderConfig,
    PredictorConfig,
    ReconstructorConfig,
    TokensAndMasks,
)
from olmoearth_pretrain.nn.utils import DistributedMixins, unpack_encoder_output


class MAE(nn.Module, DistributedMixins):
    """掩码自编码器（Masked Auto-Encoder）模块。

    核心组件：
        encoder: 编码器，处理被掩码的输入，提取潜在表示
        decoder: 解码器，从潜在表示预测被掩码 token 的值
        reconstructor: 可选的重建器，进行像素级重建

    使用场景：
        标准的 MAE 自监督预训练，适用于遥感多模态数据。
    """

    supports_multiple_modalities_at_once = True  # 标记：支持同时处理多个模态

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module | None = None,
        reconstructor: nn.Module | None = None,
    ):
        """初始化 MAE 模型。

        Args:
            encoder: 编码器模块
            decoder: 可选的解码器模块
            reconstructor: 可选的重建器模块
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reconstructor = reconstructor

    def forward(
        self, x: MaskedOlmoEarthSample, patch_size: int
    ) -> tuple[TokensAndMasks, TokensAndMasks | None, TokensAndMasks | None]:
        """MAE 前向传播。

        流程：
        1. 编码器处理被掩码的输入
        2. 解码器从编码器输出预测被掩码 token
        3. 重建器从编码器输出进行像素级重建（可选）

        Args:
            x: 被掩码的 OlmoEarth 样本
            patch_size: Patch 大小

        Returns:
            latent: 编码器的嵌入输出
            decoded: 解码器的预测结果（若启用）
            reconstructed: 重建器的重建结果（若启用）
        """
        output_dict = self.encoder(x, patch_size=patch_size)  # 编码器前向传播
        latent, _, decoder_kwargs = unpack_encoder_output(output_dict)  # 解包编码器输出
        decoded = self.decoder and self.decoder(  # 解码器前向传播（若存在）
            latent, timestamps=x.timestamps, patch_size=patch_size, **decoder_kwargs
        )
        reconstructed = self.reconstructor and self.reconstructor(  # 重建器前向传播（若存在）
            latent, timestamps=x.timestamps, patch_size=patch_size
        )

        return latent, decoded, reconstructed

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
        if self.decoder:
            self.decoder.apply_fsdp(**fsdp_config)
        if self.reconstructor:
            self.reconstructor.apply_fsdp(**fsdp_config)
        # TODO: More finegrained wrapping of the encoder transformer layers next time
        fully_shard(self, **fsdp_config)

    def apply_compile(self) -> None:
        """Apply torch.compile to the model."""
        self.encoder.apply_compile()
        if self.decoder is not None:
            self.decoder.apply_compile()
        if self.reconstructor is not None:
            self.reconstructor.apply_compile()
        # TODO: add aaply for constructor


@dataclass
class MAEConfig(Config):
    """MAE 模型的配置类。

    验证规则：
    - 编码器和解码器必须支持相同的模态
    - 编码器和解码器的最大序列长度必须一致
    - 编码器输出嵌入大小与解码器的编码器嵌入大小必须一致
    - 重建器同理需与编码器配置一致

    Attributes:
        encoder_config: 编码器配置
        decoder_config: 解码器配置（可选）
        reconstructor_config: 重建器配置（可选）
    """

    encoder_config: EncoderConfig
    decoder_config: PredictorConfig | None = None
    reconstructor_config: ReconstructorConfig | None = None

    def validate(self) -> None:
        """Validate the configuration."""
        if self.decoder_config is not None:
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
        if self.reconstructor_config is not None:
            if (
                self.encoder_config.supported_modalities
                != self.reconstructor_config.supported_modalities
            ):
                raise ValueError(
                    "Encoder and reconstructor must support the same modalities"
                )
            if (
                self.encoder_config.max_sequence_length
                != self.reconstructor_config.decoder_config.max_sequence_length
            ):
                raise ValueError(
                    "Encoder and reconstructor must have the same max sequence length"
                )
            encoder_output_size = (
                self.encoder_config.output_embedding_size
                or self.encoder_config.embedding_size
            )
            if (
                encoder_output_size
                != self.reconstructor_config.decoder_config.encoder_embedding_size
            ):
                raise ValueError("Encoder embedding size must be consistent!")

    def build(self) -> "MAE":
        """Build the MAE Predictor."""
        self.validate()
        encoder = self.encoder_config.build()
        decoder = (
            self.decoder_config.build() if self.decoder_config is not None else None
        )
        reconstructor = (
            self.reconstructor_config.build()
            if self.reconstructor_config is not None
            else None
        )
        return MAE(
            encoder=encoder,
            decoder=decoder,
            reconstructor=reconstructor,
        )
