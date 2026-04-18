"""OlmoEarth Pretrain 的核心 FlexiViT 模型代码。

本模块实现了基于 FlexiViT 架构的多模态遥感基础模型，是整个项目最核心的模块。
FlexiViT 的核心特性是支持灵活的 patch 大小，允许在推理时动态调整分辨率，
而无需重新训练模型。

主要组件：
├── 工具函数
│   ├── get_modalities_to_process: 获取可用与支持的模态交集
│   └── return_modalities_from_dict: 从字典中提取模态名称
├── ProjectAndAggregate: 线性投影 + 池化模块（用于对比学习）
├── MultiModalPatchEmbeddings: 多模态 Patch 嵌入层
├── Reconstructor: Patch 重建模块（MAE 解码头）
├── CompositeEncodings: 复合位置编码（时间+空间+月份+通道）
├── FlexiVitBase: FlexiViT 基类（提供通用注意力逻辑）
├── Encoder: 编码器（处理掩码输入，提取 token 表示）
├── PredictorBase/Predictor: 预测器（从编码 token 预测掩码 token）
└── EncoderConfig/PredictorConfig: 对应的配置类

架构概述：
    输入 → Patch嵌入 → 位置编码 → Transformer块×N → LayerNorm → 投影/池化

支持的特性：
    - 多模态输入（Sentinel-1/2、DEM、LandSAT 等）
    - 灵活 patch 大小（通过插值调整卷积核）
    - Flash Attention（支持变长序列）
    - Register Token（提升注意力质量）
    - Band Dropout（随机丢弃光谱通道，增强跨光谱学习）
    - FSDP 分布式训练
    - torch.compile 加速
"""

import logging
import math
from dataclasses import dataclass
from typing import Any

import torch
from einops import rearrange, reduce, repeat
from torch import Tensor, nn
from torch.distributed.fsdp import fully_shard

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import (
    BASE_GSD,
    Modality,
    ModalitySpec,
    get_modality_specs_from_names,
)
from olmoearth_pretrain.datatypes import (
    MaskedOlmoEarthSample,
    MaskValue,
    TokensAndMasks,
)
from olmoearth_pretrain.nn.attention import Block
from olmoearth_pretrain.nn.encodings import (
    get_1d_sincos_pos_encoding,
    get_2d_sincos_pos_encoding_with_resolution,
    get_month_encoding_table,
)
from olmoearth_pretrain.nn.flexi_patch_embed import (
    FlexiPatchEmbed,
    FlexiPatchReconstruction,
)
from olmoearth_pretrain.nn.pooling import PoolingType, pool_unmasked_tokens
from olmoearth_pretrain.nn.tokenization import TokenizationConfig
from olmoearth_pretrain.nn.utils import get_cumulative_sequence_lengths

logger = logging.getLogger(__name__)


def get_modalities_to_process(
    available_modalities: list[str], supported_modality_names: list[str]
) -> list[str]:
    """获取实际需要处理的模态列表（可用模态与支持模态的交集）。

    Args:
        available_modalities: 当前输入数据中可用的模态名称列表
        supported_modality_names: 模型配置中支持的模态名称列表

    Returns:
        交集模态名称列表
    """
    modalities_to_process = set(supported_modality_names).intersection(
        set(available_modalities)
    )
    return list(modalities_to_process)


def return_modalities_from_dict(
    per_modality_input_tokens: dict[str, Tensor],
) -> list[str]:
    """从模态字典中提取非掩码的模态名称列表。

    通过排除以 "_mask" 结尾的键，仅返回实际的模态名称。

    Args:
        per_modality_input_tokens: 模态名称到张量的映射字典

    Returns:
        不含掩码键的模态名称列表
    """
    return [
        key for key in per_modality_input_tokens.keys() if not key.endswith("_mask")
    ]


# TokensAndMasks is imported from datatypes and re-exported here for backwards compatibility
# See olmoearth_pretrain.datatypes.TokensAndMasks for the implementation


class ProjectAndAggregate(nn.Module):
    """投影与聚合模块，对 token 进行线性投影和/或池化。

    支持三种操作模式：
    1. aggregate_then_project（默认）：先池化再投影
    2. project_then_aggregate：先投影再池化
    3. only_project：仅投影，保留 token 结构

    主要用于将编码器输出投影到对比学习所需的低维空间。

    关键属性：
        projection: 多层线性投影序列（可含 ReLU 激活）
        aggregate_then_project: 是否先池化再投影
        only_project: 是否仅投影不池化
    """

    def __init__(
        self,
        embedding_size: int,
        num_layers: int,
        aggregate_then_project: bool = True,
        output_embedding_size: int | None = None,
        only_project: bool = False,
    ):
        """初始化投影与聚合模块。

        Args:
            embedding_size: 输入 TokensAndMasks 的嵌入维度
            num_layers: 投影层数。若 >1，层间使用 ReLU 激活
            aggregate_then_project: 若 True，先平均池化再投影；若 False，先投影再池化
            output_embedding_size: 若指定，最后一层输出此维度而非 embedding_size
            only_project: 若 True，仅投影不聚合，保留 token 结构
        """
        super().__init__()
        self.only_project = only_project
        out_size = (
            output_embedding_size
            if output_embedding_size is not None
            else embedding_size
        )
        # 构建投影层：所有中间层使用 embedding_size，最后一层使用 out_size
        if num_layers == 1:
            projections = [nn.Linear(embedding_size, out_size)]  # 单层直接映射
        else:
            projections = [nn.Linear(embedding_size, embedding_size)]  # 第一层
            for _ in range(1, num_layers - 1):
                projections.append(nn.ReLU())  # 层间 ReLU 激活
                projections.append(nn.Linear(embedding_size, embedding_size))
            projections.append(nn.ReLU())  # 最后一层前的激活
            projections.append(nn.Linear(embedding_size, out_size))  # 最后一层
        self.projection = nn.Sequential(*projections)
        self.aggregate_then_project = aggregate_then_project

    def apply_aggregate_then_project(
        self, x: TokensAndMasks | torch.Tensor
    ) -> torch.Tensor:
        """先池化（平均）再投影。

        Args:
            x: TokensAndMasks 对象或张量

        Returns:
            投影后的池化结果，形状 [B, D_out]
        """
        if isinstance(x, TokensAndMasks):
            pooled_for_contrastive = pool_unmasked_tokens(
                x, PoolingType.MEAN, spatial_pooling=False
            )
        elif isinstance(x, torch.Tensor):
            pooled_for_contrastive = reduce(x, "b ... d -> b  d", "mean")
        else:
            raise ValueError(f"Invalid input type: {type(x)}")
        return self.projection(pooled_for_contrastive)

    def apply_project_then_aggregate(
        self, x: TokensAndMasks | torch.Tensor
    ) -> torch.Tensor:
        """先投影再池化（平均）。

        Args:
            x: TokensAndMasks 对象或张量

        Returns:
            池化后的投影结果，形状 [B, D_out]
        """
        if isinstance(x, TokensAndMasks):
            decoder_emedded_dict = x.as_dict(include_nones=True)
            for modality in x.modalities:
                x_modality = getattr(x, modality)
                # Are these normalizations masked correctly?
                x_modality = self.projection(x_modality)
                masked_modality_name = x.get_masked_modality_name(modality)
                decoder_emedded_dict[modality] = x_modality
                decoder_emedded_dict[masked_modality_name] = getattr(
                    x, masked_modality_name
                )
            x_projected = TokensAndMasks(**decoder_emedded_dict)
            projected_pooled = pool_unmasked_tokens(
                x_projected, PoolingType.MEAN, spatial_pooling=False
            )
        elif isinstance(x, torch.Tensor):
            x_projected = self.projection(x)
            projected_pooled = reduce(x_projected, "b ... d -> b  d", "mean")
        else:
            raise ValueError(f"Invalid input type: {type(x)}")
        return projected_pooled

    def apply_project_only(
        self, x: TokensAndMasks | torch.Tensor
    ) -> TokensAndMasks | torch.Tensor:
        """仅投影不聚合，保留 token 结构。

        Args:
            x: TokensAndMasks 对象或张量

        Returns:
            投影后的 TokensAndMasks 或张量，形状与输入一致（除最后维度）
        """
        if isinstance(x, TokensAndMasks):
            decoder_emedded_dict = x._asdict()
            for modality in x.modalities:
                x_modality = getattr(x, modality)
                x_modality = self.projection(x_modality)
                masked_modality_name = x.get_masked_modality_name(modality)
                decoder_emedded_dict[modality] = x_modality
                decoder_emedded_dict[masked_modality_name] = getattr(
                    x, masked_modality_name
                )
            return TokensAndMasks(**decoder_emedded_dict)
        elif isinstance(x, torch.Tensor):
            return self.projection(x)
        else:
            raise ValueError(f"Invalid input type: {type(x)}")

    def forward(
        self, x: TokensAndMasks | torch.Tensor
    ) -> torch.Tensor | TokensAndMasks:
        """对输入 TokensAndMasks 应用（非）线性投影。

        根据 only_project 和 aggregate_then_project 参数选择操作模式：
        - only_project=True: 仅投影
        - aggregate_then_project=True: 先池化再投影
        - aggregate_then_project=False: 先投影再池化

        Args:
            x: TokensAndMasks 或张量

        Returns:
            投影结果
        """
        if self.only_project:
            return self.apply_project_only(x)
        elif self.aggregate_then_project:
            return self.apply_aggregate_then_project(x)
        else:
            return self.apply_project_then_aggregate(x)


class MultiModalPatchEmbeddings(nn.Module):
    """多模态 Patch 嵌入层，将输入数据分块并编码为 token。

    为每种模态创建独立的 FlexiPatchEmbed 或 nn.Linear 嵌入模块，
    并根据 tokenization_config 将波段分组为不同的 token。
    支持训练时的 Band Dropout（随机丢弃光谱通道），增强跨光谱学习。

    关键属性：
        per_modality_embeddings: 模态名称到嵌入模块的字典
        max_patch_size: 最大 patch 大小
        embedding_size: 嵌入维度
        band_dropout_rate: Band Dropout 概率
        tokenization_config: 分词配置

    使用场景：
        作为编码器的第一步，将多模态遥感数据转换为统一维度的 token 表示。
    """

    def __init__(
        self,
        supported_modality_names: list[str],
        max_patch_size: int,
        embedding_size: int,
        tokenization_config: TokenizationConfig | None = None,
        use_linear_patch_embed: bool = True,
        band_dropout_rate: float = 0.0,
        random_band_dropout: bool = False,
        band_dropout_modalities: list[str] | None = None,
    ):
        """初始化多模态 Patch 嵌入层。

        Args:
            supported_modality_names: 模型支持的模态名称列表
            max_patch_size: 最大 patch 大小（基准）
            embedding_size: 嵌入维度
            tokenization_config: 可选的自定义波段分组配置
            use_linear_patch_embed: 传递给 FlexiPatchEmbed，设为 False 以加载旧检查点
            band_dropout_rate: 训练时随机丢弃波段的概率。
                > 0 时在 patch 嵌入前随机将某些波段置零，
                迫使模型学习跨光谱表示。仅在训练时激活。
            random_band_dropout: 若 True，每次前向调用从 Uniform(0, band_dropout_rate)
                采样丢弃率，减少训练-推理差异，增强数据增强效果。
            band_dropout_modalities: 若指定，仅对这些模态应用 band dropout。
                若 None，对所有模态应用。
        """
        super().__init__()
        self.max_patch_size = max_patch_size
        self.embedding_size = embedding_size
        self.supported_modality_names = supported_modality_names
        self.tokenization_config = tokenization_config or TokenizationConfig()
        self.use_linear_patch_embed = use_linear_patch_embed
        self.band_dropout_rate = band_dropout_rate
        self.random_band_dropout = random_band_dropout
        self.band_dropout_modalities = band_dropout_modalities
        # 为每种模态创建独立的嵌入模块
        self.per_modality_embeddings = nn.ModuleDict({})

        for modality in self.supported_modality_names:
            self.per_modality_embeddings[modality] = (
                self._get_patch_embedding_module_for_modality(modality)
            )

        # 为每个模态的每个波段组注册索引选择缓冲区
        # 用于从输入数据张量中选择对应波段的子集
        for modality in self.supported_modality_names:
            for idx, bandset_indices in enumerate(
                self.tokenization_config.get_bandset_indices(modality)
            ):
                buffer_name = self._get_buffer_name(modality, idx)
                banset_indices_tensor = torch.tensor(bandset_indices, dtype=torch.long)
                self.register_buffer(
                    buffer_name, banset_indices_tensor, persistent=False
                )  # 非持久缓冲区，不保存到检查点

        # Create a dictionary of per modality index tensors to do  index select with registered buffer

    @staticmethod
    def _get_buffer_name(modality: str, idx: int) -> str:
        """Get the buffer name."""
        return f"{modality}__{idx}_buffer"

    @staticmethod
    def _get_embedding_module_name(modality: str, idx: int) -> str:
        """Get the embedding module name.

        Module Dicts require string keys
        """
        return f"{modality}__{idx}"

    def _get_patch_embedding_module_for_modality(self, modality: str) -> nn.Module:
        """为指定模态创建嵌入模块。

        根据模态类型选择不同的嵌入方式：
        - 空间模态（is_spatial=True）：使用 FlexiPatchEmbed（2D 卷积/线性 patch 嵌入）
        - 非空间模态（is_spatial=False）：使用 nn.Linear（线性投影）

        Args:
            modality: 模态名称

        Returns:
            包含每个波段组嵌入模块的 ModuleDict
        """
        modality_spec = Modality.get(modality)
        # 从分词配置获取波段组索引（可能已被自定义覆盖）
        bandset_indices = self.tokenization_config.get_bandset_indices(modality)

        # 根据模态名称选择嵌入方式
        # 非空间模态（如静态特征）使用线性投影
        if not modality_spec.is_spatial:
            # static in space
            return nn.ModuleDict(
                {
                    self._get_embedding_module_name(modality, idx): nn.Linear(
                        len(channel_set_idxs), self.embedding_size
                    )
                    for idx, channel_set_idxs in enumerate(bandset_indices)
                }
            )
        else:
            return nn.ModuleDict(
                {
                    self._get_embedding_module_name(modality, idx): FlexiPatchEmbed(
                        in_chans=len(channel_set_idxs),
                        embedding_size=self.embedding_size,
                        base_patch_size_at_16=self.max_patch_size,
                        modality_spec=modality_spec,
                        use_linear_patch_embed=self.use_linear_patch_embed,
                    )
                    for idx, channel_set_idxs in enumerate(bandset_indices)
                }
            )

    def apply_embedding_to_modality(
        self,
        modality: str,
        input_data: MaskedOlmoEarthSample,
        patch_size: int,
    ) -> tuple[Tensor, Tensor]:
        """对指定模态的数据应用嵌入操作。

        核心流程：
        1. 根据波段组索引从输入数据中选择对应波段子集
        2. 可选：应用 Band Dropout 随机丢弃某些波段
        3. 使用嵌入模块（FlexiPatchEmbed 或 Linear）将数据投影到嵌入空间
        4. 提取对应的掩码信息

        Args:
            modality: 模态名称
            input_data: 掩码输入样本
            patch_size: Patch 大小

        Returns:
            (modality_tokens, modality_masks) 元组
            - modality_tokens: 形状 [B, H/P, W/P, T, b_s, D]（空间模态）
              或 [B, T, b_s, D]（非空间模态）
            - modality_masks: 对应的掩码张量
        """
        logger.debug(f"applying embedding to modality:{modality}")
        masked_modality_name = input_data.get_masked_modality_name(modality)
        modality_mask = getattr(input_data, masked_modality_name)
        modality_data = getattr(input_data, modality)

        modality_spec = Modality.get(modality)
        num_band_sets = self.tokenization_config.get_num_bandsets(modality)

        modality_tokens, modality_masks = [], []
        for idx in range(num_band_sets):
            modality_specific_kwargs = {}
            if not modality_spec.is_spatial:
                # static in time
                token_mask = modality_mask[..., idx]
            else:
                token_mask = modality_mask[
                    :,
                    0 :: patch_size * modality_spec.image_tile_size_factor,
                    0 :: patch_size * modality_spec.image_tile_size_factor,
                    ...,
                    idx,
                ]
                modality_specific_kwargs = {"patch_size": patch_size}

            buffer_name = self._get_buffer_name(modality, idx)
            inp_data = torch.index_select(modality_data, -1, getattr(self, buffer_name))

            # Check if we should apply band dropout for this bandset
            apply_dropout = (
                self.band_dropout_modalities is None
                or modality in self.band_dropout_modalities
            )
            if self.training and apply_dropout and self.band_dropout_rate > 0.0:
                num_bands = inp_data.shape[-1]
                # Only apply band dropout if there are more than 1 band
                if num_bands > 1:
                    if self.random_band_dropout:
                        rate = (
                            torch.rand(1, device=inp_data.device).item()
                            * self.band_dropout_rate
                        )
                    else:
                        rate = self.band_dropout_rate
                    inp_data = self._apply_band_dropout(inp_data, rate)

            embedding_module = self.per_modality_embeddings[modality][
                self._get_embedding_module_name(modality, idx)
            ]
            patchified_data = embedding_module(inp_data, **modality_specific_kwargs)

            modality_tokens.append(patchified_data)
            modality_masks.append(token_mask)
        return torch.stack(modality_tokens, dim=-2), torch.stack(modality_masks, dim=-1)

    @staticmethod
    def _apply_band_dropout(patchified_data: Tensor, rate: float) -> Tensor:
        """随机将波段通道置零，以强制跨光谱学习。

        在批次维度上独立地对每个样本的每个波段以概率 rate 丢弃。
        确保每个样本至少保留一个波段，避免完全丢失信息。

        Args:
            patchified_data: 输入张量，最后一维为波段维度
            rate: 每个波段的丢弃概率

        Returns:
            随机置零部分波段后的张量，每个样本至少保留 1 个波段
        """
        num_bands = patchified_data.shape[-1]
        batch_size = patchified_data.shape[0]
        keep_mask = (
            torch.rand(batch_size, num_bands, device=patchified_data.device) >= rate
        )
        # If no bands are kept, randomly select one band to keep
        no_bands_kept = ~keep_mask.any(dim=1)
        if no_bands_kept.any():
            rand_idx = torch.randint(
                num_bands, (no_bands_kept.sum(),), device=keep_mask.device
            )
            keep_mask[no_bands_kept, rand_idx] = True
        # Broadcast: [B, 1, 1, ..., num_bands]
        view_shape = [batch_size] + [1] * (patchified_data.dim() - 2) + [num_bands]
        return patchified_data * keep_mask.view(*view_shape).to(patchified_data.dtype)

    @staticmethod
    def is_any_data_seen_by_encoder(modality_mask: Tensor) -> bool:
        """Check if any data is seen by the encoder."""
        return (MaskValue.ONLINE_ENCODER.value == modality_mask).any()

    def apply_compile(self) -> None:
        """Apply torch.compile to the model."""
        self.compile(dynamic=False, mode="max-autotune-no-cudagraphs", fullgraph=True)

    def forward(
        self,
        input_data: MaskedOlmoEarthSample,
        patch_size: int,
    ) -> dict[str, Tensor]:
        """对输入数据的所有模态进行灵活的 Patch 嵌入。

        将 [B, H, W, (T), C] 的输入转换为 [B, H/P, W/P, (T), b_s, D] 的输出。
        假设空间掩码与给定的 patch 大小一致（即掩码在 patch 边界对齐）。

        Args:
            input_data: 掩码输入样本
            patch_size: Patch 大小

        Returns:
            模态名称到嵌入张量的映射字典，同时包含对应的掩码
        """
        output_dict = {}
        modalities_to_process = get_modalities_to_process(
            input_data.modalities, self.supported_modality_names
        )
        for modality in modalities_to_process:
            modality_tokens, modality_masks = self.apply_embedding_to_modality(
                modality, input_data, patch_size
            )
            output_dict[modality] = modality_tokens
            modality_mask_name = input_data.get_masked_modality_name(modality)
            output_dict[modality_mask_name] = modality_masks
        return output_dict


class Reconstructor(nn.Module):
    """Patch 重建模块，将 token 解码回像素空间。

    首先通过解码器对 token 进行处理，然后使用 FlexiPatchReconstruction
    或 nn.Linear 将每个模态的 token 重建回原始像素值。

    关键属性：
        decoder: 内部解码器模块
        per_modality_reconstructions: 模态到重建模块的字典
        max_patch_size: 最大 patch 大小
        embedding_size: 嵌入维度
    """

    def __init__(
        self,
        decoder: nn.Module,
        supported_modalities: list[ModalitySpec],
        max_patch_size: int,
        tokenization_config: TokenizationConfig | None = None,
    ):
        """初始化重建模块。

        Args:
            decoder: 在重建前对 token 进行处理的预测器模块
            supported_modalities: 支持的模态列表
            max_patch_size: 最大 patch 大小
            tokenization_config: 可选的自定义波段分组配置
        """
        super().__init__()
        self.max_patch_size = max_patch_size
        self.embedding_size = decoder.output_embedding_size
        self.supported_modalities = supported_modalities
        self.tokenization_config = tokenization_config or TokenizationConfig()
        self.decoder = decoder
        # TODO: want to be able to remove certain bands and modalities
        self.per_modality_reconstructions = nn.ModuleDict({})
        for modality in self.supported_modalities:
            self.per_modality_reconstructions[modality.name] = (
                self._get_patch_reconstruction_module_for_modality(modality)
            )

    def apply_compile(self) -> None:
        """Apply torch.compile to the model."""
        self.decoder.apply_compile()

    def apply_fsdp(self, **fsdp_kwargs: Any) -> None:
        """Apply FSDP to the model."""
        self.decoder.apply_fsdp(**fsdp_kwargs)

    @staticmethod
    def _get_reconstruction_module_name(modality: str, idx: int) -> str:
        """Get the reconstruction module name.

        Module Dicts require string keys
        """
        return f"{modality}__{idx}"

    def _get_patch_reconstruction_module_for_modality(
        self, modality: ModalitySpec
    ) -> nn.Module:
        """为指定模态创建重建模块。

        根据模态类型选择不同的重建方式：
        - 非空间模态（tile_resolution==0）：使用 nn.Linear
        - 空间模态：使用 FlexiPatchReconstruction（转置卷积重建）

        Args:
            modality: 模态规格说明

        Returns:
            包含每个波段组重建模块的 ModuleDict
        """
        # Get bandset indices from tokenization config (may be overridden)
        bandset_indices = self.tokenization_config.get_bandset_indices(modality.name)

        # 根据模态类型选择重建方式
        # 非空间模态（如静态特征）使用线性投影
        if modality.get_tile_resolution() == 0:
            # static in space
            return nn.ModuleDict(
                {
                    self._get_reconstruction_module_name(modality.name, idx): nn.Linear(
                        self.embedding_size, len(channel_set_idxs)
                    )
                    for idx, channel_set_idxs in enumerate(bandset_indices)
                }
            )
        else:
            return nn.ModuleDict(
                {
                    self._get_reconstruction_module_name(
                        modality.name, idx
                    ): FlexiPatchReconstruction(
                        out_chans=len(channel_set_idxs),
                        embedding_size=self.embedding_size,
                        max_patch_size=self.max_patch_size,
                    )
                    for idx, channel_set_idxs in enumerate(bandset_indices)
                }
            )

    # TODO: Likely we want a single object that stores all the data related configuration etc per modality including channel grous bands patch size etc
    def apply_reconstruction_to_modality(
        self, modality: str, input_data: TokensAndMasks, patch_size: int
    ) -> tuple[Tensor, Tensor]:
        """对指定模态的数据应用重建操作。

        核心流程：
        1. 按波段组分离 token
        2. 使用对应的重建模块（FlexiPatchReconstruction 或 Linear）解码回像素空间
        3. 将掩码扩展到像素级（通过 repeat 扩展空间维度）

        Args:
            modality: 模态名称
            input_data: 解码后的 TokensAndMasks
            patch_size: Patch 大小

        Returns:
            (modality_tokens, modality_mask) 元组
            - modality_tokens: 重建的像素数据 [B, H*P, W*P, T, C]
            - modality_mask: 扩展到像素级的掩码
        """
        masked_modality_name = input_data.get_masked_modality_name(modality)
        modality_mask = getattr(input_data, masked_modality_name)
        modality_data = getattr(input_data, modality)

        modality_spec = Modality.get(modality)
        bandset_indices = self.tokenization_config.get_bandset_indices(modality)

        # x: Input tensor with shape [b, h, w, (t), b_s, d]
        modality_tokens, modality_masks = [], []
        for idx, channel_set_indices in enumerate(bandset_indices):
            data = modality_data[..., idx, :]
            masks = modality_mask[..., idx]
            r_model = self.per_modality_reconstructions[modality][
                self._get_reconstruction_module_name(modality, idx)
            ]
            if modality_spec.get_tile_resolution() == 0:
                data = r_model(data)
            else:
                data = r_model(data, patch_size=patch_size)
            modality_tokens.append(data)
            masks = repeat(
                masks,
                "b h w ... -> b (h p_h) (w p_w) ...",
                p_h=patch_size,
                p_w=patch_size,
            )
            modality_masks.append(masks)
        modality_mask = repeat(
            modality_mask,
            "b h w ... -> b (h p_h) (w p_w) ...",
            p_h=patch_size,
            p_w=patch_size,
        )
        return torch.cat(modality_tokens, dim=-1), modality_mask

    def forward(
        self,
        x: TokensAndMasks,
        timestamps: Tensor,
        patch_size: int,
        input_res: int = BASE_GSD,
    ) -> TokensAndMasks:
        """对输入数据的所有模态进行灵活的 Patch 重建。

        流程：先通过解码器处理 token，然后对每个模态重建回像素空间。

        Args:
            x: 编码器输出的 TokensAndMasks
            timestamps: 时间戳
            patch_size: Patch 大小
            input_res: 输入分辨率

        Returns:
            重建后的 TokensAndMasks，形状从 [B, H/P, W/P, T, b_s, D] 变为 [B, H, W, T, C]
        """
        input_data = self.decoder(x, timestamps, patch_size, input_res)
        output_dict = {}
        modalities_to_process = get_modalities_to_process(
            input_data.modalities, [m.name for m in self.supported_modalities]
        )
        for modality in modalities_to_process:
            modality_tokens, modality_masks = self.apply_reconstruction_to_modality(
                modality, input_data, patch_size
            )
            output_dict[modality] = modality_tokens
            modality_mask_name = input_data.get_masked_modality_name(modality)
            output_dict[modality_mask_name] = modality_masks
        return TokensAndMasks(**output_dict)


@dataclass
class ReconstructorConfig(Config):
    """重建器的配置类。

    包含重建器的所有超参数：
        decoder_config: 内部解码器的配置
        supported_modality_names: 支持的模态名称列表
        max_patch_size: 最大 patch 大小
        tokenization_config: 可选的分词配置
    """

    decoder_config: "Config"
    supported_modality_names: list[str]
    max_patch_size: int = 8
    tokenization_config: TokenizationConfig | None = None

    def __post_init__(self) -> None:
        """Coerce raw dicts to TokenizationConfig for old checkpoint compatibility."""
        if isinstance(self.tokenization_config, dict):
            self.tokenization_config = TokenizationConfig(**self.tokenization_config)

    def validate(self) -> None:
        """Validate the configuration."""
        if len(self.supported_modalities) == 0:
            raise ValueError("At least one modality must be added!")
        else:
            for modality in self.supported_modalities:
                if modality not in Modality.values():
                    raise ValueError(f"Modality {modality} is not supported")
        if self.tokenization_config is not None:
            self.tokenization_config.validate()

    @property
    def supported_modalities(self) -> list[ModalitySpec]:
        """Get the supported modalities."""
        return get_modality_specs_from_names(self.supported_modality_names)

    def build(self) -> "Reconstructor":
        """Build the reconstructor."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("supported_modality_names")
        kwargs["supported_modalities"] = self.supported_modalities
        kwargs.pop("decoder_config")
        kwargs["decoder"] = self.decoder_config.build()
        logger.info(f"Predictor kwargs: {kwargs}")
        return Reconstructor(**kwargs)


class CompositeEncodings(nn.Module):
    """复合位置编码模块，为 FlexiViT 模型添加多种位置编码。

    将嵌入维度平均分配给四种编码类型，各占 25%：
    1. 通道编码（channel）：标识不同模态/波段组
    2. 时间位置编码（pos_in_time）：标识时间步位置
    3. 月份编码（month）：标识观测月份
    4. 空间编码（spatial）：标识空间位置（考虑 GSD）

    关键属性：
        pos_embed: 1D 正弦-余弦时间位置编码（冻结）
        month_embed: 月份嵌入层（冻结）
        per_modality_channel_embeddings: 每个模态的通道嵌入参数
        embedding_dim_per_embedding_type: 每种编码类型的维度（embedding_size * 0.25）
    """

    def __init__(
        self,
        embedding_size: int,
        supported_modalities: list[ModalitySpec],
        max_sequence_length: int,
        learnable_channel_embeddings: bool = True,
        random_channel_embeddings: bool = False,
        tokenization_config: TokenizationConfig | None = None,
    ):
        """初始化复合位置编码模块。

        Args:
            embedding_size: 嵌入维度总数，平均分配给四种编码
            supported_modalities: 支持的模态列表
            max_sequence_length: 最大序列长度（时间维度）
            learnable_channel_embeddings: 是否使用可学习的通道嵌入
            random_channel_embeddings: 是否随机初始化通道嵌入（False 则初始化为零）
            tokenization_config: 可选的自定义波段分组配置
        """
        super().__init__()
        self.embedding_size = embedding_size
        self.supported_modalities = supported_modalities
        self.supported_modality_names = [
            modality.name for modality in supported_modalities
        ]
        self.tokenization_config = tokenization_config or TokenizationConfig()
        self.embedding_size = embedding_size
        self.max_sequence_length = (
            max_sequence_length  # This max sequence length is a time dim thing
        )
        # TODO: we need to be able to calculate the size of the param based on what types of embeddings it will get

        # 将嵌入维度平均分配给 4 种编码类型（通道、时间、月份、空间）
        self.embedding_dim_per_embedding_type = int(embedding_size * 0.25)
        # 时间位置编码：1D 正弦-余弦编码，初始化后冻结
        self.pos_embed = nn.Parameter(
            get_1d_sincos_pos_encoding(
                torch.arange(max_sequence_length),
                self.embedding_dim_per_embedding_type,
            ),
            requires_grad=False,
        )
        # 月份编码表（冻结）
        month_tab = get_month_encoding_table(self.embedding_dim_per_embedding_type)
        self.month_embed = nn.Embedding.from_pretrained(month_tab, freeze=True)
        if not learnable_channel_embeddings and not random_channel_embeddings:
            self.per_modality_channel_embeddings = nn.ParameterDict()
            for modality in self.supported_modalities:
                num_bandsets = self.tokenization_config.get_num_bandsets(modality.name)
                shape = (num_bandsets, self.embedding_dim_per_embedding_type)
                channel_embeddings = nn.Parameter(
                    torch.zeros(shape), requires_grad=False
                )
                self.per_modality_channel_embeddings[modality.name] = channel_embeddings
        else:
            # Channel embeddings
            if learnable_channel_embeddings:
                args = {"requires_grad": True}
            else:
                args = {"requires_grad": False}

            self.per_modality_channel_embeddings = nn.ParameterDict()
            for modality in self.supported_modalities:
                num_bandsets = self.tokenization_config.get_num_bandsets(modality.name)
                shape = (num_bandsets, self.embedding_dim_per_embedding_type)
                if random_channel_embeddings:
                    channel_embeddings = nn.Parameter(torch.rand(shape), **args)
                else:
                    channel_embeddings = nn.Parameter(torch.zeros(shape), **args)
                self.per_modality_channel_embeddings[modality.name] = channel_embeddings

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if getattr(m, "_skip_custom_init", False):
            return
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                # TODO: fix the dtype here
                nn.init.constant_(m.bias, 0).to(torch.float32)

    @staticmethod
    def calculate_gsd_ratio(input_res: float, patch_size: int) -> float:
        """Calculate the Ground Sample Distance ratio."""
        return input_res * patch_size / BASE_GSD

    def _apply_encodings_per_modality(
        self,
        modality_name: str,
        modality_tokens: Tensor,
        timestamps: Tensor | None = None,
        patch_size: int | None = None,
        input_res: int | None = None,
        use_modality_encodings: bool = True,
        use_temporal_encodings: bool = True,
    ) -> Tensor:
        """对单个模态的 token 应用复合位置编码。

        根据模态类型和参数，依次添加四种编码：
        1. 通道编码：标识模态和波段组（维度 0 : n）
        2. 时间位置编码：标识时间步（维度 n : 2n）
        3. 月份编码：标识观测月份（维度 2n : 3n）
        4. 空间编码：标识空间位置，考虑 GSD（维度 3n : 4n）

        Args:
            modality_name: 模态名称
            modality_tokens: 模态的 token 张量
            timestamps: 可选的时间戳（用于时间和月份编码）
            patch_size: 可选的 patch 大小（用于空间编码）
            input_res: 可选的输入分辨率（用于空间编码）
            use_modality_encodings: 是否添加通道编码
            use_temporal_encodings: 是否添加时间/月份编码

        Returns:
            添加位置编码后的 token 张量
        """
        logger.debug(
            f"use_modality_encodings: {use_modality_encodings}, use_temporal_encodings: {use_temporal_encodings}"
        )
        # TODO: Improve this implementation it is quite bad

        modality = Modality.get(modality_name)
        logger.debug(f"Applying encodings to modality {modality}")
        if not use_modality_encodings and use_temporal_encodings:
            b, h, w, t, _ = modality_tokens.shape
            ein_string, ein_dict = (
                "b h w t d",
                {"b": b, "h": h, "w": w, "t": t},
            )
        elif not use_temporal_encodings and not use_modality_encodings:
            b, h, w, _ = modality_tokens.shape
            ein_string, ein_dict = (
                "b h w d",
                {"b": b, "h": h, "w": w},
            )
        elif not use_temporal_encodings and use_modality_encodings:
            raise NotImplementedError("Not implemented")
        else:
            if modality_tokens.ndim == 3:
                # modality_tokens = [B, Band_Sets, D]; static in space, static in time
                b, b_s, _ = modality_tokens.shape
                ein_string, ein_dict = "b b_s d", {"b": b, "b_s": b_s}
            elif modality_tokens.ndim == 4:
                b, t, b_s, _ = modality_tokens.shape
                ein_string, ein_dict = "b t b_s d", {"b": b, "t": t, "b_s": b_s}
            elif modality_tokens.ndim == 5:
                b, h, w, b_s, _ = modality_tokens.shape
                ein_string, ein_dict = (
                    "b h w b_s d",
                    {"b": b, "h": h, "w": w, "b_s": b_s},
                )
            elif modality_tokens.ndim == 6:
                b, h, w, t, b_s, _ = modality_tokens.shape
                ein_string, ein_dict = (
                    "b h w t b_s d",
                    {"b": b, "h": h, "w": w, "t": t, "b_s": b_s},
                )
            else:
                raise ValueError(f"Unsupported tokens shape: {modality_tokens.shape}")

        device = modality_tokens.device
        modality_embed = torch.zeros(modality_tokens.shape, device=device)
        n = self.embedding_dim_per_embedding_type
        actual_bandsets = modality_tokens.shape[-2]

        # 通道编码：标识模态和波段组
        if use_modality_encodings:
            channel_embed = self.per_modality_channel_embeddings[modality.name]
            if channel_embed.shape[0] != actual_bandsets:
                raise ValueError(
                    f"Channel embeddings for {modality.name} expect "
                    f"{channel_embed.shape[0]} bandsets but tokens have "
                    f"{actual_bandsets}. Ensure tokenization_config is "
                    "consistently passed to the encoder/decoder and masking strategy."
                )
            channel_embed = repeat(
                channel_embed, f"b_s d -> {ein_string}", **ein_dict
            ).to(device)
            modality_embed[..., :n] += channel_embed  # 添加到前 n 维

        if modality.is_multitemporal and use_temporal_encodings:
            # 时间位置编码：标识时间步位置
            time_embed = repeat(self.pos_embed[:t], f"t d -> {ein_string}", **ein_dict)
            modality_embed[..., n : n * 2] += time_embed.to(device)  # 添加到第 n~2n 维

            # 月份编码：标识观测月份
            assert timestamps is not None
            months = timestamps[:, :, 1]  # 提取月份信息
            month_embed = self.month_embed(months)
            month_embed = repeat(month_embed, f"b t d -> {ein_string}", **ein_dict)
            modality_embed[..., n * 2 : n * 3] += month_embed.to(device)  # 添加到第 2n~3n 维
        if modality.is_spatial:
            # 空间编码：标识空间位置，考虑地面采样距离（GSD）
            assert input_res is not None
            assert patch_size is not None
            gsd_ratio = self.calculate_gsd_ratio(input_res, patch_size)  # 计算 GSD 比率
            spatial_embed = get_2d_sincos_pos_encoding_with_resolution(
                grid_size=(h, w),
                res=torch.ones(b, device=device) * gsd_ratio,
                encoding_dim=self.embedding_dim_per_embedding_type,
                device=device,
            )
            spatial_embed = rearrange(spatial_embed, "b (h w) d -> b h w d", h=h, w=w)
            spatial_embed = repeat(
                spatial_embed, f"b h w d -> {ein_string}", **ein_dict
            )
            modality_embed[..., n * 3 : n * 4] += spatial_embed  # 添加到第 3n~4n 维
        return modality_tokens + modality_embed

    def forward(
        self,
        per_modality_input_tokens: dict[str, Tensor],
        timestamps: Tensor,
        patch_size: int,
        input_res: int = BASE_GSD,
    ) -> dict[str, Tensor]:
        """Apply the encodings to the patchified data.

        Args:
            per_modality_input_tokens: Tokens only for each modality
            timestamps: Timestamps of the data
            patch_size: Size of patches
            input_res: Resolution of the input data

        Returns:
            Tokens only for each modality
        """
        output_dict = {}
        available_modalities = return_modalities_from_dict(per_modality_input_tokens)
        modalities_to_process = get_modalities_to_process(
            available_modalities, self.supported_modality_names
        )
        for modality_name in modalities_to_process:
            output_dict[modality_name] = self._apply_encodings_per_modality(
                modality_name,
                per_modality_input_tokens[modality_name],
                timestamps=timestamps,
                patch_size=patch_size,
                input_res=input_res,
            )
        return output_dict


class FlexiVitBase(nn.Module):
    """FlexiViT 基类，提供多模态 Transformer 的通用注意力逻辑。

    本类不直接使用，而是作为 Encoder 和 PredictorBase 的基类。
    提供了以下通用功能：
    - Transformer 块的初始化和 Xavier 权重初始化
    - 复合位置编码
    - Token 的折叠（collapse_and_combine_hwtc）和展开（split_and_expand_per_modality）
    - Flash Attention 的 token 打包/解包
    - FSDP 和 torch.compile 支持

    关键属性：
        blocks: Transformer 注意力块列表
        composite_encodings: 复合位置编码模块
        supported_modality_names: 支持的模态名称列表
        use_flash_attn: 是否使用 Flash Attention
    """

    cross_attn: bool = False

    def __init__(
        self,
        embedding_size: int,
        max_sequence_length: int,
        num_heads: int,
        mlp_ratio: float,
        depth: int,
        drop_path: float,
        supported_modalities: list[ModalitySpec],
        learnable_channel_embeddings: bool = True,
        random_channel_embeddings: bool = False,
        use_flash_attn: bool = False,
        qk_norm: bool = False,
        tokenization_config: TokenizationConfig | None = None,
    ) -> None:
        """Initialize the FlexiVitBase class."""
        super().__init__()

        self.embedding_size = embedding_size
        self.supported_modalities = supported_modalities
        self.supported_modality_names = [x.name for x in supported_modalities]
        logger.info(f"modalities being used by model: {self.supported_modality_names}")

        self.max_sequence_length = max_sequence_length
        self._base_tokenization_config = tokenization_config or TokenizationConfig()

        self.use_flash_attn = use_flash_attn
        self.learnable_channel_embeddings = learnable_channel_embeddings
        self.random_channel_embeddings = random_channel_embeddings
        self.blocks = nn.ModuleList(
            [
                Block(
                    embedding_size,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_norm=qk_norm,
                    norm_layer=nn.LayerNorm,  # TODO: This should be configurable
                    cross_attn=self.cross_attn,
                    drop_path=drop_path,
                    use_flash_attn=self.use_flash_attn,
                )
                for _ in range(depth)
            ]
        )

        self.composite_encodings = CompositeEncodings(
            embedding_size,
            self.supported_modalities,
            max_sequence_length,
            learnable_channel_embeddings,
            random_channel_embeddings,
            tokenization_config=self._base_tokenization_config,
        )
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if getattr(m, "_skip_custom_init", False):
            logger.debug(f"Skipping custom init for {m}")
            return
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def grab_modality_specific_dims(modality_data: Tensor) -> tuple[int, ...]:
        """Grab the modality specific dimensions from the modality data.

        Assumes [B, ..., C, D]

        Every modality will have a batch dimension, a channel dimension and embedding dimension.

        Args:
            modality_data: Modality data

        Returns:
            Modality specific dimensions
        """
        return modality_data.shape[1:-2] if modality_data.ndim > 3 else ()

    def collapse_and_combine_hwtc(self, x: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """将各模态的 token 和掩码分别折叠为两个统一张量。

        将各模态的所有 token 展平并沿 token 维度拼接，
        将各模态的所有掩码展平并沿 token 维度拼接。
        这是进入 Transformer 注意力层前的准备步骤。

        Args:
            x: 模态名称到张量的映射字典

        Returns:
            (tokens, masks) 元组，tokens 形状 [B, total_tokens, D]
        """
        tokens, masks = [], []
        available_modalities = return_modalities_from_dict(x)
        modalities_to_process = get_modalities_to_process(
            available_modalities, self.supported_modality_names
        )
        for modality in modalities_to_process:
            masked_modality_name = MaskedOlmoEarthSample.get_masked_modality_name(
                modality
            )
            x_modality = x[modality]
            x_modality_mask = x[masked_modality_name]
            tokens.append(rearrange(x_modality, "b ... d -> b (...) d"))
            masks.append(rearrange(x_modality_mask, "b ... -> b (...)"))
        tokens = torch.cat(tokens, dim=1)
        masks = torch.cat(masks, dim=1)

        return tokens, masks

    @staticmethod
    def _construct_einops_pattern(
        spatial_dims: tuple[int, ...],
    ) -> tuple[str, dict[str, int]]:
        """Given a tuple of spatial dimensions (e.g. [B, H, W, T, ...]).

        build (1) an einops rearrange pattern of the form:
            "d -> (dim0) (dim1) (dim2)... d"
        and (2) a dictionary mapping dim0..dimN to the actual sizes.

        This allows reshaping a single-dimensional tensor [D] into
        [B, H, W, T, ..., D] using einops.
        """
        dim_dict = {f"dim{i}": size for i, size in enumerate(spatial_dims)}
        # e.g., "d -> (dim0) (dim1) (dim2) (dim3) d"
        pattern_input = (
            "d -> " + " ".join(f"(dim{i})" for i in range(len(spatial_dims))) + " d"
        )
        return pattern_input, dim_dict

    def split_tokens_masks_and_dims(
        self, x: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, tuple]]:
        """Split the tokens, masks, and dimensions out into separate dicts."""
        tokens_only_dict = {}
        original_masks_dict = {}
        modalities_to_dims_dict = {}
        available_modalities = return_modalities_from_dict(x)
        modalities_to_process = get_modalities_to_process(
            available_modalities, self.supported_modality_names
        )
        for modality in modalities_to_process:
            x_modality = x[modality]
            tokens_only_dict[modality] = x_modality
            modalities_to_dims_dict[modality] = x_modality.shape
            masked_modality_name = MaskedOlmoEarthSample.get_masked_modality_name(
                modality
            )
            original_masks_dict[masked_modality_name] = x[masked_modality_name]
        return tokens_only_dict, original_masks_dict, modalities_to_dims_dict

    @staticmethod
    def split_and_expand_per_modality(
        x: Tensor, modalities_to_dims_dict: dict
    ) -> dict[str, Tensor]:
        """将统一张量按模态拆分并展开回各模态的原始形状。

        collapse_and_combine_hwtc 的逆操作。从拼接的 token 序列中
        按各模态的 token 数量切分，然后恢复为各模态的原始空间/时间形状。

        Args:
            x: 统一的 token 张量，形状 [B, total_tokens, D]
            modalities_to_dims_dict: 模态名称到原始形状的映射

        Returns:
            模态名称到恢复形状后的 token 张量的映射
        """
        tokens_only_dict = {}
        tokens_reshaped = 0
        for modality, dims in modalities_to_dims_dict.items():
            # Skip batch (first) and embedding (last) dimensions
            middle_dims = dims[1:-1]
            num_tokens_for_modality = math.prod(middle_dims)

            # Extract tokens for this modality (b n d)
            modality_tokens = x[
                :, tokens_reshaped : tokens_reshaped + num_tokens_for_modality
            ]

            # TODO: see if there  is a general and clean einops way to do this
            # Reshape to original dimensions (e.g., for 4D spatial dims: b d1 d2 d3 d4 e)
            x_modality = modality_tokens.view(x.shape[0], *middle_dims, x.shape[-1])

            tokens_reshaped += num_tokens_for_modality
            tokens_only_dict[modality] = x_modality

        return tokens_only_dict

    @staticmethod
    def pack_tokens(tokens: Tensor, mask: Tensor) -> Tensor:
        """将 token 的批次和序列维度打包为一维，移除掩码位置。

        用于 Flash Attention 的变长序列模式，将 [B, T, D] 打包为 [total_valid_tokens, D]。

        Args:
            tokens: 待打包的 token 张量 [B, T, D]
            mask: 掩码张量 [B, T]（True 表示有效 token）

        Returns:
            仅包含有效 token 的打包张量 [total_valid_tokens, D]
        """
        tokens_packed = torch.flatten(tokens, end_dim=1)
        mask = torch.flatten(mask)
        tokens = tokens_packed[mask]
        return tokens

    @staticmethod
    def unpack_tokens(tokens: Tensor, mask: Tensor, og_shape: tuple) -> Tensor:
        """将打包的一维 token 解包回原始的批次和序列维度。

        pack_tokens 的逆操作，将 [total_valid_tokens, D] 解包为 [B, T, D]，
        掩码位置填充零。

        Args:
            tokens: 打包的 token 张量 [total_valid_tokens, D]
            mask: 掩码张量 [B, T]
            og_shape: 原始张量形状 (B, T, D)

        Returns:
            解包后的 token 张量 [B, T, D]
        """
        tokens_new = tokens.new_zeros(og_shape[0] * og_shape[1], og_shape[2])
        mask = torch.flatten(mask)
        tokens_new[mask] = tokens
        tokens = tokens_new.reshape(og_shape[0], og_shape[1], -1)
        return tokens

    def apply_fsdp(self, **fsdp_kwargs: Any) -> None:
        """Apply FSDP to the model."""
        for block in self.blocks:
            block.apply_fsdp(**fsdp_kwargs)

    def apply_compile(self) -> None:
        """Apply torch.compile to the model."""
        for block in self.blocks:
            block.apply_compile()


class Encoder(FlexiVitBase):
    """FlexiViT 编码器，将掩码输入处理为 token 表示。

    完整流程：
    1. Patch 嵌入：将输入图像/数据划分为 patch 并投影到嵌入空间
    2. 位置编码：添加复合位置编码（时间+空间+月份+通道）
    3. Transformer 注意力：通过多层自注意力块处理 token
    4. LayerNorm：在最后一层应用归一化
    5. 投影/池化：将 token 投影和池化为对比学习所需的表示

    支持的高级特性：
    - Register Token：额外的可学习 token，提升注意力质量
    - Band Dropout：训练时随机丢弃光谱通道
    - Flash Attention：高效的注意力实现
    - 冻结 Patch 嵌入：保持嵌入层参数不变
    - Token Exit：支持不同深度的 token 提前退出
    - Fast Pass：推理时跳过掩码处理，直接使用全部 token

    关键属性：
        patch_embeddings: 多模态 Patch 嵌入层
        project_and_aggregate: 投影和聚合模块
        embedding_projector: 可选的嵌入投影器（改变输出维度）
        register_tokens: 可选的 Register Token 参数
    """

    cross_attn: bool = False

    def __init__(
        self,
        embedding_size: int,
        max_patch_size: int,
        min_patch_size: int,
        num_heads: int,
        mlp_ratio: float,
        depth: int,
        drop_path: float,
        supported_modalities: list[ModalitySpec],
        max_sequence_length: int,
        num_register_tokens: int = 0,
        learnable_channel_embeddings: bool = True,
        random_channel_embeddings: bool = False,
        num_projection_layers: int = 1,
        aggregate_then_project: bool = True,
        use_flash_attn: bool = False,
        frozen_patch_embeddings: bool = False,
        qk_norm: bool = False,
        log_token_norm_stats: bool = False,
        output_embedding_size: int | None = None,
        tokenization_config: TokenizationConfig | None = None,
        use_linear_patch_embed: bool = True,
        band_dropout_rate: float = 0.0,
        random_band_dropout: bool = False,
        band_dropout_modalities: list[str] | None = None,
    ):
        """Initialize the encoder.

        Args:
            embedding_size: Size of token embeddings
            max_patch_size: Maximum patch size for patchification
            min_patch_size: Minimum patch size for patchification
            num_heads: Number of attention heads
            mlp_ratio: Ratio for MLP hidden dimension
            depth: Number of transformer layers
            drop_path: Drop path rate
            supported_modalities: list documenting modalities used in a given model instantiation
            max_sequence_length: Maximum sequence length
            num_register_tokens: Number of register tokens to use
            learnable_channel_embeddings: Whether to use learnable channel embeddings
            random_channel_embeddings: Initialize channel embeddings randomly (zeros if False)
            num_projection_layers: The number of layers to use in the projection. If >1, then
                a ReLU activation will be applied between layers
            aggregate_then_project: If True, then we will average the tokens before applying
                the projection. If False, we will apply the projection first.
            use_flash_attn: Whether to use flash attention
            frozen_patch_embeddings: If True, we freeze the embedding layer, as recommended in
                https://arxiv.org/pdf/2104.02057, Section 4.2
            qk_norm: Whether to apply normalization to Q and K in attention
            log_token_norm_stats: Whether to log the token norm stats
            output_embedding_size: If set, project tokens to this size after attention
            tokenization_config: Optional config for custom band groupings
            use_linear_patch_embed: If True, use nn.Linear for patch projection (faster).
                Set False to load checkpoints trained before this flag existed (Conv2d weights).
            band_dropout_rate: Probability of dropping each band channel during training.
            random_band_dropout: If True, sample dropout rate from Uniform(0, band_dropout_rate).
            band_dropout_modalities: If provided, only apply band dropout to these
                modalities. If None, apply to all modalities. Default: None.
        """
        self.tokenization_config = tokenization_config or TokenizationConfig()
        super().__init__(
            embedding_size=embedding_size,
            depth=depth,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            max_sequence_length=max_sequence_length,
            learnable_channel_embeddings=learnable_channel_embeddings,
            drop_path=drop_path,
            supported_modalities=supported_modalities,
            use_flash_attn=use_flash_attn,
            random_channel_embeddings=random_channel_embeddings,
            qk_norm=qk_norm,
            tokenization_config=self.tokenization_config,
        )
        self.num_register_tokens = num_register_tokens
        self.has_register_tokens = num_register_tokens > 0
        self.log_token_norm_stats = log_token_norm_stats
        if self.has_register_tokens:
            self.register_tokens = nn.Parameter(
                torch.zeros(num_register_tokens, embedding_size)
            )
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.embedding_size = embedding_size
        self.use_linear_patch_embed = use_linear_patch_embed
        self.band_dropout_rate = band_dropout_rate
        self.random_band_dropout = random_band_dropout
        self.band_dropout_modalities = band_dropout_modalities
        self.patch_embeddings = MultiModalPatchEmbeddings(
            self.supported_modality_names,
            self.max_patch_size,
            self.embedding_size,
            tokenization_config=self.tokenization_config,
            use_linear_patch_embed=self.use_linear_patch_embed,
            band_dropout_rate=self.band_dropout_rate,
            random_band_dropout=self.random_band_dropout,
            band_dropout_modalities=self.band_dropout_modalities,
        )
        self.output_embedding_size = output_embedding_size
        # If output_embedding_size is set, project tokens to that size after attention
        self.embedding_projector: ProjectAndAggregate | None = None
        if output_embedding_size is not None:
            self.embedding_projector = ProjectAndAggregate(
                embedding_size=self.embedding_size,
                num_layers=1,
                output_embedding_size=output_embedding_size,
                only_project=True,
            )
            final_embedding_size = output_embedding_size
        else:
            final_embedding_size = self.embedding_size
        self.project_and_aggregate = ProjectAndAggregate(
            embedding_size=final_embedding_size,
            num_layers=num_projection_layers,
            aggregate_then_project=aggregate_then_project,
        )
        self.norm = nn.LayerNorm(self.embedding_size)

        self.apply(self._init_weights)

        if frozen_patch_embeddings:
            for p in self.patch_embeddings.parameters():
                p.requires_grad = False
        if self.has_register_tokens:
            self._init_register_tokens()

    def disable_band_dropout(self) -> None:
        """禁用 Band Dropout（用于目标/EMA 编码器）。

        目标编码器需要始终看到完整的光谱信息，
        因此需要禁用 Band Dropout。
        """
        self.patch_embeddings.band_dropout_rate = 0.0

    def _init_register_tokens(self) -> None:
        """Initialize the register tokens."""
        nn.init.xavier_uniform_(self.register_tokens)

    def create_token_exit_ids(
        self, x: dict[str, Tensor], token_exit_cfg: dict[str, int]
    ) -> dict[str, Tensor]:
        """为每个波段组创建 token 退出 ID。

        退出 ID 指示每个 token 在第几层退出 Transformer。
        假设模态通道组在 token 的倒数第二维。

        Args:
            x: 模态到 token 的映射
            token_exit_cfg: 模态到退出层数的映射

        Returns:
            模态到退出 ID 张量的映射
        """
        exit_ids_per_modality_dict = {}
        available_modalities = return_modalities_from_dict(x)
        modalities_to_process = get_modalities_to_process(
            available_modalities, self.supported_modality_names
        )
        for modality in modalities_to_process:
            num_exit_layers = token_exit_cfg[modality]
            exit_seq_modality = torch.full_like(x[modality], fill_value=num_exit_layers)
            exit_ids_per_modality_dict[modality] = exit_seq_modality
        return exit_ids_per_modality_dict

    @staticmethod
    def remove_masked_tokens(
        x: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """移除掩码位置的 token，仅保留在线编码器可见的 token。

        这是 MAE 效率优化的核心：仅对可见 token 计算注意力，
        大幅减少计算量（如 75% 掩码率时，计算量降为 25%）。

        实现方式：
        1. 按掩码值降序排序，使可见 token 排在前面
        2. 截取到最长有效序列长度
        3. 返回排序索引以便后续恢复

        Args:
            x: 待处理的 token 张量 [B, T, D]
            mask: 掩码张量 [B, T]（1=保留，0=移除）

        Returns:
            (tokens, indices, updated_mask, seqlens, max_length) 五元组
            - tokens: 仅含可见 token [B, T', D]
            - indices: 排序索引 [B, T]（用于恢复原始顺序）
            - updated_mask: 截取后的掩码 [B, T']
            - seqlens: 各样本的有效 token 数 [B]
            - max_length: 最长有效序列长度
        """
        sorted_mask, indices = torch.sort(mask, dim=1, descending=True, stable=True)  # 降序排序：可见 token 排前
        # 根据排序索引重排 token，使可见 token 排在前面
        x = x.gather(1, indices[:, :, None].expand_as(x))
        # Now all tokens that should be kept are first in the tensor

        # 将掩码位置的值置零（非必须，因为后续会忽略）
        x = x * sorted_mask.unsqueeze(-1)

        # 截取到最长有效序列长度，去除尾部全零 token
        seq_lengths = sorted_mask.sum(-1)  # 各样本的有效 token 数
        max_length = seq_lengths.max()  # 最长有效序列长度
        x = x[:, :max_length]
        # New mask chopped to the longest sequence
        updated_mask = sorted_mask[:, :max_length]

        return x, indices, updated_mask, seq_lengths, max_length

    @staticmethod
    def add_removed_tokens(
        x: Tensor, indices: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """将移除的掩码位置恢复，用零填充掩码位置的 token。

        remove_masked_tokens 的逆操作。将仅含可见 token 的张量
        恢复为原始长度，掩码位置填充零。

        Args:
            x: 仅含可见 token 的张量 [B, T', D]
            indices: 排序索引 [B, T]
            mask: 掩码张量 [B, T']

        Returns:
            (tokens, mask) 元组，tokens 形状 [B, T, D]
        """
        assert x.shape[1] > 0, (
            "x must have at least one token we should not mask all tokens"
        )
        masked_tokens = repeat(
            torch.zeros_like(x[0, 0, :]), "d -> b t d", b=x.shape[0], t=indices.shape[1]
        )
        full_mask = torch.cat(
            (
                mask,
                torch.zeros(
                    (x.shape[0], indices.shape[1] - x.shape[1]),
                    device=x.device,
                    dtype=mask.dtype,
                ),
            ),
            dim=-1,
        )
        # can't set value on leaf variable
        out = masked_tokens.clone()
        # put tokens in full masked tensor (at the first N positions in every row)
        out[full_mask] = x[mask]
        # then move them to their original positions
        out = out.scatter(1, indices[:, :, None].expand_as(out), out)
        full_mask = full_mask.scatter(1, indices.expand_as(full_mask), full_mask)
        # Values that were masked out are not returned but the values that are still there are returned to the original positions
        return out, full_mask

    def create_exit_seqs(
        self,
        tokens_only_dict: dict[str, Tensor],
        mask_only_dict: dict[str, Tensor],
        token_exit_cfg: dict[str, int] | None,
    ) -> tuple[Tensor | None]:
        """Create the exit sequences and tokens."""
        # Check that tokens_only_dict doesn't contain any mask keys
        assert all(not key.endswith("_mask") for key in tokens_only_dict), (
            "tokens_only_dict should not contain mask keys"
        )
        if token_exit_cfg:
            exit_ids_per_modality = self.create_token_exit_ids(
                tokens_only_dict, token_exit_cfg
            )
            exit_ids_per_modality.update(mask_only_dict)
            # Exit ids seqs tells us which layer to exit each token
            exit_ids_seq, _ = self.collapse_and_combine_hwtc(exit_ids_per_modality)
        else:
            exit_ids_seq = None
        return exit_ids_seq

    def _maybe_get_attn_mask(
        self,
        new_mask: Tensor,
        fast_pass: bool,
    ) -> Tensor | None:
        """Get the attention mask or None if we should pass None to the transformer."""
        if fast_pass or not self.training:
            return None
        else:
            return new_mask

    def add_register_tokens_and_masks(
        self,
        tokens: Tensor,
        attn_mask: Tensor | None,
        processed_register_tokens: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """在 token 序列前拼接 Register Token。

        Register Token 是额外的可学习 token，不对应任何输入数据，
        用作注意力汇聚点，防止自注意力退化。参考：
        Vision Transformers Need Registers (https://arxiv.org/abs/2309.16588)

        Args:
            tokens: 输入 token 张量 [B, T, D]
            attn_mask: 注意力掩码 [B, T]
            processed_register_tokens: 可选的预处理过的 register token

        Returns:
            (tokens, attn_mask) 元组，tokens 形状 [B, num_reg + T, D]
        """
        batch_size = tokens.shape[0]
        # Expand register tokens to match batch size: [num_register_tokens, embedding_size] -> [batch_size, num_register_tokens, embedding_size]
        if processed_register_tokens is None:
            reg_tokens = self.register_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            reg_tokens = processed_register_tokens
        # Concatenate register tokens at the beginning: [batch_size, seq_len, embedding_size] -> [batch_size, num_register_tokens + seq_len, embedding_size]
        tokens = torch.cat([reg_tokens, tokens], dim=1)
        if attn_mask is not None:
            # Create mask for register tokens (all True - they should participate in attention)
            reg_mask = torch.ones(
                batch_size,
                self.num_register_tokens,
                dtype=attn_mask.dtype,
                device=attn_mask.device,
            )
            attn_mask = torch.cat([reg_mask, attn_mask], dim=1)
        else:
            reg_mask = None
        return tokens, attn_mask

    def pop_register_tokens(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """从 token 序列中分离 Register Token。

        Args:
            tokens: 包含 Register Token 的张量 [B, num_reg + T, D]

        Returns:
            (tokens, register_tokens) 元组
            - tokens: 去除 Register Token 后的张量 [B, T, D]
            - register_tokens: Register Token [B, num_reg, D]
        """
        register_tokens = tokens[:, : self.num_register_tokens, :]
        tokens = tokens[:, self.num_register_tokens :, :]
        return tokens, register_tokens

    def get_token_norm_stats(
        self, tokens: Tensor, register_tokens: Tensor
    ) -> dict[str, float]:
        """计算 token 范数统计信息（用于诊断 Register Token 是否膨胀）。

        对 Register Token 和非 Register Token 分别计算 L2 范数统计：
        - Register Token: 均值、最小值、最大值
        - 非 Register Token: 均值、最小值、最大值、标准差、多个分位数

        Args:
            tokens: 非 Register Token [B, T, D]
            register_tokens: Register Token [B, num_reg, D]

        Returns:
            包含各项统计指标的字典
        """
        # Compute norms for register tokens: [batch_size, num_register_tokens]
        register_tokens_norms = torch.norm(register_tokens, dim=2)
        reg_norms_flat = register_tokens_norms.flatten()
        reg_stats = {
            "register_mean": reg_norms_flat.mean().item(),
            "register_min": reg_norms_flat.min().item(),
            "register_max": reg_norms_flat.max().item(),
        }

        # Compute norms for non-register tokens: [batch_size, seq_len]
        nonreg_tokens_norms = torch.norm(tokens, dim=2)
        nonreg_norms_flat = nonreg_tokens_norms.flatten()
        percentiles = [25.0, 75.0, 90.0, 95.0, 99.0]
        nonreg_percentiles = torch.quantile(
            nonreg_norms_flat.float(),
            torch.tensor(
                [p / 100.0 for p in percentiles], device=nonreg_norms_flat.device
            ),
        ).tolist()
        nonreg_stats = {
            "nonregister_mean": nonreg_norms_flat.mean().item(),
            "nonregister_min": nonreg_norms_flat.min().item(),
            "nonregister_max": nonreg_norms_flat.max().item(),
            "nonregister_std": nonreg_norms_flat.std().item(),
            "nonregister_25th": nonreg_percentiles[0],
            "nonregister_75th": nonreg_percentiles[1],
            "nonregister_90th": nonreg_percentiles[2],
            "nonregister_95th": nonreg_percentiles[3],
            "nonregister_99th": nonreg_percentiles[4],
        }

        token_norm_stats = {**reg_stats, **nonreg_stats}
        return token_norm_stats

    def _maybe_remove_masked_tokens(
        self,
        tokens: Tensor,
        mask: Tensor,
        fast_pass: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Remove masked tokens from the tokens and masks."""
        if fast_pass and not self.use_flash_attn:
            # This is the inference fast pass
            indices = None
            new_mask = None
            seq_lengths = None
            max_seqlen = None
            bool_mask = None
        else:
            bool_mask = mask == MaskValue.ONLINE_ENCODER.value
            tokens, indices, new_mask, seq_lengths, max_seqlen = (
                self.remove_masked_tokens(tokens, bool_mask)
            )
        return tokens, indices, new_mask, seq_lengths, max_seqlen, bool_mask

    def _maybe_add_removed_tokens(
        self,
        tokens: Tensor,
        indices: Tensor,
        mask: Tensor,
        fast_pass: bool,
    ) -> Tensor:
        """Add removed tokens to the tokens and masks."""
        if not fast_pass:
            tokens, _ = self.add_removed_tokens(tokens, indices, mask)
        return tokens

    def apply_attn(
        self,
        x: dict[str, Tensor],
        timestamps: Tensor,
        patch_size: int,
        input_res: int,
        token_exit_cfg: dict[str, int] | None = None,
        fast_pass: bool = False,
    ) -> tuple[dict[str, Tensor], dict[str, Any] | None]:
        """对 token 应用 Transformer 注意力。

        核心流程：
        1. 分离 token 和掩码
        2. 添加复合位置编码
        3. 折叠各模态 token 为统一序列
        4. 移除掩码 token（MAE 效率优化）
        5. 可选：打包 token（Flash Attention）
        6. 可选：添加 Register Token
        7. 逐层应用 Transformer 块
        8. 可选：移除 Register Token
        9. 可选：解包 token（Flash Attention）
        10. 应用 LayerNorm
        11. 恢复掩码位置（填充零）
        12. 展开回各模态原始形状

        Args:
            x: 模态到张量的映射字典
            timestamps: 时间戳
            patch_size: Patch 大小
            input_res: 输入分辨率
            token_exit_cfg: Token 退出配置（不同模态使用不同深度）
            fast_pass: 是否启用快速推理模式（跳过掩码处理）

        Returns:
            (tokens_per_modality_dict, token_norm_stats) 元组
        """
        # 步骤1：分离 token 和掩码，记录各模态的原始形状
        tokens_only_dict, original_masks_dict, modalities_to_dims_dict = (
            self.split_tokens_masks_and_dims(x)
        )
        # 已为空操作但可移除
        exit_ids_seq = self.create_exit_seqs(
            tokens_only_dict, original_masks_dict, token_exit_cfg
        )
        # 退出 token 初始值为线性投影（无编码时的值）
        exited_tokens, _ = self.collapse_and_combine_hwtc(x)

        # 步骤2：添加复合位置编码
        tokens_dict = self.composite_encodings.forward(
            tokens_only_dict,
            timestamps,
            patch_size,
            input_res,
        )
        tokens_dict.update(original_masks_dict)  # 合并掩码信息

        # 步骤3：折叠各模态 token 为统一的序列
        tokens, mask = self.collapse_and_combine_hwtc(tokens_dict)

        # 步骤4：移除掩码 token（MAE 效率优化：仅对可见 token 计算注意力）
        tokens, indices, new_mask, seq_lengths, max_seqlen, bool_mask = (
            self._maybe_remove_masked_tokens(tokens, mask, fast_pass)
        )

        if exit_ids_seq is not None:
            exit_ids_seq, _, _, _, _ = self.remove_masked_tokens(
                exit_ids_seq, bool_mask
            )
            # still linear projections
            exited_tokens, _, _, _, _ = self.remove_masked_tokens(
                exited_tokens, bool_mask
            )

        # 步骤5：可选 — 打包 token（Flash Attention 变长模式）
        if self.use_flash_attn:
            cu_seqlens = get_cumulative_sequence_lengths(seq_lengths)  # 计算累积序列长度
            og_shape = tokens.shape  # 记录原始形状
            tokens = self.pack_tokens(tokens, new_mask)  # 打包：移除填充位置
        else:
            cu_seqlens = None

        attn_mask = self._maybe_get_attn_mask(
            new_mask,
            fast_pass=fast_pass,
        )

        # 步骤6：可选 — 添加 Register Token
        if self.has_register_tokens:
            tokens, attn_mask = self.add_register_tokens_and_masks(tokens, attn_mask)

        # 步骤7：逐层应用 Transformer 注意力块
        for i_blk, blk in enumerate(self.blocks):
            # Token 退出逻辑：跳过第 0 层（允许预测共享编码的平凡解）
            if (exit_ids_seq is not None) and (i_blk > 0):
                # this should only ever be called by the target encoder,
                # in a torch.no_grad context
                assert exited_tokens is not None
                # If a token should exit, then we update the exit token with the current token at the same position
                # 若 token 应在当前层退出，更新退出 token
                exited_tokens = torch.where(
                    condition=(exit_ids_seq == i_blk),
                    input=tokens,  # 退出条件满足：使用当前层的 token
                    other=exited_tokens,  # 否则：保留之前的退出值
                )
            # we take the inverse of the mask because a value
            # of True indicates the value *should* take part in
            # attention
            # WARNING: THIS MAY CHANGE DEPENDING ON THE ATTENTION IMPLEMENTATION

            tokens = blk(
                x=tokens,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                # we will have to specify k and q lens for cross attention
                attn_mask=attn_mask,
            )

        # 步骤8：可选 — 移除 Register Token
        if self.has_register_tokens:
            tokens, register_tokens = self.pop_register_tokens(tokens)
            # 可选：记录 token 范数统计（用于诊断 Register Token 是否膨胀）
            token_norm_stats = (
                self.get_token_norm_stats(tokens, register_tokens)
                if self.log_token_norm_stats
                else None
            )
        else:
            token_norm_stats = None

        # 步骤9：可选 — 解包 token（Flash Attention）
        if self.use_flash_attn:
            tokens = self.unpack_tokens(tokens, new_mask, og_shape)

        # Token 退出：最后一层，所有 token 使用完整深度的输出
        if exit_ids_seq is not None:
            # this should only ever be called by the target encoder,
            # in a torch.no_grad context
            assert exited_tokens is not None
            # full depth
            # IMPORTANT: write this to x
            tokens = torch.where(
                condition=(exit_ids_seq == (i_blk + 1)),  # 2 for full depth
                input=tokens,
                other=exited_tokens,
            )
        # 步骤10：应用 LayerNorm（在恢复掩码位置前，仅对"真实" token 归一化）
        tokens = self.norm(tokens)
        # 步骤11：恢复掩码位置（填充零），使用原始未裁剪的掩码
        tokens = self._maybe_add_removed_tokens(tokens, indices, new_mask, fast_pass)

        # 步骤12：展开回各模态的原始空间/时间形状
        tokens_per_modality_dict = self.split_and_expand_per_modality(
            tokens, modalities_to_dims_dict
        )
        # 合并原始掩码和处理后的 token
        tokens_per_modality_dict.update(original_masks_dict)
        return tokens_per_modality_dict, token_norm_stats

    def forward(
        self,
        x: MaskedOlmoEarthSample,
        patch_size: int,
        input_res: int = BASE_GSD,
        token_exit_cfg: dict | None = None,
        fast_pass: bool = False,
    ) -> dict[str, Any]:
        """编码器前向传播：将掩码输入处理为 token 表示。

        流程：
        1. Patch 嵌入：将输入数据转换为 patch token
        2. 注意力处理：添加位置编码并通过 Transformer 块
        3. 可选：嵌入投影到 output_embedding_size
        4. 投影/池化：为对比学习生成全局表示

        Args:
            x: 掩码输入样本
            patch_size: Patch 大小
            input_res: 输入数据的地面采样距离（GSD）
            token_exit_cfg: Token 退出配置（键为模态名，值为退出层数）
            fast_pass: 快速推理模式（跳过掩码处理，启用原生 Flash Attention）

        Returns:
            输出字典，包含：
            - "tokens_and_masks": TokensAndMasks 对象
            - "project_aggregated": 投影池化后的张量（对比学习用）
            - "token_norm_stats": Token 范数统计（可选）
        """
        if fast_pass and token_exit_cfg is not None:
            raise ValueError("token_exit_cfg cannot be set when fast_pass is True")

        # 步骤1：Patch 嵌入 — 将输入数据转换为 patch token
        patchified_tokens_and_masks = self.patch_embeddings.forward(x, patch_size)

        # 步骤2：注意力处理（添加位置编码 → Transformer 块）
        if token_exit_cfg is None or any(
            [exit_depth > 0 for exit_depth in token_exit_cfg.values()]
        ):
            patchified_tokens_and_masks, token_norm_stats = self.apply_attn(
                x=patchified_tokens_and_masks,
                timestamps=x.timestamps,
                patch_size=patch_size,
                input_res=input_res,
                token_exit_cfg=token_exit_cfg,
                fast_pass=fast_pass,
            )
        else:
            token_norm_stats = {}
        output = TokensAndMasks(**patchified_tokens_and_masks)

        # 步骤3：可选 — 投影到 output_embedding_size
        if self.embedding_projector is not None:
            output = self.embedding_projector(output)

        output_dict: dict[str, Any] = {
            "tokens_and_masks": output,
        }
        if token_norm_stats:
            output_dict["token_norm_stats"] = token_norm_stats

        # 步骤4：投影和池化（为对比学习生成全局表示），仅在非快速模式时计算
        if not fast_pass:
            output_dict["project_aggregated"] = self.project_and_aggregate(output)

        return output_dict

    def apply_fsdp(self, **fsdp_kwargs: Any) -> None:
        """Apply FSDP to the model."""
        super().apply_fsdp(**fsdp_kwargs)
        # Don't Shard the small layers
        # fully_shard(self.patch_embeddings, **fsdp_kwargs)
        # register_fsdp_forward_method(self.patch_embeddings, "forward")
        # fully_shard(self.project_and_aggregate, **fsdp_kwargs)
        # register_fsdp_forward_method(self.project_and_aggregate, "forward")
        fully_shard(self, **fsdp_kwargs)

    def apply_compile(self) -> None:
        """Apply torch.compile to the model."""
        # self.compile(mode="max-autotune", dynamic=False, fullgraph=True)
        logger.info("Compiling blocks")
        # torch.compile(self.blocks, dynamic=False, mode="max-autotune", fullgraph=True)
        # individual block compile is still a lot slower
        for block in self.blocks:
            block.apply_compile()
        # torch.compile(self.patch_embeddings, dynamic=False, mode="max-autotune-no-cudagraphs", fullgraph=True)


class PredictorBase(FlexiVitBase):
    """预测器基类，从编码 token 生成预测。

    使用交叉注意力机制：待解码的 token 作为查询（Q），
    编码器输出的未掩码 token 作为键和值（K, V）。

    关键属性：
        encoder_to_decoder_embed: 编码器到解码器的维度映射
        mask_token: 可学习的掩码 token（用于替换待解码位置）
        to_output_embed: 输出投影层
        input_norm: 输入归一化层
    """

    cross_attn = True

    def __init__(
        self,
        supported_modalities: list[ModalitySpec],
        encoder_embedding_size: int = 128,
        decoder_embedding_size: int = 128,
        depth: int = 2,
        mlp_ratio: float = 2.0,
        num_heads: int = 8,
        max_sequence_length: int = 24,
        drop_path: float = 0.0,
        learnable_channel_embeddings: bool = True,
        random_channel_embeddings: bool = False,
        output_embedding_size: int | None = None,
        use_flash_attn: bool = False,
        qk_norm: bool = False,
        tokenization_config: TokenizationConfig | None = None,
    ):
        """Initialize the predictor.

        Args:
            supported_modalities: modalities this model instantiation supports
            encoder_embedding_size: Size of encoder embeddings
            decoder_embedding_size: Size of decoder embeddings
            depth: Number of transformer layers
            mlp_ratio: Ratio for MLP hidden dimension
            num_heads: Number of attention heads
            max_sequence_length: Maximum sequence length
            drop_path: Drop path rate
            learnable_channel_embeddings: Whether to use learnable channel embeddings
            random_channel_embeddings: Whether to randomly initialize channel embeddings
            output_embedding_size: Size of output embeddings
            use_flash_attn: Whether to use flash attention
            qk_norm: Whether to apply normalization to Q and K in attention
            tokenization_config: Optional config for custom band groupings
        """
        self.tokenization_config = tokenization_config or TokenizationConfig()
        super().__init__(
            embedding_size=decoder_embedding_size,
            depth=depth,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            max_sequence_length=max_sequence_length,
            drop_path=drop_path,
            learnable_channel_embeddings=learnable_channel_embeddings,
            random_channel_embeddings=random_channel_embeddings,
            supported_modalities=supported_modalities,
            use_flash_attn=use_flash_attn,
            qk_norm=qk_norm,
            tokenization_config=self.tokenization_config,
        )
        self.learnable_channel_embeddings = learnable_channel_embeddings
        self.random_channel_embeddings = random_channel_embeddings
        self.encoder_embedding_size = encoder_embedding_size
        self.encoder_to_decoder_embed = nn.Linear(
            encoder_embedding_size, decoder_embedding_size, bias=True
        )
        if output_embedding_size is None:
            output_embedding_size = encoder_embedding_size
        self.output_embedding_size = output_embedding_size
        self.to_output_embed = nn.Linear(
            decoder_embedding_size, output_embedding_size, bias=True
        )
        # THIS is the learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(decoder_embedding_size))

        self.input_norm = nn.LayerNorm(encoder_embedding_size)
        self.norm = nn.LayerNorm(decoder_embedding_size)

        self.apply(self._init_weights)

    def add_masks(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """用可学习的 mask_token 替换待解码位置的 token。

        对于掩码值为 MaskValue.DECODER 的位置，用 self.mask_token 替换原始 token，
        使解码器通过交叉注意力从可见 token 预测这些位置的值。

        Args:
            x: 模态到张量的映射字典

        Returns:
            替换掩码位置后的 token 字典
        """
        output_dict = {}
        available_modalities = return_modalities_from_dict(x)
        modalities_to_process = get_modalities_to_process(
            available_modalities, self.supported_modality_names
        )
        for modality in modalities_to_process:
            x_modality = x[modality]
            mask_name = MaskedOlmoEarthSample.get_masked_modality_name(modality)
            mask_modality = x[mask_name]
            # A boolean mask: True where tokens must be replaced by the mask token
            kept_mask = mask_modality == MaskValue.DECODER.value

            # Build the einops pattern and dimension dict
            spatial_dims = x_modality.shape[
                :-1
            ]  # all dimensions except the last (embedding)
            pattern_input, dim_dict = self._construct_einops_pattern(spatial_dims)

            mask_token_broadcasted = repeat(self.mask_token, pattern_input, **dim_dict)

            # Where kept_mask is True, use the broadcasted mask token
            x_modality = torch.where(
                kept_mask.unsqueeze(-1).bool(), mask_token_broadcasted, x_modality
            )

            output_dict[modality] = x_modality

        return output_dict

    @staticmethod
    def split_x_y(tokens: Tensor, mask: Tensor) -> tuple[Tensor, ...]:
        """根据掩码值将 token 分为解码组和上下文组。

        核心逻辑：
        1. 将 MISSING 掩码重标记为 TARGET_ENCODER_ONLY（使未使用 token 排在中间）
        2. 按掩码值降序排序 token
        3. 提取 DECODER 掩码的 token（待解码，作为交叉注意力的查询）
        4. 提取 ONLINE_ENCODER 掩码的 token（可见，作为交叉注意力的键/值）
        5. 返回两组 token 及其掩码和原始索引

        Args:
            tokens: 待分割的 token 张量 [B, T, D]
            mask: 掩码张量 [B, T]

        Returns:
            九元组：tokens_to_decode, unmasked_tokens, tokens_to_decode_mask,
            unmasked_tokens_mask, indices, seqlens_tokens_to_decode,
            seqlens_unmasked_tokens, max_length_of_decoded_tokens,
            max_length_of_unmasked_tokens
        """
        # Set Missing Masks to Target Encoder ONLY so that we can have all unused tokens in the middle
        org_mask_dtype = mask.dtype
        missing_mask = mask == MaskValue.MISSING.value
        mask[missing_mask] = MaskValue.TARGET_ENCODER_ONLY.value

        # Sort tokens by mask value (descending order)
        sorted_mask, indices = torch.sort(
            mask.int(), dim=1, descending=True, stable=True
        )
        tokens = tokens.gather(1, indices[:, :, None].expand_as(tokens))

        # Create binary masks for Encoder and Decoder
        binarized_decoder_mask = sorted_mask == MaskValue.DECODER.value
        binarized_online_encoder_mask = sorted_mask == MaskValue.ONLINE_ENCODER.value

        seqlens_unmasked_tokens = binarized_online_encoder_mask.sum(dim=-1)
        max_length_of_unmasked_tokens = seqlens_unmasked_tokens.max()
        seqlens_tokens_to_decode = binarized_decoder_mask.sum(dim=-1)
        max_length_of_decoded_tokens = seqlens_tokens_to_decode.max()

        # the y mask is going to be used to determine which of the y values take. True values
        # take part in the attention (we don't take the inverse here, unlike in the decoder)
        tokens_to_decode = tokens[:, :max_length_of_decoded_tokens]
        tokens_to_decode_mask = binarized_decoder_mask[
            :, :max_length_of_decoded_tokens
        ].to(org_mask_dtype)

        unmasked_tokens = tokens[:, -max_length_of_unmasked_tokens:]
        # the x_mask is just going to be used in the reconstruction, to know which
        # x tokens to add back into the token list. TODO is this even necessary? it could
        # get padded with noise tokens since we don't care about reconstruction at all
        # for a whole bunch of tokens
        unmasked_tokens_mask = binarized_online_encoder_mask[
            :, -max_length_of_unmasked_tokens:
        ].to(org_mask_dtype)

        return (
            tokens_to_decode,
            unmasked_tokens,
            tokens_to_decode_mask,
            unmasked_tokens_mask,
            indices,
            seqlens_tokens_to_decode,
            seqlens_unmasked_tokens,
            max_length_of_decoded_tokens,
            max_length_of_unmasked_tokens,
        )

    @staticmethod
    def combine_x_y(
        tokens_to_decode: Tensor,
        unmasked_tokens: Tensor,
        tokens_to_decode_mask: Tensor,
        unmasked_tokens_mask: Tensor,
        indices: Tensor,
    ) -> Tensor:
        """将分离的解码 token 和未掩码 token 重新合并为原始顺序。

        先将解码 token 和未掩码 token 放回全零张量的对应位置，
        然后通过 scatter 操作恢复原始排序。

        Args:
            tokens_to_decode: 解码 token [B, X_len, D]
            unmasked_tokens: 未掩码 token [B, Y_len, D]
            tokens_to_decode_mask: 解码 token 掩码 [B, X_len]
            unmasked_tokens_mask: 未掩码 token 掩码 [B, Y_len]
            indices: 原始排序索引 [B, T]

        Returns:
            合并后的 token 张量 [B, T, D]
        """
        # Get dimensions
        B, T = indices.shape[0], indices.shape[1]
        D = tokens_to_decode.shape[-1]
        tokens = torch.zeros(
            (B, T, D), dtype=tokens_to_decode.dtype, device=tokens_to_decode.device
        )
        tokens[:, -unmasked_tokens.shape[1] :] = (
            unmasked_tokens * unmasked_tokens_mask.unsqueeze(-1)
        )
        tokens[:, : tokens_to_decode.shape[1]] += (
            tokens_to_decode * tokens_to_decode_mask.unsqueeze(-1)
        )
        tokens = tokens.scatter(1, indices[:, :, None].expand_as(tokens), tokens)
        return tokens

    def is_any_data_to_be_decoded(self, modality_mask: Tensor) -> bool:
        """检查是否有数据需要解码（掩码值为 DECODER）。

        Args:
            modality_mask: 模态的掩码张量

        Returns:
            若存在需要解码的数据则返回 True
        """
        return (MaskValue.DECODER.value == modality_mask).any()

    def apply_fsdp(self, **fsdp_kwargs: Any) -> None:
        """Apply FSDP to the model."""
        super().apply_fsdp(**fsdp_kwargs)
        fully_shard(self, **fsdp_kwargs)


class Predictor(PredictorBase):
    """FlexiViT 预测器，从编码 token 生成掩码 token 的预测。

    使用交叉注意力：待解码 token（Q） attends to 编码器可见 token（K, V）。
    支持 Flash Attention 和标准 SDPA 两种后端。

    关键流程：
    1. 输入归一化 + 编码器到解码器维度映射
    2. 用 mask_token 替换待解码位置
    3. 添加位置编码
    4. 分离解码 token 和上下文 token
    5. 交叉注意力层
    6. 合并 token 并投影到输出维度
    """

    cross_attn = True

    def apply_attn(
        self,
        x: dict[str, Tensor],
        timestamps: Tensor,
        patch_size: int,
        input_res: int,
    ) -> dict[str, Tensor]:
        """预测器的注意力处理：交叉注意力解码。

        流程：
        1. 添加位置编码
        2. 折叠各模态 token
        3. 分离解码 token（Q）和上下文 token（K, V）
        4. 可选：打包 token（Flash Attention）
        5. 逐层应用交叉注意力块
        6. 可选：解包 token
        7. 合并解码和上下文 token
        8. 展开回各模态形状

        Args:
            x: 模态到张量的映射字典
            timestamps: 时间戳
            patch_size: Patch 大小
            input_res: 输入分辨率

        Returns:
            各模态的解码 token 字典
        """
        tokens_only_dict, original_masks_dict, modalities_to_dims_dict = (
            self.split_tokens_masks_and_dims(x)
        )
        tokens_dict = self.composite_encodings(
            tokens_only_dict, timestamps, patch_size, input_res
        )
        tokens_dict.update(original_masks_dict)
        all_tokens, mask = self.collapse_and_combine_hwtc(tokens_dict)
        # X contains the tokens to decode, Y contains the tokens to attend to for context
        (
            tokens_to_decode,
            unmasked_tokens,
            tokens_to_decode_mask,
            unmasked_tokens_mask,
            indices,
            seqlens_tokens_to_decode,
            seqlens_unmasked_tokens,
            max_length_of_tokens_to_decode,
            max_length_of_unmasked_tokens,
        ) = self.split_x_y(all_tokens, mask)
        # Pack x tokens
        if self.use_flash_attn:
            og_shape_tokens_to_decode = tokens_to_decode.shape
            tokens_to_decode = self.pack_tokens(
                tokens_to_decode, tokens_to_decode_mask.bool()
            )
            og_shape_unmasked_tokens = unmasked_tokens.shape
            unmasked_tokens = self.pack_tokens(
                unmasked_tokens, unmasked_tokens_mask.bool()
            )
            cu_seqlens_tokens_to_decode = get_cumulative_sequence_lengths(
                seqlens_tokens_to_decode
            )
            cu_seqlens_unmasked_tokens = get_cumulative_sequence_lengths(
                seqlens_unmasked_tokens
            )
        else:
            cu_seqlens_tokens_to_decode = None
            cu_seqlens_unmasked_tokens = None

        for blk in self.blocks:
            # note that we are not taking the inverse of the mask, since split_x_y gives us
            # true values for values we want to take part in attention
            tokens_to_decode = blk(
                x=tokens_to_decode,
                y=unmasked_tokens,
                attn_mask=(
                    unmasked_tokens_mask.bool() if not self.use_flash_attn else None
                ),  # only for flash attn though this should not be left in
                cu_seqlens_q=cu_seqlens_tokens_to_decode,
                cu_seqlens_k=cu_seqlens_unmasked_tokens,
                max_seqlen_q=max_length_of_tokens_to_decode,
                max_seqlen_k=max_length_of_unmasked_tokens,
            )

        if self.use_flash_attn:
            tokens_to_decode = self.unpack_tokens(
                tokens_to_decode,
                tokens_to_decode_mask.bool(),
                og_shape_tokens_to_decode,
            )
            unmasked_tokens = self.unpack_tokens(
                unmasked_tokens, unmasked_tokens_mask.bool(), og_shape_unmasked_tokens
            )

        x = self.combine_x_y(
            tokens_to_decode=tokens_to_decode,
            unmasked_tokens=unmasked_tokens,
            tokens_to_decode_mask=tokens_to_decode_mask,
            unmasked_tokens_mask=unmasked_tokens_mask,
            indices=indices,
        )
        tokens_per_modality_dict = self.split_and_expand_per_modality(
            x, modalities_to_dims_dict
        )
        tokens_per_modality_dict.update(original_masks_dict)
        return tokens_per_modality_dict

    def forward(
        self,
        x: TokensAndMasks,
        timestamps: Tensor,
        patch_size: int,
        input_res: int = BASE_GSD,
    ) -> TokensAndMasks:
        """预测器前向传播：从编码 token 生成掩码位置的预测。

        流程：
        1. 输入归一化 + 编码器到解码器维度映射
        2. 用 mask_token 替换待解码位置
        3. 交叉注意力解码
        4. 逐波段组投影到输出维度

        Args:
            x: 编码器输出的 TokensAndMasks
            timestamps: 时间戳
            patch_size: Patch 大小
            input_res: 输入分辨率

        Returns:
            预测的 TokensAndMasks
        """
        decoder_emedded_dict = x.as_dict()
        # 步骤1：对每个模态应用输入归一化和编码器到解码器的维度映射
        available_modalities = x.modalities
        modalities_to_process = get_modalities_to_process(
            available_modalities, self.supported_modality_names
        )
        for modality in modalities_to_process:
            x_modality = getattr(x, modality)
            # 注意：归一化和投影在 token 维度上操作，不会与缺失 token 混合
            x_modality = self.input_norm(x_modality)  # 输入归一化
            x_modality = self.encoder_to_decoder_embed(x_modality)  # 维度映射
            masked_modality_name = x.get_masked_modality_name(modality)
            decoder_emedded_dict[modality] = x_modality
            decoder_emedded_dict[masked_modality_name] = getattr(
                x, masked_modality_name
            )

        # 步骤2：用可学习的 mask_token 替换待解码位置
        tokens_only_dict = self.add_masks(decoder_emedded_dict)
        decoder_emedded_dict.update(tokens_only_dict)
        # 步骤3：交叉注意力解码
        tokens_and_masks = self.apply_attn(
            decoder_emedded_dict, timestamps, patch_size, input_res
        )
        # 步骤4：逐波段组投影到输出维度
        output_dict = {}
        available_modalities = return_modalities_from_dict(tokens_and_masks)
        modalities_to_process = get_modalities_to_process(
            available_modalities, self.supported_modality_names
        )
        for modality in modalities_to_process:
            masked_modality_name = MaskedOlmoEarthSample.get_masked_modality_name(
                modality
            )
            modality_mask = tokens_and_masks[masked_modality_name]
            # patchify masked data
            per_modality_output_tokens = []
            modality_data = tokens_and_masks[modality]

            num_band_sets = self.tokenization_config.get_num_bandsets(modality)
            for idx in range(num_band_sets):
                per_channel_modality_data = modality_data[..., idx, :]
                # 归一化 + 输出投影
                output_data = self.to_output_embed(self.norm(per_channel_modality_data))
                per_modality_output_tokens.append(output_data)
            output_dict[modality] = torch.stack(per_modality_output_tokens, dim=-2)
            output_dict[masked_modality_name] = modality_mask
        return TokensAndMasks(**output_dict)


@dataclass
class EncoderConfig(Config):
    """FlexiViT 编码器的配置类。

    包含编码器的所有超参数，并提供验证和构建方法。

    关键参数：
        supported_modality_names: 支持的模态名称列表
        embedding_size: 嵌入维度
        max_patch_size: 最大 patch 大小（基准）
        depth: Transformer 层数
        num_register_tokens: Register Token 数量
        use_flash_attn: 是否使用 Flash Attention
        frozen_patch_embeddings: 是否冻结 Patch 嵌入层
        band_dropout_rate: Band Dropout 概率
        output_embedding_size: 可选的输出嵌入维度（改变最终投影维度）
    """

    supported_modality_names: list[str]

    embedding_size: int = 16
    # This is the base patch size for the patch embedder
    max_patch_size: int = 8
    min_patch_size: int = 1
    num_heads: int = 2
    mlp_ratio: float = 1.0
    depth: int = 2
    drop_path: float = 0.1
    max_sequence_length: int = 12
    num_register_tokens: int = 0
    learnable_channel_embeddings: bool = True
    random_channel_embeddings: bool = False
    num_projection_layers: int = 1
    aggregate_then_project: bool = True
    use_flash_attn: bool = False
    frozen_patch_embeddings: bool = False
    qk_norm: bool = False
    log_token_norm_stats: bool = False
    output_embedding_size: int | None = None
    tokenization_config: TokenizationConfig | None = None
    use_linear_patch_embed: bool = True
    band_dropout_rate: float = 0.0
    random_band_dropout: bool = False
    band_dropout_modalities: list[str] | None = None

    def __post_init__(self) -> None:
        """Coerce raw dicts to TokenizationConfig for old checkpoint compatibility."""
        if isinstance(self.tokenization_config, dict):
            self.tokenization_config = TokenizationConfig(**self.tokenization_config)

    def validate(self) -> None:
        """Validate the configuration."""
        if len(self.supported_modalities) == 0:
            raise ValueError("At least one modality must be added!")
        else:
            for modality in self.supported_modalities:
                if modality not in Modality.values():
                    raise ValueError(f"Modality {modality} is not supported")
        if self.band_dropout_modalities is not None:
            unknown = set(self.band_dropout_modalities) - set(
                self.supported_modality_names
            )
            if unknown:
                raise ValueError(
                    f"band_dropout_modalities contains modalities not in "
                    f"supported_modality_names: {unknown}"
                )
        if self.tokenization_config is not None:
            self.tokenization_config.validate()

    @property
    def supported_modalities(self) -> list[ModalitySpec]:
        """Get the supported modalities."""
        return get_modality_specs_from_names(self.supported_modality_names)

    def build(self) -> "Encoder":
        """Build the encoder."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        # supported_modality_names is replaced by supported_modalities
        kwargs.pop("supported_modality_names")
        kwargs["supported_modalities"] = self.supported_modalities
        logger.info(f"Encoder kwargs: {kwargs}")
        return Encoder(**kwargs)


@dataclass
class PredictorConfig(Config):
    """FlexiViT 预测器的配置类。

    包含预测器的所有超参数，并提供验证和构建方法。

    关键参数：
        supported_modality_names: 支持的模态名称列表
        encoder_embedding_size: 编码器嵌入维度（用于维度映射）
        decoder_embedding_size: 解码器嵌入维度
        depth: Transformer 层数
        output_embedding_size: 可选的输出嵌入维度
        use_flash_attn: 是否使用 Flash Attention
    """

    supported_modality_names: list[str]
    encoder_embedding_size: int = 16
    decoder_embedding_size: int = 16
    depth: int = 2
    mlp_ratio: float = 1.0
    num_heads: int = 2
    max_sequence_length: int = 12
    drop_path: float = 0.0
    learnable_channel_embeddings: bool = True
    random_channel_embeddings: bool = False
    output_embedding_size: int | None = None
    use_flash_attn: bool = False
    qk_norm: bool = False
    tokenization_config: TokenizationConfig | None = None

    def __post_init__(self) -> None:
        """Coerce raw dicts to TokenizationConfig for old checkpoint compatibility."""
        if isinstance(self.tokenization_config, dict):
            self.tokenization_config = TokenizationConfig(**self.tokenization_config)

    def validate(self) -> None:
        """Validate the configuration."""
        if len(self.supported_modalities) == 0:
            raise ValueError("At least one modality must be added!")
        else:
            for modality in self.supported_modalities:
                if modality not in Modality.values():
                    raise ValueError(f"Modality {modality} is not supported")
        if self.tokenization_config is not None:
            self.tokenization_config.validate()

    @property
    def supported_modalities(self) -> list[ModalitySpec]:
        """Get the supported modalities."""
        return get_modality_specs_from_names(self.supported_modality_names)

    def build(self) -> "PredictorBase":
        """Build the predictor."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        # supported_modality_names is replaced by supported_modalities
        kwargs.pop("supported_modality_names")
        kwargs["supported_modalities"] = self.supported_modalities
        logger.info(f"Predictor kwargs: {kwargs}")
        return Predictor(**kwargs)
