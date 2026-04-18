"""统一评估包装器模块。

本模块定义了所有模型评估的统一接口 EvalWrapper 及其各模型特定的子类。
核心设计思想是通过统一的包装器接口，使不同基线模型能够在相同的评估流程中运行。

主要组件：
- EvalWrapper: 评估包装器基类，定义统一接口
- OlmoEarthEvalWrapper: OlmoEarth 预训练模型的包装器
- 各基线模型包装器: AnySat, Clay, Croma, DINOv3, Galileo, Panopticon,
  Presto, PrithviV2, Satlas, Terramind, Tessera
- get_eval_wrapper: 工厂函数，根据模型类型自动选择对应的包装器

使用场景：
  在评估流程中，通过 get_eval_wrapper(model) 自动获取对应的包装器，
  然后调用包装器的 __call__ 方法获取模型嵌入和标签。
"""

from logging import getLogger
from typing import Any

import torch
from einops import rearrange, reduce
from torch import nn

from olmoearth_pretrain._compat import deprecated_class_alias as _deprecated_class_alias
from olmoearth_pretrain.evals.datasets.configs import TaskType
from olmoearth_pretrain.evals.models import (
    AnySat,
    Clay,
    Croma,
    DINOv3,
    GalileoWrapper,
    Panopticon,
    PrestoWrapper,
    PrithviV2,
    Satlas,
    Terramind,
    Tessera,
)
from olmoearth_pretrain.nn.flexi_vit import (
    FlexiVitBase,
    TokensAndMasks,
)
from olmoearth_pretrain.nn.pooled_modality_predictor import EncodeEarlyAttnPool
from olmoearth_pretrain.nn.pooling import PoolingType, pool_unmasked_tokens
from olmoearth_pretrain.nn.st_model import STBase
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = getLogger(__name__)


class EvalWrapper:
    """评估包装器基类，定义所有模型评估的统一接口。

    这是运行评估的通用接口，所有具体的模型包装器都继承自此类。

    关键属性：
        model: 被评估的模型 (nn.Module)
        task_type: 任务类型 (分类/分割)
        patch_size: 模型使用的 patch 大小
        pooling_type: 池化类型 (mean/max等)
        concat_features: 是否跨模态拼接特征
        spatial_pool: 是否进行空间池化（分割任务时为 True）
        use_pooled_tokens: 是否使用池化后的 token（仅 EncodeEarlyAttnPool 支持）

    使用场景：
        作为基类，不直接实例化，而是通过 get_eval_wrapper() 工厂函数
        获取对应模型的具体子类实例。
    """

    def __init__(
        self,
        model: nn.Module,
        task_type: TaskType,
        patch_size: int,
        pooling_type: PoolingType,
        concat_features: bool = False,
        use_pooled_tokens: bool = False,
    ):
        """初始化评估包装器。

        Args:
            model: 被评估的模型实例
            task_type: 任务类型 (分类/分割)
            patch_size: 模型使用的 patch 大小
            pooling_type: 池化类型 (mean/max等)
            concat_features: 是否跨模态拼接特征，默认 False
            use_pooled_tokens: 是否使用池化后的 token，默认 False
        """
        super().__init__()
        self.model = model
        self.task_type = task_type
        self.patch_size = patch_size
        self.pooling_type = pooling_type
        self.concat_features = concat_features
        # 分割任务时需要进行空间池化以保留空间维度
        self.spatial_pool = task_type == TaskType.SEGMENTATION
        self.use_pooled_tokens = use_pooled_tokens
        if self.use_pooled_tokens:
            # 只有 EncodeEarlyAttnPool 类型模型支持池化 token
            assert isinstance(self.model, EncodeEarlyAttnPool), (
                "Pooled tokens are only supported for EncodeEarlyAttnPool"
            )

    @property
    def device(self) -> torch.device:
        """获取模型所在设备。

        依次尝试以下方式获取设备：
        1. 模型的 device 属性（如果是 torch.device 类型）
        2. 字符串类型转换为 torch.device
        3. 对于 FSDP 包装的模型，从模型参数推断设备

        Returns:
            torch.device: 模型所在的设备
        """
        dev = getattr(self.model, "device", None)

        if isinstance(dev, torch.device):
            return dev

        if isinstance(dev, str):
            return torch.device(dev)

        # 对于 FSDP 包装的模型，从模型参数推断设备
        return next(self.model.parameters()).device

    def __getattr__(self, name: str) -> Any:
        """属性访问代理：当包装器上找不到属性时，委托给底层模型。

        这使得可以直接通过包装器访问模型的属性和方法，
        例如 wrapper.config 会自动转发到 model.config。
        """
        return getattr(self.model, name)

    def __call__(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播，根据初始化配置生成嵌入。

        Args:
            masked_olmoearth_sample: 掩码后的 OlmoEarth 样本
            labels: 标签张量
            is_train: 是否为训练模式

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (嵌入张量, 标签张量)

        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("Subclasses must implement this method")


class OlmoEarthEvalWrapper(EvalWrapper):
    """OlmoEarth 预训练模型的评估包装器。

    处理两种嵌入提取模式：
    1. 标准 token 模式：从模型获取 tokens_and_masks，然后进行池化
    2. 池化 token 模式：从模型获取 pooled_tokens_and_masks，适用于 EncodeEarlyAttnPool

    对于分割任务，空间池化会保留空间维度（仅沿时间维度池化）。
    """

    def __call__(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播，根据初始化配置生成嵌入。

        Args:
            masked_olmoearth_sample: 掩码后的 OlmoEarth 样本
            labels: 标签张量
            is_train: 是否为训练模式

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (嵌入张量, 标签张量)
        """
        if not self.use_pooled_tokens:
            # 标准 token 模式：获取所有 token 和掩码
            batch_embeddings: TokensAndMasks = self.model(
                masked_olmoearth_sample, patch_size=self.patch_size, fast_pass=True
            )["tokens_and_masks"]  # (bsz, dim)
            # 跨模态拼接特征，在空间维度上取平均，沿时间维度池化
            batch_embeddings = pool_unmasked_tokens(
                batch_embeddings,
                self.pooling_type,
                spatial_pooling=self.spatial_pool,
                concat_features=self.concat_features,
            )
        else:
            # 池化 token 模式：获取预池化的 token
            pooled_tokens_dict = self.model(
                masked_olmoearth_sample, patch_size=self.patch_size, fast_pass=True
            )["pooled_tokens_and_masks"]
            pooled_tokens = pooled_tokens_dict["modality_pooled_tokens"]
            # spatial_pool 为 True 表示需要保留空间维度，因此仅沿时间维度池化
            logger.info(f"pooled tokens shape in eval wrapper: {pooled_tokens.shape}")

            if self.spatial_pool:
                # 分割任务：保留空间维度 (B H W D)，仅沿时间维度池化
                # B H W T C -> B H W D
                if pooled_tokens.shape[1] == 1 and pooled_tokens.ndim == 3:
                    # 如果只有一个空间维度，增加维度以获得 W H C T 格式
                    pooled_tokens = pooled_tokens.unsqueeze(1)
                pooled_tokens = reduce(
                    pooled_tokens, "b h w ... d -> b h w d", self.pooling_type
                )
            else:
                # 分类任务：沿所有中间维度取平均，仅保留批次和特征维度
                # B ... D -> B D
                pooled_tokens = reduce(
                    pooled_tokens, "b ... d -> b d", self.pooling_type
                )
            batch_embeddings = pooled_tokens
        return batch_embeddings, labels


# 向后兼容别名：HeliosEvalWrapper 已弃用，重定向到 OlmoEarthEvalWrapper
HeliosEvalWrapper = _deprecated_class_alias(
    OlmoEarthEvalWrapper, "helios.evals.eval_wrapper.HeliosEvalWrapper"
)


class TerramindEvalWrapper(EvalWrapper):
    """Terramind 模型评估包装器。

    直接调用模型的前向传播，传递池化类型和空间池化参数。
    """

    def __call__(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the embedding specified by initialization."""
        batch_embeddings = self.model(
            masked_olmoearth_sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        return batch_embeddings, labels


class PanopticonEvalWrapper(EvalWrapper):
    """Panopticon 模型评估包装器。

    对于分割任务（需要空间池化），调用 forward_features 获取中间特征；
    对于分类任务，调用模型的标准前向传播。
    注意：空间池化模式下使用 forward_features 是因为模型内部存在已知 bug。
    """

    def __call__(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播，根据初始化配置生成嵌入。"""
        if self.spatial_pool:
            # 分割任务：使用 forward_features 获取中间特征
            # 注意：由于模型内部的已知 bug，不能直接调用 __call__
            batch_embeddings = self.model.forward_features(
                masked_olmoearth_sample, pooling=self.pooling_type
            )
        else:
            # 分类任务：使用标准前向传播
            batch_embeddings = self.model(
                masked_olmoearth_sample, pooling=self.pooling_type
            )
        return batch_embeddings, labels


class GalileoEvalWrapper(EvalWrapper):
    """Galileo 模型评估包装器。直接调用模型前向传播。"""

    def __call__(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the embedding specified by initialization."""
        embeddings = self.model(
            masked_olmoearth_sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        return embeddings, labels


class AnySatEvalWrapper(EvalWrapper):
    """AnySat 模型评估包装器。

    AnySat 输出的是逐像素嵌入（而非逐 patch 嵌入），因此在训练分割任务时
    需要对像素特征进行子采样以控制内存消耗。
    参考: https://arxiv.org/abs/2502.09356
    训练时采样 6.25% (1/16) 的像素特征，测试时使用全部像素特征。
    """

    def __call__(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播，根据初始化配置生成嵌入。

        对于训练阶段的分割任务，对像素特征进行子采样以节省内存。
        """
        embeddings = self.model(
            masked_olmoearth_sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        if is_train and (self.task_type == TaskType.SEGMENTATION):
            # AnySat 特殊处理：训练时对逐像素嵌入进行子采样
            # 仅保留 6.25% (1/16) 的像素特征以控制训练成本
            subsample_by = 1 / 16
            # 将空间维度展平: B H W D -> B (H*W) D
            embeddings = rearrange(embeddings, "b h w d -> b (h w) d")
            labels = rearrange(labels, "b h w -> b (h w)")

            assert embeddings.shape[1] == labels.shape[1]
            num_tokens = embeddings.shape[1]
            num_tokens_to_keep = int(num_tokens * subsample_by)  # 保留的 token 数量
            # 随机采样指定数量的 token 索引
            sampled_indices = torch.randperm(num_tokens)[:num_tokens_to_keep]
            embeddings = embeddings[:, sampled_indices]
            labels = labels[:, sampled_indices]

            # 将子采样后的 token 重塑为正方形空间维度
            new_hw = int(num_tokens_to_keep**0.5)
            embeddings = rearrange(
                embeddings, "b (h w) d -> b h w d", h=new_hw, w=new_hw
            )
            labels = rearrange(labels, "b (h w) -> b h w", h=new_hw, w=new_hw)
        return embeddings, labels


class PrithviV2EvalWrapper(EvalWrapper):
    """PrithviV2 模型评估包装器。直接调用模型前向传播。"""

    def __call__(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the embedding specified by initialization."""
        embeddings = self.model(
            masked_olmoearth_sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        return embeddings, labels


class ClayEvalWrapper(EvalWrapper):
    """Clay 模型评估包装器。直接调用模型前向传播。"""

    def __call__(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the embedding specified by initialization."""
        batch_embeddings = self.model(
            masked_olmoearth_sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        return batch_embeddings, labels


class CromaEvalWrapper(EvalWrapper):
    """Croma 模型评估包装器。直接调用模型前向传播。"""

    def __call__(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the embedding specified by initialization."""
        batch_embeddings = self.model(
            masked_olmoearth_sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        return batch_embeddings, labels


class PrestoEvalWrapper(EvalWrapper):
    """Presto 模型评估包装器。直接调用模型前向传播。"""

    def __call__(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the embedding specified by initialization."""
        batch_embeddings = self.model(
            masked_olmoearth_sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        return batch_embeddings, labels


class DINOv3EvalWrapper(EvalWrapper):
    """DINOv3 模型评估包装器。

    对于分割任务（需要空间池化），调用 forward_features 获取中间特征；
    对于分类任务，调用模型的标准前向传播。
    注意：空间池化模式下使用 forward_features 是因为模型内部存在已知 bug。
    """

    def __call__(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播，根据初始化配置生成嵌入。"""
        if self.spatial_pool:
            # 分割任务：使用 forward_features 获取中间特征
            batch_embeddings = self.model.forward_features(
                masked_olmoearth_sample,
                pooling=self.pooling_type,
            )
        else:
            # 分类任务：使用标准前向传播
            batch_embeddings = self.model(
                masked_olmoearth_sample,
                pooling=self.pooling_type,
            )
        return batch_embeddings, labels


class SatlasEvalWrapper(EvalWrapper):
    """Satlas 模型评估包装器。直接调用模型前向传播。"""

    def __call__(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the embedding specified by initialization."""
        batch_embeddings = self.model(
            masked_olmoearth_sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        return batch_embeddings, labels


class TesseraEvalWrapper(EvalWrapper):
    """Tessera 模型评估包装器。直接调用模型前向传播。"""

    def __call__(
        self,
        masked_olmoearth_sample: MaskedOlmoEarthSample,
        labels: torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model produces the embedding specified by initialization."""
        batch_embeddings = self.model(
            masked_olmoearth_sample,
            pooling=self.pooling_type,
            spatial_pool=self.spatial_pool,
        )
        return batch_embeddings, labels


def get_eval_wrapper(model: nn.Module, **kwargs: Any) -> EvalWrapper:
    """工厂函数：根据模型类型自动选择对应的评估包装器。

    通过 isinstance 检查模型类型，返回对应的具体 EvalWrapper 子类实例。

    Args:
        model: 被评估的模型实例
        **kwargs: 传递给 EvalWrapper 构造函数的额外参数
            包括 task_type, patch_size, pooling_type 等

    Returns:
        EvalWrapper: 对应模型类型的评估包装器实例

    Raises:
        NotImplementedError: 如果模型类型没有对应的包装器

    支持的模型类型：
        - FlexiVitBase / STBase -> OlmoEarthEvalWrapper
        - Panopticon -> PanopticonEvalWrapper
        - DINOv3 -> DINOv3EvalWrapper
        - Croma -> CromaEvalWrapper
        - Clay -> ClayEvalWrapper
        - GalileoWrapper -> GalileoEvalWrapper
        - Terramind -> TerramindEvalWrapper
        - PrestoWrapper -> PrestoEvalWrapper
        - AnySat -> AnySatEvalWrapper
        - Satlas -> SatlasEvalWrapper
        - Tessera -> TesseraEvalWrapper
        - PrithviV2 -> PrithviV2EvalWrapper
    """
    if isinstance(model, FlexiVitBase) or isinstance(model, STBase):
        logger.info("Using OlmoEarthEvalWrapper")
        return OlmoEarthEvalWrapper(model=model, **kwargs)
    elif isinstance(model, Panopticon):
        logger.info("Using PanopticonEvalWrapper")
        return PanopticonEvalWrapper(model=model, **kwargs)
    elif isinstance(model, DINOv3):
        logger.info("Using DINOv3EvalWrapper")
        return DINOv3EvalWrapper(model=model, **kwargs)
    elif isinstance(model, Croma):
        logger.info("Using CromaEvalWrapper")
        return CromaEvalWrapper(model=model, **kwargs)
    elif isinstance(model, Clay):
        logger.info("Using ClayEvalWrapper")
        return ClayEvalWrapper(model=model, **kwargs)
    elif isinstance(model, GalileoWrapper):
        logger.info("Using GalileoEvalWrapper")
        return GalileoEvalWrapper(model=model, **kwargs)
    elif isinstance(model, Terramind):
        logger.info("Using TerramindEvalWrapper")
        return TerramindEvalWrapper(model=model, **kwargs)
    elif isinstance(model, PrestoWrapper):
        logger.info("Using PrestoEvalWrapper")
        return PrestoEvalWrapper(model=model, **kwargs)
    elif isinstance(model, AnySat):
        logger.info("Using AnySatEvalWrapper")
        return AnySatEvalWrapper(model=model, **kwargs)
    elif isinstance(model, Satlas):
        logger.info("Using SatlasEvalWrapper")
        return SatlasEvalWrapper(model=model, **kwargs)
    elif isinstance(model, Tessera):
        logger.info("Using TesseraEvalWrapper")
        return TesseraEvalWrapper(model=model, **kwargs)
    elif isinstance(model, PrithviV2):
        logger.info("Using PrithviEvalWrapper")
        return PrithviV2EvalWrapper(model=model, **kwargs)
    else:
        raise NotImplementedError(f"No EvalWrapper for model type {type(model)}")
