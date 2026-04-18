"""核心数据类型模块 - 定义 OlmoEarth Pretrain 的数据结构。

本模块定义了 OlmoEarth 预训练框架中使用的核心数据类型，包括：

- MaskValue: 掩码值枚举，定义了四种掩码状态（在线编码器、目标编码器、解码器、缺失）
- OlmoEarthSample: 未掩码的数据样本，包含多种地球观测模态数据
- MaskedOlmoEarthSample: 带掩码的数据样本，用于自监督学习中的掩码策略
- TokensAndMasks: 编码器输出的嵌入令牌和掩码，用于计算损失

这些数据结构均基于 NamedTuple 实现，具有不可变性和内存效率，
同时提供了丰富的属性和方法用于数据操作和形状查询。

使用场景：
    - 数据加载和预处理：使用 OlmoEarthSample 封装多模态数据
    - 自监督预训练：使用 MaskedOlmoEarthSample 实现掩码策略
    - 模型前向传播：使用 TokensAndMasks 传递编码器输出
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from enum import Enum
from typing import Any, NamedTuple, cast

import numpy as np
import torch
from einops import rearrange
from torch import Tensor
from torch.distributed import DeviceMesh
from torch.distributed.tensor import distribute_tensor

from olmoearth_pretrain.data.constants import MISSING_VALUE, TIMESTAMPS, Modality
from olmoearth_pretrain.types import ArrayTensor

logger = logging.getLogger(__name__)


class MaskValue(Enum):
    """掩码值枚举，定义了掩码可以取的四种可能值。

    在自监督预训练（如 BYOL/MAE 风格）中，每个令牌（token）的掩码值
    决定了它在训练流程中的可见性：

    - ONLINE_ENCODER: 令牌对在线编码器可见（主要训练路径）
    - TARGET_ENCODER_ONLY: 令牌仅对目标编码器可见（用于生成训练目标）
    - DECODER: 令牌仅对解码器可见（用于重建任务）
    - MISSING: 令牌缺失（该模态数据不可用）

    使用场景：
        - 在掩码策略中标记每个令牌的可见性
        - 在损失计算时区分不同类型的令牌
    """

    ONLINE_ENCODER = 0      # 在线编码器可见
    TARGET_ENCODER_ONLY = 1  # 仅目标编码器可见
    DECODER = 2              # 解码器可见
    MISSING = 3              # 数据缺失


# timestamps 从不被视为"模态" - 它是关于样本采集时间的元数据
TIMESTAMPS_FIELD = "timestamps"


# =============================================================================
# 共享的独立辅助函数（由 NamedTuple 方法调用以避免代码重复）
# =============================================================================


def _as_dict(obj: NamedTuple, include_nones: bool = False) -> dict[str, Any]:
    """将 NamedTuple 转换为字典，可选择是否包含 None 值。

    Args:
        obj: 要转换的 NamedTuple 实例。
        include_nones: 是否包含值为 None 的字段。默认为 False。

    Returns:
        字典表示，键为字段名，值为字段值。
    """
    result = {}
    for name in obj._fields:
        val = getattr(obj, name)
        # 仅在 include_nones 为 True 或值不为 None 时包含该字段
        if include_nones or val is not None:
            result[name] = val
    return result


def _modalities(obj: NamedTuple) -> list[str]:
    """获取当前存在的模态名称列表（排除掩码字段和 timestamps）。

    遍历 NamedTuple 的所有字段，过滤掉以 _mask 结尾的字段、
    timestamps 字段，以及值为 None 的字段。

    Args:
        obj: 要查询的 NamedTuple 实例。

    Returns:
        当前存在的模态名称列表。
    """
    return [
        name
        for name in obj._fields
        if not name.endswith("_mask")           # 排除掩码字段
        and name != TIMESTAMPS_FIELD            # 排除时间戳字段
        and getattr(obj, name) is not None      # 排除 None 值字段
    ]


def _get_masked_modality_name(modality: str) -> str:
    """根据模态名称获取对应的掩码字段名称。

    Args:
        modality: 模态名称（如 "sentinel2_l2a"）。

    Returns:
        掩码字段名称（如 "sentinel2_l2a_mask"）。
    """
    return f"{modality}_mask"


def _get_unmasked_modality_name(modality_mask_name: str) -> str:
    """根据掩码字段名称获取对应的模态名称。

    Args:
        modality_mask_name: 掩码字段名称（如 "sentinel2_l2a_mask"）。

    Returns:
        模态名称（如 "sentinel2_l2a"）。
    """
    return modality_mask_name.replace("_mask", "")


class OlmoEarthSample(NamedTuple):
    """OlmoEarth Pretrain 数据集的单个样本或批量样本。

    这是一个 NamedTuple，包含来自多种地球观测模态的数据。
    每个模态由一个 ArrayTensor（numpy 数组或 PyTorch 张量）表示，
    同时包含经纬度（latlon）和时间戳（timestamps）信息。

    关键属性：
        - 各种模态数据字段（如 sentinel2_l2a, sentinel1 等），形状为 [B, H, W, T, C]
          或 [B, H, W, 1, C]，其中 B=批大小, H=高度, W=宽度, T=时间步, C=通道数
        - latlon: 经纬度坐标，形状为 [B, 2]
        - timestamps: 时间戳信息，形状为 [B, T, 3]，3维为 [日, 月, 年]

    使用场景：
        - 数据加载器返回的原始数据样本
        - 多模态数据在预处理流水线中的传递
        - 作为 MaskedOlmoEarthSample 的来源
    """

    # ==================== 模态数据字段 ====================
    sentinel2_l2a: ArrayTensor | None = None  # Sentinel-2 L2A 多光谱影像 [B, H, W, T, len(S2_bands)]
    sentinel1: ArrayTensor | None = None  # Sentinel-1 雷达影像 [B, H, W, T, len(S1_bands)]
    worldcover: ArrayTensor | None = None  # WorldCover 土地覆盖分类 [B, H, W, 1, len(WC_bands)]
    openstreetmap_raster: ArrayTensor | None = None  # OpenStreetMap 栅格化数据 [B, H, W, 1, len(OSM_bands)]
    srtm: ArrayTensor | None = None  # SRTM 数字高程模型 [B, H, W, 1, len(SRTM_bands)]
    landsat: ArrayTensor | None = None  # Landsat 多光谱影像 [B, H, W, T, len(LANDSAT_bands)]
    # naip（不同瓦片分辨率）目前不再使用，已由 naip_10 替代
    naip: ArrayTensor | None = None  # NAIP 航空影像 [B, H, W, T, len(NAIP_bands)]
    # naip_10 的空间分辨率是 sentinel2_l2a 的 4 倍（高度/宽度为 4 倍）
    naip_10: ArrayTensor | None = None  # NAIP 10m 影像 [B, H, W, T, len(NAIP_bands)]
    gse: ArrayTensor | None = None  # GSE 全球地表数据 [B, H, W, 1, len(GSE_bands)]
    cdl: ArrayTensor | None = None  # CDL 作物数据层 [B, H, W, 1, len(CDL_bands)]
    worldpop: ArrayTensor | None = None  # WorldPop 人口密度 [B, H, W, 1, len(WORLDPOP_bands)]
    worldcereal: ArrayTensor | None = None  # WorldCereal 作物类型 [B, H, W, 1, len(CDL_bands)]
    wri_canopy_height_map: ArrayTensor | None = None  # WRI 林冠高度图 [B, H, W, 1, 1]
    # era5_10 没有空间维度变化（不随地理位置改变），因此没有高度/宽度维度
    era5_10: ArrayTensor | None = None  # ERA5 气象再分析数据 [B, T, len(ERA5_bands)]
    # ndvi 由 S2 L2A 的 B04（红光）和 B08（近红外）波段计算得到，不从文件加载
    ndvi: ArrayTensor | None = None  # NDVI 植被指数 [B, H, W, T, 1]
    eurocrops: ArrayTensor | None = None  # EuroCrops 欧洲作物分类 [B, H, W, 1, 1]
    # ==================== 元数据字段 ====================
    latlon: ArrayTensor | None = None  # 经纬度坐标 [B, 2]
    timestamps: ArrayTensor | None = None  # 时间戳 [B, T, D=3]，D=[日, 月, 年]

    def as_dict(self, include_nones: bool = False) -> dict[str, ArrayTensor | None]:
        """将样本转换为字典。

        Args:
            include_nones: 是否包含值为 None 的字段。默认为 False。

        Returns:
            字典表示，键为字段名，值为字段值。
        """
        return _as_dict(self, include_nones=include_nones)

    @property
    def modalities(self) -> list[str]:
        """获取当前存在的模态名称列表（排除掩码和 timestamps）。

        Returns:
            存在的模态名称列表。
        """
        return _modalities(self)

    @property
    def modalities_with_timestamps(self) -> list[str]:
        """获取所有存在的模态名称列表，包括 timestamps（排除掩码字段）。

        与 modalities 属性不同，此属性会在结果中包含 timestamps（如果存在）。

        Returns:
            包含 timestamps 在内的所有存在模态名称列表。
        """
        result = []
        for name in self._fields:
            # 排除掩码字段，但保留 timestamps
            if not name.endswith("_mask") and getattr(self, name) is not None:
                result.append(name)
        return result

    @property
    def batch_size(self) -> int:
        """获取数据的批大小。

        从所有非 None 字段中提取第 0 维大小，如果所有字段一致则返回该值，
        否则返回 1（表示单个样本）。

        Returns:
            批大小。
        """
        # 收集所有非 None 字段的第 0 维大小
        vals = [
            cast(ArrayTensor, x).shape[0]
            for x in self.as_dict(include_nones=False).values()
        ]
        if len(set(vals)) == 1:
            # 所有字段的批大小一致
            return vals[0]
        else:
            # 批大小不一致，返回 1（单个样本）
            return 1

    def shape(self, attribute: str, mask: bool = False) -> Sequence[int]:
        """返回指定属性的期望形状。

        根据属性名称和是否为掩码，返回对应的数据形状。
        timestamps 字段有特殊处理，不支持掩码。

        Args:
            attribute: 属性名称（如 "sentinel2_l2a"、"timestamps" 等）。
            mask: 是否查询掩码的形状。默认为 False。

        Returns:
            属性的形状元组。

        Raises:
            ValueError: 当 timestamps 不存在时查询其形状，或尝试查询 timestamps 掩码时。
        """
        if attribute == "timestamps":
            # timestamps 有特殊的形状处理逻辑
            if not mask:
                if self.timestamps is None:
                    raise ValueError("Timestamps are not present in the sample")
                return self.timestamps.shape
            else:
                raise ValueError("Timestamps are not maskable")
        else:
            # 其他模态使用通用形状计算
            return self.get_expected_shape(attribute, mask)

    @staticmethod
    def num_bands(attribute: str) -> int:
        """获取指定属性的通道数。

        Args:
            attribute: 属性名称（如 "sentinel2_l2a" 或 "timestamps"）。

        Returns:
            该属性的通道数。对于 timestamps，返回时间戳维度数（日、月、年）；
            对于其他模态，返回模态定义的通道数。
        """
        if attribute == "timestamps":
            return len(TIMESTAMPS)
        else:
            return Modality.get(attribute).num_bands

    def to_device(
        self, device: torch.device, non_blocking: bool = True
    ) -> OlmoEarthSample:
        """将所有张量移动到指定设备。

        遍历所有非 None 字段，将其中的张量移动到目标设备。
        仅移动具有 .to() 方法的对象（即 PyTorch 张量）。

        Args:
            device: 目标设备（如 torch.device("cuda")）。
            non_blocking: 是否使用非阻塞传输。默认为 True。

        Returns:
            所有张量已移动到目标设备的新 OlmoEarthSample 实例。
        """
        return OlmoEarthSample(
            **{
                key: val.to(device, non_blocking=non_blocking)
                for key, val in self.as_dict(include_nones=False).items()
                if val is not None
            }
        )

    def distribute_tensors(self, device_mesh: DeviceMesh) -> OlmoEarthSample:
        """将张量分布到指定的设备网格上。

        用于分布式训练，将每个张量按照 device_mesh 进行分片分布。

        Args:
            device_mesh: 分布式设备网格，定义了张量的分布策略。

        Returns:
            张量已分布到设备网格上的新 OlmoEarthSample 实例。
        """
        return OlmoEarthSample(
            **{
                key: distribute_tensor(val, device_mesh)
                for key, val in self.as_dict(include_nones=False).items()
            }
        )

    @property
    def height(self) -> int:
        """获取数据在分辨率因子为 16 时的高度（像素数）。

        遍历所有空间模态，找到第一个非 None 的空间数据，
        根据其形状和 image_tile_size_factor 计算基础高度。

        Returns:
            基础高度（分辨率因子为 16 时的像素数）。

        Raises:
            ValueError: 当没有任何空间模态存在时。
        """
        for modality in self.modalities:
            modality_spec = Modality.get(modality)
            if not modality_spec.is_spatial:
                continue  # 跳过非空间模态
            x = getattr(self, modality)
            if x is not None:
                if len(x.shape) == 5:
                    # 5维形状 [B, H*factor, W*factor, T, C]，高度在第1维
                    return x.shape[1] // modality_spec.image_tile_size_factor
                else:
                    # 4维形状 [H*factor, W*factor, T, C]（无批维度），高度在第0维
                    if len(x.shape) != 4:
                        raise ValueError(f"Unexpected shape {x.shape} for {modality}")
                    return x.shape[0] // modality_spec.image_tile_size_factor
        raise ValueError("No modality with height or width present")

    @property
    def width(self) -> int:
        """获取数据在分辨率因子为 16 时的宽度（像素数）。

        遍历所有空间模态，找到第一个非 None 的空间数据，
        根据其形状和 image_tile_size_factor 计算基础宽度。

        Returns:
            基础宽度（分辨率因子为 16 时的像素数）。

        Raises:
            ValueError: 当没有任何空间模态存在时。
        """
        for modality in self.modalities:
            modality_spec = Modality.get(modality)
            if not modality_spec.is_spatial:
                continue  # 跳过非空间模态
            x = getattr(self, modality)
            if x is not None:
                if len(x.shape) == 5:
                    # 5维形状 [B, H*factor, W*factor, T, C]，宽度在第2维
                    return x.shape[2] // modality_spec.image_tile_size_factor
                else:
                    # 4维形状 [H*factor, W*factor, T, C]（无批维度），宽度在第1维
                    if len(x.shape) != 4:
                        raise ValueError(f"Unexpected shape {x.shape} for {modality}")
                    return x.shape[1] // modality_spec.image_tile_size_factor
        raise ValueError("No modality with height or width present")

    @property
    def time(self) -> int:
        """获取数据的时间步数。

        Returns:
            时间步数。

        Raises:
            ValueError: 当 timestamps 不存在时。
        """
        if self.timestamps is None:
            raise ValueError("Timestamps are not present in the sample")
        # timestamps 形状为 [B, T, D]，取倒数第2维
        return self.timestamps.shape[-2]

    @property
    def valid_time(self) -> int:
        """获取批次中有效的最小时间步数。

        有效时间步是指至少有一个模态存在数据的时间步。

        Returns:
            有效时间步数。
        """
        return self.timesteps_with_at_least_one_modality.shape[0]

    @property
    def timesteps_with_at_least_one_modality(self) -> torch.Tensor:
        """获取至少有一个模态存在数据的时间步索引。

        遍历所有多时间步模态，检查每个时间步是否有非缺失值的数据，
        然后返回至少一个模态有数据的时间步索引。

        Returns:
            包含有效时间步索引的一维张量。

        Raises:
            ValueError: 当多时间步模态的数据为 numpy 数组时（暂不支持）。
        """
        per_modality_present_masks = []
        for modality in self.modalities:
            modality_spec = Modality.get(modality)
            if modality_spec.is_multitemporal:
                # 仅处理多时间步模态
                data = getattr(self, modality)
                if isinstance(data, np.ndarray):
                    raise ValueError(
                        "timesteps_with_at_least_one_modality is not yet supported for numpy arrays"
                    )
                # 检查每个时间步是否所有数据都非缺失值（形状为 [T]）
                # 排除批维度(0)、高度(1)、宽度(2)、通道(4)维度，沿时间维度(3)检查
                present_mask = (data != MISSING_VALUE).all(dim=(0, 1, 2, 4))
                per_modality_present_masks.append(present_mask)
        # 在模态维度上取"或"操作，得到至少一个模态存在的时间步掩码
        at_least_one_modality_present_timestep_mask = torch.stack(
            per_modality_present_masks, dim=1
        ).any(dim=1)
        # 提取满足条件的时间步索引
        timesteps_with_at_least_one_modality = torch.where(
            at_least_one_modality_present_timestep_mask
        )[0]
        return timesteps_with_at_least_one_modality

    @staticmethod
    def compute_expected_shape(
        attribute: str,
        height: int | None,
        width: int | None,
        time: int,
        mask: bool = False,
    ) -> tuple[int, ...]:
        """根据给定的维度参数计算模态的期望形状。

        根据模态的空间/时间特性，返回不同维度的形状元组：
        - 空间+时间模态（如 sentinel2_l2a）: (H*factor, W*factor, T, C)
        - 仅空间模态（如 worldcover）: (H*factor, W*factor, 1, C)
        - 仅时间模态（如 era5_10）: (T, C)
        - 其他: (C,)

        Args:
            attribute: 模态名称。
            height: 高度像素数（空间模态必需）。
            width: 宽度像素数（空间模态必需）。
            time: 时间步数。
            mask: 是否计算掩码的形状。为 True 时使用 num_band_sets
                  代替 num_bands 作为通道数。默认为 False。

        Returns:
            模态的期望形状元组。
        """
        modality_spec = Modality.get(attribute)
        # 掩码使用 band_sets 数量，数据使用完整 bands 数量
        num_bands = modality_spec.num_band_sets if mask else modality_spec.num_bands

        if modality_spec.is_spacetime_varying:
            # 空间+时间模态：形状为 (H*factor, W*factor, T, C)
            assert height is not None and width is not None, (
                f"height and width required for spatial modality {attribute}"
            )
            return (
                height * modality_spec.image_tile_size_factor,
                width * modality_spec.image_tile_size_factor,
                time,
                num_bands,
            )
        elif modality_spec.is_space_only_varying:
            # 仅空间模态：形状为 (H*factor, W*factor, 1, C)，时间维度固定为 1
            assert height is not None and width is not None, (
                f"height and width required for spatial modality {attribute}"
            )
            return (
                height * modality_spec.image_tile_size_factor,
                width * modality_spec.image_tile_size_factor,
                1,
                num_bands,
            )
        elif modality_spec.is_time_only_varying:
            # 仅时间模态：形状为 (T, C)，无空间维度
            return (time, num_bands)
        else:
            # 其他模态：形状为 (C,)，仅有通道维度
            return (num_bands,)

    def get_expected_shape(self, attribute: str, mask: bool = False) -> tuple[int, ...]:
        """使用当前样本的维度计算指定属性的期望形状。

        这是 compute_expected_shape 的实例方法封装，自动使用
        当前样本的 height、width 和 time 属性。

        Args:
            attribute: 模态名称。
            mask: 是否计算掩码的形状。默认为 False。

        Returns:
            模态的期望形状元组。
        """
        return OlmoEarthSample.compute_expected_shape(
            attribute, self.height, self.width, self.time, mask
        )

    def scale(self, s: float) -> OlmoEarthSample:
        """将 OlmoEarthSample 中的所有数据乘以一个浮点数。

        用于缩放数据，例如在梯度累积时除以累积步数。

        Args:
            s: 缩放因子。

        Returns:
            缩放后的新 OlmoEarthSample 实例。
        """
        return OlmoEarthSample(
            **{k: cast(ArrayTensor, v) * s for k, v in self.as_dict().items()}
        )

    def add(
        self, other: OlmoEarthSample, timestamps_to_keep: ArrayTensor
    ) -> OlmoEarthSample:
        """将两个 OlmoEarthSample 逐元素相加。

        要求两个样本具有相同的模态（所有字段必须都存在且非 None），
        结果中的 timestamps 由参数指定。

        Args:
            other: 要相加的另一个 OlmoEarthSample。
            timestamps_to_keep: 相加后要使用的时间戳数据。

        Returns:
            逐元素相加后的新 OlmoEarthSample 实例。

        Raises:
            ValueError: 当 other 不是 OlmoEarthSample，或两者模态不一致时。
        """
        if not isinstance(other, OlmoEarthSample):
            raise ValueError("Addition only supported for OlmoEarthSamples")
        summed_dict: dict[str, ArrayTensor] = {}
        for key, val in self.as_dict(include_nones=False).items():
            assert val is not None
            other_val = getattr(other, key)
            if other_val is None:
                raise ValueError(
                    f"Add requires both OlmoEarthSamples to have the same modalities, other is missing {key}"
                )
            # 逐元素相加
            summed_dict[key] = val + other_val
        # 使用指定的时间戳替代相加后的时间戳
        summed_dict["timestamps"] = timestamps_to_keep
        return OlmoEarthSample(**summed_dict)

    def rotate(self) -> OlmoEarthSample:
        """将批次中的样本循环旋转一个位置。

        例如，如果原来有三个样本 [B1, B2, B3]，
        旋转后变为 [B2, B3, B1]。

        这在自监督学习中用于创建不同视角的配对数据，
        使同一批次的样本互相作为目标编码器的输入。

        Returns:
            旋转后的新 OlmoEarthSample 实例。
        """
        output_dict: dict[str, ArrayTensor] = {}
        for key, v in self.as_dict().items():
            if isinstance(v, np.ndarray):
                # numpy 数组：沿第 0 维（批次维）旋转
                output_dict[key] = np.concatenate((v[1:], v[:1]), axis=0)
            elif isinstance(v, torch.Tensor):
                # torch 张量：沿第 0 维（批次维）旋转
                output_dict[key] = torch.cat((v[1:], v[:1]), dim=0)
        return OlmoEarthSample(**output_dict)


class MaskedOlmoEarthSample(NamedTuple):
    """带掩码的 OlmoEarth 数据样本。

    与 OlmoEarthSample 类似，但每个模态都附带一个掩码张量，
    掩码的值由 MaskValue 枚举定义，标识每个令牌在训练流程中的可见性。

    关键属性：
        - timestamps: 时间戳数据（必填字段），形状为 [B, T, 3]
        - 各种模态数据字段及其对应的掩码字段（如 sentinel2_l2a 和 sentinel2_l2a_mask）
        - latlon 和 latlon_mask: 经纬度及其掩码

    使用场景：
        - 自监督预训练中实现掩码策略（如随机掩码、模态间掩码）
        - 区分在线编码器、目标编码器和解码器可见的令牌
        - 标记缺失的模态数据
    """

    # ==================== 必填字段 ====================
    timestamps: (
        ArrayTensor  # [B, T, D=3]，D=[日, 月, 年]（月份从 0 开始计数）
    )
    # ==================== 模态数据字段及对应掩码 ====================
    sentinel2_l2a: Tensor | None = None
    sentinel2_l2a_mask: Tensor | None = None
    sentinel1: Tensor | None = None
    sentinel1_mask: Tensor | None = None
    worldcover: Tensor | None = None
    worldcover_mask: Tensor | None = None
    latlon: Tensor | None = None  # [B, 2]
    latlon_mask: Tensor | None = None
    openstreetmap_raster: Tensor | None = None
    openstreetmap_raster_mask: Tensor | None = None
    srtm: Tensor | None = None
    srtm_mask: Tensor | None = None
    landsat: Tensor | None = None
    landsat_mask: Tensor | None = None
    naip: Tensor | None = None
    naip_mask: Tensor | None = None
    naip_10: Tensor | None = None
    naip_10_mask: Tensor | None = None
    gse: Tensor | None = None
    gse_mask: Tensor | None = None
    cdl: Tensor | None = None
    cdl_mask: Tensor | None = None
    worldpop: Tensor | None = None
    worldpop_mask: Tensor | None = None
    worldcereal: Tensor | None = None
    worldcereal_mask: Tensor | None = None
    wri_canopy_height_map: Tensor | None = None
    wri_canopy_height_map_mask: Tensor | None = None
    era5_10: Tensor | None = None
    era5_10_mask: Tensor | None = None
    ndvi: Tensor | None = None
    ndvi_mask: Tensor | None = None
    eurocrops: Tensor | None = None
    eurocrops_mask: Tensor | None = None

    def as_dict(self, include_nones: bool = False) -> dict[str, Any]:
        """将掩码样本转换为字典。

        Args:
            include_nones: 是否包含值为 None 的字段。默认为 False。

        Returns:
            字典表示，键为字段名，值为字段值。
        """
        return _as_dict(self, include_nones=include_nones)

    @property
    def modalities(self) -> list[str]:
        """获取当前存在的模态名称列表（排除掩码和 timestamps）。

        Returns:
            存在的模态名称列表。
        """
        return _modalities(self)

    @staticmethod
    def get_masked_modality_name(modality: str) -> str:
        """根据模态名称获取对应的掩码字段名称。

        Args:
            modality: 模态名称（如 "sentinel2_l2a"）。

        Returns:
            掩码字段名称（如 "sentinel2_l2a_mask"）。
        """
        return _get_masked_modality_name(modality)

    @staticmethod
    def get_unmasked_modality_name(modality_mask_name: str) -> str:
        """根据掩码字段名称获取对应的模态名称。

        Args:
            modality_mask_name: 掩码字段名称（如 "sentinel2_l2a_mask"）。

        Returns:
            模态名称（如 "sentinel2_l2a"）。
        """
        return _get_unmasked_modality_name(modality_mask_name)

    @property
    def batch_size(self) -> int:
        """获取样本的批大小。

        直接从 timestamps 的第 0 维获取，因为 timestamps 是必填字段。

        Returns:
            批大小。
        """
        return self.timestamps.shape[0]

    def to_device(
        self, device: torch.device, non_blocking: bool = True
    ) -> MaskedOlmoEarthSample:
        """将所有张量移动到指定设备。

        遍历所有非 None 且具有 .to() 方法的字段，将其移动到目标设备。

        Args:
            device: 目标设备（如 torch.device("cuda")）。
            non_blocking: 是否使用非阻塞传输。默认为 True。

        Returns:
            所有张量已移动到目标设备的新 MaskedOlmoEarthSample 实例。
        """
        return MaskedOlmoEarthSample(
            **{
                key: val.to(device, non_blocking=non_blocking)
                for key, val in self.as_dict(include_nones=False).items()
                if val is not None and hasattr(val, "to")
            }
        )

    def unmask(self) -> MaskedOlmoEarthSample:
        """返回一个全部解除掩码的 MaskedOlmoEarthSample。

        将所有掩码值设为 MaskValue.ONLINE_ENCODER（对在线编码器可见），
        但保留 MaskValue.MISSING（缺失数据）不变，因为缺失数据无法变为可见。

        Returns:
            解除掩码后的新 MaskedOlmoEarthSample 实例。
        """
        updates = {}
        for name in _MASKED_SAMPLE_MASK_FIELDS:
            val = getattr(self, name)
            if val is not None:
                # 保留 MISSING 值不变，其余设为 ONLINE_ENCODER (0)
                # val == MaskValue.MISSING.value 的位置保持原值，其余变为 0
                updates[name] = val * (val == MaskValue.MISSING.value)
        return self._replace(**updates)

    @classmethod
    def from_olmoearthsample(
        cls,
        sample: OlmoEarthSample,
    ) -> MaskedOlmoEarthSample:
        """将 OlmoEarthSample 转换为 MaskedOlmoEarthSample。

        此函数假设模态数据是均匀缺失的（即整个模态要么存在要么缺失）。
        对于存在的模态，其掩码初始化为全部 ONLINE_ENCODER；
        对于缺失的模态，数据和掩码均设为 None。

        Args:
            sample: 原始的未掩码 OlmoEarthSample 样本。

        Returns:
            转换后的 MaskedOlmoEarthSample，所有存在模态的掩码均为 ONLINE_ENCODER。
        """
        masked_sample_dict: dict[str, Any] = {}
        for key, t in sample.as_dict(include_nones=True).items():
            if key == "timestamps":
                # timestamps 直接复制，不需要掩码
                masked_sample_dict[key] = t
            else:
                if t is None:
                    # 模态缺失：数据和掩码均设为 None
                    masked_sample_dict[key] = None
                    masked_sample_dict[cls.get_masked_modality_name(key)] = None
                else:
                    # 模态存在：复制数据，掩码初始化为全 ONLINE_ENCODER
                    masked_sample_dict[key] = t
                    masked_sample_dict[cls.get_masked_modality_name(key)] = (
                        torch.ones(sample.shape(key, mask=False))
                        * MaskValue.ONLINE_ENCODER.value
                    )

        return MaskedOlmoEarthSample(**masked_sample_dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MaskedOlmoEarthSample:
        """从字典创建 MaskedOlmoEarthSample。

        Args:
            d: 包含字段名和值的字典。

        Returns:
            从字典构造的 MaskedOlmoEarthSample 实例。
        """
        return cls(**d)


# 预计算的掩码字段名元组，用于 unmask() 中更快的迭代
_MASKED_SAMPLE_MASK_FIELDS: tuple[str, ...] = tuple(
    f for f in MaskedOlmoEarthSample._fields if f.endswith("_mask")
)


class TokensAndMasks(NamedTuple):
    """编码器输出的嵌入令牌和掩码，用于计算损失。

    这是编码器阶段的输出格式，包含每个模态的嵌入令牌（tokens）和
    对应的掩码（masks）。嵌入令牌经过编码器处理后的高维表示，
    将用于下游的损失计算和解码器输入。

    形状说明：
        - 空间+时间模态（如 sentinel2_l2a）:
            - modality: (B, P_H, P_W, T, Band_Sets, D)
            - modality_mask: (B, P_H, P_W, T, Band_Sets)
        - 仅时间模态（如 era5_10）: (B, T, Band_Sets, D) -- 无空间维度
        - latlon: (B, D) -- 无空间和时间维度

    其中 P_H、P_W 为 patch 化后的高度和宽度，D 为嵌入维度，
    Band_Sets 为波段组数。

    使用场景：
        - 编码器前向传播的输出
        - 传递给解码器或损失函数进行训练
    """

    # ==================== 模态令牌字段及对应掩码（不含 timestamps） ====================
    sentinel2_l2a: Tensor | None = None
    sentinel2_l2a_mask: Tensor | None = None
    sentinel1: Tensor | None = None
    sentinel1_mask: Tensor | None = None
    worldcover: Tensor | None = None
    worldcover_mask: Tensor | None = None
    openstreetmap_raster: Tensor | None = None
    openstreetmap_raster_mask: Tensor | None = None
    srtm: Tensor | None = None
    srtm_mask: Tensor | None = None
    landsat: Tensor | None = None
    landsat_mask: Tensor | None = None
    naip: Tensor | None = None
    naip_mask: Tensor | None = None
    naip_10: Tensor | None = None
    naip_10_mask: Tensor | None = None
    gse: Tensor | None = None
    gse_mask: Tensor | None = None
    cdl: Tensor | None = None
    cdl_mask: Tensor | None = None
    worldpop: Tensor | None = None
    worldpop_mask: Tensor | None = None
    worldcereal: Tensor | None = None
    worldcereal_mask: Tensor | None = None
    wri_canopy_height_map: Tensor | None = None
    wri_canopy_height_map_mask: Tensor | None = None
    era5_10: Tensor | None = None
    era5_10_mask: Tensor | None = None
    ndvi: Tensor | None = None
    ndvi_mask: Tensor | None = None
    eurocrops: Tensor | None = None
    eurocrops_mask: Tensor | None = None
    latlon: Tensor | None = None
    latlon_mask: Tensor | None = None

    def as_dict(self, include_nones: bool = False) -> dict[str, Any]:
        """将令牌和掩码转换为字典。

        Args:
            include_nones: 是否包含值为 None 的字段。默认为 False。

        Returns:
            字典表示，键为字段名，值为字段值。
        """
        return _as_dict(self, include_nones=include_nones)

    @property
    def modalities(self) -> list[str]:
        """获取当前存在的模态名称列表（排除掩码和 timestamps）。

        Returns:
            存在的模态名称列表。
        """
        return _modalities(self)

    @staticmethod
    def get_masked_modality_name(modality: str) -> str:
        """根据模态名称获取对应的掩码字段名称。

        Args:
            modality: 模态名称（如 "sentinel2_l2a"）。

        Returns:
            掩码字段名称（如 "sentinel2_l2a_mask"）。
        """
        return _get_masked_modality_name(modality)

    @staticmethod
    def get_unmasked_modality_name(modality_mask_name: str) -> str:
        """根据掩码字段名称获取对应的模态名称。

        Args:
            modality_mask_name: 掩码字段名称（如 "sentinel2_l2a_mask"）。

        Returns:
            模态名称（如 "sentinel2_l2a"）。
        """
        return _get_unmasked_modality_name(modality_mask_name)

    @property
    def batch_size(self) -> int:
        """获取批次大小。

        从第一个非 None 字段的第 0 维获取。

        Returns:
            批次大小。

        Raises:
            ValueError: 当没有数据可获取批次大小时。
        """
        for name in self._fields:
            val = getattr(self, name)
            if val is not None:
                return val.shape[0]
        raise ValueError("No data to get batch size from")

    def to_device(
        self, device: torch.device, non_blocking: bool = True
    ) -> TokensAndMasks:
        """将所有张量移动到指定设备。

        遍历所有非 None 且具有 .to() 方法的字段，将其移动到目标设备。

        Args:
            device: 目标设备（如 torch.device("cuda")）。
            non_blocking: 是否使用非阻塞传输。默认为 True。

        Returns:
            所有张量已移动到目标设备的新 TokensAndMasks 实例。
        """
        return TokensAndMasks(
            **{
                key: val.to(device, non_blocking=non_blocking)
                for key, val in self.as_dict(include_nones=False).items()
                if val is not None and hasattr(val, "to")
            }
        )

    @property
    def device(self) -> torch.device:
        """获取令牌和掩码所在的设备。

        从第一个非 None 字段获取设备信息。

        Returns:
            张量所在的设备。

        Raises:
            ValueError: 当没有数据可获取设备信息时。
        """
        for name in self._fields:
            val = getattr(self, name)
            if val is not None:
                return val.device
        raise ValueError("No data to get device from")

    def get_shape_dict(self) -> dict[str, tuple]:
        """返回所有非 None 字段的形状字典。

        Returns:
            字典，键为字段名，值为对应的形状元组。
        """
        return {
            name: getattr(self, name).shape
            for name in self._fields
            if getattr(self, name) is not None
        }

    @staticmethod
    def _flatten(x: Tensor) -> Tensor:
        """将张量的空间/时间维度展平为单一序列维度。

        将形状为 (B, ..., D) 的张量展平为 (B, S, D)，
        其中 S 是所有中间维度的乘积。

        Args:
            x: 输入张量，形状为 (B, ..., D)。

        Returns:
            展平后的张量，形状为 (B, S, D)。
        """
        return rearrange(x, "b ... d -> b (...) d")

    def _flatten_per_modality(
        self,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """按模态展平令牌和掩码（不跨模态拼接）。

        对每个模态，将令牌和掩码的空间/时间维度展平，
        返回按模态分组的列表。

        Returns:
            元组 (flattened_x, flattened_masks):
            - flattened_x: 每个模态的展平令牌列表，每个元素形状为 (B, S, D)
            - flattened_masks: 每个模态的展平掩码列表，每个元素形状为 (B, S)
        """
        flattened_x, flattened_masks = [], []
        for attr_name in self.modalities:
            mask_attr_name = self.get_masked_modality_name(attr_name)
            attr = getattr(self, attr_name)
            masked_attr = getattr(self, mask_attr_name)
            if attr is not None:
                if masked_attr is None:
                    raise ValueError(
                        f"Can't have present {attr_name} but None {mask_attr_name}"
                    )
                # 为掩码增加最后一维以便展平，形状从 (B, ..., S) 变为 (B, ..., S, 1)
                masked_attr = masked_attr.unsqueeze(dim=-1)
                # 展平令牌：(B, P_H, P_W, T, Band_Sets, D) -> (B, S, D)
                flattened_x.append(self._flatten(attr))
                # 展平掩码：(B, P_H, P_W, T, Band_Sets, 1) -> (B, S, 1)
                flattened_masks.append(self._flatten(masked_attr))
        # 移除掩码的最后一维：(B, S, 1) -> (B, S)
        flattened_masks = [mask[:, :, 0] for mask in flattened_masks]
        return flattened_x, flattened_masks

    def flatten_tokens_and_masks_per_modality(
        self,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """按模态展平令牌和掩码，返回每个模态独立的张量列表。

        这是 _flatten_per_modality 的公开接口。

        Returns:
            元组 (flattened_x, flattened_masks)：
            - flattened_x: 每个模态的展平令牌列表，每个元素形状为 (B, S, D)
            - flattened_masks: 每个模态的展平掩码列表，每个元素形状为 (B, S)
        """
        return self._flatten_per_modality()

    def flatten_all_tokens_and_masks(self) -> tuple[Tensor, Tensor]:
        """展平并跨模态拼接所有令牌和掩码。

        先按模态展平，然后在序列维度上拼接所有模态的令牌和掩码，
        返回统一的张量对。

        Returns:
            元组 (x, masks)：
            - x: 所有模态拼接后的令牌张量，形状为 [B, total_S, D]
            - masks: 所有模态拼接后的掩码张量，形状为 [B, total_S]
            其中 total_S 是所有模态序列长度的总和。
        """
        flattened_x, flattened_masks = self._flatten_per_modality()
        # 在序列维度（dim=1）上拼接所有模态
        x = torch.cat(flattened_x, dim=1)
        masks = torch.cat(flattened_masks, dim=1)
        return x, masks
