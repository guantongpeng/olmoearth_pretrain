"""
OlmoEarthSample 的数据增强变换模块。

本模块提供了可应用于 OlmoEarthSample 的数据增强变换，包括：
- NoTransform: 恒等变换，不做任何修改
- FlipAndRotateSpace: 从 8 种对称变换（4 种旋转 + 4 种翻转）中随机选择一种，
    仅作用于空间变化数据
- Mixup: Mixup 数据增强（https://arxiv.org/abs/1710.09412），
    对两个样本进行凸组合

所有变换都通过 TRANSFORM_REGISTRY 注册，可通过 TransformConfig 配置和构建。
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torchvision.transforms.v2.functional as F
from class_registry import ClassRegistry
from einops import rearrange
from torch.distributions import Beta

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import OlmoEarthSample
from olmoearth_pretrain.types import ArrayTensor


class Transform(ABC):
    """可应用于 OlmoEarthSample 的变换基类。

    所有变换必须实现 apply 方法。
    """

    @abstractmethod
    def apply(self, batch: OlmoEarthSample) -> "OlmoEarthSample":
        """对批次应用变换。

        Args:
            batch: 批次化的 OlmoEarthSample。

        Returns:
            变换后的 OlmoEarthSample。
        """
        pass


# 变换注册表，用于通过名称查找和构建变换
TRANSFORM_REGISTRY = ClassRegistry[Transform]()


@TRANSFORM_REGISTRY.register("no_transform")
class NoTransform(Transform):
    """恒等变换，不做任何修改，直接返回原始样本。"""

    def apply(self, batch: OlmoEarthSample) -> "OlmoEarthSample":
        """应用恒等变换，直接返回原始批次。"""
        return batch


@TRANSFORM_REGISTRY.register("flip_and_rotate")
class FlipAndRotateSpace(Transform):
    """从 8 种对称变换中随机选择一种，仅应用于空间变化数据。

    8 种变换包括：
    - 恒等（no_transform）
    - 旋转 90/180/270 度
    - 水平翻转、垂直翻转
    - 水平翻转+旋转 90 度
    - 垂直翻转+旋转 90 度

    仅对 is_spacetime_varying 或 is_space_only_varying 的模态应用变换，
    时间戳等非空间数据保持不变。
    """

    def __init__(self) -> None:
        """初始化 FlipAndRotateSpace，定义 8 种可选变换。"""
        self.transformations = [
            self.no_transform,       # 恒等
            self.rotate_90,          # 旋转 90 度
            self.rotate_180,         # 旋转 180 度
            self.rotate_270,         # 旋转 270 度
            self.hflip,              # 水平翻转
            self.vflip,              # 垂直翻转
            self.hflip_rotate_90,    # 水平翻转后旋转 90 度
            self.vflip_rotate_90,    # 垂直翻转后旋转 90 度
        ]

    def no_transform(self, x: ArrayTensor) -> ArrayTensor:
        """恒等变换。"""
        return x

    def rotate_90(self, x: ArrayTensor) -> ArrayTensor:
        """旋转 90 度。"""
        return F.rotate(x, 90)

    def rotate_180(self, x: ArrayTensor) -> ArrayTensor:
        """旋转 180 度。"""
        return F.rotate(x, 180)

    def rotate_270(self, x: ArrayTensor) -> ArrayTensor:
        """旋转 270 度。"""
        return F.rotate(x, 270)

    def hflip(self, x: ArrayTensor) -> ArrayTensor:
        """水平翻转。"""
        return F.hflip(x)

    def vflip(self, x: ArrayTensor) -> ArrayTensor:
        """垂直翻转。"""
        return F.vflip(x)

    def hflip_rotate_90(self, x: ArrayTensor) -> ArrayTensor:
        """水平翻转后旋转 90 度。"""
        return F.hflip(F.rotate(x, 90))

    def vflip_rotate_90(self, x: ArrayTensor) -> ArrayTensor:
        """垂直翻转后旋转 90 度。"""
        return F.vflip(F.rotate(x, 90))

    def apply(
        self,
        batch: OlmoEarthSample,
    ) -> "OlmoEarthSample":
        """对批次应用随机空间变换。

        随机选择一种变换，仅对空间变化数据（spacetime_varying 和 space_only_varying）
        应用，timestamps 和非空间数据保持不变。

        Args:
            batch: 批次化的 OlmoEarthSample。

        Returns:
            变换后的 OlmoEarthSample。
        """
        # 随机选择一种变换
        transformation = random.choice(self.transformations)
        new_data_dict: dict[str, ArrayTensor] = {}
        for attribute, modality_data in batch.as_dict().items():
            if attribute == "timestamps":
                new_data_dict[attribute] = modality_data  # 时间戳不变
            else:
                modality_spec = Modality.get(attribute)
                # 仅对空间变化数据应用变换
                if (
                    modality_spec.is_spacetime_varying
                    or modality_spec.is_space_only_varying
                ):
                    # 将数据从 (B, H, W, T, C) 重排为 (B, T, C, H, W) 以适配 torchvision
                    modality_data = rearrange(modality_data, "b h w t c -> b t c h w")
                    modality_data = transformation(modality_data)
                    # 重排回原始维度顺序
                    modality_data = rearrange(modality_data, "b t c h w -> b h w t c")
                new_data_dict[attribute] = modality_data
        return OlmoEarthSample(**new_data_dict)


@TRANSFORM_REGISTRY.register("mixup")
class Mixup(Transform):
    """Mixup 数据增强（https://arxiv.org/abs/1710.09412）。

    对两个样本进行凸组合：new = (1-lam) * sample + lam * other_sample，
    其中 lam 从 Beta(alpha, alpha) 分布中采样。

    启动训练时的示例参数：
    --train_module.transform_config.transform_type=mixup
    --train_module.transform_config.transform_kwargs={"alpha": 1.3}
    """

    def __init__(self, alpha: float) -> None:
        """初始化 Mixup 变换。

        Args:
            alpha: Beta 分布的参数，用于采样混合比例 lam。
        """
        self.alpha = alpha
        self.dist = Beta(torch.tensor([alpha]), torch.tensor([alpha]))  # Beta 分布

    def apply(self, batch: OlmoEarthSample) -> OlmoEarthSample:
        """对批次应用 Mixup 增强。

        将当前批次与自身旋转后的版本进行凸组合。
        时间戳取 lam 较大一方的时间戳。

        Args:
            batch: 批次化的 OlmoEarthSample。

        Returns:
            Mixup 后的 OlmoEarthSample。
        """
        other_microbatch = batch.rotate()  # 获取旋转后的另一个微批次

        lam = float(self.dist.sample())  # 从 Beta 分布采样混合比例
        if lam >= 0.5:
            ts_to_keep = other_microbatch.timestamps  # lam 较大时使用另一个批次的时间戳
        else:
            ts_to_keep = batch.timestamps  # lam 较小时使用当前批次的时间戳
        return batch.scale(1 - lam).add(other_microbatch.scale(lam), ts_to_keep)  # 凸组合


@dataclass
class TransformConfig(Config):
    """变换配置类。

    属性:
        transform_type: 变换类型名称，必须在 TRANSFORM_REGISTRY 中注册。
        transform_kwargs: 传递给变换构造函数的关键字参数。
    """

    transform_type: str = "no_transform"
    transform_kwargs: dict[str, Any] = field(default_factory=lambda: {})

    def validate(self) -> None:
        """验证配置，检查变换类型是否已注册。

        Raises:
            ValueError: 如果变换类型未注册。
        """
        if self.transform_type not in TRANSFORM_REGISTRY:
            raise ValueError(f"Invalid transform type: {self.transform_type}")

    def build(self) -> Transform:
        """构建变换实例。

        Returns:
            配置好的 Transform 实例。
        """
        self.validate()
        return TRANSFORM_REGISTRY.get_class(self.transform_type)(
            **self.transform_kwargs
        )
