"""
OlmoEarth Pretrain 数据集拼接模块。

本模块实现了 OlmoEarthConcatDataset，用于将多个 OlmoEarthDataset 拼接为一个统一数据集。
需要自定义实现而非直接使用 PyTorch ConcatDataset 的原因：
- 自定义的 __getitem__ 访问方式（GetItemArgs 而非整数索引）
- 需要支持 OlmoEarthDataLoader 和各种回调期望的函数和属性
- 需要统一的指纹（fingerprint）和地理分布（latlon_distribution）管理
"""

import bisect
import hashlib
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from torch.utils.data import ConcatDataset, Dataset

from olmoearth_pretrain._compat import deprecated_class_alias as _deprecated_class_alias
from olmoearth_pretrain.config import Config

from .dataset import GetItemArgs

logger = logging.getLogger(__name__)


class OlmoEarthConcatDataset(ConcatDataset):
    """基于 ConcatDataset 的拼接数据集，用于拼接多个 OlmoEarthDataset。

    该拼接数据集作为各子数据集的统一版本，支持：
    - 自定义 __getitem__（通过 GetItemArgs 而非整数索引）
    - 统一的指纹和版本管理
    - 统一的地理分布属性

    关键属性:
        datasets: 子数据集列表
        cumulative_sizes: 累积大小列表（继承自 ConcatDataset）
        training_modalities: 训练模态列表（所有子数据集必须一致）
        latlon_distribution: 拼接后的地理分布数组
    """

    def __getitem__(self, args: GetItemArgs) -> Any:
        """获取指定索引的样本。

        适配自 ConcatDataset，主要变化是从 GetItemArgs 中提取索引，
        并将更新后的索引传递给子数据集。

        Args:
            args: GetItemArgs 命名元组，包含 idx、patch_size、sampled_hw_p 等。

        Returns:
            子数据集返回的样本数据。
        """
        idx = args.idx
        if idx < 0:
            # 处理负索引
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        # 使用二分查找确定索引属于哪个子数据集
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx  # 第一个子数据集，索引直接使用
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]  # 转换为子数据集内的索引
        # 更新 GetItemArgs 中的索引，传递给子数据集
        sample_args = args._replace(idx=sample_idx)
        return self.datasets[dataset_idx][sample_args]

    @property
    def fingerprint_version(self) -> str:
        """数据集指纹版本号（所有子数据集必须一致）。

        Returns:
            指纹版本字符串。
        """
        version = self.datasets[0].fingerprint_version
        for dataset in self.datasets:
            if dataset.fingerprint_version != version:
                raise ValueError(
                    "expected all sub datasets to have the same fingerprint_version"
                )
        return version

    @property
    def fingerprint(self) -> str:
        """拼接数据集的指纹，基于所有子数据集指纹的组合 SHA256 哈希。

        Returns:
            SHA256 哈希字符串。
        """
        sha256_hash = hashlib.sha256()
        for dataset in self.datasets:
            if not hasattr(dataset, "fingerprint"):
                raise ValueError(
                    "expected all sub datasets to have fingerprint property"
                )
            sha256_hash.update(dataset.fingerprint.encode())  # 将子数据集指纹加入哈希
        return sha256_hash.hexdigest()

    def _set_latlon_distribution(self) -> None:
        """根据子数据集的地理分布设置拼接数据集的地理分布。"""
        dataset_latlons = []
        for dataset in self.datasets:
            dataset_latlons.append(dataset.latlon_distribution)
        self.latlon_distribution = np.concatenate(dataset_latlons, axis=0)  # 拼接所有子数据集的地理分布

    def prepare(self) -> None:
        """准备数据集。

        子数据集应在初始化 OlmoEarthConcatDataset 之前已准备完毕
        （否则它们没有定义的长度），但此处仍调用 prepare 以防万一。
        """
        for dataset in self.datasets:
            dataset.prepare()  # 确保每个子数据集已准备

        # 计算地理分布属性（某些回调需要）
        self._set_latlon_distribution()

        # 设置训练模态属性（DataLoader 需要访问），所有子数据集必须一致
        self.training_modalities = self.datasets[0].training_modalities
        for dataset in self.datasets:
            if self.training_modalities != dataset.training_modalities:
                raise ValueError(
                    "expected all sub datasets to have same training modalities"
                )


@dataclass
class OlmoEarthConcatDatasetConfig(Config):
    """OlmoEarthConcatDataset 的配置类。

    属性:
        dataset_configs: 子数据集配置列表。
        dataset_percentage: 可选的数据集百分比覆盖（应用于所有子数据集）。
        seed: 可选的随机种子覆盖（应用于所有子数据集）。
    """

    dataset_configs: list[Config]

    # 可选的子数据集覆盖参数
    dataset_percentage: float | None = None
    seed: int | None = None

    def validate(self) -> None:
        """验证配置，确保至少有一个子数据集配置。"""
        if len(self.dataset_configs) == 0:
            raise ValueError("at least one dataset config must be provided")

    def build(self) -> OlmoEarthConcatDataset:
        """构建拼接数据集。

        依次构建每个子数据集，准备后传入 OlmoEarthConcatDataset。

        Returns:
            配置好的 OlmoEarthConcatDataset 实例。
        """
        self.validate()
        logging.info(f"concatenating {len(self.dataset_configs)} sub datasets")
        datasets: list[Dataset] = []
        for dataset_config in self.dataset_configs:
            # 应用覆盖参数
            if self.dataset_percentage is not None:
                dataset_config.dataset_percentage = self.dataset_percentage
            if self.seed is not None:
                dataset_config.seed = self.seed
            dataset = dataset_config.build()
            # 子数据集必须在传入 OlmoEarthConcatDataset 之前准备，以便有定义的长度
            dataset.prepare()
            datasets.append(dataset)
        return OlmoEarthConcatDataset(datasets)


# 向后兼容的废弃别名
HeliosConcatDataset = _deprecated_class_alias(
    OlmoEarthConcatDataset, "helios.data.concat.HeliosConcatDataset"
)
HeliosConcatDatasetConfig = _deprecated_class_alias(
    OlmoEarthConcatDatasetConfig, "helios.data.concat.HeliosConcatDatasetConfig"
)
