"""
OlmoEarth Pretrain 数据集模块。

本模块实现了基于 H5 文件格式的 OlmoEarthDataset 数据集类，用于加载和预处理
多模态地球观测数据。核心功能包括：
- 从 H5 文件中读取多模态栅格数据
- 处理缺失模态和缺失时间步的填充
- 数据归一化
- 支持默认矩形裁剪（subset_sample_default）和 CutMix 裁剪（subset_sample_cutmix）
- NDVI 指数计算
- 数据集指纹生成用于版本控制

主要类：
- OlmoEarthDataset: 核心数据集类，继承自 torch.utils.data.Dataset
- OlmoEarthDatasetConfig: 数据集配置类
- GetItemArgs: __getitem__ 方法的参数命名元组
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import time
from dataclasses import dataclass
from typing import Any, NamedTuple

import h5py

# hdf5 plugin 需要导入以解压某些压缩类型的数据
import hdf5plugin  # noqa: F401
import numpy as np
import pandas as pd
from olmo_core.data.utils import get_rng
from torch.utils.data import Dataset
from upath import UPath

from olmoearth_pretrain._compat import (
    deprecated_class_alias as _deprecated_class_alias,
)
from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import (
    MAX_SEQUENCE_LENGTH,
    MISSING_VALUE,
    Modality,
    ModalitySpec,
)
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.dataset.convert_to_h5py import ConvertToH5py
from olmoearth_pretrain.datatypes import (
    OlmoEarthSample,
)
from olmoearth_pretrain.nn.tokenization import TokenizationConfig
from olmoearth_pretrain.types import ArrayTensor

logger = logging.getLogger(__name__)


# =============================================================================
# 子采样函数（Subsetting Functions）
# =============================================================================


def _get_max_t_within_token_budget(
    sample: OlmoEarthSample,
    h_w_p: int,
    max_tokens_per_instance: int,
    tokenization_config: TokenizationConfig | None = None,
) -> int:
    """在 token 预算内计算允许的最大时间步数。

    给定采样的 h_w_p（高度和宽度方向上的 token 数），
    返回在 max_tokens 预算内允许的最大时间步数 t，
    使得 patchify 后的 OlmoEarthSample 的 token 总数不超过 max_tokens。

    本函数假设采用 (H, W, T=1 patchifying) 的分块方式。

    Args:
        sample: 待子采样的 OlmoEarthSample 实例。
        h_w_p: 高度和宽度方向上的 patch 数量。
        max_tokens_per_instance: 每个 instance 的最大 token 预算。
        tokenization_config: 可选的 tokenization 配置，用于自定义波段分组。

    Returns:
        在 token 预算内允许的最大时间步数 t。
    """
    from math import floor

    used_tokens = 0  # 已使用的 token 数（静态模态）
    time_multiply_tokens = 0  # 随时间线性增长的 token 数（多时相模态）
    for attribute in sample.as_dict().keys():
        if attribute in ("timestamps", "latlon"):
            continue  # 时间戳和经纬度不占 token
        modality_spec = Modality.get(attribute)
        num_band_sets = (
            tokenization_config.get_num_bandsets(attribute)
            if tokenization_config is not None
            else modality_spec.num_band_sets
        )
        if modality_spec.is_spacetime_varying:
            # 时空变化模态：token 数 = h_w_p^2 * band_sets数 * t
            time_multiply_tokens += (h_w_p**2) * num_band_sets
        elif modality_spec.is_space_only_varying:
            # 仅空间变化模态：token 数 = h_w_p^2 * band_sets数
            used_tokens += (h_w_p**2) * num_band_sets
        elif modality_spec.is_time_only_varying:
            # 仅时间变化模态：token 数 = band_sets数 * t
            time_multiply_tokens += num_band_sets
        elif modality_spec.is_static_in_space_and_time:
            # 空间和时间均不变模态：token 数 = band_sets数
            used_tokens += num_band_sets
    if time_multiply_tokens == 0:
        return 1  # 没有多时相模态，t 默认为 1
    remaining_tokens = max_tokens_per_instance - used_tokens  # 剩余 token 预算
    max_t_within_budget = remaining_tokens / time_multiply_tokens  # 最大允许 t
    if max_t_within_budget < 1:
        raise ValueError(
            f"patch_size too small for this sample and budget, h_w_p: {h_w_p}, max_tokens: {max_tokens_per_instance}"
        )

    return min(floor(max_t_within_budget), sample.time)  # 取预算和实际时间步的较小值


def get_valid_start_ts(
    missing_timesteps: dict[str, Any], max_t: int, current_length: int
) -> list[int]:
    """获取有效的时间步起始位置列表。

    在子采样时，需要选择一个起始时间步 t，使得从该位置开始的 max_t 个时间步
    都是有效的（非缺失的）。

    Args:
        missing_timesteps: 字典，键为模态名，值为该模态的缺失时间步掩码。
        max_t: 最大时间步数。
        current_length: 当前序列长度。

    Returns:
        有效起始时间步索引的有序列表。
    """
    if current_length > max_t:
        if not missing_timesteps:
            # 无缺失信息时，所有位置均可作为起始
            valid_start_ts = list(range(current_length - max_t + 1))
        else:
            # 有缺失信息时，需要找到所有模态都有有效数据的位置
            start_ts = set()
            for modality in missing_timesteps:
                valid_timesteps = np.flatnonzero(missing_timesteps[modality])
                # 筛选从该起始位置开始的 max_t 个时间步都在有效范围内
                valid_timesteps = valid_timesteps[
                    valid_timesteps + max_t <= current_length
                ]
                start_ts.update(valid_timesteps)
            valid_start_ts = list(start_ts)
    else:
        # 当前序列长度不超过 max_t，只能从位置 0 开始
        valid_start_ts = [0]
    if len(valid_start_ts) == 0:
        logger.warning(
            f"No valid start timesteps found for {missing_timesteps} with max_t {max_t} and current_length {current_length}"
        )
        raise ValueError(
            f"No valid start timesteps found for {missing_timesteps} with max_t {max_t} and current_length {current_length}"
        )
    return sorted(valid_start_ts)


def subset_sample_default(
    sample: OlmoEarthSample,
    patch_size: int,
    max_tokens_per_instance: int | None,
    sampled_hw_p: int,
    current_length: int,
    missing_timesteps_masks: dict[str, Any] | None = None,
    tokenization_config: TokenizationConfig | None = None,
) -> OlmoEarthSample:
    """使用默认矩形裁剪方式对 OlmoEarthSample 进行子采样。

    从样本中随机裁剪一块矩形区域（空间维度）和一段连续时间步（时间维度），
    使得 patchify 后的 token 总数不超过预算。

    Args:
        sample: 待子采样的 OlmoEarthSample 实例。
        patch_size: 当前样本的 patch 大小。
        max_tokens_per_instance: 每个 instance 的 token 预算。若为 None，则不做子采样。
        sampled_hw_p: 高度和宽度方向上的 patch 数量。
        current_length: 当前样本的最大序列长度。
        missing_timesteps_masks: 缺失时间步掩码字典。
        tokenization_config: 可选的 tokenization 配置。

    Returns:
        子采样后的 OlmoEarthSample。
    """
    if max_tokens_per_instance is None:
        return sample  # 无 token 预算限制，不做子采样
    if missing_timesteps_masks is None:
        missing_timesteps_masks = {}

    # 计算 token 预算内允许的最大时间步数
    max_t = _get_max_t_within_token_budget(
        sample, sampled_hw_p, max_tokens_per_instance, tokenization_config
    )
    # 获取有效起始时间步并随机选择一个
    valid_start_ts = get_valid_start_ts(missing_timesteps_masks, max_t, current_length)
    start_t = np.random.choice(valid_start_ts)
    new_data_dict: dict[str, ArrayTensor] = {}

    # 计算采样的空间范围（像素单位）
    sampled_hw = sampled_hw_p * patch_size
    # 随机选择空间起始位置
    start_h = np.random.choice(sample.height - sampled_hw + 1)
    start_w = np.random.choice(sample.width - sampled_hw + 1)

    for attribute, modality in sample.as_dict().items():
        assert modality is not None
        if attribute == "timestamps":
            # 时间戳按时间维度裁剪
            new_data_dict[attribute] = modality[start_t : start_t + max_t]
            continue
        if attribute == "latlon":
            # 经纬度不参与空间裁剪
            new_data_dict[attribute] = modality
            continue
        modality_spec = Modality.get(attribute)
        if modality_spec.is_spacetime_varying:
            # 时空变化模态：裁剪空间和时间维度，注意 image_tile_size_factor 缩放
            new_data_dict[attribute] = modality[
                start_h * modality_spec.image_tile_size_factor : (start_h + sampled_hw)
                * modality_spec.image_tile_size_factor,
                start_w * modality_spec.image_tile_size_factor : (start_w + sampled_hw)
                * modality_spec.image_tile_size_factor,
                start_t : start_t + max_t,
            ]
        elif modality_spec.is_space_only_varying:
            # 仅空间变化模态：只裁剪空间维度
            new_data_dict[attribute] = modality[
                start_h * modality_spec.image_tile_size_factor : (start_h + sampled_hw)
                * modality_spec.image_tile_size_factor,
                start_w * modality_spec.image_tile_size_factor : (start_w + sampled_hw)
                * modality_spec.image_tile_size_factor,
            ]
        elif modality_spec.is_time_only_varying:
            # 仅时间变化模态：只裁剪时间维度
            new_data_dict[attribute] = modality[start_t : start_t + max_t]
        elif modality_spec.is_static_in_space_and_time:
            # 空间和时间均不变的模态：不做裁剪
            new_data_dict[attribute] = modality

    return OlmoEarthSample(**new_data_dict)


def subset_sample_cutmix(
    sample: OlmoEarthSample,
    patch_size: int,
    max_tokens_per_instance: int | None,
    sampled_hw_p: int,
    current_length: int,
    missing_timesteps_masks: dict[str, Any] | None = None,
    tokenization_config: TokenizationConfig | None = None,
) -> OlmoEarthSample:
    """使用 CutMix patch 采样方式对 OlmoEarthSample 进行子采样。

    与默认矩形裁剪不同，CutMix 在空间维度上随机选择不连续的 patch，
    实现更丰富的空间数据增强。

    Args:
        sample: 待子采样的 OlmoEarthSample 实例。
        patch_size: 当前样本的 patch 大小。
        max_tokens_per_instance: 每个 instance 的 token 预算。若为 None，则不做子采样。
        sampled_hw_p: 高度和宽度方向上的 patch 数量。
        current_length: 当前样本的最大序列长度。
        missing_timesteps_masks: 缺失时间步掩码字典。
        tokenization_config: 可选的 tokenization 配置。

    Returns:
        子采样后的 OlmoEarthSample（使用 CutMix 采样）。
    """
    if max_tokens_per_instance is None:
        return sample  # 无 token 预算限制，不做子采样
    if missing_timesteps_masks is None:
        missing_timesteps_masks = {}

    # 计算 token 预算内允许的最大时间步数
    max_t = _get_max_t_within_token_budget(
        sample, sampled_hw_p, max_tokens_per_instance, tokenization_config
    )
    # 获取有效起始时间步并随机选择一个
    valid_start_ts = get_valid_start_ts(missing_timesteps_masks, max_t, current_length)
    start_t = np.random.choice(valid_start_ts)
    new_data_dict: dict[str, ArrayTensor] = {}

    # 计算高度和宽度方向的 patch 数量
    height_p, width_p = sample.height // patch_size, sample.width // patch_size
    # 随机选择不连续的 patch 索引（无放回采样）
    h_p_indices = np.random.choice(height_p, size=sampled_hw_p, replace=False)
    w_p_indices = np.random.choice(width_p, size=sampled_hw_p, replace=False)
    # 将 patch 索引展平为像素索引
    h_indices = [
        i
        for h_p in h_p_indices
        for i in range(h_p * patch_size, (h_p + 1) * patch_size)
    ]
    w_indices = [
        i
        for w_p in w_p_indices
        for i in range(w_p * patch_size, (w_p + 1) * patch_size)
    ]
    # 构建二维网格索引，用于高级索引
    hh, ww = np.meshgrid(h_indices, w_indices, indexing="ij")

    for attribute, modality in sample.as_dict().items():
        assert modality is not None
        if attribute == "timestamps":
            new_data_dict[attribute] = modality[start_t : start_t + max_t]
            continue
        if attribute == "latlon":
            new_data_dict[attribute] = modality
            continue
        modality_spec = Modality.get(attribute)
        if modality_spec.is_spacetime_varying:
            # 使用高级索引选取非连续的 patch
            new_data_dict[attribute] = modality[
                hh * modality_spec.image_tile_size_factor,
                ww * modality_spec.image_tile_size_factor,
                start_t : start_t + max_t,
            ]
        elif modality_spec.is_space_only_varying:
            new_data_dict[attribute] = modality[
                hh * modality_spec.image_tile_size_factor,
                ww * modality_spec.image_tile_size_factor,
            ]
        elif modality_spec.is_time_only_varying:
            new_data_dict[attribute] = modality[start_t : start_t + max_t]
        elif modality_spec.is_static_in_space_and_time:
            new_data_dict[attribute] = modality

    return OlmoEarthSample(**new_data_dict)


class GetItemArgs(NamedTuple):
    """OlmoEarthDataset.__getitem__ 方法的参数命名元组。

    属性:
        idx: 样本索引。
        patch_size: patch 大小（像素）。
        sampled_hw_p: 采样的高度和宽度方向上的 patch 数量。
        token_budget: 可选的 token 预算限制。
        tokenization_config: 可选的 tokenization 配置。
    """

    idx: int
    patch_size: int
    sampled_hw_p: int
    token_budget: int | None = None
    tokenization_config: TokenizationConfig | None = None


# TODO: training_modalities 应该是 str 还是 modality_spec？
class OlmoEarthDataset(Dataset):
    """OlmoEarth Pretrain 数据集类，基于 H5 文件格式加载多模态地球观测数据。

    该数据集支持：
    - 从 H5 文件中读取多种模态的栅格数据
    - 处理缺失模态和缺失时间步的填充
    - 数据归一化（预定义和计算两种策略）
    - 支持默认矩形裁剪和 CutMix 裁剪
    - NDVI 指数计算
    - 数据集指纹生成用于版本控制
    - 可选的数据缓存和读取限速

    关键属性:
        h5py_dir: H5 文件目录路径
        training_modalities: 训练使用的模态列表
        dtype: 数据类型
        normalize: 是否应用归一化
        sample_indices: 过滤后的样本索引数组
        latlon_distribution: 样本的地理分布（经纬度数组）

    使用场景:
        用于 OlmoEarth 预训练任务的数据加载，通常与 OlmoEarthDataLoader 配合使用。
    """

    def __init__(
        self,
        h5py_dir: UPath,
        training_modalities: list[str],
        dtype: np.dtype,
        max_sequence_length: int = MAX_SEQUENCE_LENGTH,
        normalize: bool = True,
        cache_dir: UPath | None = None,
        samples_per_sec: float | None = None,
        dataset_percentage: float = 1.0,
        seed: int = 0,
        apply_cutmix: bool = False,
        filter_idx_file: str | None = None,
    ):
        """初始化数据集。

        使用已有的 H5 目录时，设置 h5py_dir 为 H5 目录路径。
        使用原始瓦片目录时，设置 tile_path 为瓦片目录路径，将在训练前的准备步骤中创建 H5 文件。

        来自 OLMo-core 的警告：
            在分布式设置中，确保 work_dir 在所有本地 rank 之间共享，
            并相应设置 fs_local_rank。设置这些字段后，应在做任何其他操作之前
            在主进程中调用 prepare() 方法。

        Args:
            h5py_dir: 包含预处理数据的 H5 目录路径。
            training_modalities: 训练使用的模态名称列表。
            dtype: 数据的 numpy dtype。
            max_sequence_length: 所有时间维度填充到的最大序列长度。
            normalize: 是否对数据应用归一化。
            cache_dir: 可选的本地缓存目录，用于缓存 H5 文件。
            samples_per_sec: 限制每秒读取的样本数（限速），仅在从 h5py_dir 读取时生效。
            dataset_percentage: 使用的数据集百分比（0.0~1.0）。
            seed: 选择数据集百分比时的随机种子。
            apply_cutmix: 是否在子采样时应用 CutMix 增强。
            filter_idx_file: 若非 None，则使用该 numpy 文件中的索引过滤样本。
        """
        self.h5py_dir = h5py_dir
        if not self.h5py_dir.exists():
            raise FileNotFoundError(f"H5PY directory does not exist: {self.h5py_dir}")
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)  # 创建缓存目录

        self.training_modalities = training_modalities

        self.dtype = dtype
        self.normalize = normalize
        self.dataset_percentage = dataset_percentage
        self.seed = seed
        if self.normalize:
            # 初始化两种归一化器：预定义（min-max）和计算（mean-std）
            self.normalizer_predefined = Normalizer(Strategy.PREDEFINED)
            self.normalizer_computed = Normalizer(Strategy.COMPUTED)
        self.max_sequence_length = max_sequence_length

        if samples_per_sec is None:
            self.sec_per_sample = None
        else:
            self.sec_per_sample = 1 / samples_per_sec  # 计算每个样本的读取间隔
        self.last_read_time = time.time()

        self.sample_indices: np.ndarray | None = None  # 准备后设置的样本索引
        self.latlon_distribution: np.ndarray | None = None  # 样本的地理分布
        self.apply_cutmix = apply_cutmix
        self.filter_idx_file = filter_idx_file
        if filter_idx_file is not None:
            # 加载过滤索引文件
            self.indices_to_filter: np.ndarray | None = np.load(filter_idx_file)
            assert isinstance(self.indices_to_filter, np.ndarray), (
                f"Expected filter_idx_file to point to a np.ndarray, got {type(self.indices_to_filter)} instead."
            )
        else:
            self.indices_to_filter = None

    @property
    def fingerprint_version(self) -> str:
        """数据集指纹的版本号，用于版本控制。"""
        return "v0.1"

    @property
    def fingerprint(self) -> str:
        """数据集指纹，可用于识别和比较数据集。

        基于瓦片路径、支持的模态、样本数量和数据类型生成 SHA256 哈希值。

        Returns:
            数据集的 SHA256 哈希字符串。

        Raises:
            RuntimeError: 如果数据集尚未准备。
        """
        if not self.is_dataset_prepared:
            raise RuntimeError("Dataset must be prepared before creating a fingerprint")
        sha256_hash = hashlib.sha256()
        # 从 h5py_dir 路径解析支持的模态信息
        supported_modalities_folder = self.h5py_dir.parent.name
        supported_modalities = supported_modalities_folder.split("_")
        # 将拆分后的模态名重新合并（如 sentinel2 + l2a -> sentinel2_l2a）
        if "l2a" in supported_modalities:
            supported_modalities.remove("l2a")
            supported_modalities.remove("sentinel2")
            supported_modalities.append("sentinel2_l2a")
        if "raster" in supported_modalities:
            supported_modalities.remove("raster")
            supported_modalities.remove("openstreetmap")
            supported_modalities.append("openstreetmap_raster")

        if "naip" in supported_modalities and "10" in supported_modalities:
            supported_modalities.remove("naip")
            supported_modalities.remove("10")
            supported_modalities.append("naip_10")
        if "rgb" in supported_modalities and "2" in supported_modalities and "5" in supported_modalities:
            supported_modalities.remove("rgb")
            supported_modalities.remove("2")
            supported_modalities.remove("5")
            supported_modalities.append("rgb_2_5")
        # 经纬度随每个 h5py 文件保存
        supported_modalities.append("latlon")
        num_samples = int(self.h5py_dir.name)  # 目录名即为样本数量

        tile_path = self.h5py_dir.parent.parent.parent

        if self.filter_idx_file is not None:
            filter_file_string = f",filter_idx_file={self.filter_idx_file}"
        else:
            filter_file_string = ""

        # 基于关键信息生成哈希
        sha256_hash.update(
            f"tile_path={tile_path},"
            f"supported_modalities={sorted(supported_modalities)},"
            f"sample_size={num_samples},"
            f"dtype={self.dtype}"
            f"{filter_file_string}".encode()
        )
        return sha256_hash.hexdigest()

    @property
    def sample_metadata_path(self) -> UPath:
        """获取样本元数据文件的路径。"""
        return self.h5py_dir / ConvertToH5py.sample_metadata_fname

    @property
    def latlon_distribution_path(self) -> UPath:
        """获取经纬度分布文件的路径。"""
        return self.h5py_dir / ConvertToH5py.latlon_distribution_fname

    @property
    def is_dataset_prepared(self) -> bool:
        """检查数据集是否已准备（sample_indices 是否已设置）。"""
        return self.sample_indices is not None

    def _filter_sample_indices_for_training(self) -> None:
        """过滤训练用的样本索引。

        更新 sample_indices 数组，仅保留包含至少一个时空变化训练模态的样本。
        同时根据 filter_idx_file 进一步过滤索引。
        """
        # 读取元数据 CSV
        # TODO: Pandas 无法读取 GCS upaths
        metadata_df = pd.read_csv(str(self.sample_metadata_path))
        logger.info(f"Metadata CSV has {len(metadata_df)} samples")
        logger.info(f"columns: {metadata_df.columns}")

        # 获取不包含任何时空变化训练模态的样本索引，这些样本需要移除
        # 跳过派生模态（ignore_when_parsing=True），因为它们在元数据 CSV 中没有列
        spacetime_varying_training_modalities = [
            modality
            for modality in self.training_modalities
            if Modality.get(modality).is_spacetime_varying
            and not Modality.get(modality).ignore_when_parsing
        ]
        if len(spacetime_varying_training_modalities) == 0:
            raise ValueError(
                "no spacetime varying modalities are specified for training"
            )
        # 找到所有时空变化模态列之和为 0 的行（即无任何训练模态的样本）
        no_spacetime_varying_indices = metadata_df[
            metadata_df[spacetime_varying_training_modalities].sum(axis=1) == 0
        ].index

        # 从样本索引中移除这些无效样本
        logger.info(
            f"Filtering out {len(no_spacetime_varying_indices)} samples without any training modalities"
        )
        self.sample_indices = np.setdiff1d(
            self.sample_indices, no_spacetime_varying_indices
        )
        logger.info(
            f"Filtered {len(no_spacetime_varying_indices)} samples to {self.sample_indices.shape} samples"
        )
        # 如果提供了过滤索引文件，进一步取交集
        if self.indices_to_filter is not None:
            self.sample_indices = np.intersect1d(
                self.sample_indices, self.indices_to_filter
            )

            logger.info(
                f"Intersected {len(self.indices_to_filter)} samples to yield {self.sample_indices.shape} samples"
            )

    def _filter_sample_indices_by_dataset_percentage(self) -> None:
        """根据数据集百分比过滤样本索引。

        当 dataset_percentage < 1.0 时，随机选择一定比例的样本。

        Raises:
            AssertionError: 如果 sample_indices 尚未设置。
        """
        assert self.sample_indices is not None, (
            "Sample indices must be set before filtering by dataset percentage"
        )
        if self.dataset_percentage < 1.0:
            rng = get_rng(self.seed)  # 使用确定性随机数生成器
            num_samples = len(self.sample_indices)
            self.sample_indices = rng.choice(
                self.sample_indices,
                size=int(len(self.sample_indices) * self.dataset_percentage),
                replace=False,  # 无放回采样
            )
            logger.info(
                f"Picked {len(self.sample_indices)} samples from {num_samples} samples"
            )

    def prepare(self) -> None:
        """准备数据集。

        此方法应仅由主进程调用，且应在任何其他进程尝试使用数据集之前执行。
        准备步骤包括：加载地理分布、初始化样本索引、过滤无效样本、
        按百分比采样、更新地理分布。
        """
        logger.info("Preparing dataset...")
        if self.is_dataset_prepared:
            logger.info("Dataset is already prepared")
            return

        num_samples = int(self.h5py_dir.name)  # 从目录名获取样本数量
        self.latlon_distribution = self.get_geographic_distribution()
        self.sample_indices = np.arange(num_samples)  # 初始化所有样本索引
        self._filter_sample_indices_for_training()  # 过滤无效样本
        self._filter_sample_indices_by_dataset_percentage()  # 按百分比采样
        self.latlon_distribution = self.latlon_distribution[self.sample_indices]  # 同步更新地理分布

    def get_geographic_distribution(self) -> np.ndarray:
        """获取数据集的地理分布（经纬度坐标）。

        Returns:
            形状为 (N, 2) 的 numpy 数组，包含 N 个样本的 [纬度, 经度] 坐标。
        """
        if self.latlon_distribution_path.exists():
            with self.latlon_distribution_path.open("rb") as f:
                return np.load(f)

    def __len__(self) -> int:
        """获取数据集的样本数量。

        Raises:
            ValueError: 如果数据集尚未准备。
        """
        if self.sample_indices is None:
            raise ValueError("Dataset is not prepared")
        return self.sample_indices.shape[0]

    def normalize_image(self, modality: ModalitySpec, image: np.ndarray) -> np.ndarray:
        """对图像数据进行归一化。

        优先尝试计算策略（mean-std），若失败则回退到预定义策略（min-max）。

        Args:
            modality: 模态规格。
            image: 待归一化的图像数据。

        Returns:
            归一化后的图像数据。
        """
        # TODO: 可以后续将模态归一化策略设为可配置
        try:
            return self.normalizer_computed.normalize(modality, image)
        except Exception:
            return self.normalizer_predefined.normalize(modality, image)

    def _compute_ndvi(
        self,
        s2_data: np.ndarray,
        missing_modalities: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        """从原始 Sentinel-2 L2A 波段计算 NDVI（归一化植被指数）。

        NDVI = (NIR - Red) / (NIR + Red)，其中 NIR=B08（索引 3），Red=B04（索引 2）。
        如果某个像素的 Red 或 NIR 波段为 MISSING_VALUE，则 NDVI 也设为 MISSING_VALUE。

        Args:
            s2_data: 原始（未归一化）的 S2 L2A 数据，形状为 [H, W, T, C]。
            missing_modalities: 完全缺失的模态列表。

        Returns:
            元组：(ndvi 数组 [H, W, T, 1], 更新后的 missing_modalities 列表)。
        """
        s2_band_order = Modality.SENTINEL2_L2A.band_order
        red = s2_data[..., s2_band_order.index("B04")]  # 红光波段
        nir = s2_data[..., s2_band_order.index("B08")]  # 近红外波段

        # 标记缺失像素
        missing = (red == MISSING_VALUE) | (nir == MISSING_VALUE)

        # 安全计算 NDVI，避免除零
        denom = nir + red
        safe_denom = np.where(np.abs(denom) < 1e-10, 1.0, denom)  # 除零保护
        ndvi = (nir - red) / safe_denom
        ndvi = np.where(np.abs(denom) < 1e-10, 0.0, ndvi)  # 分母为零时 NDVI 设为 0
        ndvi = np.where(missing, MISSING_VALUE, ndvi)  # 缺失像素恢复为 MISSING_VALUE

        # 从 missing_modalities 中移除 "ndvi"，因为已计算
        updated_missing = [m for m in missing_modalities if m != "ndvi"]
        return ndvi[..., np.newaxis].astype(self.dtype), updated_missing

    def _fill_missing_timesteps(
        self,
        modality_data: np.ndarray,
        missing_timestep_mask: np.ndarray,
    ) -> np.ndarray:
        """用缺失值填充缺失的时间步。

        将模态数据的时间维度扩展到 max_sequence_length，
        仅在有效时间步位置填入实际数据，其余位置填入 MISSING_VALUE。

        Args:
            modality_data: 原始模态数据，形状为 [H, W, T, C]。
            missing_timestep_mask: 布尔掩码，True 表示该时间步有效。

        Returns:
            填充后的模态数据，形状为 [H, W, max_sequence_length, C]。
        """
        # 转换为适当的数据类型以防止缺失值溢出
        modality_data = modality_data.astype(self.dtype)
        h, w, t, c = modality_data.shape

        # 创建全为 MISSING_VALUE 的完整时间步数组
        full_timesteps_data = np.full(
            (h, w, self.max_sequence_length, c),
            MISSING_VALUE,
            dtype=self.dtype,
        )

        # 将有效数据复制到对应的时间步位置
        present_indices = np.where(missing_timestep_mask)[0]
        num_to_copy = min(len(present_indices), t)
        if num_to_copy > 0:
            full_timesteps_data[:, :, present_indices[:num_to_copy], :] = modality_data[
                :, :, :num_to_copy, :
            ]

        return full_timesteps_data

    def _fill_missing_modality(
        self, modality: str, height: int | None, width: int | None, time: int
    ) -> np.ndarray:
        """用缺失值填充整个模态数组。

        当某个模态完全缺失时，创建一个全为 MISSING_VALUE 的数组来占位。

        Args:
            modality: 模态名称。
            height: 空间高度（像素）。
            width: 空间宽度（像素）。
            time: 时间步数。

        Returns:
            全为 MISSING_VALUE 的模态数组。
        """
        expected_shape = OlmoEarthSample.compute_expected_shape(
            modality, height, width, time
        )
        logger.debug(f"Filling {modality} with shape {expected_shape}")
        return np.full(
            expected_shape,
            fill_value=MISSING_VALUE,
            dtype=self.dtype,
        )

    @staticmethod
    def extract_hwt_from_sample_dict(
        sample_dict: dict[str, Any],
    ) -> tuple[int, int, int]:
        """从样本字典中提取高度（h）、宽度（w）和时间步数（t）。

        遍历样本字典中的模态数据，找到第一个空间模态来获取 h 和 w，
        并从 timestamps 中获取 t。

        Args:
            sample_dict: 包含各模态数据和 timestamps 的字典。

        Returns:
            元组 (height, width, time)。

        Raises:
            ValueError: 如果样本字典中没有空间模态。
        """
        time = sample_dict["timestamps"].shape[0]
        for mod_name, mod_data in sample_dict.items():
            if mod_name == "timestamps":
                continue
            mod_spec = Modality.get(mod_name)
            if mod_spec.is_spatial and mod_data is not None:
                # 形状为 (H, W, T, C)，无 batch 维度
                height = mod_data.shape[0] // mod_spec.image_tile_size_factor
                width = mod_data.shape[1] // mod_spec.image_tile_size_factor
                return height, width, time
        raise ValueError("Expected sample dict to have at least one spatial modality")

    def fill_sample_with_missing_values(
        self, sample_dict: dict[str, Any], missing_timesteps_masks: dict[str, Any]
    ) -> tuple[OlmoEarthSample, list[str]]:
        """用缺失值填充样本中缺失的模态和时间步。

        遍历所有训练模态：
        - 对于完全缺失的模态，用全 MISSING_VALUE 数组填充
        - 对于部分时间步缺失的模态，用 MISSING_VALUE 填充缺失时间步

        Args:
            sample_dict: 包含各模态数据的字典。
            missing_timesteps_masks: 缺失时间步掩码字典，True 表示有效。

        Returns:
            元组：(填充后的 OlmoEarthSample, 缺失模态名称列表)。
        """
        assert sample_dict["timestamps"].shape[0] == self.max_sequence_length, (
            f"Timestamps shape {sample_dict['timestamps'].shape[0]} does not match max_sequence_length {self.max_sequence_length}"
        )
        missing_modalities = []

        height, width, time = self.extract_hwt_from_sample_dict(sample_dict)

        for modality in self.training_modalities:
            # 模态完全缺失：用 MISSING_VALUE 填充
            if modality not in sample_dict.keys():
                logger.debug(f"Filling {modality} with missing values")
                sample_dict[modality] = self._fill_missing_modality(
                    modality, height, width, time
                )
                missing_modalities.append(modality)
                continue

            # 多时相模态：处理缺失时间步
            # missing_timesteps_masks 中 True 表示有效，False 表示缺失
            if modality in missing_timesteps_masks:
                mask = missing_timesteps_masks[modality]
                modality_data = sample_dict[modality]
                # 转换为适当的数据类型以防止缺失值溢出
                modality_data = modality_data.astype(self.dtype)

                # 如果存在缺失时间步或时间步数不足，用 MISSING_VALUE 填充
                has_missing_timesteps = (
                    not np.all(mask) or len(mask) < self.max_sequence_length
                )
                if has_missing_timesteps:
                    modality_data = self._fill_missing_timesteps(modality_data, mask)
                # 更新样本字典
                sample_dict[modality] = modality_data
        return OlmoEarthSample(**sample_dict), missing_modalities

    def _pad_timestamps(
        self, sample_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], int]:
        """将时间戳填充到 max_sequence_length。

        如果当前时间步数不足，则在末尾复制最后一个时间步进行填充（edge padding）。

        Args:
            sample_dict: 包含 timestamps 数据的字典。

        Returns:
            元组：(更新后的 sample_dict, 填充前的原始序列长度)。
        """
        timestamps_data = sample_dict["timestamps"]
        current_length = timestamps_data.shape[0]
        if current_length < self.max_sequence_length:
            pad_width = ((0, self.max_sequence_length - current_length), (0, 0))
            # 在末尾用最后一个时间步的副本填充
            padded_timestamps = np.pad(
                timestamps_data, pad_width=pad_width, mode="edge"
            )
            sample_dict["timestamps"] = padded_timestamps
        return sample_dict, current_length

    def _apply_throttling(self) -> None:
        """应用读取限速。

        当从 h5py_dir 读取样本时调用，确保不超过配置的每秒读取速率。
        仅对 h5py_dir 读取生效，缓存读取不限速。
        """
        if self.sec_per_sample is None:
            return  # 无限速要求
        elapsed = time.time() - self.last_read_time
        time_to_sleep = self.sec_per_sample - elapsed
        self.last_read_time = time.time()
        logger.info(f"{elapsed} elapsed since last read, sleeping for {time_to_sleep}")
        if time_to_sleep <= 0:
            return  # 已超过限速间隔，无需等待
        time.sleep(time_to_sleep)

    def read_h5_file(
        self, h5_file_path: UPath
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """读取 H5 文件，返回样本数据和缺失时间步掩码。

        如果配置了缓存目录，会先将文件缓存到本地再读取，以提高后续读取速度。
        缓存使用原子重命名以避免并发问题。

        Args:
            h5_file_path: H5 文件路径。

        Returns:
            元组：(sample_dict 包含各模态数据, missing_timesteps_masks 字典)。
        """
        if self.cache_dir is not None:
            cache_file_path = self.cache_dir / h5_file_path.name
            logger.debug(f"Caching H5 file {h5_file_path} to {cache_file_path}")
            if not cache_file_path.exists():
                self._apply_throttling()  # 从远程读取时应用限速
                # 先复制到临时文件，然后原子重命名以避免并发问题
                tmp_file_path = self.cache_dir / (h5_file_path.name + ".tmp")
                with h5_file_path.open("rb") as src, tmp_file_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                tmp_file_path.rename(cache_file_path)
            h5_file_path = cache_file_path  # 使用缓存文件路径

        else:
            self._apply_throttling()  # 无缓存时直接限速

        sample_dict = {}
        with h5_file_path.open("rb") as f:
            with h5py.File(f, "r") as h5file:
                logger.debug(
                    f"Reading h5 file {h5_file_path} with keys {h5file.keys()}"
                )
                # 读取训练模态的数据和时间戳
                sample_dict = {
                    k: v[()]
                    for k, v in h5file.items()
                    if k in self.training_modalities
                    # TODO: 修复浮动字符串问题
                    or k in ["timestamps"]
                }

                # 读取缺失时间步掩码
                if (
                    missing_mask_group_name
                    := ConvertToH5py.missing_timesteps_mask_group_name
                ) in h5file:
                    missing_timesteps_masks = {
                        k: v[()]
                        for k, v in h5file[missing_mask_group_name].items()
                        if k in self.training_modalities
                    }
                else:
                    # 兼容旧版本：如果文件中不存在掩码组，设为空字典
                    missing_timesteps_masks = {}
        return sample_dict, missing_timesteps_masks

    def _get_h5_file_path(self, index: int) -> UPath:
        """根据索引获取 H5 文件路径。"""
        return self.h5py_dir / ConvertToH5py.sample_file_pattern.format(index=index)

    @staticmethod
    def _crop_timestamps_and_masks(
        timestamps: np.ndarray, missing_timesteps_masks: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """将时间戳和掩码裁剪到现存模态的首尾有效时间步之间。

        Args:
            timestamps: 时间戳数组。
            missing_timesteps_masks: 缺失时间步掩码字典。

        Returns:
            元组：(裁剪后的 timestamps, 裁剪后的 missing_timesteps_masks)。
        """
        # 假设 missing_timesteps_masks 已经过滤为仅包含训练模态
        if not missing_timesteps_masks:
            first_valid_timestep = 0
            last_valid_timestep = MAX_SEQUENCE_LENGTH
        else:
            # 找到所有模态中最早和最晚的有效时间步
            first_valid_timestep = MAX_SEQUENCE_LENGTH
            last_valid_timestep = 0
            for timestep_mask in missing_timesteps_masks.values():
                valid_timesteps = np.where(timestep_mask)[0]
                if len(valid_timesteps) > 0:
                    first_valid_timestep = min(first_valid_timestep, valid_timesteps[0])
                    last_valid_timestep = max(last_valid_timestep, valid_timesteps[-1])
        # 裁剪时间戳和掩码
        timestamps = timestamps[first_valid_timestep : last_valid_timestep + 1]
        for modality, timestep_mask in missing_timesteps_masks.items():
            missing_timesteps_masks[modality] = timestep_mask[
                first_valid_timestep : last_valid_timestep + 1
            ]
        return timestamps, missing_timesteps_masks

    def __getitem__(self, args: GetItemArgs) -> tuple[int, OlmoEarthSample]:
        """获取指定索引的样本。

        完整的数据加载流程：
        1. 将 args.idx 映射到过滤后的样本索引
        2. 读取 H5 文件获取原始数据
        3. 裁剪时间戳和掩码到有效范围
        4. 填充时间戳到 max_sequence_length
        5. 用缺失值填充缺失的模态和时间步
        6. 子采样（矩形裁剪或 CutMix）
        7. 计算派生模态（如 NDVI）
        8. 归一化

        Args:
            args: GetItemArgs 命名元组，包含 idx、patch_size、sampled_hw_p 等。

        Returns:
            元组：(patch_size, OlmoEarthSample)。
        """
        if hasattr(self, "sample_indices") and self.sample_indices is not None:
            index = self.sample_indices[args.idx]  # 使用过滤后的索引
        else:
            index = args.idx
        h5_file_path = self._get_h5_file_path(index)

        # 读取 H5 文件
        sample_dict, missing_timesteps_masks = self.read_h5_file(h5_file_path)
        # 裁剪时间戳和掩码
        timestamps, missing_timesteps_masks = self._crop_timestamps_and_masks(
            sample_dict["timestamps"], missing_timesteps_masks
        )
        sample_dict["timestamps"] = timestamps
        # 填充时间戳到 max_sequence_length
        sample_dict, current_length = self._pad_timestamps(sample_dict)
        # 用缺失值填充缺失的模态和时间步（当前耗时约 0.08 秒，可能成为小模型的瓶颈）
        sample, missing_modalities = self.fill_sample_with_missing_values(
            sample_dict, missing_timesteps_masks
        )

        # 子采样
        if self.apply_cutmix:
            subset_sample = subset_sample_cutmix(
                sample,
                patch_size=args.patch_size,
                max_tokens_per_instance=args.token_budget,
                sampled_hw_p=args.sampled_hw_p,
                current_length=current_length,
                missing_timesteps_masks=missing_timesteps_masks,
                tokenization_config=args.tokenization_config,
            )
        else:
            subset_sample = subset_sample_default(
                sample,
                patch_size=args.patch_size,
                max_tokens_per_instance=args.token_budget,
                sampled_hw_p=args.sampled_hw_p,
                current_length=current_length,
                missing_timesteps_masks=missing_timesteps_masks,
                tokenization_config=args.tokenization_config,
            )

        sample_dict = subset_sample.as_dict()

        # 如果请求了 NDVI 且有 S2 L2A 数据，从原始（未归一化）波段计算 NDVI
        if (
            "ndvi" in sample_dict
            and "sentinel2_l2a" in sample_dict
            and "sentinel2_l2a" not in missing_modalities
        ):
            sample_dict["ndvi"], missing_modalities = self._compute_ndvi(
                sample_dict["sentinel2_l2a"], missing_modalities
            )

        if self.normalize:
            for modality_name in sample_dict.keys():
                if modality_name == "timestamps":
                    continue  # 时间戳不归一化
                # 不要归一化缺失模态，否则 MISSING_VALUE 会被归一化
                if modality_name in missing_modalities:
                    logger.debug(
                        f"Skipping normalization for {modality_name} because it is in missing_modalities"
                    )
                    continue
                logger.debug(f"Normalizing {modality_name}")
                modality_data = sample_dict[modality_name]
                # 记录缺失值位置
                missing_mask = modality_data == MISSING_VALUE
                normalized_data = self.normalize_image(
                    Modality.get(modality_name), modality_data
                )
                # 归一化后必须恢复缺失值标记，以便缺失掩码能正确识别
                sample_dict[modality_name] = np.where(
                    missing_mask, modality_data, normalized_data
                ).astype(self.dtype)

        return args.patch_size, OlmoEarthSample(**sample_dict)


@dataclass
class OlmoEarthDatasetConfig(Config):
    """OlmoEarthDataset 的配置类。

    属性:
        h5py_dir: H5 文件目录路径字符串。
        training_modalities: 训练使用的模态名称列表。
        dtype: 数据类型字符串（如 "float32"）。
        normalize: 是否应用归一化。
        cache_dir: 可选的本地缓存目录路径字符串。
        samples_per_sec: 每秒读取样本数的限速。
        dataset_percentage: 使用的数据集百分比。
        seed: 随机种子。
        apply_cutmix: 是否应用 CutMix 增强。
        filter_idx_file: 可选的索引过滤文件路径。
    """

    h5py_dir: str
    training_modalities: list[str]
    dtype: str = "float32"
    normalize: bool = True
    cache_dir: str | None = None
    samples_per_sec: float | None = None
    dataset_percentage: float = 1.0
    seed: int = 0
    apply_cutmix: bool = False
    filter_idx_file: str | None = None

    def get_numpy_dtype(self) -> np.dtype:
        """获取 numpy 数据类型。

        Returns:
            对应的 numpy.dtype。

        Raises:
            ValueError: 如果 dtype 不受支持。
        """
        if self.dtype == "float16":
            return np.float16
        elif self.dtype == "float32":
            return np.float32
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

    def validate(self) -> None:
        """验证配置参数的合法性。

        Raises:
            ValueError: 如果 training_modalities 不是列表。
        """
        if not isinstance(self.training_modalities, list):
            raise ValueError("training_modalities must be a list")

    @property
    def h5py_dir_upath(self) -> UPath:
        """获取 H5 目录的 UPath 对象。"""
        return UPath(self.h5py_dir)

    @property
    def cache_dir_upath(self) -> UPath:
        """获取缓存目录的 UPath 对象。"""
        return UPath(self.cache_dir)

    def build(self) -> OlmoEarthDataset:
        """构建 OlmoEarthDataset 实例。

        Returns:
            配置好的 OlmoEarthDataset 实例。
        """
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs["h5py_dir"] = self.h5py_dir_upath  # 转换为 UPath
        kwargs["cache_dir"] = (
            self.cache_dir_upath if self.cache_dir is not None else None
        )
        kwargs["dtype"] = self.get_numpy_dtype()  # 转换为 numpy dtype
        logger.info(f"OlmoEarthDataset kwargs: {kwargs}")
        return OlmoEarthDataset(**kwargs)


# 向后兼容的废弃别名
HeliosSample = _deprecated_class_alias(
    OlmoEarthSample, "helios.data.dataset.HeliosSample"
)
HeliosDataset = _deprecated_class_alias(
    OlmoEarthDataset, "helios.data.dataset.HeliosDataset"
)
HeliosDatasetConfig = _deprecated_class_alias(
    OlmoEarthDatasetConfig, "helios.data.dataset.HeliosDatasetConfig"
)
