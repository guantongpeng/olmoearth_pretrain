"""GeoTIFF 数据集转换为 H5PY 格式模块。

本模块负责将原始的 GeoTIFF 格式遥感数据集转换为 H5PY 格式的训练数据集。
转换过程包括：解析 CSV 元数据、加载图像、过滤无效样本、创建 H5 文件，
并保存经纬度分布和样本元数据等辅助信息。

主要类:
    ConvertToH5pyConfig: H5 转换的配置类，包含压缩、分块等参数。
    ConvertToH5py: 核心转换器，执行从 GeoTIFF 到 H5 的完整转换流程。

使用场景:
    1. 配置转换参数（支持多种压缩算法、分块策略等）。
    2. 解析原始数据集并过滤无效样本。
    3. 将每个样本的多种模态数据保存到独立的 H5 文件中。
    4. 处理缺失时间步掩码，确保多时相数据的一致性。

参考:
    https://docs.h5py.org/en/stable/high/dataset.html (H5 压缩设置文档)
"""

import json
import logging
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from typing import Any

import h5py
import hdf5plugin
import numpy as np
import pandas as pd
from einops import rearrange
from tqdm import tqdm
from upath import UPath

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import (
    IMAGE_TILE_SIZE,
    SENTINEL1_NODATA,
    YEAR_NUM_TIMESTEPS,
    Modality,
    ModalitySpec,
    TimeSpan,
    get_modality_specs_from_names,
)
from olmoearth_pretrain.data.utils import convert_to_db
from olmoearth_pretrain.dataset.parse import parse_dataset
from olmoearth_pretrain.dataset.sample import (
    ModalityTile,
    SampleInformation,
    image_tiles_to_samples,
    load_image_for_sample,
)

logger = logging.getLogger(__name__)

mp.set_start_method("spawn", force=True)


mp.set_start_method("spawn", force=True)  # 使用 spawn 方式启动多进程，避免 CUDA 和 fork 的兼容性问题


@dataclass
class ConvertToH5pyConfig(Config):
    """H5 格式转换配置类，定义从 GeoTIFF 转换为 H5PY 文件的所有参数。

    关键属性:
        tile_path: 原始瓦片数据目录路径
        supported_modality_names: 需要转换的模态名称列表
        multiprocessed_h5_creation: 是否并行创建 H5 文件
        compression: 压缩算法（None, "gzip", "zstd", "lz4"）
        compression_opts: 压缩级别（gzip 为 0-9）
        shuffle: 是否启用 shuffle 滤波器（仅在启用压缩时有效）
        chunk_options: 分块配置（None=禁用, True=自动, tuple=指定形状）
        tile_size: 瓦片大小，基于 IMAGE_TILE_SIZE
        reserved_cores: 预留的核心数（不用于多进程）
        required_modality_names: 必需的模态列表，缺少这些模态的样本将被跳过

    使用场景:
        通过 build() 方法构建 ConvertToH5py 实例，然后调用 run() 执行转换。

    参考:
        https://docs.h5py.org/en/stable/high/dataset.html
    """

    tile_path: str
    supported_modality_names: list[str]  # List of modality names
    multiprocessed_h5_creation: bool = True
    compression: str | None = None  # Compression algorithm
    compression_opts: int | None = None  # Compression level (0-9 for gzip)
    shuffle: bool | None = None  # Enable shuffle filter (only used with compression)
    chunk_options: tuple | None = (
        None  # Chunking configuration. None: disabled. True: auto (data_item.shape). tuple: specific shape.
    )
    tile_size: int = IMAGE_TILE_SIZE
    # Processes may go to sleep state if we use too many processes
    reserved_cores: int = (
        10  # Number of cores to reserve and not used for multiprocessing
    )
    required_modality_names: list[str] = field(
        default_factory=lambda: list()
    )  # Samples without all of these are skipped

    def build(self) -> "ConvertToH5py":
        """根据配置构建 ConvertToH5py 转换器对象。

        Returns:
            ConvertToH5py: 构建好的 H5 转换器实例
        """
        return ConvertToH5py(
            tile_path=UPath(self.tile_path),
            supported_modalities=get_modality_specs_from_names(
                self.supported_modality_names
            ),
            multiprocessed_h5_creation=self.multiprocessed_h5_creation,
            compression=self.compression,
            compression_opts=self.compression_opts,
            shuffle=self.shuffle,
            chunk_options=self.chunk_options,
            tile_size=self.tile_size,
            reserved_cores=self.reserved_cores,
            required_modalities=get_modality_specs_from_names(
                self.required_modality_names
            ),
        )


class ConvertToH5py:
    """GeoTIFF 到 H5PY 格式转换器。

    该类负责将原始的 GeoTIFF 遥感数据集转换为 H5PY 格式，供训练时高效读取。

    关键属性:
        h5py_folder: H5 数据的输出子目录名称
        latlon_distribution_fname: 经纬度分布文件的文件名
        sample_metadata_fname: 样本元数据 CSV 文件名
        sample_file_pattern: 单个样本 H5 文件的命名模式
        compression_settings_fname: 压缩设置 JSON 文件名
        missing_timesteps_mask_group_name: 缺失时间步掩码的 H5 组名

    使用场景:
        1. 初始化转换器并设置数据路径和模态列表。
        2. 调用 run() 方法执行完整的转换流程。
        3. 转换器会自动过滤无效样本、处理缺失时间步、保存辅助元数据。

    核心流程:
        解析数据集 -> 获取样本 -> 过滤样本 -> 处理样本 -> 创建 H5 数据集
    """

    h5py_folder: str = "h5py_data_w_missing_timesteps"
    latlon_distribution_fname: str = "latlon_distribution.npy"
    sample_metadata_fname: str = "sample_metadata.csv"
    sample_file_pattern: str = "sample_{index}.h5"
    compression_settings_fname: str = "compression_settings.json"
    missing_timesteps_mask_group_name: str = "missing_timesteps_masks"

    def __init__(
        self,
        tile_path: UPath,
        supported_modalities: list[ModalitySpec],
        multiprocessed_sample_processing: bool = True,
        multiprocessed_h5_creation: bool = True,
        compression: str | None = None,
        compression_opts: int | None = None,
        shuffle: bool | None = None,
        chunk_options: tuple | bool | None = None,
        tile_size: int = IMAGE_TILE_SIZE,
        reserved_cores: int = 10,
        required_modalities: list[ModalitySpec] = [],
    ) -> None:
        """初始化 ConvertToH5py 转换器。

        Args:
            tile_path: 瓦片目录路径，包含各模态的 CSV 文件和瓦片数据
            supported_modalities: 需要转换的模态规格列表
            multiprocessed_sample_processing: 是否并行处理样本（去除坏模态）
            multiprocessed_h5_creation: 是否并行创建 H5 文件
            compression: 压缩算法（None, "gzip", "lzf", "szip"）
            compression_opts: 压缩级别（0-9，仅 gzip 使用）
            shuffle: 是否启用 shuffle 滤波器，仅在启用压缩时有效
            chunk_options: 分块配置。
                         None: 禁用分块。
                         True: 自动分块（分块大小匹配数据集形状）。
                         tuple: 指定分块形状。如果 tuple 维度与数据维度不同，
                                会自动调整（用完整维度大小填充或截断）。
            tile_size: 瓦片大小，基于 IMAGE_TILE_SIZE，高分辨率模态如 NAIP 会相应分割
            reserved_cores: 预留的核心数，不用于多进程（防止进程休眠）
            required_modalities: 必需的模态列表，缺少这些模态的样本将被跳过
        """
        self.tile_path = tile_path
        self.supported_modalities = supported_modalities
        logger.info(f"Supported modalities: {self.supported_modalities}")
        self.multiprocessed_sample_processing = multiprocessed_sample_processing
        self.multiprocessed_h5_creation = multiprocessed_h5_creation
        self.compression = compression
        self.compression_opts = compression_opts
        self.shuffle = shuffle
        self.chunk_options = chunk_options
        self.h5py_dir: UPath | None = None
        self.required_modalities = required_modalities
        if IMAGE_TILE_SIZE % tile_size != 0:
            raise ValueError(
                f"Tile size {tile_size} must be a factor of {IMAGE_TILE_SIZE}"
            )
        self.tile_size = tile_size
        # Tile_size_split_factor 是瓦片被分割为子瓦片的因子
        self.num_subtiles_per_dim = IMAGE_TILE_SIZE // tile_size  # 每个维度上的子瓦片数量
        self.num_subtiles = self.num_subtiles_per_dim**2  # 子瓦片总数
        self.reserved_cores = reserved_cores

    @property
    def compression_settings_suffix(self) -> str:
        """获取压缩设置的字符串表示，用于文件夹命名。

        Returns:
            str: 压缩设置的后缀字符串，格式如 "_gzip_3_shuffle"
        """
        compression_str = ""
        if self.compression is not None:
            compression_str = f"_{self.compression}"
        if self.compression_opts is not None:
            compression_str += f"_{self.compression_opts}"
        if self.shuffle is not None:
            compression_str += "_shuffle"
        return compression_str

    @property
    def image_tile_size_suffix(self) -> str:
        """获取图像瓦片大小的字符串表示，格式为 "_{tile_size}_x_{num_subtiles}"。

        Returns:
            str: 图像瓦片大小后缀字符串
        """
        return f"_{self.tile_size}_x_{self.num_subtiles}"

    def _get_samples(self) -> list[SampleInformation]:
        """从原始数据集（图像瓦片目录）解析获取样本信息。

        核心逻辑:
            1. 使用 parse_dataset() 解析 CSV 元数据
            2. 使用 image_tiles_to_samples() 将瓦片信息转换为样本列表
            3. 记录样本模态分布统计

        Returns:
            list[SampleInformation]: 解析得到的样本信息列表
        """
        tiles = parse_dataset(self.tile_path, self.supported_modalities)
        samples = image_tiles_to_samples(tiles, self.supported_modalities)
        logger.info(f"Total samples: {len(samples)}")
        logger.info("Distribution of samples before filtering:\n")
        self._log_modality_distribution(samples)
        return samples

    def process_sample_into_h5(
        self, index_sample_tuple: tuple[int, tuple[int, SampleInformation]]
    ) -> None:
        """将单个样本处理并保存为 H5 文件。

        Args:
            index_sample_tuple: 元组，包含 (全局索引, (子瓦片索引, 样本信息))
        """
        i, (sublock_index, sample) = index_sample_tuple
        h5_file_path = self._get_h5_file_path(i)
        # 注释掉存在性检查，因为如果 H5 生成中断后重新运行，
        # 可能会有一些损坏的 H5 文件（部分写入、无效），需要覆盖
        # if h5_file_path.exists():
        #     return
        self._create_h5_file(sample, h5_file_path, sublock_index)

    def create_h5_dataset(self, samples: list[tuple[int, SampleInformation]]) -> None:
        """创建 H5 格式的样本数据集，保存到共享 weka 目录下。

        支持多进程并行创建和单进程顺序创建两种模式。

        Args:
            samples: 样本列表，每个元素为 (子瓦片索引, 样本信息) 的元组
        """
        total_sample_indices = len(samples)

        if self.multiprocessed_h5_creation:
            num_processes = max(1, mp.cpu_count() - self.reserved_cores)
            logger.info(f"Creating H5 dataset using {num_processes} processes")
            with mp.Pool(processes=num_processes) as pool:
                # Process samples in parallel and track progress with tqdm
                _ = list(
                    tqdm(
                        pool.imap(self.process_sample_into_h5, enumerate(samples)),
                        total=total_sample_indices,
                        desc="Creating H5 files",
                    )
                )
        else:
            for i, (sublock_index, sample) in enumerate(samples):
                logger.info(f"Processing sample {i}")
                self.process_sample_into_h5((i, (sublock_index, sample)))

    def save_sample_metadata(
        self, samples: list[tuple[int, SampleInformation]]
    ) -> None:
        """保存样本元数据到 CSV 文件，记录每个样本包含哪些模态。

        Args:
            samples: 样本列表，每个元素为 (子瓦片索引, 样本信息) 的元组
        """
        if self.h5py_dir is None:
            raise ValueError("h5py_dir is not set")
        csv_path = self.h5py_dir / self.sample_metadata_fname
        logger.info(f"Writing metadata CSV to {csv_path}")

        # 创建 DataFrame 存储元数据
        metadata_dict: dict = {
            "sample_index": [],
        }

        # 为每个支持的模态添加列
        for modality in self.supported_modalities:
            metadata_dict[modality.name] = []

        # 填充 DataFrame 中每个样本的元数据
        for i, (_, sample) in enumerate(samples):
            metadata_dict["sample_index"].append(i)

            # 设置模态存在性（存在为1，不存在为0）
            for modality in self.supported_modalities:
                metadata_dict[modality.name].append(
                    1 if modality in sample.modalities else 0
                )

        # 将 DataFrame 写入 CSV 文件
        df = pd.DataFrame(metadata_dict)
        df.to_csv(csv_path, index=False)

    def _get_h5_file_path(self, index: int) -> UPath:
        """根据样本索引获取对应的 H5 文件路径。

        Args:
            index: 样本索引

        Returns:
            UPath: H5 文件的完整路径
        """
        if self.h5py_dir is None:
            raise ValueError("h5py_dir is not set")
        return self.h5py_dir / self.sample_file_pattern.format(index=index)

    @property
    def latlon_distribution_path(self) -> UPath:
        """获取经纬度分布文件的路径。

        Returns:
            UPath: 经纬度分布 .npy 文件的完整路径
        """
        if self.h5py_dir is None:
            raise ValueError("h5py_dir is not set")
        return self.h5py_dir / self.latlon_distribution_fname

    def save_latlon_distribution(
        self, samples: list[tuple[int, SampleInformation]]
    ) -> None:
        """将样本的经纬度分布保存到 .npy 文件。

        Args:
            samples: 样本列表，每个元素为 (子瓦片索引, 样本信息) 的元组
        """
        logger.info(f"Saving latlon distribution to {self.latlon_distribution_path}")
        latlons = np.array([sample.get_latlon() for _, sample in samples])
        with self.latlon_distribution_path.open("wb") as f:
            np.save(f, latlons)

    def _find_longest_timestamps_array(
        self, spacetime_varying_modalities: dict[ModalitySpec, np.ndarray]
    ) -> np.ndarray:
        """找到具有最多时间步的模态对应的时间戳数组。

        用于处理不同模态时间步数量不一致的情况，以最长的时间戳数组为基准。

        Args:
            spacetime_varying_modalities: 时空变化模态到其时间戳数组的映射

        Returns:
            np.ndarray: 时间步最多的模态的时间戳数组
        """
        return spacetime_varying_modalities[
            max(
                spacetime_varying_modalities,
                key=lambda k: len(spacetime_varying_modalities[k]),
            )
        ]

    def _create_missing_timesteps_masks(
        self,
        spacetime_varying_modalities: dict[ModalitySpec, np.ndarray],
        longest_timestamps_array: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """为每个模态创建缺失时间步的掩码。

        核心逻辑:
            对最长时间戳数组中的每个时间步，检查当前模态是否存在该时间步，
            生成布尔掩码标记缺失的时间步。

        Args:
            spacetime_varying_modalities: 时空变化模态到其时间戳数组的映射
            longest_timestamps_array: 最长的时间戳数组（作为基准）

        Returns:
            dict[str, np.ndarray]: 模态名称到缺失时间步掩码数组的映射
        """
        missing_timesteps_masks_data: dict[str, np.ndarray] = {}
        for mod_spec, mod_timestamps in spacetime_varying_modalities.items():
            # 创建布尔掩码，标记最长时间戳数组中每个时间步在当前模态时间戳中是否存在
            # np.all(..., axis=1) 检查完整的行匹配（日、月、年）
            # np.any(...) 检查当前模态的任何行是否与 longest_ts 匹配
            mask = np.array(
                [
                    np.any(np.all(longest_ts == mod_timestamps, axis=1))
                    for longest_ts in longest_timestamps_array
                ],
                dtype=bool,
            )
            missing_timesteps_masks_data[mod_spec.name] = mask
        return missing_timesteps_masks_data

    def _remove_bad_modalities_from_sample(
        self, sample: SampleInformation
    ) -> SampleInformation:
        """从样本中移除有问题的模态数据。

        移除规则:
            1. 包含 NaN 值的模态
            2. ERA5 和 OSM Raster 全为零的模态
            3. Sentinel-1 包含 nodata 值的模态

        Args:
            sample: 待处理的样本信息

        Returns:
            SampleInformation: 移除坏模态后的样本信息
        """
        modalities_to_remove = set()
        for modality in sample.modalities:
            sample_modality = sample.modalities[modality]
            image = self.load_sample(sample_modality, sample)
            # 移除包含 NaN 的模态
            if np.any(np.isnan(image)):
                logger.warning(
                    f"Image for modality {modality.name} contains NaN values, removing this modality"
                )
                modalities_to_remove.add(modality)
            # 移除 ERA5 和 OSM Raster 全为零的数据
            if (
                modality == Modality.ERA5_10
                or modality == Modality.OPENSTREETMAP_RASTER
            ) and np.all(image == 0):
                logger.warning(
                    f"Image for modality {modality.name} is all zeros, removing this modality"
                )
                modalities_to_remove.add(modality)
            # 移除 Sentinel-1 包含 nodata 值的数据
            if modality == Modality.SENTINEL1 and np.any(image == SENTINEL1_NODATA):
                logger.warning(
                    f"Image for modality {modality.name} contains nodata values, removing this modality"
                )
                modalities_to_remove.add(modality)
        for modality in modalities_to_remove:
            del sample.modalities[modality]
        return sample

    def _process_samples(
        self, samples: list[SampleInformation]
    ) -> list[SampleInformation]:
        """在过滤之前处理样本，移除有问题的模态。

        支持多进程并行处理和单进程顺序处理两种模式。

        Args:
            samples: 待处理的样本列表

        Returns:
            list[SampleInformation]: 处理后的样本列表
        """
        total_sample_indices = len(samples)
        if self.multiprocessed_sample_processing:
            num_processes = max(1, mp.cpu_count() - self.reserved_cores)
            logger.info(f"Processing samples using {num_processes} processes")
            with mp.Pool(processes=num_processes) as pool:
                # 并行处理样本并使用 tqdm 跟踪进度
                processed_samples = list(
                    tqdm(
                        pool.imap(self._remove_bad_modalities_from_sample, samples),
                        total=total_sample_indices,
                        desc="Processing samples",
                    )
                )
        else:
            processed_samples = []
            for i, sample in enumerate(samples):
                logger.info(f"Processing sample {i}")
                processed_sample = self._remove_bad_modalities_from_sample(sample)
                processed_samples.append(processed_sample)
        return processed_samples

    def _create_h5_file(
        self, sample: SampleInformation, h5_file_path: UPath, sublock_index: int
    ) -> dict[str, Any]:
        """创建单个样本的 H5 文件。

        核心逻辑:
            1. 加载样本的所有模态图像数据
            2. 处理时空变化模态的时间戳对齐
            3. 创建缺失时间步掩码
            4. 对空间模态进行子瓦片切片
            5. 将数据写入 H5 文件，支持多种压缩算法

        Args:
            sample: 样本信息
            h5_file_path: H5 文件输出路径
            sublock_index: 子瓦片索引，用于从大瓦片中提取子区域

        Returns:
            dict[str, Any]: 包含样本所有模态数据的字典
        """
        sample_dict = {}
        sample_dict["latlon"] = sample.get_latlon().astype(np.float32)
        multi_temporal_timestamps_dict = sample.get_timestamps()

        # 从时空变化模态中计算最长时间戳
        # 处理 ERA5 始终有完整 12 个月但其他模态可能少得多的情况
        spacetime_varying_modalities = {
            modality: timestamps
            for modality, timestamps in multi_temporal_timestamps_dict.items()
            if modality.is_spacetime_varying
        }
        # 注意：使用最长时间戳数组，所有模态都被截断到该范围
        # 该范围之外的数据被视为缺失
        longest_timestamps_array = self._find_longest_timestamps_array(
            spacetime_varying_modalities
        )
        missing_timesteps_masks_data = self._create_missing_timesteps_masks(
            spacetime_varying_modalities, longest_timestamps_array
        )

        sample_dict["timestamps"] = longest_timestamps_array

        # 加载样本中所有模态的图像数据
        for modality in sample.modalities:
            sample_modality = sample.modalities[modality]
            image = self.load_sample(sample_modality, sample)

            if modality == Modality.SENTINEL1:
                # 将 Sentinel-1 数据转换为分贝(dB)值
                image = convert_to_db(image)

            if modality.is_spatial:
                # 计算网格的行列索引，用于子瓦片切片
                if image.shape[0] != image.shape[1]:
                    raise ValueError("Expected image width to match image height")
                if image.shape[0] % self.num_subtiles_per_dim != 0:
                    raise ValueError(
                        f"Got image size {image.shape[0]} which is not multiple of subtile count {self.num_subtiles_per_dim}"
                    )
                tile_size = image.shape[0] // self.num_subtiles_per_dim
                row = (sublock_index // self.num_subtiles_per_dim) * tile_size
                col = (sublock_index % self.num_subtiles_per_dim) * tile_size
                logger.info(f"Sublock index: {sublock_index}, row: {row}, col: {col}")
                logger.info(f"Image shape: {image.shape}")
                image = image[row : row + tile_size, col : col + tile_size, ...]
                logger.info(f"Image shape after slicing: {image.shape}")

            sample_dict[modality.name] = image

        # w+b 模式打开，因为有时需要读取元数据以确定不同的分块/压缩设置
        with h5_file_path.open("w+b") as f:
            with h5py.File(f, "w") as h5file:
                # 写入经纬度、时间戳和模态图像数据集
                for item_name, data_item in sample_dict.items():
                    logger.info(
                        f"Writing item {item_name} to h5 file path {h5_file_path}"
                    )
                    # 创建数据集，可选压缩
                    create_kwargs: dict[str, Any] = {}
                    # 也许应该将压缩逻辑移到单独的类中以便使用
                    if self.compression is not None:
                        # Gzip 是 h5py 原生支持的压缩算法
                        if self.compression == "gzip":
                            create_kwargs["compression"] = self.compression
                            if self.compression_opts is not None:
                                create_kwargs["compression_opts"] = (
                                    self.compression_opts
                                )
                            if self.shuffle is not None:
                                create_kwargs["shuffle"] = self.shuffle
                        # 对于其他压缩算法，切换到 hdf5plugin
                        elif self.compression == "zstd":
                            create_kwargs["compression"] = hdf5plugin.Zstd(
                                clevel=self.compression_opts
                            )
                        elif self.compression == "lz4":
                            create_kwargs["compression"] = hdf5plugin.LZ4(nbytes=0)
                        else:
                            raise ValueError(
                                f"Unsupported compression: {self.compression}"
                            )

                        # 根据配置应用分块策略
                        if self.chunk_options is True:  # 自动分块
                            create_kwargs["chunks"] = True  # need to configure
                        elif (
                            isinstance(self.chunk_options, tuple)
                            and self.chunk_options is not None
                        ):  # 指定分块形状
                            num_data_dims = len(data_item.shape)
                            final_chunks_list = []
                            for i in range(num_data_dims):
                                if i < len(self.chunk_options):
                                    final_chunks_list.append(self.chunk_options[i])
                                else:
                                    # 如果 chunk_options 更短，用完整数据维度大小填充
                                    final_chunks_list.append(data_item.shape[i])
                            logger.info(f"Final chunks list: {final_chunks_list}")
                            create_kwargs["chunks"] = tuple(final_chunks_list)
                        else:
                            logger.info(
                                f"Chunk options: using chunk size {data_item.shape}"
                            )
                            create_kwargs["chunks"] = (
                                data_item.shape
                            )  # 使用数据集项形状作为分块大小，实际上不分块

                    # 为每个数据项创建 H5 数据集
                    logger.info(
                        f"Creating dataset for {item_name} with kwargs: {create_kwargs}"
                    )
                    h5file.create_dataset(item_name, data=data_item, **create_kwargs)

                # 将缺失时间步掩码存储到专用 H5 组中
                if missing_timesteps_masks_data:
                    masks_group = h5file.create_group(
                        self.missing_timesteps_mask_group_name
                    )
                    for mod_name, mask_array in missing_timesteps_masks_data.items():
                        logger.info(
                            f"Writing missing timesteps mask for {mod_name} to h5 file path {h5_file_path}"
                        )
                        # 布尔掩码通常不需要压缩/shuffle
                        masks_group.create_dataset(mod_name, data=mask_array)
        return sample_dict

    def _log_modality_distribution(self, samples: list[SampleInformation]) -> None:
        """记录样本的模态分布统计信息，包括单个模态和模态组合的分布。

        Args:
            samples: 样本信息列表
        """
        # 记录模态分布
        modality_counts: dict[str, int] = {}
        modality_combinations: dict[frozenset[str], int] = {}

        for sample in samples:
            # 统计单个模态出现次数
            for modality in sample.modalities:
                modality_counts[modality.name] = (
                    modality_counts.get(modality.name, 0) + 1
                )

            # 统计模态组合出现次数
            combination = frozenset(m.name for m in sample.modalities)
            modality_combinations[combination] = (
                modality_combinations.get(combination, 0) + 1
            )

        # 记录单个模态计数
        for modality_name, count in modality_counts.items():
            percentage = (count / len(samples)) * 100
            logger.info(
                f"Modality {modality_name}: {count} samples ({percentage:.1f}%)"
            )

        # 记录模态组合统计
        logger.info("\nModality combinations:")
        for combination, count in modality_combinations.items():
            percentage = (count / len(samples)) * 100
            logger.info(
                f"{'+'.join(sorted(combination))}: {count} samples ({percentage:.1f}%)"
            )

    def set_h5py_dir(self, num_samples: int) -> None:
        """设置 H5 数据输出目录，只能设置一次以确保一致性。

        Args:
            num_samples: 数据集中样本的数量
        """
        if self.h5py_dir is not None:
            logger.warning("h5py_dir is already set, ignoring new value")
            return

        required_modalities_suffix = ""
        if self.required_modalities:
            required_modalities_suffix = "_required_" + "_".join(
                sorted([modality.name for modality in self.required_modalities])
            )
        h5py_dir = (
            self.tile_path
            / f"{self.h5py_folder}{self.compression_settings_suffix}{self.image_tile_size_suffix}"
            / (
                "_".join(
                    sorted([modality.name for modality in self.supported_modalities])
                )
                + required_modalities_suffix
            )
            / str(num_samples)
        )
        self.h5py_dir = h5py_dir
        logger.info(f"Setting h5py_dir to {self.h5py_dir}")
        os.makedirs(self.h5py_dir, exist_ok=True)

    @classmethod
    def load_sample(
        cls, sample_modality: ModalityTile, sample: SampleInformation
    ) -> np.ndarray:
        """加载样本的图像数据并调整维度顺序。

        核心逻辑:
            根据图像的维度数量调整维度顺序：
            - 4D: (T, C, H, W) -> (H, W, T, C)（时空变化模态）
            - 3D: (C, H, W) -> (H, W, C)（仅空间变化模态）
            - 2D: (T, C) -> (T, C)（非空间模态，如 ERA5）

        Args:
            sample_modality: 样本的模态瓦片信息
            sample: 样本信息

        Returns:
            np.ndarray: 调整维度顺序后的图像数据
        """
        image = load_image_for_sample(sample_modality, sample)

        if image.ndim == 4:
            modality_data = rearrange(image, "t c h w -> h w t c")
        elif image.ndim == 3:
            modality_data = rearrange(image, "c h w -> h w c")
        elif image.ndim == 2:
            # 已经是正确的形状 (t, c)
            modality_data = image
        else:
            raise ValueError(
                f"Unexpected image shape {image.shape} for modality {sample_modality.modality.name}"
            )
        return modality_data

    def _filter_samples(
        self, samples: list[SampleInformation]
    ) -> list[SampleInformation]:
        """过滤样本以适配 OlmoEarthSample 格式要求。

        过滤规则:
            1. 移除坏模态（NaN、全零 ERA5/OSM、含 nodata 的 S1）
            2. 跳过包含不支持模态的样本
            3. 跳过缺少必需模态的样本
            4. 跳过非年度频率数据的样本
            5. 跳过没有时空变化模态的样本
            6. 跳过最长时间步少于 12 的样本（需与 ERA5 对齐）

        Args:
            samples: 待过滤的样本列表

        Returns:
            list[SampleInformation]: 过滤后的样本列表
        """
        logger.info(f"Number of samples before filtering: {len(samples)}")

        # 移除样本中的坏模态
        # 确保元数据与 H5 中保存的实际数据一致
        processed_samples = self._process_samples(samples)
        filtered_samples = []
        for sample in processed_samples:
            if not all(
                modality in self.supported_modalities
                for modality in sample.modalities
                # TODO: clarify usage of ignore when parsing
                if not modality.ignore_when_parsing
            ):
                logger.info("Skipping sample because it has unsupported modalities")
                continue
            if any(
                modality not in sample.modalities
                for modality in self.required_modalities
            ):
                logger.info(
                    "Skipping sample because it does not have a required modality"
                )
                continue

            if sample.time_span != TimeSpan.YEAR:
                logger.debug(
                    "Skipping sample because it is not the yearly frequency data"
                )
                continue

            multi_temporal_timestamps_dict = sample.get_timestamps()
            spacetime_varying_modalities = {
                modality: timestamps
                for modality, timestamps in multi_temporal_timestamps_dict.items()
                if modality.is_spacetime_varying
            }

            if len(spacetime_varying_modalities) == 0:
                logger.info(
                    "Skipping sample because it has no spacetime varying modalities"
                )
                continue

            # 为了与 ERA5 对齐（要么缺失即海洋区域，要么有 12 个时间步），
            # 我们要求至少一个时空变化模态有 12 个时间步
            # 例如，Presto 数据集中只有 43 个样本不满足此要求
            longest_timestamps_array = self._find_longest_timestamps_array(
                spacetime_varying_modalities
            )
            if len(longest_timestamps_array) < YEAR_NUM_TIMESTEPS:
                logger.info(
                    "Skipping sample because it does not have at least 12 timesteps"
                )
                continue

            filtered_samples.append(sample)

        logger.info("Distribution of samples after filtering:")
        self._log_modality_distribution(filtered_samples)
        return filtered_samples

    def get_and_filter_samples(self) -> list[SampleInformation]:
        """获取并过滤样本。

        解析 CSV 文件、加载图像、过滤样本以适配 OlmoEarthSample 格式。

        Returns:
            list[SampleInformation]: 过滤后的样本信息列表
        """
        samples = self._get_samples()
        return self._filter_samples(samples)

    def save_compression_settings(self) -> None:
        """将压缩设置保存到 JSON 文件。"""
        if self.h5py_dir is None:
            raise ValueError("h5py_dir is not set")

        settings = {
            "compression": (
                str(self.compression) if self.compression is not None else None
            ),
            "compression_opts": (
                int(self.compression_opts)
                if self.compression_opts is not None
                else None
            ),
            "shuffle": bool(self.shuffle) if self.shuffle is not None else None,
        }

        settings_path = self.h5py_dir / self.compression_settings_fname
        logger.info(f"Saving compression settings to {settings_path}")
        with settings_path.open("w") as f:
            json.dump(settings, f, indent=2)

    def prepare_h5_dataset(self, samples: list[SampleInformation]) -> None:
        """准备 H5 数据集，包括设置输出目录、保存元数据和创建 H5 文件。

        Args:
            samples: 过滤后的样本信息列表
        """
        tuples = []
        for sample in samples:
            for j in range(self.num_subtiles):
                tuples.append((j, sample))
        self.set_h5py_dir(len(tuples))
        self.save_compression_settings()  # 在创建数据之前保存设置
        self.save_sample_metadata(tuples)
        self.save_latlon_distribution(tuples)
        logger.info("Attempting to create H5 files may take some time...")
        self.create_h5_dataset(tuples)

    def run(self) -> None:
        """执行完整的 GeoTIFF 到 H5PY 的转换流程。"""
        samples = self.get_and_filter_samples()
        self.prepare_h5_dataset(samples)
