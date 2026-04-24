"""
OlmoEarth Pretrain 数据加载器模块。

本模块实现了 OlmoEarthDataLoader，用于在分布式训练环境中高效加载多模态地球观测数据。
核心功能包括：
- 支持分布式数据并行（DDP）和多种进程上下文
- 全局索引管理和数据重排
- 多种 patch 大小和采样尺寸的动态配置
- 批次级的数据增强（Transform）和掩码策略（Masking）
- 单视图和双视图掩码（用于 Galileo 风格训练）
- 模拟批次生成用于训练前的 dry-run
- 训练状态保存与恢复

主要类：
- OlmoEarthDataLoader: 核心数据加载器
- OlmoEarthDataLoaderConfig: 数据加载器配置类
- _IterableDatasetWrapper: 迭代数据集包装器
- iter_batched: 批次迭代辅助函数
"""

import functools
import logging
import math
import multiprocessing as mp
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from olmo_core.data.data_loader import DataLoaderBase
from olmo_core.data.utils import get_rng, memmap_to_write
from olmo_core.distributed.utils import (
    barrier,
    get_fs_local_rank,
    get_rank,
    get_world_size,
)
from olmo_core.utils import get_default_device
from torch.utils.data import default_collate
from upath import UPath

from olmoearth_pretrain._compat import deprecated_class_alias as _deprecated_class_alias
from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.collate import (
    collate_double_masked_batched,
    collate_single_masked_batched,
)
from olmoearth_pretrain.data.concat import OlmoEarthConcatDataset
from olmoearth_pretrain.data.constants import IMAGE_TILE_SIZE, Modality
from olmoearth_pretrain.data.dataset import (
    GetItemArgs,
    OlmoEarthDataset,
    OlmoEarthSample,
    subset_sample_default,
)
from olmoearth_pretrain.data.transform import Transform, TransformConfig
from olmoearth_pretrain.nn.tokenization import TokenizationConfig
from olmoearth_pretrain.train.masking import MaskingConfig, MaskingStrategy

logger = logging.getLogger(__name__)


class OlmoEarthDataLoader(DataLoaderBase):
    """OlmoEarth Pretrain 数据加载器。

    适配自 OLMo-core 的 TextDataLoaderBase 和 NumpyDataLoaderBase，
    整合了 DDP、多线程和多进程的核心功能。

    关键属性:
        dataset: 数据集实例（OlmoEarthDataset 或 OlmoEarthConcatDataset）
        min_patch_size / max_patch_size: patch 大小范围
        sampled_hw_p_list: 采样的高度/宽度 patch 数候选列表
        token_budget: 每个 instance 的 token 预算
        transform: 数据增强变换
        masking_strategy / masking_strategy_b: 掩码策略（支持双视图）
        num_masked_views: 掩码视图数（1=单视图，2=双视图）

    使用场景:
        用于 OlmoEarth 预训练任务的数据迭代，通常由 OlmoEarthDataLoaderConfig.build() 构建。
    """

    def __init__(
        self,
        dataset: OlmoEarthDataset | OlmoEarthConcatDataset,
        work_dir: UPath,
        global_batch_size: int,
        min_patch_size: int,
        max_patch_size: int,
        sampled_hw_p_list: list[int],
        token_budget: int | None = None,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        fs_local_rank: int = 0,
        seed: int = 0,
        shuffle: bool = True,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        collator: Callable = default_collate,
        target_device_type: str = "cpu",
        drop_last: bool = True,
        persistent_workers: bool = True,
        multiprocessing_context: str = "spawn",
        num_dataset_repeats_per_epoch: int = 1,
        # Dataloader-side masking
        transform: Transform | None = None,
        masking_strategy: MaskingStrategy | None = None,
        masking_strategy_b: MaskingStrategy | None = None,
        num_masked_views: int = 1,
        tokenization_config: TokenizationConfig | None = None,
    ):
        """初始化 OlmoEarthDataLoader。

        Args:
            dataset: 数据集实例。
            work_dir: 工作目录，用于存储索引文件。
            global_batch_size: 全局批次大小（跨所有 worker）。
            min_patch_size: 训练的最小 patch 大小。
            max_patch_size: 训练的最大 patch 大小。
            sampled_hw_p_list: 可采样的 height/width patch 数候选列表。
            token_budget: 可选的每个 instance 的 token 预算。
            dp_world_size: 数据并行的 world size。
            dp_rank: 数据并行的 rank。
            fs_local_rank: 文件系统本地 rank。
            seed: 随机种子。
            shuffle: 是否打乱数据。
            num_workers: DataLoader 的 worker 数量。
            prefetch_factor: DataLoader 的预取因子。
            collator: 批次整理函数。
            target_device_type: 目标设备类型（"cpu" 或 "cuda"）。
            drop_last: 是否丢弃最后一个不完整批次。
            persistent_workers: worker 是否在 epoch 之间保持存活。
            multiprocessing_context: 多进程上下文（"spawn" 或 "forkserver"）。
            num_dataset_repeats_per_epoch: 每个 epoch 中数据集重复次数。
            transform: 可选的数据增强变换。
            masking_strategy: 掩码策略（必须提供）。
            masking_strategy_b: 可选的第二掩码策略（用于 Galileo 风格训练）。
            num_masked_views: 掩码视图数（1=单视图，2=双视图）。
            tokenization_config: 可选的 tokenization 配置。
        """
        super().__init__(
            work_dir=work_dir,
            global_batch_size=global_batch_size,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            fs_local_rank=fs_local_rank,
        )
        self.dataset = dataset
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        if token_budget is None:
            logger.warning("No token budget provided ALL PIXELS WILL BE USED")  # 无 token 预算将使用所有像素
        self.token_budget = token_budget
        self.patch_sizes = np.arange(min_patch_size, max_patch_size + 1)  # patch 大小候选数组
        self.sampled_hw_p_list = sampled_hw_p_list
        self.collator = collator
        self.seed = seed
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.target_device_type = target_device_type
        self.drop_last = drop_last
        self._global_indices: np.ndarray | None = None  # 全局索引缓存
        self.persistent_workers = persistent_workers
        self.multiprocessing_context = multiprocessing_context
        self.num_dataset_repeats_per_epoch = num_dataset_repeats_per_epoch

        # DataLoader 侧掩码配置
        self.transform = transform
        self.masking_strategy = masking_strategy
        self.masking_strategy_b = masking_strategy_b
        self.num_masked_views = num_masked_views
        self.tokenization_config = tokenization_config

        # 验证配置
        if masking_strategy is None:
            raise ValueError("masking_strategy must be provided")
        if num_masked_views not in (1, 2):
            raise ValueError(f"num_masked_views must be 1 or 2, got {num_masked_views}")

        if self.num_workers > 0 and self.multiprocessing_context == "forkserver":
            # 预加载模块以减少 forkserver 启动开销
            mp.set_forkserver_preload(["torch", "rasterio"])

    @property
    def total_unique_batches(self) -> int:
        """一个 epoch 中唯一的批次总数。"""
        return len(self.dataset) // (self.global_batch_size)

    @property
    def total_unique_size(self) -> int:
        """一个 epoch 中唯一的 instance 总数。"""
        return self.total_unique_batches * self.global_batch_size

    @property
    def total_batches(self) -> int:
        """一个 epoch 中的批次总数（含重复）。"""
        return self.total_unique_batches * self.num_dataset_repeats_per_epoch

    @property
    def total_size(self) -> int:
        """一个 epoch 中的 instance 总数（含重复）。"""
        return self.total_batches * self.global_batch_size

    @property
    def _global_indices_file(self) -> UPath:
        """全局索引文件的路径。

        文件名包含 seed、epoch 和 size 信息，确保不同配置的索引文件不会冲突。
        """
        global_indices_fname = self._format_fname_from_fields(
            "global_indices",
            seed=self.seed if self.shuffle else None,
            epoch=self.epoch if self.shuffle else None,  # type: ignore
            size=self.total_size,
        )
        return (
            Path(self.work_dir)
            / f"dataset-{self.dataset.fingerprint}"
            / f"{global_indices_fname}.npy"
        )

    def _build_global_indices(self) -> np.ndarray:
        """构建全局索引数组。

        创建样本索引数组，如果配置了 shuffle 则基于 seed 和 epoch 进行确定性打乱。
        重复 num_dataset_repeats_per_epoch 次并裁剪到 total_unique_size。

        Returns:
            全局索引的 numpy 数组。
        """
        assert len(self.dataset) < np.iinfo(np.uint32).max

        rng: np.random.Generator | None = None
        if self.shuffle:
            # 基于 epoch 和 seed 确定性打乱
            rng = get_rng(self.seed + self.epoch)  # type: ignore
        indices_list = []
        for _ in range(self.num_dataset_repeats_per_epoch):
            indices = np.arange(len(self.dataset), dtype=np.uint32)
            if rng is not None:
                rng.shuffle(indices)
            # 裁剪尾部以使其能被批次大小整除
            cropped_indices = indices[: self.total_unique_size]
            indices_list.append(cropped_indices)
        indices = np.concatenate(indices_list)
        return indices

    def build_and_save_global_indices(self, in_memory: bool = False) -> None:
        """构建并保存全局索引。

        根据 in_memory 参数选择将索引保存在内存或文件中。
        在分布式设置中，只有 rank 0 负责构建索引文件。

        Args:
            in_memory: 是否将索引保存在内存中。
        """
        if in_memory:
            self._global_indices = self._build_global_indices()
        else:
            self._global_indices = None
            if self.fs_local_rank == 0:
                # 从文件加载或构建并保存到文件
                if self._global_indices_file.is_file():
                    logger.info(
                        f"Using existing global indices file for seed {self.seed} and epoch {self.epoch}"  # type: ignore
                        f"at:\n'{self._global_indices_file}'"
                    )
                else:
                    global_indices = self._build_global_indices()
                    assert (
                        len(global_indices) < np.iinfo(np.int32).max
                    )  # 注意：OLMo 使用 uint32
                    # 使用内存映射写入索引文件
                    with memmap_to_write(
                        self._global_indices_file,
                        shape=global_indices.shape,
                        dtype=np.int32,
                    ) as global_indices_mmap:
                        global_indices_mmap[:] = global_indices
                    logger.info(
                        f"Global data order indices saved to:\n'{self._global_indices_file}'"
                    )
        barrier()  # 等待所有进程同步

    def reshuffle(self, epoch: int | None = None, in_memory: bool = False) -> None:
        """重新打乱数据。

        更新 epoch 并重新构建全局索引。

        Args:
            epoch: 目标 epoch 编号。若为 None，则自动递增。
            in_memory: 是否将索引保存在内存中。
        """
        if epoch is None:
            epoch = 1 if self._epoch is None else self._epoch + 1  # type: ignore
        if epoch <= 0:
            raise ValueError(f"'epoch' must be at least 1, got {epoch}")
        self._epoch = epoch
        # epoch 更新后需要重建全局索引
        self.build_and_save_global_indices(in_memory=in_memory)

    def get_global_indices(self) -> np.ndarray:
        """获取全局索引数组。

        从内存或文件中加载全局索引。

        Returns:
            全局索引的 numpy 数组。

        Raises:
            RuntimeError: 如果全局索引文件不存在。
        """
        if self._global_indices is not None:
            return self._global_indices  # 从内存返回
        if not self._global_indices_file.is_file():
            raise RuntimeError(
                f"Missing global indices file {self._global_indices_file}, did you forget to call 'reshuffle()'?"
            )
        return np.memmap(self._global_indices_file, mode="r", dtype=np.uint32)  # 从文件内存映射

    def _iter_batches(self) -> Iterable[OlmoEarthSample]:
        """迭代数据集的批次。

        使用 PyTorch DataLoader 包装 _IterableDatasetWrapper 进行批次迭代。
        """
        return torch.utils.data.DataLoader(
            _IterableDatasetWrapper(self),
            batch_size=None,  # 批次大小由 iter_batched 控制
            num_workers=self.num_workers,
            pin_memory=self.target_device_type == "cuda" and self.num_workers > 0,  # CUDA 时启用 pinned memory
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=(
                self.persistent_workers if self.num_workers > 0 else False
            ),
            multiprocessing_context=(
                self.multiprocessing_context if self.num_workers > 0 else None
            ),
            timeout=0,
        )

    @property
    def worker_info(self):  # type: ignore
        """获取当前 DataLoader worker 信息。"""
        return torch.utils.data.get_worker_info()

    def _get_local_instance_indices(self, indices: np.ndarray) -> Iterable[int]:
        """获取当前本地 rank 的 instance 索引。

        将全局索引按批次分组，跳过已处理的批次，
        然后按 worker ID 切片避免重复，最后按 DP rank 切片分配给本地进程。

        Args:
            indices: 全局 instance 索引数组。

        Returns:
            本地 rank 的 instance 索引迭代器。
        """
        # 'indices' 是全局 instance 索引
        instances_per_batch = self.global_batch_size
        indices = indices.reshape(-1, instances_per_batch)  # 按批次大小重塑

        if self.batches_processed > 0:  # type: ignore
            indices = indices[self.batches_processed :]  # type: ignore  # 跳过已处理的批次

        # 按 DataLoader worker rank 切片以避免重复
        if (worker_info := self.worker_info) is not None:
            indices = indices[worker_info.id :: worker_info.num_workers]

        # 按 DP rank 切片分配给本地进程
        indices = indices[:, self.dp_rank :: self.dp_world_size].reshape((-1,))
        return indices

    def _get_dataset_item(
        self, idx: int, patch_size: int, sampled_hw_p: int
    ) -> tuple[int, OlmoEarthSample]:
        """从数据集获取单个样本。

        Args:
            idx: 样本索引。
            patch_size: patch 大小。
            sampled_hw_p: 采样的高度/宽度 patch 数。

        Returns:
            元组 (patch_size, OlmoEarthSample)。
        """
        args = GetItemArgs(
            idx=idx,
            patch_size=patch_size,
            sampled_hw_p=sampled_hw_p,
            token_budget=self.token_budget,
            tokenization_config=self.tokenization_config,
        )
        item = self.dataset[args]
        return item

    def state_dict(self) -> dict[str, Any]:
        """获取 DataLoader 的状态字典，用于训练检查点保存。

        Returns:
            包含数据集指纹、批次进度、种子和 epoch 的状态字典。
        """
        return {
            "dataset_fingerprint_version": self.dataset.fingerprint_version,
            "dataset_fingerprint": self.dataset.fingerprint,
            "batches_processed": self.batches_processed,  # type: ignore
            "seed": self.seed,
            "epoch": self._epoch,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """从状态字典恢复 DataLoader 状态。

        Args:
            state_dict: 包含训练状态的字典。
        """
        if (
            state_dict["dataset_fingerprint_version"]
            != self.dataset.fingerprint_version
        ):
            logger.warning(
                "Dataset fingerprint version does not match the version in the checkpoint, "
                "this could mean the data has changed"
            )
        elif state_dict["dataset_fingerprint"] != self.dataset.fingerprint:
            logger.warning(
                "Restoring state from a different dataset! If this is not expected, please check the dataset fingerprint(fingerprint doesn't match)"
                f"old fingerprint: {state_dict['dataset_fingerprint']}, new fingerprint: {self.dataset.fingerprint}"
            )

        if state_dict["seed"] != self.seed:
            logger.warning(
                "Restoring data loading state with a different data seed, "
                "will use data seed from state dict for data order consistency."
            )
            self.seed = state_dict["seed"]  # 使用检查点中的种子以保持数据顺序一致性

        self.batches_processed = state_dict["batches_processed"]
        self._epoch = state_dict["epoch"] or self._epoch  # type: ignore

    def _format_fname_from_fields(self, prefix: str, **fields: Any) -> str:
        """根据字段格式化文件名。

        按字段名排序，仅包含非 None 值的字段。

        Args:
            prefix: 文件名前缀。
            **fields: 文件名字段键值对。

        Returns:
            格式化后的文件名字符串。
        """
        parts = [prefix]
        for key in sorted(fields):
            value = fields[key]
            if value is not None:
                parts.append(f"{key}{value}")
        return "_".join(parts)

    def _get_mock_sample(self, rng: np.random.Generator) -> OlmoEarthSample:
        """生成模拟样本，用于 dry-run 测试。

        根据训练模态列表生成随机数据的 OlmoEarthSample，
        用于在无需真实数据的情况下测试前向和反向传播。

        Args:
            rng: 随机数生成器。

        Returns:
            包含随机数据的 OlmoEarthSample。
        """
        output_dict = {}
        standard_hw = 64  # 标准空间尺寸
        # 生成各模态的随机模拟数据
        if Modality.SENTINEL2_L2A.name in self.dataset.training_modalities:
            mock_sentinel2_l2a = rng.random(
                (standard_hw, standard_hw, 12, 12), dtype=np.float32
            )
            output_dict["sentinel2_l2a"] = mock_sentinel2_l2a
        if Modality.NAIP_10.name in self.dataset.training_modalities:
            mock_naip_10 = rng.random((1024, 1024, 1, 4), dtype=np.float32)
            output_dict["naip_10"] = mock_naip_10
        if Modality.SENTINEL1.name in self.dataset.training_modalities:
            mock_sentinel1 = rng.random(
                (standard_hw, standard_hw, 12, 2), dtype=np.float32
            )
            output_dict[Modality.SENTINEL1.name] = mock_sentinel1
        if Modality.WORLDCOVER.name in self.dataset.training_modalities:
            mock_worldcover = rng.random(
                (standard_hw, standard_hw, 1, 1), dtype=np.float32
            )
            output_dict["worldcover"] = mock_worldcover
        if Modality.LATLON.name in self.dataset.training_modalities:
            mock_latlon = rng.random((2,), dtype=np.float32)
            output_dict["latlon"] = mock_latlon
        if Modality.OPENSTREETMAP_RASTER.name in self.dataset.training_modalities:
            mock_openstreetmap_raster = rng.random(
                (standard_hw, standard_hw, 1, 30), dtype=np.float32
            )
            output_dict["openstreetmap_raster"] = mock_openstreetmap_raster
        if Modality.SRTM.name in self.dataset.training_modalities:
            mock_srtm = rng.random((standard_hw, standard_hw, 1, 1), dtype=np.float32)
            output_dict["srtm"] = mock_srtm
        if Modality.LANDSAT.name in self.dataset.training_modalities:
            mock_landsat = rng.random(
                (standard_hw, standard_hw, 12, Modality.LANDSAT.num_bands),
                dtype=np.float32,
            )
            output_dict["landsat"] = mock_landsat
        if Modality.GSE.name in self.dataset.training_modalities:
            mock_gse = rng.random(
                (standard_hw, standard_hw, 1, Modality.GSE.num_bands), dtype=np.float32
            )
            output_dict["gse"] = mock_gse
        if Modality.CDL.name in self.dataset.training_modalities:
            mock_cdl = rng.random(
                (standard_hw, standard_hw, 1, Modality.CDL.num_bands), dtype=np.float32
            )
            output_dict["cdl"] = mock_cdl
        if Modality.WORLDPOP.name in self.dataset.training_modalities:
            mock_worldpop = rng.random(
                (standard_hw, standard_hw, 1, Modality.WORLDPOP.num_bands),
                dtype=np.float32,
            )
            output_dict["worldpop"] = mock_worldpop
        if Modality.WRI_CANOPY_HEIGHT_MAP.name in self.dataset.training_modalities:
            mock_wri_canopy_height_map = rng.random(
                (standard_hw, standard_hw, 1, Modality.WRI_CANOPY_HEIGHT_MAP.num_bands),
                dtype=np.float32,
            )
            output_dict["wri_canopy_height_map"] = mock_wri_canopy_height_map
        if Modality.ERA5_10.name in self.dataset.training_modalities:
            mock_era5_10 = rng.random(
                (12, Modality.ERA5_10.num_bands), dtype=np.float32
            )
            output_dict["era5_10"] = mock_era5_10
        if Modality.EUROCROPS.name in self.dataset.training_modalities:
            mock_eurocrops = rng.random(
                (standard_hw, standard_hw, 1, Modality.EUROCROPS.num_bands),
                dtype=np.float32,
            )
            output_dict["eurocrops"] = mock_eurocrops
        if Modality.RGB_2_5.name in self.dataset.training_modalities:
            mock_rgb_2_5 = rng.random(
                (standard_hw * Modality.RGB_2_5.image_tile_size_factor,
                 standard_hw * Modality.RGB_2_5.image_tile_size_factor,
                 1,
                 Modality.RGB_2_5.num_bands),
                dtype=np.float32,
            )
            output_dict[Modality.RGB_2_5.name] = mock_rgb_2_5

        # 生成随机时间戳：日期(0-24)、月份(0-11)、年份(2018-2019)
        days = rng.integers(0, 25, (12, 1))
        months = rng.integers(0, 12, (12, 1))
        years = rng.integers(2018, 2020, (12, 1))
        timestamps = np.concatenate([days, months, years], axis=1)  # 形状: (12, 3)

        output_dict["timestamps"] = timestamps
        return OlmoEarthSample(**output_dict)

    def get_mock_batch(self) -> Any:
        """获取模拟批次，用于训练前向和反向传播的 dry-run 测试。

        根据 num_masked_views 返回对应格式的批次：
        - 1: (patch_size, MaskedOlmoEarthSample) - 单掩码视图
        - 2: (patch_size, MaskedOlmoEarthSample, MaskedOlmoEarthSample) - 双掩码视图

        Returns:
            模拟批次数据。
        """
        logger.info("Getting mock batch NOT FROM DATASET")
        logger.info(f"Training modalities: {self.dataset.training_modalities}")
        logger.info(f"num_masked_views: {self.num_masked_views}")
        rng = get_rng(42)
        batch_size = self.global_batch_size // self.dp_world_size  # 本地批次大小
        patch_size = 1

        # Generate mock samples
        mock_samples = [
            subset_sample_default(
                self._get_mock_sample(rng),
                patch_size=patch_size,
                max_tokens_per_instance=1500,
                sampled_hw_p=6,
                current_length=12,
            )
            for _ in range(batch_size)
        ]

        # Pass raw samples to the collator - the batched collators handle
        # transform + masking internally when num_masked_views > 0
        collated_sample = self.collator(
            [(patch_size, sample) for sample in mock_samples]
        )

        return collated_sample

    def fast_forward(self, global_step: int) -> np.ndarray:
        """快进 DataLoader 到指定的全局步数，并返回对应的批次索引。

        Args:
            global_step: 目标全局步数。

        Returns:
            该步对应的样本索引数组。

        Raises:
            NotImplementedError: 在 DDP 环境中不支持快进。
        """
        logger.warning(
            "Fast forward does not yet support returning to indices for multiple GPUs"
        )
        if get_world_size() > 1:
            raise NotImplementedError("Fast forward is not supported in DDP")
        # 如果使用多 GPU 训练，需要更新此逻辑以获取所有 rank 的起始位置
        self.batches_processed = global_step
        epoch = math.ceil(global_step / self.total_batches)
        step_in_epoch = global_step % self.total_batches
        logger.info(f"epoch: {epoch}, step in epoch: {step_in_epoch}")
        self.reshuffle(epoch=epoch)
        batch_start = int(self.get_global_indices()[step_in_epoch])
        batch_end = batch_start + self.global_batch_size
        sample_indices = np.arange(batch_start, batch_end)
        return sample_indices


def iter_batched(
    iterable: Iterable[tuple[int, OlmoEarthSample]],
    batch_size: int,
    drop_last: bool = True,
) -> Iterable[tuple[tuple[int, OlmoEarthSample], ...]]:
    """将迭代器中的 item 按批次大小分组。

    这是 olmo_core.data.data_loader.iter_batched 的修改版本，
    从 item 迭代器中为本地 rank 创建大小为 local_batch_size 的批次。

    Args:
        iterable: item 的迭代器。
        batch_size: 批次大小。
        drop_last: 是否丢弃最后一个不完整批次。

    Returns:
        批次元组的迭代器，每个批次包含 batch_size 个 item。
    """
    assert batch_size > 0
    batch: list[tuple[int, OlmoEarthSample]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield tuple(batch)  # 达到批次大小则输出
            batch.clear()

    # 如果剩余部分批次且 drop_last 为 False，则输出
    if not drop_last and batch:
        yield tuple(batch)


class _IterableDatasetWrapper(torch.utils.data.IterableDataset[OlmoEarthSample]):
    """可迭代数据集包装器。

    这是 olmo_core.data.data_loader._IterableDatasetWrapper 的修改版本，
    用于将 OlmoEarthDataLoader 包装为 PyTorch IterableDataset。

    关键属性:
        data_loader: 被包装的 OlmoEarthDataLoader 实例
        rngs: 每个 worker 的随机数生成器列表
        transform / masking_strategy: 数据增强和掩码策略
    """

    def __init__(self, data_loader: OlmoEarthDataLoader):
        """初始化 IterableDatasetWrapper。

        为每个 worker 创建独立的随机数生成器，确保不同 worker 的随机性互不干扰。
        """
        self.data_loader = data_loader
        workers = data_loader.num_workers or 1
        # 为每个 worker 创建独立的随机数生成器
        self.rngs = [
            get_rng(
                data_loader.seed + data_loader.epoch + data_loader.dp_rank * workers + i
            )
            for i in range(workers)
        ]
        # 从 DataLoader 继承掩码配置
        self.transform = data_loader.transform
        self.masking_strategy = data_loader.masking_strategy
        self.masking_strategy_b = data_loader.masking_strategy_b
        self.num_masked_views = data_loader.num_masked_views

    def _get_batch_item_params_iterator(
        self,
        indices: np.ndarray,
        patch_size_list: list[int],
        hw_p_to_sample: list[int],
        rank_batch_size: int,
    ) -> Iterator[tuple[int, int, int]]:
        """获取批次 item 参数的生成器，产生 (idx, patch_size, sampled_hw_p) 元组。

        每 rank_batch_size 个 item 更新一次 patch_size 和 sampled_hw_p。

        Args:
            indices: instance 索引数组。
            patch_size_list: patch 大小候选列表。
            hw_p_to_sample: 采样 hw_p 候选列表。
            rank_batch_size: 本地 rank 的批次大小。

        Yields:
            (idx, patch_size, sampled_hw_p) 元组。
        """
        patch_size_array = np.array(patch_size_list)
        hw_p_to_sample_array = np.array(hw_p_to_sample)
        instances_processed = 0

        # TODO: 需要在此处维护状态和可复现性
        worker_id = self.worker_info.id if self.worker_info is not None else 0
        rng = self.rngs[worker_id]

        for idx in indices:
            if instances_processed % rank_batch_size == 0:
                # 每个批次随机选择 patch_size 和 sampled_hw_p
                patch_size = rng.choice(patch_size_array)
                max_height_width_tokens = int(IMAGE_TILE_SIZE / patch_size)
                # 过滤出不超过最大 token 数且大于 0 的 hw_p 候选
                filtered_hw_p_to_sample_array = hw_p_to_sample_array[
                    hw_p_to_sample_array <= max_height_width_tokens
                ]
                filtered_hw_p_to_sample_array = filtered_hw_p_to_sample_array[
                    filtered_hw_p_to_sample_array > 0
                ]
                sampled_hw_p = rng.choice(filtered_hw_p_to_sample_array)
            yield idx, int(patch_size), int(sampled_hw_p)
            instances_processed += 1

    @property
    def dataset(self) -> OlmoEarthDataset:
        """获取包装的数据集。"""
        return self.data_loader.dataset

    @property
    def worker_info(self):  # type: ignore
        """获取当前 DataLoader worker 信息。"""
        return torch.utils.data.get_worker_info()

    def __iter__(self) -> Iterator[Any]:
        """迭代数据集，产生批次数据。

        产生格式取决于 num_masked_views：
        - 1: (patch_size, MaskedOlmoEarthSample) - 单掩码视图
        - 2: (patch_size, MaskedOlmoEarthSample, MaskedOlmoEarthSample) - 双掩码视图

        数据增强和掩码在批次整理器中应用以实现更好的向量化。
        """
        global_indices = self.data_loader.get_global_indices()
        indices = self.data_loader._get_local_instance_indices(global_indices)

        # 创建从数据集获取样本的迭代器
        instance_iterator = (
            self.data_loader._get_dataset_item(int(idx), patch_size, sampled_hw_p)
            for idx, patch_size, sampled_hw_p in self._get_batch_item_params_iterator(
                indices,
                self.data_loader.patch_sizes,
                self.data_loader.sampled_hw_p_list,
                self.data_loader.rank_batch_size,
            )
        )

        # 通过 iter_batched 分组后由 collator 处理
        return (
            self.data_loader.collator(batch)  # type: ignore[arg-type]
            for batch in iter_batched(
                instance_iterator,  # type: ignore[arg-type]
                self.data_loader.rank_batch_size,
                self.data_loader.drop_last,
            )
        )


@dataclass
class OlmoEarthDataLoaderConfig(Config):
    """OlmoEarthDataLoader 的配置类。

    属性:
        work_dir: 工作目录路径。
        global_batch_size: 全局批次大小。
        min_patch_size: 最小 patch 大小。
        max_patch_size: 最大 patch 大小。
        sampled_hw_p_list: 采样的 hw_p 候选列表。
        seed: 随机种子。
        token_budget: 可选的 token 预算。
        shuffle: 是否打乱数据。
        num_workers: DataLoader worker 数量。
        prefetch_factor: 预取因子。
        target_device_type: 目标设备类型。
        drop_last: 是否丢弃最后不完整批次。
        num_dataset_repeats_per_epoch: 每个 epoch 数据集重复次数。
        transform_config: 可选的数据增强配置。
        masking_config: 掩码配置（必须提供）。
        masking_config_b: 可选的第二掩码配置。
        num_masked_views: 掩码视图数（1=单，2=双）。
        tokenization_config: 可选的 tokenization 配置。
    """

    work_dir: str
    global_batch_size: int
    min_patch_size: int
    max_patch_size: int
    sampled_hw_p_list: list[int]
    seed: int
    token_budget: int | None = None  # 若为 None 则不做子采样
    shuffle: bool = True
    num_workers: int = 0
    prefetch_factor: int | None = None
    target_device_type: str | None = None
    drop_last: bool = True
    num_dataset_repeats_per_epoch: int = 1
    # DataLoader 侧掩码配置
    transform_config: TransformConfig | None = None
    masking_config: MaskingConfig | None = None
    masking_config_b: MaskingConfig | None = None
    num_masked_views: int = 1  # 1 = 单视图，2 = 双视图
    tokenization_config: TokenizationConfig | None = None

    def validate(self) -> None:
        """验证配置参数的合法性。

        Raises:
            ValueError: 如果 work_dir 未设置、min_patch_size > max_patch_size、
                masking_config 未提供或 num_masked_views 不在 1 或 2 中。
        """
        if self.work_dir is None:
            raise ValueError("Work directory is not set")
        if self.min_patch_size > self.max_patch_size:
            raise ValueError("min_patch_size must be less than max_patch_size")
        if self.masking_config is None:
            raise ValueError("masking_config must be provided")
        if self.num_masked_views not in (1, 2):
            raise ValueError(
                f"num_masked_views must be 1 or 2, got {self.num_masked_views}"
            )

    @property
    def work_dir_upath(self) -> UPath:
        """获取工作目录的 UPath 对象。"""
        return UPath(self.work_dir)

    def build(
        self,
        dataset: OlmoEarthDataset,
        dp_process_group: dist.ProcessGroup | None = None,
    ) -> "OlmoEarthDataLoader":
        """构建 OlmoEarthDataLoader 实例。

        Args:
            dataset: 数据集实例。
            dp_process_group: 可选的分布式数据并行进程组。

        Returns:
            配置好的 OlmoEarthDataLoader 实例。
        """
        self.validate()
        dataset.prepare()

        # 构建数据增强和掩码策略
        transform = (
            self.transform_config.build() if self.transform_config is not None else None
        )
        # masking_config 是必须的（上面已验证）
        assert self.masking_config is not None
        masking_strategy = self.masking_config.build()
        masking_strategy_b = (
            self.masking_config_b.build() if self.masking_config_b is not None else None
        )

        # 根据 num_masked_views 选择对应的批次整理器
        # 使用批次级整理器对整个批次一次性应用增强和掩码以实现更好的向量化
        collator: Callable
        if self.num_masked_views == 1:
            # 单掩码视图
            collator = functools.partial(
                collate_single_masked_batched,
                transform=transform,
                masking_strategy=masking_strategy,
            )
        else:  # num_masked_views == 2，双掩码视图
            collator = functools.partial(
                collate_double_masked_batched,
                transform=transform,
                masking_strategy=masking_strategy,
                masking_strategy_b=masking_strategy_b,
            )

        return OlmoEarthDataLoader(
            dataset=dataset,
            work_dir=self.work_dir_upath,
            global_batch_size=self.global_batch_size,
            dp_world_size=get_world_size(dp_process_group),
            dp_rank=get_rank(dp_process_group),
            fs_local_rank=get_fs_local_rank(),
            seed=self.seed,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            target_device_type=self.target_device_type or get_default_device().type,
            collator=collator,
            drop_last=self.drop_last,
            min_patch_size=self.min_patch_size,
            max_patch_size=self.max_patch_size,
            sampled_hw_p_list=self.sampled_hw_p_list,
            token_budget=self.token_budget,
            num_dataset_repeats_per_epoch=self.num_dataset_repeats_per_epoch,
            transform=transform,
            masking_strategy=masking_strategy,
            masking_strategy_b=masking_strategy_b,
            num_masked_views=self.num_masked_views,
            tokenization_config=self.tokenization_config,
        )


# 向后兼容的废弃别名
HeliosDataLoader = _deprecated_class_alias(
    OlmoEarthDataLoader, "helios.data.dataloader.HeliosDataLoader"
)
HeliosDataLoaderConfig = _deprecated_class_alias(
    OlmoEarthDataLoaderConfig, "helios.data.dataloader.HeliosDataLoaderConfig"
)
