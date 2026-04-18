"""
Weights & Biases (W&B) 日志回调模块。

本模块提供 OlmoEarthWandBCallback，用于在训练过程中：
- 初始化 W&B 实验运行（支持从上次中断处恢复）
- 上传数据集地理分布图到 W&B
- 上传各模态数据的归一化分布直方图
- 管理 W&B API 密钥和环境变量

使用场景：在分布式训练中，仅 rank 0 进程执行 W&B 操作。
"""

"""OlmoEarth Pretrain specific wandb callback."""

import logging  # 日志记录
import os  # 环境变量访问
import random  # 随机采样
from dataclasses import dataclass
from pathlib import Path  # 路径操作
from typing import Any

import matplotlib.pyplot as plt
from olmo_core.distributed.utils import get_rank
from olmo_core.exceptions import OLMoEnvironmentError
from olmo_core.train.callbacks.wandb import WANDB_API_KEY_ENV_VAR, WandBCallback
from tqdm import tqdm

from olmoearth_pretrain._compat import deprecated_class_alias as _deprecated_class_alias
from olmoearth_pretrain.data.constants import IMAGE_TILE_SIZE, Modality
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoader
from olmoearth_pretrain.data.dataset import GetItemArgs, OlmoEarthDataset
from olmoearth_pretrain.data.utils import (
    plot_latlon_distribution,
    plot_modality_data_distribution,
)

logger = logging.getLogger(__name__)


def get_sample_data_for_histogram(
    dataset: OlmoEarthDataset, num_samples: int = 100, num_values: int = 100
) -> dict[str, Any]:
    """从数据集中采样数据，用于绘制各模态各波段的直方图。

    随机选取 num_samples 个样本，对每个样本的每个模态的每个波段，
    随机采样 num_values 个像素值，用于后续绘制数据分布直方图并上传到 W&B。

    Args:
        dataset: 要采样的数据集。
        num_samples: 从数据集中采样的样本数量。
        num_values: 从每个模态每个波段中采样的像素值数量。

    Returns:
        dict: 嵌套字典，结构为 {模态名: {波段名: [像素值列表]}}。
    """
    if num_samples > len(dataset):
        raise ValueError(
            f"num_samples {num_samples} is greater than the number of samples in the dataset {len(dataset)}"
        )
    # 随机选择要采样的样本索引
    indices_to_sample = random.sample(list(range(len(dataset))), k=num_samples)
    sample_data: dict[str, Any] = {}

    # 遍历采样索引，收集各模态各波段的数据
    # TODO: 直接计算每个模态和波段的直方图，而非采样
    for i in tqdm(indices_to_sample):
        get_item_args = GetItemArgs(idx=i, patch_size=1, sampled_hw_p=IMAGE_TILE_SIZE)
        _, sample = dataset[get_item_args]
        for modality in sample.modalities:
            if modality == "latlon":
                continue  # 跳过经纬度模态
            modality_data = sample.as_dict()[modality]
            if modality_data is None:
                continue  # 跳过缺失模态
            modality_spec = Modality.get(modality)
            modality_bands = modality_spec.band_order  # 获取波段顺序
            if modality not in sample_data:
                sample_data[modality] = {band: [] for band in modality_bands}
            # 对每个波段，展平数据并随机采样像素值
            for idx, band in enumerate(modality_bands):
                sample_data[modality][band].extend(
                    random.sample(
                        modality_data[:, :, :, idx].flatten().tolist(), num_values
                    )
                )
    return sample_data


@dataclass
class OlmoEarthWandBCallback(WandBCallback):
    """OlmoEarth Pretrain 的 W&B 日志回调。

    继承自 olmo-core 的 WandBCallback，扩展了以下功能：
    - 训练前上传数据集地理分布图
    - 训练前上传各模态数据的归一化分布直方图
    - 支持在同一 W&B 运行中恢复训练（restart_on_same_run）

    关键属性:
        upload_dataset_distribution_pre_train: 是否在训练前上传数据集地理分布（默认True）
        upload_modality_data_band_distribution_pre_train: 是否上传各模态波段分布（默认False）
        restart_on_same_run: 是否在同一 W&B 运行中恢复（默认True）
    """

    upload_dataset_distribution_pre_train: bool = True  # 训练前上传数据集地理分布
    upload_modality_data_band_distribution_pre_train: bool = False  # 训练前上传模态波段分布
    restart_on_same_run: bool = True  # 在同一 W&B 运行中恢复

    def pre_train(self) -> None:
        """训练开始前的回调：初始化 W&B 运行并上传数据分布图。

        执行流程：
        1. 检查 W&B API 密钥是否已设置
        2. 初始化 W&B 运行（支持从上次运行恢复）
        3. 上传数据集地理分布图
        4. 上传各模态数据的归一化分布直方图（可选）
        5. 清理内存中的分布数据，避免被 pickle 序列化到数据工作进程
        """
        if self.enabled and get_rank() == 0:  # 仅在 rank 0 且 W&B 启用时执行
            self.wandb
            if WANDB_API_KEY_ENV_VAR not in os.environ:
                raise OLMoEnvironmentError(f"missing env var '{WANDB_API_KEY_ENV_VAR}'")

            # 创建 W&B 目录
            wandb_dir = Path(self.trainer.save_folder) / "wandb"
            wandb_dir.mkdir(parents=True, exist_ok=True)
            resume_id = None
            if self.restart_on_same_run:
                # 如果启用同一运行恢复，读取上次运行的 ID
                runid_file = wandb_dir / "wandb_runid.txt"
                if runid_file.exists():
                    resume_id = runid_file.read_text().strip()

            # 初始化 W&B 运行
            self.wandb.init(
                dir=wandb_dir,
                project=self.project,
                entity=self.entity,
                group=self.group,
                name=self.name,
                tags=self.tags,
                notes=self.notes,
                config=self.config,
                id=resume_id,  # 若有 resume_id 则恢复该运行
                resume="allow",
                settings=self.wandb.Settings(init_timeout=240),
            )

            # 保存运行 ID 以便后续恢复
            if not resume_id and self.restart_on_same_run:
                runid_file.write_text(self.run.id)

            self._run_path = self.run.path  # type: ignore
            if self.upload_dataset_distribution_pre_train:
                # 上传数据集地理分布图
                assert isinstance(self.trainer.data_loader, OlmoEarthDataLoader)
                dataset = self.trainer.data_loader.dataset
                logger.info("Gathering locations of entire dataset")
                latlons = dataset.latlon_distribution  # 获取经纬度分布
                assert latlons is not None
                logger.info(f"Uploading dataset distribution to wandb: {latlons.shape}")
                # 绘制地理分布图并上传到 W&B
                fig = plot_latlon_distribution(
                    latlons, "Geographic Distribution of Dataset"
                )
                self.wandb.log(
                    {
                        "dataset/pretraining_geographic_distribution": self.wandb.Image(
                            fig
                        )
                    }
                )
                plt.close(fig)  # 关闭图形以释放内存
                # 从数据集中删除经纬度分布数据，避免被 pickle 序列化到数据工作进程
                del dataset.latlon_distribution
                if self.upload_modality_data_band_distribution_pre_train:
                    # 上传各模态数据的归一化分布直方图
                    logger.info("Gathering normalized data distribution")
                    sample_data = get_sample_data_for_histogram(dataset)
                    for modality, modality_data in sample_data.items():
                        fig = plot_modality_data_distribution(modality, modality_data)
                        self.wandb.log(
                            {
                                f"dataset/pretraining_{modality}_distribution": self.wandb.Image(
                                    fig
                                )
                            }
                        )
                        plt.close(fig)


HeliosWandBCallback = _deprecated_class_alias(
    OlmoEarthWandBCallback, "helios.train.callbacks.wandb.HeliosWandBCallback"
)
