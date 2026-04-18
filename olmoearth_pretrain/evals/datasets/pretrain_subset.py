"""预训练数据子集评估数据集模块。

本模块从预训练数据集中加载一个子集，用于嵌入质量诊断评估。
返回的样本与预训练时完全相同的格式（MaskedOlmoEarthSample），
支持各种评估模式（KNN、线性探针等）。

主要组件：
- PretrainSubsetDataset: 预训练数据子集数据集类
  包装 OlmoEarthDataset 以适配评估接口 (返回 MaskedOlmoEarthSample + dummy_label)

使用场景：
  评估预训练模型在训练分布数据上的嵌入质量，
  检测表示坍缩等训练问题。
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from upath import UPath

from olmoearth_pretrain.data.dataset import GetItemArgs, OlmoEarthDataset
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)

# 默认参数
DEFAULT_PATCH_SIZE = 4     # 默认 patch 大小
DEFAULT_HW_P = 8           # 默认 hw_p (高宽 patch 数)
DEFAULT_MAX_SAMPLES = 512  # 默认最大样本数


class PretrainSubsetDataset(Dataset):
    """预训练数据子集数据集类。

    包装 OlmoEarthDataset 以适配评估数据集接口，
    返回 (MaskedOlmoEarthSample, dummy_label) 格式。
    使用固定种子选择训练索引的子集以确保可复现性。

    关键属性：
        _dataset: OlmoEarthDataset 实例
        _indices: 固定随机种子选择的索引子集
        patch_size: patch 大小
        hw_p: 高宽 patch 数
        max_samples: 最大样本数
    """

    def __init__(
        self,
        h5py_dir: str,
        training_modalities: list[str],
        max_samples: int = DEFAULT_MAX_SAMPLES,
        patch_size: int = DEFAULT_PATCH_SIZE,
        hw_p: int = DEFAULT_HW_P,
        seed: int = 42,
    ) -> None:
        """初始化预训练数据子集。

        使用固定随机种子从完整数据集中选择索引子集，确保可复现性。

        Args:
            h5py_dir: h5py 数据文件目录路径
            training_modalities: 训练模态名称列表
            max_samples: 最大样本数，默认 512
            patch_size: patch 大小，默认 4
            hw_p: 高宽 patch 数，默认 8
            seed: 随机种子，默认 42
        """
        self.patch_size = patch_size
        self.hw_p = hw_p
        self.max_samples = max_samples

        self._dataset = OlmoEarthDataset(
            h5py_dir=UPath(h5py_dir),
            training_modalities=training_modalities,
            dtype=np.float32,
            normalize=True,
        )
        self._dataset.prepare()

        total = len(self._dataset)
        n = min(max_samples, total)
        rng = np.random.RandomState(seed)
        self._indices = rng.choice(total, size=n, replace=False).tolist()

    def __len__(self) -> int:
        """Return number of samples in the subset."""
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[MaskedOlmoEarthSample, torch.Tensor]:
        """Return (MaskedOlmoEarthSample, dummy_label) for the given index."""
        real_idx = self._indices[idx]
        args = GetItemArgs(
            idx=real_idx,
            patch_size=self.patch_size,
            sampled_hw_p=self.hw_p,
        )
        _, sample = self._dataset[args]
        masked = MaskedOlmoEarthSample.from_olmoearthsample(sample)
        dummy_label = torch.tensor(0, dtype=torch.long)
        return masked, dummy_label
