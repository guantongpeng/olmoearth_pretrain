"""计算 H5 数据集的归一化统计量。

本脚本以流式方式计算指定 H5 数据集中各模态各波段的均值、方差和标准差，
用于训练时的数据归一化。

使用示例:
    python3 olmoearth_pretrain/internal/compute_h5_norm.py --h5py_dir /path/to/h5pydir  --supported_modalities "era5_10,landsat,naip_10,sentinel1,sentinel2_l2a,srtm,worldcover" --estimate_from 100 --output_path /weka/dfive-default/yawenz/helios/helios/data/norm_configs/computed_20250722.json

模块功能:
    1. 加载 H5 数据集并采样指定数量的样本
    2. 逐模态逐波段计算流式统计量（均值、方差）
    3. 计算标准差并保存为 JSON 格式
"""

import argparse
import json
import logging
import random
from typing import Any

from olmo_core.utils import prepare_cli_environment
from tqdm import tqdm

from olmoearth_pretrain.data.constants import (
    IMAGE_TILE_SIZE,
    MISSING_VALUE,
    Modality,
)
from olmoearth_pretrain.data.dataset import (
    GetItemArgs,
    OlmoEarthDataset,
    OlmoEarthDatasetConfig,
)
from olmoearth_pretrain.data.utils import update_streaming_stats

logger = logging.getLogger(__name__)


def compute_normalization_values(
    dataset: OlmoEarthDataset,
    estimate_from: int | None = None,
) -> dict[str, Any]:
    """以流式方式计算数据集的归一化统计值。

    核心逻辑:
        1. 随机采样指定数量的样本（或使用全部样本）
        2. 对每个样本的每个模态，逐波段使用 Welford 在线算法更新均值和方差
        3. 跳过包含缺失值和 latlon 模态的样本
        4. 计算标准差并记录数据集元信息

    Args:
        dataset: 要计算归一化统计值的数据集
        estimate_from: 估计归一化值所用的样本数量，None 表示使用全部样本

    Returns:
        dict: 包含数据集归一化值的字典，结构为:
            {modality_name: {band_name: {mean, var, std, count}}, total_n, sampled_n, tile_path}
    """
    dataset_len = len(dataset)
    if estimate_from is not None:
        indices_to_sample = random.sample(list(range(dataset_len)), k=estimate_from)  # 随机采样指定数量的样本
    else:
        indices_to_sample = list(range(dataset_len))
    norm_dict: dict[str, Any] = {}
    for i in tqdm(indices_to_sample):
        get_item_args = GetItemArgs(idx=i, patch_size=1, sampled_hw_p=IMAGE_TILE_SIZE)
        _, sample = dataset[get_item_args]
        for modality in sample.modalities:
            # 是否应该计算 worldcover 的归一化统计？
            if modality == "latlon":
                continue  # 跳过经纬度模态
            modality_data = sample.as_dict()[modality]
            modality_spec = Modality.get(modality)
            modality_bands = modality_spec.band_order
            if modality_data is None:
                continue  # 跳过缺失的模态数据
            # 避免将缺失值纳入统计
            if (modality_data == MISSING_VALUE).any():
                logger.info(
                    f"Skipping sample {i} because modality {modality} contains missing values."
                )
                continue
            if modality not in norm_dict:
                norm_dict[modality] = {}
                for band in modality_bands:
                    norm_dict[modality][band] = {
                        "mean": 0.0,
                        "var": 0.0,
                        "std": 0.0,
                        "count": 0,
                    }
            # 逐波段计算归一化统计量（使用流式 Welford 算法）
            for idx, band in enumerate(modality_bands):
                modality_band_data = modality_data[..., idx]
                current_stats = norm_dict[modality][band]
                new_count, new_mean, new_var = update_streaming_stats(
                    current_stats["count"],
                    current_stats["mean"],
                    current_stats["var"],
                    modality_band_data,
                )
                # 更新归一化统计量
                norm_dict[modality][band]["count"] = int(new_count)
                norm_dict[modality][band]["mean"] = float(new_mean)
                norm_dict[modality][band]["var"] = float(new_var)

    # 计算标准差（方差 / 计数 的平方根）
    for modality in norm_dict:
        for band in norm_dict[modality]:
            norm_dict[modality][band]["std"] = (
                norm_dict[modality][band]["var"] / norm_dict[modality][band]["count"]
            ) ** 0.5

    norm_dict["total_n"] = dataset_len  # 数据集总样本数
    norm_dict["sampled_n"] = len(indices_to_sample)  # 实际采样样本数
    path = dataset.h5py_dir or dataset.tile_path
    norm_dict["tile_path"] = str(path)

    return norm_dict


if __name__ == "__main__":
    prepare_cli_environment()
    args = argparse.ArgumentParser()
    args.add_argument("--h5py_dir", type=str, required=True)
    args.add_argument("--supported_modalities", type=str, required=True)
    args.add_argument("--estimate_from", type=int, required=False, default=None)
    args.add_argument("--output_path", type=str, required=True)
    args_dict = args.parse_args().__dict__  # type: ignore

    logger.info(
        f"Computing normalization stats with modalities {args_dict['supported_modalities']}"
    )

    def parse_supported_modalities(supported_modalities: str) -> list[str]:
        """Parse the supported modalities from a string."""
        return supported_modalities.split(",")

    # FOr some reason landsat and naip were missi g from every sample
    supported_modalities = parse_supported_modalities(args_dict["supported_modalities"])
    logger.info(f"Supported modalities: {supported_modalities}")
    # Use the config to build the dataset
    dataset_config = OlmoEarthDatasetConfig(
        h5py_dir=args_dict["h5py_dir"],
        training_modalities=supported_modalities,
        normalize=False,
    )
    dataset = dataset_config.build()
    dataset.prepare()
    logger.info(f"Dataset: {dataset.normalize}")

    norm_dict = compute_normalization_values(
        dataset=dataset,
        estimate_from=args_dict["estimate_from"],
    )
    logger.info(f"Normalization stats: {norm_dict}")

    with open(args_dict["output_path"], "w") as f:
        json.dump(norm_dict, f)
