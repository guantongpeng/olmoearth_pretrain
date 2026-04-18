"""OlmoEarth 评估数据集模块。

本模块提供所有评估数据集的统一访问接口，包括：
- GeoBench 数据集（m-eurosat, m-bigearthnet, m-so2sat 等）
- MADOS 分割数据集
- Sen1Floods11 洪水分割数据集
- PASTIS-R 农作物分割时间序列数据集
- BreizhCrops 农作物分类时间序列数据集
- 预训练数据子集（用于嵌入诊断）
- rslearn 注册表数据集

核心函数：
- get_eval_dataset: 根据数据集名称获取对应的评估数据集实例
- EvalDatasetPartition: 数据集分区枚举（支持不同比例的训练子集）
"""

import logging
from typing import Any

from olmo_core.config import StrEnum
from torch.utils.data import Dataset

import olmoearth_pretrain.evals.datasets.paths as paths
from olmoearth_pretrain.evals.studio_ingest.registry import get_dataset_entry

from .breizhcrops import BreizhCropsDataset
from .floods_dataset import Sen1Floods11Dataset
from .geobench_dataset import GeobenchDataset
from .mados_dataset import MADOSDataset
from .normalize import NormMethod
from .pastis_dataset import PASTISRDataset
from .pretrain_subset import PretrainSubsetDataset
from .rslearn_dataset import from_registry_entry

logger = logging.getLogger(__name__)


class EvalDatasetPartition(StrEnum):
    """评估数据集分区枚举，支持不同比例的训练子集。

    用于数据效率评估，测试不同训练数据量下的模型性能。

    Attributes:
        TRAIN1X: 默认，使用全部训练数据
        TRAIN_001X: 使用 1% 的训练数据
        TRAIN_002X: 使用 2% 的训练数据
        TRAIN_005X: 使用 5% 的训练数据
        TRAIN_010X: 使用 10% 的训练数据
        TRAIN_020X: 使用 20% 的训练数据
        TRAIN_050X: 使用 50% 的训练数据
    """

    TRAIN1X = "default"
    TRAIN_001X = "0.01x_train"  # 不适用于非训练划分
    TRAIN_002X = "0.02x_train"
    TRAIN_005X = "0.05x_train"
    TRAIN_010X = "0.10x_train"
    TRAIN_020X = "0.20x_train"
    TRAIN_050X = "0.50x_train"


def get_eval_dataset(
    eval_dataset: str,
    split: str,
    norm_stats_from_pretrained: bool = False,
    input_modalities: list[str] = [],
    partition: str = EvalDatasetPartition.TRAIN1X,
    # 默认使用 2std no clip 归一化，与模型预训练时看到的归一化方式一致
    # 当使用数据集统计量（如 MADOS）时，一致性很重要
    norm_method: str = NormMethod.NORM_NO_CLIP_2_STD,
    **kwargs: Any,
) -> Dataset:
    """根据数据集名称获取对应的评估数据集实例。

    支持的数据集类型：
    - "pretrain_subset": 预训练数据子集，用于嵌入诊断
    - "m-*": GeoBench 修改版数据集（m-eurosat, m-bigearthnet 等）
    - "mados": MADOS 海洋分割数据集
    - "sen1floods11": Sen1Floods11 洪水分割数据集
    - "pastis*": PASTIS-R 农作物分割数据集
    - "breizhcrops": BreizhCrops 农作物分类数据集
    - 其他: 从 rslearn 注册表加载的数据集

    Args:
        eval_dataset: 数据集名称
        split: 数据集划分 (train/valid/test)
        norm_stats_from_pretrained: 是否使用预训练模型的归一化统计量
        input_modalities: 输入模态列表
        partition: 数据集分区（支持不同比例的训练子集）
        norm_method: 归一化方法，默认为 2std no clip
        **kwargs: 传递给数据集构造函数的额外参数

    Returns:
        Dataset: 对应的评估数据集实例
    """
    if eval_dataset == "pretrain_subset":
        return PretrainSubsetDataset(
            h5py_dir=kwargs["h5py_dir"],
            training_modalities=kwargs.get("training_modalities", input_modalities),
            max_samples=kwargs.get("max_samples", 512),
            patch_size=kwargs.get("pretrain_patch_size", 4),
            hw_p=kwargs.get("pretrain_hw_p", 8),
            seed=kwargs.get("pretrain_seed", 42),
        )
    elif eval_dataset.startswith("m-"):
        # m- == "modified for geobench"
        return GeobenchDataset(
            geobench_dir=paths.GEOBENCH_DIR,
            dataset=eval_dataset,
            split=split,
            partition=partition,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            norm_method=norm_method,
        )
    elif eval_dataset == "mados":
        if norm_stats_from_pretrained:
            logger.warning(
                "MADOS has very different norm stats than our pretraining dataset"
            )
        return MADOSDataset(
            path_to_splits=paths.MADOS_DIR,
            split=split,
            partition=partition,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            norm_method=norm_method,
        )
    elif eval_dataset == "sen1floods11":
        return Sen1Floods11Dataset(
            path_to_splits=paths.FLOODS_DIR,
            split=split,
            partition=partition,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            norm_method=norm_method,
        )
    elif eval_dataset.startswith("pastis"):
        kwargs = {
            "split": split,
            "partition": partition,
            "norm_stats_from_pretrained": norm_stats_from_pretrained,
            "input_modalities": input_modalities,
            "norm_method": norm_method,
            "dir_partition": paths.PASTIS_DIR_PARTITION,
        }
        if "128" in eval_dataset:
            # "pastis128"
            kwargs["path_to_splits"] = paths.PASTIS_DIR_ORIG
        else:
            kwargs["path_to_splits"] = paths.PASTIS_DIR
        return PASTISRDataset(**kwargs)  # type: ignore
    elif eval_dataset == "breizhcrops":
        return BreizhCropsDataset(
            path_to_splits=paths.BREIZHCROPS_DIR,
            split=split,
            partition=partition,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            norm_method=norm_method,
        )
    else:
        eval_dataset_entry = get_dataset_entry(eval_dataset)
        return from_registry_entry(
            entry=eval_dataset_entry,
            split=split,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            norm_method=norm_method,
            input_modalities_override=input_modalities if input_modalities else None,
        )
