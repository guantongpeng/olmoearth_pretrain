"""评估数据集配置中心。

本模块定义了所有评估数据集的配置信息，包括任务类型、类别数、
是否多标签、支持的模态、归一化插补规则等。

主要组件：
- EvalDatasetConfig: 数据集配置数据类
- DATASET_TO_CONFIG: 硬编码的数据集配置字典
- dataset_to_config: 根据名称获取配置（优先查硬编码，再查注册表）
- get_eval_mode: 根据任务类型获取默认评估方法

已配置的数据集包括：
  pretrain_subset, m-eurosat, m-bigearthnet, m-so2sat, m-brick-kiln,
  m-sa-crop-type, m-cashew-plant, m-forestnet, mados, sen1floods11,
  pastis, pastis128, breizhcrops, nandi, awf
"""

from dataclasses import asdict, dataclass
from typing import Any

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.studio_ingest import get_dataset_entry
from olmoearth_pretrain.evals.task_types import TaskType


def get_eval_mode(task_type: TaskType) -> str:
    """根据任务类型获取默认评估方法。

    分类任务默认使用 KNN，分割任务默认使用线性探针。

    Args:
        task_type: 任务类型

    Returns:
        str: 评估方法名称 ("knn" 或 "linear_probe")
    """
    if task_type == TaskType.CLASSIFICATION:
        return "knn"
    else:
        return "linear_probe"


__all__ = ["TaskType", "get_eval_mode", "EvalDatasetConfig"]


@dataclass
class EvalDatasetConfig:
    """评估数据集配置数据类，包含数据集的所有元信息。

    Attributes:
        task_type: 任务类型（分类/分割）
        imputes: 缺失波段的插补规则列表，每项为 (源波段, 目标波段) 元组
        num_classes: 类别数量
        is_multilabel: 是否为多标签分类
        supported_modalities: 支持的模态列表
        height_width: 输入/输出的高度和宽度（仅分割任务需要）
        timeseries: 是否为时间序列数据集
    """

    task_type: TaskType
    imputes: list[tuple[str, str]]
    num_classes: int
    is_multilabel: bool
    supported_modalities: list[str]
    # this is only necessary for segmentation tasks,
    # and defines the input / output height width.
    height_width: int | None = None
    timeseries: bool = False

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典，将 TaskType 枚举转换为其值字符串。"""
        d = asdict(self)
        d["task_type"] = self.task_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvalDatasetConfig":
        """从字典反序列化，将字符串转换回 TaskType 枚举和元组。"""
        d = d.copy()
        d["task_type"] = TaskType(d["task_type"])
        d["imputes"] = [tuple(x) for x in d["imputes"]]
        return cls(**d)


# 所有硬编码的评估数据集配置
# 键: 数据集名称，值: EvalDatasetConfig 实例
DATASET_TO_CONFIG = {
    # Dummy config — only used for embedding diagnostics, not actual classification.
    "pretrain_subset": EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[],
        num_classes=1,
        is_multilabel=False,
        supported_modalities=[
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
            Modality.LANDSAT.name,
        ],
    ),
    "m-eurosat": EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[],
        num_classes=10,
        is_multilabel=False,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
    ),
    "m-bigearthnet": EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[("11 - SWIR", "10 - SWIR - Cirrus")],
        num_classes=43,
        is_multilabel=True,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
    ),
    "m-so2sat": EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[
            ("02 - Blue", "01 - Coastal aerosol"),
            ("08A - Vegetation Red Edge", "09 - Water vapour"),
            ("11 - SWIR", "10 - SWIR - Cirrus"),
        ],
        num_classes=17,
        is_multilabel=False,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
    ),
    "m-brick-kiln": EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[],
        num_classes=2,
        is_multilabel=False,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
    ),
    "m-sa-crop-type": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[("11 - SWIR", "10 - SWIR - Cirrus")],
        num_classes=10,
        is_multilabel=False,
        height_width=256,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
    ),
    "m-cashew-plant": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[("11 - SWIR", "10 - SWIR - Cirrus")],
        num_classes=7,
        is_multilabel=False,
        height_width=256,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
    ),
    "m-forestnet": EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[
            # src (we have), tgt (we want), using the geobench L8 names
            # we don't need to impute B8 since our band name conversion does it for us
            ("02 - Blue", "01 - Coastal aerosol"),
            ("07 - SWIR2", "09 - Cirrus"),
            ("07 - SWIR2", "10 - Tirs1"),
        ],
        num_classes=12,
        is_multilabel=False,
        supported_modalities=[Modality.LANDSAT.name],
    ),
    "mados": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[
            ("05 - Vegetation Red Edge", "06 - Vegetation Red Edge"),
            ("08A - Vegetation Red Edge", "09 - Water vapour"),
            ("11 - SWIR", "10 - SWIR - Cirrus"),
        ],
        num_classes=15,
        is_multilabel=False,
        height_width=80,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
    ),
    "sen1floods11": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=2,
        is_multilabel=False,
        height_width=64,
        supported_modalities=[Modality.SENTINEL1.name],
    ),
    "pastis": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=19,
        is_multilabel=False,
        height_width=64,
        supported_modalities=[Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name],
        timeseries=True,
    ),
    "pastis128": EvalDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=19,
        is_multilabel=False,
        height_width=128,
        supported_modalities=[Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name],
        timeseries=True,
    ),
    "breizhcrops": EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[],
        num_classes=9,
        is_multilabel=False,
        height_width=1,
        supported_modalities=[Modality.SENTINEL2_L2A.name],
        timeseries=True,
    ),
    "nandi": EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[],
        num_classes=6,
        is_multilabel=False,
        supported_modalities=[
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
            Modality.LANDSAT.name,
        ],
        timeseries=True,
    ),
    "awf": EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[],
        num_classes=9,
        is_multilabel=False,
        supported_modalities=[
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
            Modality.LANDSAT.name,
        ],
        timeseries=True,
    ),
}


def dataset_to_config(dataset: str) -> EvalDatasetConfig:
    """Get EvalDatasetConfig by name, checking both hardcoded and registry.

    First checks DATASET_TO_CONFIG dict, then falls back to registry.

    Args:
        dataset: Dataset name to look up.

    Returns:
        EvalDatasetConfig for the dataset.

    Raises:
        ValueError: If dataset not found in either location.
    """
    if dataset in DATASET_TO_CONFIG:
        return DATASET_TO_CONFIG[dataset]

    entry = get_dataset_entry(dataset)
    return entry.to_eval_config()
