"""任务类型和数据集划分枚举定义。

本模块定义了评估流程中使用的核心枚举类型：
- TaskType: 任务类型（分类/分割）
- SplitName: 数据集划分名称（训练/验证/测试）

这些枚举被整个评估框架广泛引用，是任务配置的基础类型。
"""

from enum import StrEnum


class TaskType(StrEnum):
    """任务类型枚举。

    Attributes:
        CLASSIFICATION: 分类任务
        SEGMENTATION: 分割任务
    """

    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"


class SplitName(StrEnum):
    """标准数据集划分名称枚举。

    Attributes:
        TRAIN: 训练集
        VAL: 验证集
        TEST: 测试集
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
