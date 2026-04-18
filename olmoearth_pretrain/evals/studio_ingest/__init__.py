"""Studio 数据集摄取模块（仅限内部使用）。

本模块提供从 Studio 平台摄取数据集到 OlmoEarth 评估系统的工具。

核心功能：
1. 从 GCS 复制数据到 Weka 存储
2. 计算归一化统计量（通过 band_stats.py）
3. 在评估注册表中注册数据集

对外部用户的说明：
  设置 OLMOEARTH_EVAL_DATASETS 环境变量指向本地下载的 rslearn 数据集目录。

使用方式：
    python -m olmoearth_pretrain.evals.studio_ingest.cli ingest ...

    # 单独计算波段统计量：
    python -m olmoearth_pretrain.evals.studio_ingest.band_stats ...
"""

from olmoearth_pretrain.evals.studio_ingest.registry import (
    Registry,
    get_dataset_entry,
    list_dataset_names,
    load_registry,
)
from olmoearth_pretrain.evals.studio_ingest.schema import EvalDatasetEntry

__all__ = [
    "EvalDatasetEntry",
    "Registry",
    "get_dataset_entry",
    "list_dataset_names",
    "load_registry",
]
