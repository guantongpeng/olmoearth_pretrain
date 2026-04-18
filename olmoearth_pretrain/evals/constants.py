"""评估模块共享常量。

本模块定义了评估流程中使用的常量映射，特别是将 rslearn 数据集层名称
映射到 OlmoEarth 内部的 ModalitySpec 类型，作为模态类型的唯一真实来源。

关键映射：
- RSLEARN_TO_OLMOEARTH: rslearn 层名 -> OlmoEarth ModalitySpec 的映射字典
  支持的映射包括：sentinel2/sentinel2_l2a -> S2 L2A, sentinel1 -> S1, landsat -> Landsat
"""

from olmoearth_pretrain.data.constants import Modality, ModalitySpec

# rslearn 层名称到 OlmoEarth ModalitySpec 的映射字典
# 这是模态类型转换的唯一真实来源 (single source of truth)
# 键: rslearn 数据集中的层名称
# 值: OlmoEarth 内部对应的 ModalitySpec 枚举值
RSLEARN_TO_OLMOEARTH: dict[str, ModalitySpec] = {
    "sentinel2": Modality.SENTINEL2_L2A,          # Sentinel-2 L1C 层 -> 映射为 L2A
    "sentinel2_l2a": Modality.SENTINEL2_L2A,      # Sentinel-2 L2A 层
    "sentinel1": Modality.SENTINEL1,              # Sentinel-1 层（不区分升降轨）
    "sentinel1_ascending": Modality.SENTINEL1,    # Sentinel-1 升轨 -> 映射为通用 S1
    "sentinel1_descending": Modality.SENTINEL1,   # Sentinel-1 降轨 -> 映射为通用 S1
    "landsat": Modality.LANDSAT,                   # Landsat 层
}
