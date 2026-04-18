"""评估基线模型包。

本模块定义了所有参与评估对比的基线模型，包括模型类、配置类和启动脚本路径。

支持的基线模型：
- DINOv3: Meta 的 DINOv3 视觉基础模型
- Panopticon: 多模态遥感基础模型
- Galileo: Galileo 地球观测基础模型
- Satlas: Satlas 遥感基础模型
- Croma: CROMA 遥感基础模型（多尺寸）
- Presto: Presto 时间序列基础模型
- AnySat: AnySat 通用卫星基础模型
- Tessera: Tessera 遥感基础模型
- PrithviV2: IBM 的 Prithvi v2 遥感基础模型（多尺寸）
- Terramind: Terramind 地球观测基础模型（多尺寸）
- Clay: Clay 遥感基础模型

主要组件：
- BaselineModelName: 基线模型名称枚举
- MODELS_WITH_MULTIPLE_SIZES: 具有多个尺寸变体的模型映射
- get_launch_script_path: 获取模型的启动脚本路径
"""

from enum import StrEnum
from typing import Any

from olmoearth_pretrain.evals.models.anysat.anysat import AnySat, AnySatConfig
from olmoearth_pretrain.evals.models.clay.clay import Clay, ClayConfig
from olmoearth_pretrain.evals.models.croma.croma import CROMA_SIZES, Croma, CromaConfig
from olmoearth_pretrain.evals.models.dinov3.constants import DinoV3Models
from olmoearth_pretrain.evals.models.dinov3.dinov3 import DINOv3, DINOv3Config
from olmoearth_pretrain.evals.models.galileo import GalileoConfig, GalileoWrapper
from olmoearth_pretrain.evals.models.galileo.single_file_galileo import (
    MODEL_SIZE_TO_WEKA_PATH as GALILEO_MODEL_SIZE_TO_WEKA_PATH,
)
from olmoearth_pretrain.evals.models.panopticon.panopticon import (
    Panopticon,
    PanopticonConfig,
)
from olmoearth_pretrain.evals.models.presto.presto import PrestoConfig, PrestoWrapper
from olmoearth_pretrain.evals.models.prithviv2.prithviv2 import (
    PrithviV2,
    PrithviV2Config,
    PrithviV2Models,
)
from olmoearth_pretrain.evals.models.satlas.satlas import Satlas, SatlasConfig
from olmoearth_pretrain.evals.models.terramind.terramind import (
    TERRAMIND_SIZES,
    Terramind,
    TerramindConfig,
)
from olmoearth_pretrain.evals.models.tessera.tessera import Tessera, TesseraConfig


class BaselineModelName(StrEnum):
    """基线模型名称枚举。

    Attributes:
        DINO_V3: Meta DINOv3 视觉基础模型
        PANOPTICON: Panopticon 多模态遥感基础模型
        GALILEO: Galileo 地球观测基础模型
        SATLAS: Satlas 遥感基础模型
        CROMA: CROMA 遥感基础模型
        PRESTO: Presto 时间序列基础模型
        ANYSAT: AnySat 通用卫星基础模型
        TESSERA: Tessera 遥感基础模型
        PRITHVI_V2: IBM Prithvi v2 遥感基础模型
        TERRAMIND: Terramind 地球观测基础模型
        CLAY: Clay 遥感基础模型
    """

    DINO_V3 = "dino_v3"
    PANOPTICON = "panopticon"
    GALILEO = "galileo"
    SATLAS = "satlas"
    CROMA = "croma"
    PRESTO = "presto"
    ANYSAT = "anysat"
    TESSERA = "tessera"
    PRITHVI_V2 = "prithvi_v2"
    TERRAMIND = "terramind"
    CLAY = "clay"


# 具有多个尺寸变体的模型映射
# 键: 模型名称枚举值，值: 可用尺寸列表
MODELS_WITH_MULTIPLE_SIZES: dict[BaselineModelName, Any] = {
    BaselineModelName.CROMA: CROMA_SIZES,          # CROMA 多尺寸
    BaselineModelName.DINO_V3: list(DinoV3Models),  # DINOv3 多尺寸
    BaselineModelName.GALILEO: GALILEO_MODEL_SIZE_TO_WEKA_PATH.keys(),  # Galileo 多尺寸
    BaselineModelName.PRITHVI_V2: list(PrithviV2Models),  # Prithvi v2 多尺寸
    BaselineModelName.TERRAMIND: TERRAMIND_SIZES,  # Terramind 多尺寸
}


def get_launch_script_path(model_name: str) -> str:
    """获取模型的启动脚本路径。

    Args:
        model_name: 模型名称

    Returns:
        str: 启动脚本的相对路径

    Raises:
        ValueError: 如果模型名称无效
    """
    if model_name == BaselineModelName.DINO_V3:
        return "olmoearth_pretrain/evals/models/dinov3/dino_v3_launch.py"
    elif model_name == BaselineModelName.GALILEO:
        return "olmoearth_pretrain/evals/models/galileo/galileo_launch.py"
    elif model_name == BaselineModelName.PANOPTICON:
        return "olmoearth_pretrain/evals/models/panopticon/panopticon_launch.py"
    elif model_name == BaselineModelName.TERRAMIND:
        return "olmoearth_pretrain/evals/models/terramind/terramind_launch.py"
    elif model_name == BaselineModelName.SATLAS:
        return "olmoearth_pretrain/evals/models/satlas/satlas_launch.py"
    elif model_name == BaselineModelName.CROMA:
        return "olmoearth_pretrain/evals/models/croma/croma_launch.py"
    elif model_name == BaselineModelName.CLAY:
        return "olmoearth_pretrain/evals/models/clay/clay_launch.py"
    elif model_name == BaselineModelName.PRESTO:
        return "olmoearth_pretrain/evals/models/presto/presto_launch.py"
    elif model_name == BaselineModelName.ANYSAT:
        return "olmoearth_pretrain/evals/models/anysat/anysat_launch.py"
    elif model_name == BaselineModelName.TESSERA:
        return "olmoearth_pretrain/evals/models/tessera/tessera_launch.py"
    elif model_name == BaselineModelName.PRITHVI_V2:
        return "olmoearth_pretrain/evals/models/prithviv2/prithviv2_launch.py"
    else:
        raise ValueError(f"Invalid model name: {model_name}")


# TODO: assert that they all store a patch_size variable and supported modalities
__all__ = [
    "Panopticon",
    "PanopticonConfig",
    "GalileoWrapper",
    "GalileoConfig",
    "DINOv3",
    "DINOv3Config",
    "Terramind",
    "TerramindConfig",
    "Satlas",
    "SatlasConfig",
    "Croma",
    "CromaConfig",
    "Clay",
    "ClayConfig",
    "PrestoWrapper",
    "PrestoConfig",
    "AnySat",
    "AnySatConfig",
    "Tessera",
    "TesseraConfig",
    "PrithviV2",
    "PrithviV2Config",
]
