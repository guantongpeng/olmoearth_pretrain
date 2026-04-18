"""Satlas 模型评估启动脚本。

配置模型参数、数据加载器和评估流程，
将 Satlas 模型集成到 OlmoEarth 评估框架中运行。
"""

import logging

from olmoearth_pretrain.evals.models.satlas.satlas import SatlasConfig
from olmoearth_pretrain.internal.experiment import (
    CommonComponents,
)

logger = logging.getLogger(__name__)


def build_model_config(common: CommonComponents) -> SatlasConfig:
    """Build the model config for an experiment."""
    model_config = SatlasConfig()
    return model_config
