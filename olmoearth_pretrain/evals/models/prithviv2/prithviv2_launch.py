"""Prithvi v2 模型评估启动脚本。

配置模型参数、数据加载器和评估流程，
将 Prithvi v2 模型集成到 OlmoEarth 评估框架中运行。
"""

import logging

from olmoearth_pretrain.evals.models.prithviv2.prithviv2 import PrithviV2Config
from olmoearth_pretrain.internal.experiment import (
    CommonComponents,
)

logger = logging.getLogger(__name__)


def build_model_config(common: CommonComponents) -> PrithviV2Config:
    """Build the model config for an experiment."""
    model_config = PrithviV2Config()
    return model_config
