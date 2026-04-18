"""AnySat 模型评估启动脚本。

配置模型参数、数据加载器和评估流程，
将 AnySat 模型集成到 OlmoEarth 评估框架中运行。
"""

import logging

from olmoearth_pretrain.evals.models import AnySatConfig
from olmoearth_pretrain.internal.experiment import (
    CommonComponents,
)

logger = logging.getLogger(__name__)


def build_model_config(common: CommonComponents) -> AnySatConfig:
    """Build the model config for an experiment."""
    model_config = AnySatConfig()
    return model_config
