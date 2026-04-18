"""Presto 模型评估启动脚本。

配置模型参数、数据加载器和评估流程，
将 Presto 模型集成到 OlmoEarth 评估框架中运行。
"""

import logging

from olmoearth_pretrain.evals.models import PrestoConfig
from olmoearth_pretrain.internal.experiment import (
    CommonComponents,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config for an experiment."""
    model_config = PrestoConfig()
    return model_config
