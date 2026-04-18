"""
训练回调模块初始化文件。

本模块导出 OlmoEarth Pretrain 训练过程中使用的所有回调类，包括：
- DownstreamEvaluatorCallbackConfig: 下游任务评估回调配置
- OlmoEarthSpeedMonitorCallback: 训练吞吐量（TPS/BPS）监控回调
- OlmoEarthWandBCallback: Weights & Biases 实验跟踪回调
- HeliosSpeedMonitorCallback / HeliosWandBCallback: 旧名称的兼容别名
"""

"""Callbacks for the trainer specific to OlmoEarth Pretrain."""

# 导入下游评估回调配置
from .evaluator_callback import DownstreamEvaluatorCallbackConfig
# 导入速度监控回调（OlmoEarth 版本和 Helios 兼容别名）
from .speed_monitor import HeliosSpeedMonitorCallback, OlmoEarthSpeedMonitorCallback
# 导入 W&B 日志回调（OlmoEarth 版本和 Helios 兼容别名）
from .wandb import HeliosWandBCallback, OlmoEarthWandBCallback

__all__ = [
    "DownstreamEvaluatorCallbackConfig",
    "OlmoEarthSpeedMonitorCallback",
    "OlmoEarthWandBCallback",
    "HeliosSpeedMonitorCallback",
    "HeliosWandBCallback",
]
