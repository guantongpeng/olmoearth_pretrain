"""已废弃的模块别名，请从 olmoearth_pretrain.nn.flexivit 导入。

本模块仅为向后兼容旧检查点而保留，实际功能已迁移到 flexi_vit.py。
导入时会发出 DeprecationWarning 警告，并将模块引用替换为 flexivit 模块。

使用方式：
    # 旧方式（已废弃）：
    from olmoearth_pretrain.nn.flexihelios import Encoder

    # 新方式（推荐）：
    from olmoearth_pretrain.nn.flexi_vit import Encoder
"""

import sys
import warnings

import olmoearth_pretrain.nn.flexi_vit as flexivit

from .flexi_vit import *  # noqa: F403

warnings.warn(
    "olmoearth_pretrain.nn.flexi_vit is deprecated. "
    "Please import from olmoearth_pretrain.nn.flexivit instead.",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = flexivit
