"""类型别名定义模块。

本模块定义了 OlmoEarth Pretrain 库中使用的核心类型别名，
统一 numpy 数组和 PyTorch 张量的类型标注，使函数签名更加简洁，
同时支持数据处理流水线中 numpy 和 torch 之间的灵活切换。

使用场景：
    - 数据预处理阶段通常使用 numpy 进行高效数组操作
    - 模型训练和推理阶段使用 PyTorch 张量
    - ArrayTensor 类型别名允许函数同时接受两种类型，提高代码复用性
"""

from typing import TypeAlias

import numpy as np
import torch

# 统一的数组/张量类型别名，表示既可以是 numpy.ndarray 也可以是 torch.Tensor
# 在数据处理流水线中，数据可能在 numpy 和 torch 之间转换，
# 使用此类型别名可以简化类型标注，避免重复写 np.ndarray | torch.Tensor
ArrayTensor: TypeAlias = np.ndarray | torch.Tensor
