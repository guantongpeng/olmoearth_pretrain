"""OlmoEarth Pretrain 模型的位置编码模块。

本模块提供多种位置编码函数，用于为 Transformer 输入添加位置信息：
- 1D 正弦-余弦位置编码：用于时间维度
- 2D 正弦-余弦位置编码：用于空间维度（H, W）
- 2D 带分辨率的位置编码：考虑地面采样距离（GSD）的空间编码
- 月份编码：用于时间维度中的月份信息

参考实现：
https://github.com/bair-climate-initiative/scale-mae/blob/main/mae/util/pos_embed.py
"""

import numpy as np
import torch


def get_1d_sincos_pos_encoding(pos: torch.Tensor, encoding_dim: int) -> torch.Tensor:
    """计算 1D 正弦-余弦位置编码。

    对给定的位置序列计算正弦-余弦编码，前半维度使用 sin，后半维度使用 cos。
    频率从 1 到 10000^(2/D) 呈对数等间距分布。

    Args:
        pos: 位置序列，形状 (L,)，可以是时间或空间坐标
        encoding_dim: 编码输出维度 D（必须为偶数）

    Returns:
        位置编码，形状 (L, D)
    """
    assert encoding_dim % 2 == 0, f"encoding_dim must be even, got {encoding_dim}"
    omega = torch.arange(encoding_dim // 2, device=pos.device) / encoding_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,) — 频率序列，从 1 到 1/10000

    pos = pos.reshape(-1)  # (L,)
    out = torch.einsum("l,d->ld", pos, omega)  # (L, D/2) — 外积：每个位置 × 每个频率
    encoding_sin = torch.sin(out)  # (L, D/2) — 正弦部分
    encoding_cos = torch.cos(out)  # (L, D/2) — 余弦部分

    encoding = torch.cat([encoding_sin, encoding_cos], dim=1)  # (L, D) — 拼接 sin 和 cos
    return encoding


def get_2d_sincos_pos_encoding(grid: torch.Tensor, encoding_dim: int) -> torch.Tensor:
    """计算 2D 正弦-余弦位置编码。

    将编码维度平均分配给水平和垂直方向，分别计算 1D 编码后拼接。
    前 D/2 维用于水平位置编码，后 D/2 维用于垂直位置编码。

    Args:
        grid: 位置网格，形状 2 × H × W，grid[0] 为水平坐标，grid[1] 为垂直坐标
        encoding_dim: 编码输出维度 D（必须为偶数）

    Returns:
        位置编码，形状 (H*W, D)
    """
    assert encoding_dim % 2 == 0

    # 将维度平均分配给 h 和 w 方向
    encoding_dim_1d = encoding_dim // 2
    emb_h = get_1d_sincos_pos_encoding(grid[0], encoding_dim_1d)  # (H*W, D/2) — 水平编码
    emb_w = get_1d_sincos_pos_encoding(grid[1], encoding_dim_1d)  # (H*W, D/2) — 垂直编码

    emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D) — 拼接两个方向
    return emb


def get_2d_sincos_pos_encoding_with_resolution(
    grid_size: int | tuple[int, int],
    res: torch.Tensor,
    encoding_dim: int,
    device: torch.device,
    cls_token: bool = False,
) -> torch.Tensor:
    """计算带分辨率信息的 2D 正弦-余弦位置编码。

    在标准 2D 位置编码的基础上，将空间坐标乘以分辨率（GSD），
    使得编码能够感知不同空间分辨率下的实际地面距离。
    这对于遥感数据特别重要，因为不同传感器的空间分辨率可能不同。

    Args:
        grid_size: 网格大小。若为整数则使用方形网格 (H=W=grid_size)；
            若为元组则解释为 (H, W)
        res: 分辨率数组，形状 (n,)，表示每个像素的地面采样距离（如米），
            n 为空间维度数量
        encoding_dim: 编码输出维度 D
        cls_token: 是否在编码前添加零值的 CLS token 位置
        device: 计算设备

    Returns:
        位置编码，形状 (n, H*W, D) 或 (n, 1+H*W, D)（含 CLS token 时）
    """
    # TODO: What happens when the res array is bigger than 1?
    if isinstance(grid_size, tuple):
        grid_h_size, grid_w_size = grid_size
    else:
        grid_h_size = grid_w_size = grid_size

    grid_h = torch.arange(grid_h_size, device=device)  # 水平坐标
    grid_w = torch.arange(grid_w_size, device=device)  # 垂直坐标
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")  # (h_grid, w_grid) — 生成网格
    grid = torch.stack(grid, dim=0)  # 2 x h x w

    # 将坐标乘以分辨率，得到以地面距离为单位的坐标
    grid = torch.einsum("chw,n->cnhw", grid, res)  # 2 x n x h x w
    _, n, h, w = grid.shape
    pos_embed = get_2d_sincos_pos_encoding(grid, encoding_dim)  # (nxH*W, D/2)
    pos_embed = pos_embed.reshape(n, h * w, encoding_dim)
    if cls_token:
        pos_embed = torch.cat(
            [
                torch.zeros([n, 1, encoding_dim], device=pos_embed.device),
                pos_embed,
            ],
            dim=1,
        )
    return pos_embed


def get_month_encoding_table(encoding_dim: int) -> torch.Tensor:
    """生成月份的正弦编码表，用于 12 个月份（索引 0-11）。

    对 0-12 共 13 个角度值计算正弦和余弦编码，
    然后去掉最后一个（索引 12），保留 12 个月份的编码。

    Args:
        encoding_dim: 编码输出维度 D（必须为偶数）

    Returns:
        月份编码表，形状 (M, D)，M=12
    """
    assert encoding_dim % 2 == 0
    angles = torch.arange(0, 13) / (12 / (2 * np.pi))

    dim_per_table = encoding_dim // 2
    sin_table = torch.sin(torch.stack([angles for _ in range(dim_per_table)], axis=-1))
    cos_table = torch.cos(torch.stack([angles for _ in range(dim_per_table)], axis=-1))
    month_table = torch.concatenate([sin_table[:-1], cos_table[:-1]], axis=-1)

    return month_table  # (M, D)
