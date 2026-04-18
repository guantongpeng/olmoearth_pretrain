"""灵活 Patch 嵌入模块 (Flexible Patch Embedding)。

本模块实现了 FlexiViT 架构中的核心 patch 嵌入和重建操作，支持在推理时
动态调整 patch 大小（通过双三次插值调整卷积核权重），而无需重新训练模型。

主要组件：
- FlexiPatchEmbed: 将 2D 图像转换为灵活大小的 patch token 嵌入
- FlexiPatchReconstruction: 将 patch token 重建为 2D 图像（解码器端）

扩展自：https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py#L24
以及：https://github.com/bwconrad/flexivit/

设计思路：
- 基准 patch 大小决定了卷积核参数的实际尺寸
- 当请求不同的 patch 大小时，通过 F.interpolate 对权重进行插值调整
- 支持 nn.Linear（默认，利用 cuBLAS GEMM 加速）和 nn.Conv2d 两种投影方式
"""

import logging
from collections.abc import Iterable

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from olmoearth_pretrain.data.constants import ModalitySpec

logger = logging.getLogger(__name__)


def _to_2tuple(x: int | tuple[int, ...]) -> tuple[int, int]:
    """将标量或二元可迭代对象转换为 (h, w) 二元组。

    Args:
        x: 整数或包含两个元素的元组/列表。若为整数，则转换为 (x, x)。

    Returns:
        (h, w) 形式的二元组。

    Raises:
        TypeError: 当输入既非整数也非二元可迭代对象时。
    """
    if isinstance(x, int):
        return (x, x)  # 整数转对称二元组
    if isinstance(x, Iterable) and not isinstance(x, str):
        values = tuple(x)
        assert len(values) == 2, "x must be a 2-tuple"
        return (int(values[0]), int(values[1]))
    raise TypeError(f"Expected int or tuple[int, int], got {type(x)}")


class FlexiPatchEmbed(nn.Module):
    """灵活 Patch 嵌入模块，将 2D 图像划分为 patch 并投影到嵌入空间。

    支持在推理时动态改变 patch 大小，通过插值调整投影权重实现。
    这是 FlexiViT 的核心组件，使得模型可以在不同分辨率/patch 大小下运行。

    关键属性：
        base_patch_size: 基准 patch 大小（由 max_patch_size_at_16 和 modality 的 tile_size_factor 决定）
        proj: 投影层（nn.Linear 或 nn.Conv2d）
        norm: 可选的归一化层
        interpolation: 插值方式（默认 bicubic）
        antialias: 是否启用抗锯齿

    使用场景：
        作为编码器的 patch 嵌入层，将输入图像 [B, H, W, C] 转换为
        patch token [B, H/P, W/P, D]，其中 P 为 patch 大小，D 为嵌入维度。
    """

    def __init__(
        self,
        modality_spec: ModalitySpec,
        base_patch_size_at_16: int | tuple[int, int],
        in_chans: int = 3,
        embedding_size: int = 128,
        norm_layer: nn.Module | None = None,
        bias: bool = True,
        interpolation: str = "bicubic",
        antialias: bool = True,
        use_linear_patch_embed: bool = True,
    ) -> None:
        """2D 图像到 patch 嵌入的转换，支持灵活的 patch 大小。

        核心原理：卷积核的参数以基准 patch 大小存储，当请求不同的 patch 大小时，
        通过双三次插值调整卷积核大小，使得模型无需重新训练即可适配不同分辨率。

        Args:
            modality_spec: 该模态的规格说明，包含通道顺序、band 分组等信息
            base_patch_size_at_16: 在分辨率为 16 时的基准 patch 大小，
                实际基准 patch 大小 = base_patch_size_at_16 * image_tile_size_factor
            in_chans: 输入图像的通道数
            embedding_size: 网络嵌入维度大小
            norm_layer: 可选的归一化层（如 LayerNorm）
            bias: 卷积/线性层是否使用偏置
            interpolation: 权重插值方式（'bicubic' 或 'bilinear'）
            antialias: 插值时是否启用抗锯齿（推荐开启以提高质量）
            use_linear_patch_embed: 若为 True，使用 nn.Linear（reshape + matmul，
                利用 cuBLAS GEMM 加速，在 TensorCore 上始终高效）；
                若为 False，使用 nn.Conv2d（用于加载旧检查点兼容性）。
        """
        super().__init__()

        self.embedding_size = embedding_size
        self.use_linear_patch_embed = use_linear_patch_embed

        self.modality_spec = modality_spec
        self.base_patch_size = _to_2tuple(
            base_patch_size_at_16 * modality_spec.image_tile_size_factor
        )  # 实际基准 patch 大小 = 基准值 × 模态分块因子

        p_h, p_w = self.base_patch_size
        if use_linear_patch_embed:
            # 将 patch 展平为 (p1*p2*c) 维向量后通过线性层投影
            # 优势：利用 cuBLAS GEMM（在 TensorCore 上始终高效）
            # 相比 Conv2d 在小 in_chans 时走慢速 cuDNN 路径更快
            self.proj = nn.Linear(in_chans * p_h * p_w, embedding_size, bias=bias)
            # 保持 PyTorch 默认的 nn.Linear 初始化（kaiming_uniform_）
            # 以匹配先前 Conv2d 的行为；用编码器级 Xavier 初始化覆盖此值
            # 曾导致 PASTIS 回归问题
            self.proj._skip_custom_init = True
        else:
            self.proj = nn.Conv2d(
                in_chans,
                embedding_size,
                kernel_size=self.base_patch_size,
                stride=self.base_patch_size,
                bias=bias,
            )
        self.norm = norm_layer(embedding_size) if norm_layer else nn.Identity()
        self.interpolation = interpolation
        self.antialias = antialias

    def _resolve_patch_size(
        self, patch_size: int | tuple[int, int] | None
    ) -> tuple[int, int]:
        """解析有效的 patch 大小，应用模态分块因子。

        如果未指定 patch_size，则返回基准 patch 大小；
        否则将请求的 patch_size 乘以模态的 image_tile_size_factor。

        Args:
            patch_size: 请求的 patch 大小，若为 None 则使用基准大小。

        Returns:
            解析后的 (p_h, p_w) 二元组。
        """
        if not patch_size:
            return self.base_patch_size
        if isinstance(patch_size, tuple):
            patch_size = (
                patch_size[0] * self.modality_spec.image_tile_size_factor,
                patch_size[1] * self.modality_spec.image_tile_size_factor,
            )
        else:
            patch_size = patch_size * self.modality_spec.image_tile_size_factor
        resolved = _to_2tuple(patch_size)
        assert isinstance(resolved, tuple) and len(resolved) == 2
        return resolved

    def _project_linear(
        self,
        x: Tensor,
        h_patches: int,
        w_patches: int,
        batch_size: int,
        has_time_dim: bool,
        num_timesteps: int,
    ) -> Tensor:
        """使用 nn.Linear 投影 patch（reshape → cuBLAS GEMM → reshape）。

        核心逻辑：将输入 [B, C, H, W] 重组为 [B, H/P*W/P, P*P*C]，
        然后通过线性层投影到嵌入维度，最后重组回空间维度。

        Args:
            x: 输入张量 [B*C, C, H, W]（时间维已折叠到批次维）
            h_patches: 高度方向的 patch 数量
            w_patches: 宽度方向的 patch 数量
            batch_size: 原始批次大小（时间维折叠前）
            has_time_dim: 输入是否包含时间维度
            num_timesteps: 时间步数

        Returns:
            投影后的张量，形状为 [B, H/P, W/P, D] 或 [B, H/P, W/P, T, D]
        """
        p_h, p_w = self.base_patch_size
        x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p_h, p2=p_w)
        x = self.proj(x)
        if has_time_dim:
            return rearrange(
                x,
                "(b t) (h w) d -> b h w t d",
                b=batch_size,
                t=num_timesteps,
                h=h_patches,
                w=w_patches,
            )
        return rearrange(x, "b (h w) d -> b h w d", h=h_patches, w=w_patches)

    def _project_conv(
        self,
        x: Tensor,
        batch_size: int,
        has_time_dim: bool,
        num_timesteps: int,
    ) -> Tensor:
        """使用 nn.Conv2d 投影 patch（用于加载旧版 Linear 前的检查点兼容）。

        Args:
            x: 输入张量 [B*C, C, H, W]
            batch_size: 原始批次大小
            has_time_dim: 输入是否包含时间维度
            num_timesteps: 时间步数

        Returns:
            投影后的张量，形状为 [B, H/P, W/P, D] 或 [B, H/P, W/P, T, D]
        """
        x = self.proj(x)  # b c h w -> b d h_out w_out
        if has_time_dim:
            _, d, h, w = x.shape
            return rearrange(
                x,
                "(b t) d h w -> b h w t d",
                b=batch_size,
                t=num_timesteps,
                h=h,
                w=w,
            )
        return rearrange(x, "b d h w -> b h w d")

    def forward(
        self,
        x: Tensor,
        patch_size: int | tuple[int, int] | None = None,
    ) -> Tensor:
        """FlexiPatchEmbed 前向传播。

        核心流程：
        1. 将输入 [B, H, W, (T), C] 重排为标准卷积输入格式 [B*T, C, H, W]
        2. 若请求的 patch 大小与基准不同，通过插值调整图像大小以匹配基准 patch
        3. 使用投影层（Linear 或 Conv2d）进行 patch 嵌入
        4. 应用归一化层

        Args:
            x: 输入张量，形状为 [B, H, W, C] 或 [B, H, W, T, C]
            patch_size: 请求使用的 patch 大小。若为 None，使用基准 patch 大小。
                当与基准不同时，通过插值调整输入图像大小。

        Returns:
            Patch 嵌入张量，形状为 [B, H/P, W/P, D] 或 [B, H/P, W/P, T, D]
        """
        batch_size = x.shape[0]
        has_time_dim = len(x.shape) == 5
        num_timesteps = x.shape[3] if has_time_dim else 0

        if has_time_dim:
            x = rearrange(x, "b h w t c -> (b t) c h w")  # 将时间维折叠到批次维
        else:
            x = rearrange(x, "b h w c -> b c h w")  # 转为标准卷积输入格式

        req_patch_size = self._resolve_patch_size(patch_size)  # 解析实际 patch 大小

        if req_patch_size != self.base_patch_size:
            # 当请求 patch 大小与基准不同时，通过插值调整图像大小
            # 使得基准 patch 能整除调整后的图像大小
            shape = x.shape[-2:]
            new_shape = (
                shape[0] // req_patch_size[0] * self.base_patch_size[0],
                shape[1] // req_patch_size[1] * self.base_patch_size[1],
            )
            x = F.interpolate(
                x, size=new_shape, mode=self.interpolation, antialias=self.antialias
            )

        p_h, p_w = self.base_patch_size
        h_patches, w_patches = x.shape[-2] // p_h, x.shape[-1] // p_w  # 计算 patch 数量

        if self.use_linear_patch_embed:
            x = self._project_linear(  # 使用线性投影（推荐，更快）
                x, h_patches, w_patches, batch_size, has_time_dim, num_timesteps
            )
        else:
            x = self._project_conv(x, batch_size, has_time_dim, num_timesteps)  # 使用卷积投影（旧检查点兼容）

        return self.norm(x)  # 应用归一化


class FlexiPatchReconstruction(nn.Module):
    """灵活 Patch 重建模块，将 patch token 解码回 2D 图像空间。

    使用转置卷积（ConvTranspose2d）将低分辨率的 patch token 重建为
    像素级输出。支持灵活的 patch 大小，当请求大小与基准不同时，
    通过插值调整重建结果。

    关键属性：
        max_patch_size: 基准（最大）patch 大小
        proj: 转置卷积层（ConvTranspose2d）
        norm: 可选的归一化层

    使用场景：
        作为解码器的重建头，将编码器输出的 patch token [B, H, W, D]
        重建为图像 [B, H*P, W*P, C]，用于 MAE 重建任务。
    """

    def __init__(
        self,
        max_patch_size: int | tuple[int, int],
        out_chans: int = 3,
        embedding_size: int = 128,
        norm_layer: nn.Module | None = None,
        bias: bool = True,
        interpolation: str = "bicubic",
        antialias: bool = True,
    ) -> None:
        """Patch 嵌入到 2D 图像的重建，支持灵活的 patch 大小。

        Args:
            max_patch_size: 基准 patch 大小，即转置卷积核参数的实际尺寸
            out_chans: 输出图像的通道数
            embedding_size: 网络嵌入维度大小
            norm_layer: 可选的归一化层
            bias: 转置卷积是否使用偏置
            interpolation: 重建插值方式
            antialias: 插值时是否启用抗锯齿
        """
        super().__init__()

        self.embedding_size = embedding_size

        self.max_patch_size = _to_2tuple(max_patch_size)

        self.proj = nn.ConvTranspose2d(
            embedding_size,
            out_chans,
            kernel_size=max_patch_size,
            stride=max_patch_size,
            bias=bias,
        )
        self.norm = norm_layer(embedding_size) if norm_layer else nn.Identity()
        self.interpolation = interpolation
        self.antialias = antialias

    def _resize(self, x: Tensor, shape: tuple[int, int]) -> Tensor:
        """将输入张量插值到目标形状。

        通过在张量前后添加虚拟维度使其符合 F.interpolate 的 4D 输入要求。

        Args:
            x: 输入张量
            shape: 目标空间形状 (H, W)

        Returns:
            插值后的张量
        """
        x_resized = F.interpolate(
            x[None, None, ...],
            shape,
            mode=self.interpolation,
            antialias=self.antialias,
        )
        return x_resized[0, 0, ...]

    def forward(
        self,
        x: Tensor,
        patch_size: int | tuple[int, int] | None = None,
    ) -> Tensor | tuple[Tensor, tuple[int, int]]:
        """FlexiPatchReconstruction 前向传播。

        核心流程：
        1. 将输入 [B, H, W, (T), D] 转为转置卷积格式
        2. 使用 ConvTranspose2d 重建到像素空间
        3. 若请求的 patch 大小与基准不同，通过插值调整重建结果
        4. 应用归一化层

        Args:
            x: 输入张量，形状为 [B, H, W, D] 或 [B, H, W, T, D]
            patch_size: 请求使用的 patch 大小。若为 None，使用基准 patch 大小。

        Returns:
            重建张量，形状为 [B, H*P, W*P, C] 或 [B, H*P, W*P, T, C]
        """
        # x has input shape [b, h, w, (t), d]
        if len(x.shape) == 4:
            has_time_dimension = False
            b, h, w, d = x.shape
            t = 1
        else:
            has_time_dimension = True
            b, h, w, t, d = x.shape

        if not patch_size:
            # 评估时若未指定，使用基准 patch 大小
            patch_size = self.max_patch_size

        patch_size = _to_2tuple(patch_size)

        if has_time_dimension:
            x = rearrange(x, "b h w t d -> (b t) d h w", b=b, t=t)
        else:
            x = rearrange(x, "b h w d -> b d h w")

        x = self.proj(x)

        if patch_size != self.max_patch_size:
            # 当请求 patch 大小与基准不同时，需要插值调整重建结果
            # 先将重建结果拆分为单独的 patch，再对每个 patch 插值到目标大小
            x = rearrange(
                x,
                "b c (h p_h) (w p_w) -> b h w c p_h p_w",
                p_h=self.max_patch_size[0],
                p_w=self.max_patch_size[1],
            )
            bl, hl, wl, cl = x.shape[:4]
            x = rearrange(x, "b h w c p_h p_w -> (b h w) c p_h p_w")
            x = F.interpolate(
                x, patch_size, mode=self.interpolation, antialias=self.antialias
            )
            x = rearrange(
                x, "(b h w) c p_h p_w -> b c (h p_h) (w p_w)", b=bl, h=hl, w=wl
            )

        if has_time_dimension:
            x = rearrange(x, "(b t) c h w -> b h w t c", b=b, t=t)
        else:
            x = rearrange(x, "b c h w -> b h w c")

        x = self.norm(x)

        return x
