"""OlmoEarth Pretrain 的注意力组件模块。

本模块包含 Transformer 架构中的核心组件：
- dispatch_flash_attn: Flash Attention 调度函数（支持变长序列）
- Attention: 多头注意力模块（支持自注意力和交叉注意力）
- Mlp: 前馈网络（MLP）模块
- LayerScale: 可学习的层缩放（用于深层 ViT 的稳定训练）
- DropPath: 随机深度（Stochastic Depth）正则化
- Block: 完整的 Transformer 块（注意力 + MLP + 残差连接）
"""

from logging import getLogger
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributed.fsdp import fully_shard
from torch.jit import Final

try:
    import flash_attn
except ImportError:
    flash_attn = None

logger = getLogger(__name__)


@torch._dynamo.disable()  # 禁用 torch.compile 以避免与 flash attention 的兼容问题
def dispatch_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """调度 Flash Attention 计算。

    根据输入是否包含变长序列信息（cu_seqlens），自动选择使用
    flash_attn_varlen_func（变长序列）或 flash_attn_func（定长序列）。

    Args:
        q: 查询张量，形状为 [B, N, H, D] 或 [total_tokens, H, D]（变长时）
        k: 键张量
        v: 值张量
        cu_seqlens: 累积序列长度（用于变长 flash attention）
        cu_seqlens_q: 查询的累积序列长度（交叉注意力时使用）
        cu_seqlens_k: 键的累积序列长度（交叉注意力时使用）
        max_seqlen: 最大序列长度
        max_seqlen_q: 查询的最大序列长度
        max_seqlen_k: 键的最大序列长度
        dropout_p: 注意力 dropout 概率
        softmax_scale: softmax 缩放因子（默认为 1/sqrt(d)）
        causal: 是否使用因果注意力掩码

    Returns:
        注意力输出张量

    Raises:
        RuntimeError: 当 flash_attn 未安装时
    """
    if flash_attn is None:
        raise RuntimeError("flash-attn is required!")

    if cu_seqlens is not None:
        if cu_seqlens_q is None:
            cu_seqlens_q = cu_seqlens
        if cu_seqlens_k is None:
            cu_seqlens_k = cu_seqlens
    if max_seqlen is not None:
        if max_seqlen_q is None:
            max_seqlen_q = max_seqlen
        if max_seqlen_k is None:
            max_seqlen_k = max_seqlen

    varlen = all(
        x is not None for x in (cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    )  # 判断是否为变长序列模式

    if varlen:
        assert q.ndim == 3, "q must be pre-packed"  # 变长模式下 q 必须是预打包的
        logger.debug("using varlen")

        return flash_attn.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
        )
    else:
        return flash_attn.flash_attn_func(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
        )


class Attention(nn.Module):
    """多头注意力模块，支持自注意力和交叉注意力。

    该模块实现了标准的多头注意力机制，支持以下三种计算后端：
    1. Flash Attention（使用 flash_attn 库，最高效）
    2. PyTorch SDPA（scaled_dot_product_attention，次高效）
    3. 手动实现（兼容旧版 PyTorch）

    关键属性：
        num_heads: 注意力头数
        head_dim: 每个头的维度（dim / num_heads）
        scale: 注意力缩放因子（1 / sqrt(head_dim)）
        q, k, v: 查询、键、值的线性投影层
        q_norm, k_norm: QK 归一化层（可选，用于稳定训练）
        proj: 输出投影层

    使用场景：
        - 编码器中的自注意力（y=None）
        - 解码器中的交叉注意力（y=encoder_output）
    """

    fast_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        cross_attn: bool = False,
        use_flash_attn: bool = False,
    ) -> None:
        """初始化注意力模块。

        Args:
            dim: 输入维度
            num_heads: 注意力头数
            qkv_bias: QKV 投影是否使用偏置
            qk_norm: 是否对 Q 和 K 应用归一化（QK-Norm，提升训练稳定性）
            attn_drop: 注意力权重 dropout 概率
            proj_drop: 输出投影 dropout 概率
            norm_layer: 归一化层类型
            cross_attn: 是否启用交叉注意力
            use_flash_attn: 是否使用 Flash Attention（需要 flash-attn 库）
        """
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 每个头的维度
        self.scale = self.head_dim**-0.5  # 缩放因子 1/sqrt(d_k)

        self.cross_attn = cross_attn
        self.use_flash_attn = use_flash_attn
        self.fast_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")  # 检测 PyTorch SDPA 是否可用
        self.q = nn.Linear(dim, dim, bias=qkv_bias)  # 查询投影
        self.k = nn.Linear(dim, dim, bias=qkv_bias)  # 键投影
        self.v = nn.Linear(dim, dim, bias=qkv_bias)  # 值投影

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        n: int,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """计算缩放点积注意力（Scaled Dot-Product Attention）。

        根据配置自动选择最高效的计算后端：
        1. Flash Attention（使用 flash_attn 库，支持变长序列）
        2. PyTorch SDPA（使用 F.scaled_dot_product_attention）
        3. 手动实现（兼容旧版 PyTorch，不支持 attn_mask）

        Args:
            q: 查询张量，形状 (B, H, N, D) 或变长格式
            k: 键张量，形状 (B, H, N, D) 或变长格式
            v: 值张量，形状 (B,; H, N, D) 或变长格式
            n: token 数量（用于构建注意力掩码）
            cu_seqlens: 变长 flash attention 的累积序列长度
            cu_seqlens_q: 交叉注意力中查询的累积序列长度
            cu_seqlens_k: 交叉注意力中键的累积序列长度
            max_seqlen: 变长序列的最大长度
            max_seqlen_q: 查询的最大序列长度
            max_seqlen_k: 键的最大序列长度
            attn_mask: 注意力掩码

        Returns:
            注意力输出张量，形状 (B, H, N, D)
        """
        if self.use_flash_attn:
            # 使用 Flash Attention（最高效，支持变长序列）
            x = dispatch_flash_attn(
                q,
                k,
                v,
                cu_seqlens=cu_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen=max_seqlen,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=False,
            )
            # Flash Attention 输出形状为 (B, Nq, H, D)，需要转置回 (B, H, Nq, D)
            # 以匹配其他注意力实现的输出格式
            x = x.transpose(1, 2)
        elif self.fast_attn:
            # 使用 PyTorch SDPA（次高效）
            if attn_mask is not None:
                attn_mask = attn_mask[:, None, None].repeat((1, self.num_heads, n, 1))  # 将掩码扩展到所有头
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                # True 表示该位置参与注意力计算
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p,
            )
        else:
            # 手动实现（兼容旧版 PyTorch）
            if attn_mask is not None:
                raise NotImplementedError
            q = q * self.scale  # 缩放查询
            attn = q @ k.transpose(-2, -1)  # 计算注意力分数
            attn = attn.softmax(dim=-1)  # softmax 归一化
            attn = self.attn_drop(attn)  # 应用 dropout
            x = attn @ v  # 加权求和

        return x

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """注意力模块前向传播。

        核心流程：
        1. 对输入 x 投影得到 Q；对 x 或 y 投影得到 K, V
        2. 将 Q, K, V 重排为多头格式
        3. 可选：对 Q, K 应用归一化
        4. 计算缩放点积注意力
        5. 输出投影 + dropout

        Args:
            x: 输入张量，形状 (B, N, C)；若为打包格式则为 (B*N, C)
            y: 交叉注意力的第二输入（编码器输出）；自注意力时为 None
            cu_seqlens: 变长 flash attention 的累积序列长度
            cu_seqlens_q: 交叉注意力中查询的累积序列长度
            cu_seqlens_k: 交叉注意力中键的累积序列长度
            max_seqlen: 变长序列的最大长度
            max_seqlen_q: 查询的最大序列长度
            max_seqlen_k: 键的最大序列长度
            attn_mask: 注意力掩码

        Returns:
            输出张量，形状与输入 x 相同
        """
        original_shape = x.shape

        q = self.q(x)  # 查询投影

        if y is None:
            assert not self.cross_attn  # 自注意力模式不应提供 y
            k = self.k(x)  # 键投影（自注意力：从同一输入）
            v = self.v(x)  # 值投影
        else:
            assert self.cross_attn  # 交叉注意力模式必须提供 y
            k = self.k(y)  # 键投影（交叉注意力：从编码器输出）
            v = self.v(y)  # 值投影
        if not self.use_flash_attn:
            # 非 Flash Attention 模式：重排为 (B, H, N, D) 格式
            q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
            k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
            v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        else:
            # Flash Attention 模式：重排为 (B*N, H, D) 打包格式
            q = rearrange(q, "bn (h d) -> bn h d", h=self.num_heads)
            # Flash attention 仅支持 KV 头数整除查询头数
            k = rearrange(k, "bn (h d) -> bn h d", h=self.num_heads)
            v = rearrange(v, "bn (h d) -> bn h d", h=self.num_heads)
        # logger.info(f"q shape: {q.shape} k shape: {k.shape} v shape: {v.shape}")

        q, k = self.q_norm(q), self.k_norm(k)  # 可选的 QK 归一化
        x = self.sdpa(
            q,
            k,
            v,
            n=original_shape[
                -2
            ],  # supposed to be the number of tokens in each sample with padding
            cu_seqlens=cu_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen=max_seqlen,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            attn_mask=attn_mask,
        )
        x = x.transpose(1, 2).reshape(original_shape)  # 将多头输出合并回原始形状
        x = self.proj(x)  # 输出投影
        x = self.proj_drop(x)  # 输出 dropout
        return x


class Mlp(nn.Module):
    """MLP（前馈网络）模块，用于 Vision Transformer 和相关网络。

    标准的两层 MLP 结构：Linear → Activation → Dropout → Linear → Dropout。
    隐藏维度通常为输入维度的 4 倍（由 mlp_ratio 控制）。

    Args:
        in_features: 输入特征维度
        hidden_features: 隐藏层维度（默认等于 in_features）
        out_features: 输出特征维度（默认等于 in_features）
        act_layer: 激活函数类型（默认 GELU）
        bias: 线性层是否使用偏置
        drop: Dropout 概率
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: nn.Module = nn.GELU,
        bias: bool = True,
        drop: float = 0.0,
    ) -> None:
        """Initialize the MLP module.

        Args:
            in_features: Number of input features
            hidden_features: Hidden dimension. Defaults to None.
            out_features: Output dimension. Defaults to None.
            act_layer: Activation layer. Defaults to nn.GELU.
            bias: Enable bias in linear layers. Defaults to True.
            drop: Dropout rate. Defaults to 0.0.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LayerScale(nn.Module):
    """可学习的层缩放（LayerScale）。

    对输入进行逐元素缩放，缩放因子为可学习参数。
    初始值通常设为很小的数（如 1e-5），使深层网络在训练初期
    接近恒等映射，有助于稳定深层 Transformer 的训练。

    参考：Scaling Vision Transformers to 22 billion parameters (https://arxiv.org/abs/2302.02318)

    Args:
        dim: 输入维度（缩放参数的数量）
        init_values: 缩放参数的初始值（默认 1e-5）
        inplace: 是否原地执行缩放操作（节省内存）
    """

    def __init__(
        self, dim: int, init_values: float = 1e-5, inplace: bool = False
    ) -> None:
        """Initialize the LayerScale module.

        Args:
            dim: Input dimension
            init_values: Initial scaling value
            inplace: Perform scaling operation in-place
        """
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Scaled output tensor
        """
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class DropPath(nn.Module):
    """随机深度（Stochastic Depth）模块，在残差块的主路径中随机丢弃整条路径。

    这是一种正则化技术，在训练时以一定概率将整个残差分支置零
    （相当于跳过该层），防止过拟合。推理时所有路径都保留。

    效果等同于 Dropout，但作用于整个残差路径而非单个神经元。
    被丢弃时，输出为 x / keep_prob（保持期望值不变）。

    Args:
        drop_prob: 丢弃路径的概率

    参考：
        Deep Networks with Stochastic Depth (https://arxiv.org/abs/1603.09382)
    """

    def __init__(self, drop_prob: float) -> None:
        """Initialize the DropPath module.

        Args:
            drop_prob: Probability of dropping the path. Defaults to None.
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying stochastic depth to input tensor.

        Args:
            x: Input tensor of any shape (B, ...)

        Returns:
            Tensor with same shape as input, with paths randomly dropped during training
        """
        if self.drop_prob is None or self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1, ...) — 只在样本维度随机
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 二值化：0（丢弃）或 1（保留）
        return x.div(keep_prob) * random_tensor  # 缩放以保持期望值不变


class Block(nn.Module):
    """Transformer 块，包含自/交叉注意力和 MLP。

    标准的 Transformer 编码器块结构：
        x = x + DropPath(LayerScale(Attention(LayerNorm(x))))
        x = x + DropPath(LayerScale(MLP(LayerNorm(x))))

    支持：
    - 自注意力和交叉注意力
    - LayerScale（可学习的层缩放）
    - DropPath（随机深度正则化）
    - QK 归一化
    - Flash Attention

    Args:
        dim: 输入维度
        num_heads: 注意力头数
        mlp_ratio: MLP 隐藏维度与输入维度的比率（默认 4.0）
        qkv_bias: QKV 投影是否使用偏置
        qk_norm: 是否对 Q, K 应用归一化
        drop: Dropout 概率
        attn_drop: 注意力 dropout 概率
        drop_path: 随机深度丢弃概率
        init_values: LayerScale 初始值（None 表示不使用）
        act_layer: 激活函数类型
        norm_layer: 归一化层类型
        cross_attn: 是否启用交叉注意力
        use_flash_attn: 是否使用 Flash Attention
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: float | None = None,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        cross_attn: bool = False,
        use_flash_attn: bool = False,
    ) -> None:
        """Initialize the Transformer block.

        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of mlp hidden dim to input dim
            qkv_bias: Add bias to qkv projections
            qk_norm: Apply normalization to q,k
            drop: Dropout rate
            attn_drop: Attention dropout rate
            drop_path: Drop path rate
            init_values: Layer scale initialization value
            act_layer: Activation layer
            norm_layer: Normalization layer
            cross_attn: Whether to use cross attention
            use_flash_attn: Whether to use flash attention
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
            cross_attn=cross_attn,
            use_flash_attn=use_flash_attn,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, N, C)
            y: Optional context tensor for cross attention of shape (B, M, C)
            attn_mask: Optional attention mask tensor
            cu_seqlens: Optional cumulative sequence lengths for the input tensor needed for varlen flash attention
            cu_seqlens_q: Optional cumulative sequence lengths for the query tensor, needed for cross varlen flash attention
            cu_seqlens_k: Optional cumulative sequence lengths for the key tensor, needed for cross varlen flash attention
            max_seqlen: Optional maximum sequence length for the input tensor, needed for varlen flash attention
            max_seqlen_q: Optional maximum sequence length for the query tensor, needed for cross varlen flash attention
            max_seqlen_k: Optional maximum sequence length for the key tensor, needed for cross varlen flash attention

        Returns:
            Output tensor of shape (B, N, C)
        """
        x = x + self.drop_path(
            self.ls1(
                self.attn(
                    x=self.norm1(x),
                    y=y,
                    cu_seqlens=cu_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen=max_seqlen,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    attn_mask=attn_mask,
                )
            )
        )

        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x

    def apply_fsdp(self, **fsdp_kwargs: Any) -> None:
        """Apply FSDP to the model."""
        fully_shard(self, **fsdp_kwargs)

    def apply_compile(self) -> None:
        """Apply torch.compile to the model."""
        self.compile(dynamic=True)
