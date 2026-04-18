"""神经网络模块的通用工具函数和混入类。

本模块提供编码器-解码器架构中常用的工具函数和分布式训练混入类：
- unpack_encoder_output: 解包编码器输出字典
- get_cumulative_sequence_lengths: 计算变长序列的累积长度（用于 Flash Attention）
- DistributedMixins: 分布式训练混入类（DDP 支持）
"""

from typing import Any

import torch
from torch.distributed import DeviceMesh


def unpack_encoder_output(
    output_dict: dict[str, Any],
) -> tuple:
    """解包编码器的输出字典，分离关键组件。

    编码器输出包含多种信息，此函数将其解包为：
    1. tokens_and_masks: 编码后的 token 和掩码
    2. project_aggregated: 投影和池化后的 token（用于对比损失）
    3. decoder_kwargs: 其他传递给解码器的参数

    Args:
        output_dict: 编码器的输出字典，包含以下键：
            - "tokens_and_masks": TokensAndMasks 对象
            - "project_aggregated": 投影池化后的张量
            - "token_norm_stats": token 范数统计（可选，会被移除）
            - 其他: 传递给解码器的额外参数

    Returns:
        (latent, latent_projected_and_pooled, decoder_kwargs) 三元组
    """
    latent = output_dict.pop("tokens_and_masks", None)  # 提取 token 和掩码
    latent_projected_and_pooled = output_dict.pop("project_aggregated", None)  # 提取投影池化结果
    # 移除 token_norm_stats（不在解码器中使用）
    output_dict.pop("token_norm_stats", None)
    decoder_kwargs = output_dict  # 剩余的键值对作为解码器参数
    return latent, latent_projected_and_pooled, decoder_kwargs


def get_cumulative_sequence_lengths(seq_lengths: torch.Tensor) -> torch.Tensor:
    """计算变长序列的累积长度，用于 Flash Attention 的变长模式。

    返回的累积长度以 0 开头，例如 [0, len_1, len_1+len_2, ...]。
    保留长度为 0 的序列，使得 cu_seqlens_q 和 cu_seqlens_k 在
    交叉注意力中保持对齐（flash_attn_varlen_func 要求两者条目数相同）。

    Args:
        seq_lengths: 各序列的长度，形状 (batch_size,)

    Returns:
        累积序列长度，形状 (batch_size + 1,)，以 [0] 开头
    """
    return torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=seq_lengths.device),  # 开头的 0
            torch.cumsum(seq_lengths, 0, dtype=torch.int32),  # 累积求和
        ]
    )


class DistributedMixins:
    """分布式训练混入类，提供 DDP（分布式数据并行）支持。

    提供 apply_ddp 方法用于将模型包装为 DDP 模式，
    支持与 torch.compile 的兼容性配置。

    使用方式：与其他 nn.Module 一起多重继承，如：
        class MyModel(nn.Module, DistributedMixins): ...
    """

    def apply_ddp(
        self,
        dp_mesh: DeviceMesh | None = None,
        compile_enabled: bool = False,
        autograd_compile_enabled: bool = False,
        find_unused_parameters: bool = True,
    ) -> None:
        """对模型应用 DDP（分布式数据并行）包装。

        Args:
            dp_mesh: 设备网格（用于 HSDP 等高级并行策略）
            compile_enabled: 是否启用 torch.compile
            autograd_compile_enabled: 是否启用自动梯度编译
            find_unused_parameters: 是否查找未使用参数（MAE 需要 True）

        注意：
            通常不需要直接调用此方法，TransformerConfig.build() 会自动调用。
        """
        from torch.distributed._composable.replicate import replicate

        # Adapted from
        # https://github.com/pytorch/torchtitan/blob/90c889e972b56b9faadebbb78fc985dedc537ed9/torchtitan/parallelisms/parallelize_llama.py#L328
        if compile_enabled:
            if autograd_compile_enabled:
                torch._dynamo.config.optimize_ddp = (
                    "python_reducer_without_compiled_forward"  # type: ignore
                )
            else:
                torch._dynamo.config.optimize_ddp = "ddp_optimizer"  # type: ignore
        # Forwards kwargs to torch DDP class, find_unused_parameters=True is required for MAE
        # Small performance hit could be possible for other models
        replicate(
            self,
            device_mesh=dp_mesh,
            bucket_cap_mb=100,
            find_unused_parameters=find_unused_parameters,
        )
