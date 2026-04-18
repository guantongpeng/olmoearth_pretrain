"""嵌入后处理变换模块（量化和降维）。

本模块提供嵌入提取后的变换操作：

1. 量化 (Quantization):
   - quantize_embeddings: 将浮点嵌入量化为 int8，使用幂次变换保持非均匀分布信息
   - dequantize_embeddings: 将 int8 嵌入反量化回浮点型
   量化方案与 AlphaEarth 一致，使用幂次变换 (power=2.0) 和缩放因子 (scale=127.5)

2. 降维 (Dimensionality Reduction):
   - reduce_embedding_dim: 通过 PCA 降低嵌入维度，支持空间维度 (N,H,W,C)
"""

import torch
from sklearn.decomposition import PCA

# === 量化参数 ===
# 与 AlphaEarth 的量化方案保持一致
QUANTIZE_POWER = 2.0    # 幂次变换的指数，用于在量化前压缩数值范围
QUANTIZE_SCALE = 127.5  # int8 范围缩放因子 (255/2)


def quantize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """将浮点嵌入量化为 int8，使用幂次变换方案。

    量化步骤：
    1. 应用幂次变换: sat = |x|^(1/power) * sign(x)，保留符号，压缩数值范围
    2. 缩放到 int8 范围 [-127, 127]
    3. 四舍五入并转换为 int8

    这种幂次变换可以在非均匀嵌入分布下更好地保留信息。

    Args:
        embeddings: 浮点张量，形状 (N, dim) 或 (N, H, W, dim)

    Returns:
        int8 张量，形状与输入相同
    """
    # 幂次变换：保留符号，压缩数值范围
    sat = embeddings.abs().pow(1.0 / QUANTIZE_POWER) * embeddings.sign()
    # 缩放并量化到 int8 范围
    quantized = (sat * QUANTIZE_SCALE).clamp(-127, 127).round().to(torch.int8)
    return quantized


def dequantize_embeddings(quantized: torch.Tensor) -> torch.Tensor:
    """将 int8 嵌入反量化回 float32，是量化操作的逆过程。

    反量化步骤：
    1. 从 int8 范围重新缩放
    2. 应用逆幂次变换: x = |rescaled|^power * sign(rescaled)

    Args:
        quantized: int8 张量，形状 (N, dim) 或 (N, H, W, dim)

    Returns:
        float32 张量，形状与输入相同
    """
    # 从 int8 范围重新缩放
    rescaled = quantized.float() / QUANTIZE_SCALE
    # 逆幂次变换：保留符号，恢复原始数值范围
    dequantized = rescaled.abs().pow(QUANTIZE_POWER) * rescaled.sign()
    return dequantized


# === 降维 ===


def reduce_embedding_dim(
    train_embeddings: torch.Tensor,
    val_embeddings: torch.Tensor,
    test_embeddings: torch.Tensor | None,
    target_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, float]:
    """通过 PCA 降低嵌入维度。

    在训练嵌入上拟合 PCA，然后将相同变换应用到验证/测试嵌入。
    支持空间维度 (N, H, W, C)：先展平再 PCA，之后重塑回原始形状。

    Args:
        train_embeddings: 训练嵌入，形状 (N, dim) 或 (N, H, W, dim)
        val_embeddings: 验证嵌入，形状与训练嵌入结构相同
        test_embeddings: 测试嵌入（可选），形状与训练嵌入结构相同
        target_dim: PCA 后的目标维度

    Returns:
        tuple: (train_reduced, val_reduced, test_reduced, variance_retained)
            - train_reduced: 降维后的训练嵌入
            - val_reduced: 降维后的验证嵌入
            - test_reduced: 降维后的测试嵌入（可能为 None）
            - variance_retained: 保留的方差比例（解释方差比之和）
    """
    original_dim = train_embeddings.shape[-1]
    train_shape = train_embeddings.shape
    val_shape = val_embeddings.shape
    test_shape = test_embeddings.shape if test_embeddings is not None else None

    # Flatten spatial dimensions if present (for segmentation tasks)
    if len(train_shape) > 2:
        # Shape is (N, H, W, C) or similar - flatten to (N*H*W, C)
        train_flat = train_embeddings.reshape(-1, original_dim)
        val_flat = val_embeddings.reshape(-1, original_dim)
        test_flat = (
            test_embeddings.reshape(-1, original_dim)
            if test_embeddings is not None
            else None
        )
    else:
        train_flat = train_embeddings
        val_flat = val_embeddings
        test_flat = test_embeddings

    # Fit PCA on train embeddings
    pca = PCA(n_components=target_dim)
    train_reduced = pca.fit_transform(train_flat.cpu().numpy())
    val_reduced = pca.transform(val_flat.cpu().numpy())
    test_reduced = (
        pca.transform(test_flat.cpu().numpy()) if test_flat is not None else None
    )

    variance_retained = float(sum(pca.explained_variance_ratio_))

    # Convert back to tensors and reshape if needed
    device = train_embeddings.device
    dtype = train_embeddings.dtype

    if len(train_shape) > 2:
        new_train_shape = train_shape[:-1] + (target_dim,)
        new_val_shape = val_shape[:-1] + (target_dim,)
        train_out = (
            torch.from_numpy(train_reduced)
            .to(device=device, dtype=dtype)
            .reshape(new_train_shape)
        )
        val_out = (
            torch.from_numpy(val_reduced)
            .to(device=device, dtype=dtype)
            .reshape(new_val_shape)
        )
        if test_reduced is not None and test_shape is not None:
            new_test_shape = test_shape[:-1] + (target_dim,)
            test_out = (
                torch.from_numpy(test_reduced)
                .to(device=device, dtype=dtype)
                .reshape(new_test_shape)
            )
        else:
            test_out = None
    else:
        train_out = torch.from_numpy(train_reduced).to(device=device, dtype=dtype)
        val_out = torch.from_numpy(val_reduced).to(device=device, dtype=dtype)
        test_out = (
            torch.from_numpy(test_reduced).to(device=device, dtype=dtype)
            if test_reduced is not None
            else None
        )

    return train_out, val_out, test_out, variance_retained
