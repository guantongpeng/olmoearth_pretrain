"""嵌入质量诊断模块，用于检测表示坍缩 (representation collapse)。

计算嵌入矩阵的几何度量以诊断自监督预训练中的失败模式
（维度坍缩、拥挤等）。

支持两种嵌入形状：
- [N, D]: 图像级别（分类任务），每个样本一个嵌入向量
- [N, P, D] 或 [N, H, W, D]: patch 级别（分割任务），每个样本多个 patch 嵌入
  计算全局 (global)、样本间 (inter-sample) 和样本内 (intra-sample) 三级诊断

可独立用于任意嵌入张量，也可通过评估回调集成到评估流程中。

核心诊断指标：
- effective_rank: 有效秩，通过奇异值的 Shannon 熵计算
- uniformity: 均匀性度量 (Wang & Isola 2020)
- pairwise_cosine_stats: 成对余弦相似度统计
- embedding_norm_stats: L2 范数统计
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# 采样上限常量，防止大规模嵌入导致内存溢出
MAX_PAIRWISE_SAMPLES = 2048   # 成对计算的最大样本数
MAX_SVD_SAMPLES = 4096        # SVD 计算的最大样本数
MAX_INTRA_SAMPLE_IMAGES = 256 # 样本内诊断的最大图像数


def effective_rank(embeddings: Tensor) -> float:
    """通过奇异值的 Shannon 熵计算有效秩。

    有效秩衡量嵌入矩阵的实际维度利用率，返回值范围：
    - 1: 完全坍缩（所有嵌入相同方向）
    - min(N, D): 最大化分布（嵌入均匀分布在所有维度）

    参考: Roy & Bhattacharyya (2007)

    Args:
        embeddings: 嵌入张量，形状 [N, D]

    Returns:
        float: 有效秩值
    """
    n = embeddings.shape[0]
    if n > MAX_SVD_SAMPLES:
        # 采样以控制计算量
        idx = torch.randperm(n, device=embeddings.device)[:MAX_SVD_SAMPLES]
        embeddings = embeddings[idx]
    S = torch.linalg.svdvals(embeddings.float())  # 计算奇异值
    S = S[S > 0]  # 仅保留正奇异值
    if S.numel() == 0:
        return 0.0
    p = S / S.sum()  # 归一化为概率分布
    entropy = -(p * p.log()).sum()  # Shannon 熵
    return entropy.exp().item()  # 返回 exp(entropy) 作为有效秩


def uniformity(embeddings: Tensor, t: float = 2.0) -> float:
    """均匀性度量 (Wang & Isola 2020)。

    衡量嵌入在超球面上的均匀分布程度。
    值越小（越负）表示越均匀。

    Args:
        embeddings: 嵌入张量，形状 [N, D]
        t: 温度参数，默认 2.0

    Returns:
        float: 均匀性值，越负越均匀
    """
    z = torch.nn.functional.normalize(embeddings.float(), dim=-1)  # L2 归一化
    n = z.shape[0]
    if n > MAX_PAIRWISE_SAMPLES:
        idx = torch.randperm(n, device=z.device)[:MAX_PAIRWISE_SAMPLES]
        z = z[idx]
        n = MAX_PAIRWISE_SAMPLES
    sq_dists = torch.cdist(z, z, p=2).pow(2)  # 成对平方距离
    # 取上三角部分，避免重复计算和自距离
    mask = torch.triu(torch.ones(n, n, device=z.device, dtype=torch.bool), diagonal=1)
    sq_dists_upper = sq_dists[mask]
    return torch.log(torch.exp(-t * sq_dists_upper).mean()).item()


def pairwise_cosine_stats(embeddings: Tensor) -> dict[str, float]:
    """成对余弦相似度统计。

    高均值 + 低标准差 表示拥挤现象（嵌入过于集中）。

    Args:
        embeddings: 嵌入张量，形状 [N, D]

    Returns:
        dict: 包含 cosine_sim_mean, cosine_sim_std, cosine_sim_min, cosine_sim_max
    """
    z = torch.nn.functional.normalize(embeddings.float(), dim=-1)  # L2 归一化
    n = z.shape[0]
    if n > MAX_PAIRWISE_SAMPLES:
        idx = torch.randperm(n, device=z.device)[:MAX_PAIRWISE_SAMPLES]
        z = z[idx]
        n = MAX_PAIRWISE_SAMPLES
    sim = z @ z.T  # 成对余弦相似度矩阵
    # 取上三角部分，避免重复和自相似度
    mask = torch.triu(torch.ones(n, n, device=z.device, dtype=torch.bool), diagonal=1)
    sims = sim[mask]
    return {
        "cosine_sim_mean": sims.mean().item(),   # 均值
        "cosine_sim_std": sims.std().item(),      # 标准差
        "cosine_sim_min": sims.min().item(),      # 最小值
        "cosine_sim_max": sims.max().item(),      # 最大值
    }


def embedding_norm_stats(embeddings: Tensor) -> dict[str, float]:
    """L2 范数统计，计算样本间的嵌入范数分布。

    Args:
        embeddings: 嵌入张量，形状 [N, D]

    Returns:
        dict: 包含 norm_mean, norm_std, norm_min, norm_max
    """
    norms = embeddings.float().norm(dim=-1)  # 每个样本的 L2 范数
    return {
        "norm_mean": norms.mean().item(),
        "norm_std": norms.std().item(),
        "norm_min": norms.min().item(),
        "norm_max": norms.max().item(),
    }


def compute_embedding_diagnostics(embeddings: Tensor) -> dict[str, float]:
    """计算所有嵌入质量诊断指标（图像级嵌入 [N, D]）。

    Args:
        embeddings: 嵌入张量，形状 [N, D]

    Returns:
        dict: 包含 effective_rank, embedding_dim, num_samples,
              范数统计，以及（当样本数>=4时）均匀性和余弦统计

    Raises:
        ValueError: 如果嵌入不是 2 维张量
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings [N, D], got shape {embeddings.shape}")
    n, d = embeddings.shape
    if n < 2:
        logger.warning("Need at least 2 samples for embedding diagnostics")
        return {}

    metrics: dict[str, float] = {}
    metrics["effective_rank"] = effective_rank(embeddings)  # 有效秩
    metrics["embedding_dim"] = float(d)   # 嵌入维度
    metrics["num_samples"] = float(n)     # 样本数
    metrics.update(embedding_norm_stats(embeddings))  # 范数统计

    if n >= 4:
        # 样本数足够时计算均匀性和余弦统计
        metrics["uniformity"] = uniformity(embeddings)
        metrics.update(pairwise_cosine_stats(embeddings))

    return metrics


def _compute_intra_sample_diagnostics(embeddings: Tensor) -> dict[str, float]:
    """计算样本内（per-image patch）诊断指标，在所有图像上取平均。

    衡量同一图像内的 patch 嵌入是否具有多样性：
    - 多样性好：适合分割任务
    - 坍缩（所有 patch 相同）：分割不可能进行

    Args:
        embeddings: 嵌入张量，形状 [N, P, D]，P 为每图像的 patch 数

    Returns:
        dict: 包含 norm_std, num_patches, num_images_sampled,
              cosine_sim_mean, cosine_sim_std
    """
    n, p, d = embeddings.shape
    if p < 2:
        logger.warning("Need at least 2 patches per image for intra-sample diagnostics")
        return {}

    # 限制图像数量以控制计算量
    num_images = min(n, MAX_INTRA_SAMPLE_IMAGES)
    if num_images < n:
        idx = torch.randperm(n, device=embeddings.device)[:num_images]
        embeddings = embeddings[idx]

    # 批量计算余弦相似度: 归一化后 bmm -> [num_images, P, P]
    z = torch.nn.functional.normalize(embeddings.float(), dim=-1)
    sim_matrices = torch.bmm(z, z.transpose(1, 2))
    # 上三角掩码，排除对角线（自相似度）
    tri_mask = torch.triu(
        torch.ones(p, p, device=z.device, dtype=torch.bool), diagonal=1
    )

    # 计算每张图像的余弦相似度均值和标准差
    cosine_means = []
    cosine_stds = []
    for i in range(num_images):
        sims = sim_matrices[i][tri_mask]
        cosine_means.append(sims.mean().item())
        cosine_stds.append(sims.std().item())

    # 计算每张图像的范数标准差
    norms = embeddings.float().norm(dim=-1)  # [num_images, P]
    norm_stds = norms.std(dim=1)  # [num_images]

    metrics: dict[str, float] = {
        "norm_std": norm_stds.mean().item(),          # 范数标准差的均值
        "num_patches": float(p),                       # 每图像 patch 数
        "num_images_sampled": float(num_images),       # 采样的图像数
    }
    if cosine_means:
        metrics["cosine_sim_mean"] = sum(cosine_means) / len(cosine_means)  # 余弦相似度均值
        metrics["cosine_sim_std"] = sum(cosine_stds) / len(cosine_stds)     # 余弦相似度标准差
    return metrics


def compute_spatial_embedding_diagnostics(embeddings: Tensor) -> dict[str, float]:
    """计算空间（patch 级别）嵌入的诊断指标。

    接受 [N, *, D] 形状的张量，其中 * 为一个或多个空间维度
    （例如 [N, H, W, D] 或 [N, P, D]）。

    返回带有前缀 (global_, inter_, intra_) 的指标，避免 wandb 中深层嵌套。

    Args:
        embeddings: 嵌入张量，形状 [N, *, D]

    Returns:
        dict: 包含三级诊断指标：
            - global_*: 全局指标（所有 patch 展平后计算）
            - inter_*: 样本间指标（每图像平均池化后计算）
            - intra_*: 样本内指标（每图像 patch 多样性）

    Raises:
        ValueError: 如果嵌入维度小于 3
    """
    if embeddings.ndim < 3:
        raise ValueError(
            f"Expected 3+ dim embeddings [N, *, D], got shape {embeddings.shape}"
        )

    n = embeddings.shape[0]
    d = embeddings.shape[-1]
    patches = embeddings.reshape(n, -1, d)  # 展平空间维度: [N, P, D]
    p = patches.shape[1]

    if n < 2:
        logger.warning("Need at least 2 samples for spatial embedding diagnostics")
        return {}

    metrics: dict[str, float] = {}

    # 全局诊断：将所有 patch 展平为 [N*P, D]，必要时子采样
    flat = patches.reshape(-1, d)
    if flat.shape[0] > MAX_SVD_SAMPLES:
        idx = torch.randperm(flat.shape[0], device=flat.device)[:MAX_SVD_SAMPLES]
        flat = flat[idx]
    for k, v in compute_embedding_diagnostics(flat).items():
        metrics[f"global_{k}"] = v

    # 样本间诊断：每图像的 patch 平均池化 -> [N, D]
    pooled = patches.float().mean(dim=1)
    for k, v in compute_embedding_diagnostics(pooled).items():
        metrics[f"inter_{k}"] = v

    # 样本内诊断：每图像的 patch 多样性
    if p >= 2:
        for k, v in _compute_intra_sample_diagnostics(patches).items():
            metrics[f"intra_{k}"] = v

    return metrics
