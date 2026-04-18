"""
训练工具模块。

本模块提供 OlmoEarth Pretrain 训练过程中使用的实用工具函数，包括：
- 批次拆分（split_masked_batch）：将大批次拆分为微批次，用于梯度累积
- 内存使用日志（log_memory_usage_for_process, log_total_memory_usage）：
  监控主进程及子进程的内存消耗，用于调试内存泄漏和OOM问题
"""

"""Training utilities specific to OlmoEarth Pretrain."""

import logging  # 日志记录
import os  # 操作系统接口，获取进程ID

import psutil  # 进程和系统监控库

from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample  # 掩码样本数据类型

logger = logging.getLogger(__name__)


def split_masked_batch(
    batch: MaskedOlmoEarthSample, microbatch_size: int
) -> list[MaskedOlmoEarthSample]:
    """将一个大批次 MaskedOlmoEarthSample 拆分为多个微批次列表。

    当批次大小超过 microbatch_size 时，将数据沿批次维度拆分为多个子批次，
    最后一个子批次可能较小。用于梯度累积场景，即前向/反向传播以更小的
    微批次进行，梯度在多个微批次上累积后再执行优化器步进。

    Args:
        batch (MaskedOlmoEarthSample): 输入的掩码样本批次，
            第一维度(B)为批次大小。
        microbatch_size (int): 每个微批次的最大批次大小。

    Returns:
        list[MaskedOlmoEarthSample]: 拆分后的微批次列表。
            若批次大小 <= microbatch_size，则直接返回原批次的单元素列表。
    """
    batch_size = batch.batch_size  # 获取当前批次大小

    # 如果批次大小不超过微批次大小，无需拆分
    if batch_size <= microbatch_size:
        return [batch]

    # 计算需要的微批次数目（向上取整）
    num_microbatches = (batch_size + microbatch_size - 1) // microbatch_size

    # 计算每个微批次的拆分大小（最后一个可能更小）
    split_sizes = [microbatch_size] * (num_microbatches - 1)
    split_sizes.append(batch_size - microbatch_size * (num_microbatches - 1))  # 最后一个微批次的大小

    # 沿批次维度(dim=0)拆分每个字段
    splits: dict[str, tuple] = {}
    for field in MaskedOlmoEarthSample._fields:
        data = getattr(batch, field)
        if data is not None:
            splits[field] = data.split(split_sizes, dim=0)  # 沿第0维拆分

    # 将拆分后的数据重新组装为 MaskedOlmoEarthSample 对象
    return [
        MaskedOlmoEarthSample(**{f: chunks[i] for f, chunks in splits.items()})
        for i in range(num_microbatches)
    ]


def log_memory_usage_for_process(process: psutil.Process) -> tuple[int, int, int, int]:
    """记录给定进程的内存使用情况并返回内存统计信息。

    遍历进程的所有内存映射区域，分别统计 PSS（比例集大小）、
    USS（唯一集大小）和共享内存。

    Args:
        process (psutil.Process): 要监控的进程对象。

    Returns:
        tuple[int, int, int, int]: 包含四个值：
            - rss: 常驻集大小（Resident Set Size），进程使用的物理内存总量
            - pss: 比例集大小（Proportional Set Size），考虑共享内存的按比例分配
            - uss: 唯一集大小（Unique Set Size），进程独占的内存
            - shared: 共享内存大小
    """
    try:
        memory_info = process.memory_info()
        rss = memory_info.rss  # 常驻集大小
        pss = 0  # 比例集大小，累加计算
        uss = 0  # 唯一集大小，累加计算
        shared = 0  # 共享内存大小，累加计算

        # 遍历进程的所有内存映射区域
        for mmap in process.memory_maps():
            pss += mmap.pss  # 累加比例集大小
            uss += mmap.private_clean + mmap.private_dirty  # 独占内存 = 私有干净页 + 私有脏页
            shared += mmap.shared_clean + mmap.shared_dirty  # 共享内存 = 共享干净页 + 共享脏页

        return rss, pss, uss, shared

    except psutil.NoSuchProcess:
        # 进程可能在获取列表和当前时刻之间已终止
        return 0, 0, 0, 0


def log_total_memory_usage() -> float:
    """记录主进程及其所有子进程的总内存使用情况。

    依次统计主进程和所有递归子进程的 RSS、PSS、USS 和共享内存，
    最终返回以 GB 为单位的总 PSS 内存。

    Returns:
        float: 总 PSS 内存，单位为 GB（吉字节）。
    """
    # 获取当前主进程
    main_process = psutil.Process(os.getpid())

    # 初始化总内存计数器
    total_rss = 0
    total_pss = 0
    total_uss = 0
    total_shared = 0

    # 记录主进程的内存使用
    logger.info("Logging memory usage for main process")
    rss, pss, uss, shared = log_memory_usage_for_process(main_process)
    total_rss += rss
    total_pss += pss
    total_uss += uss
    total_shared += shared

    # 遍历所有子进程并记录其内存使用
    logger.info("Logging memory usage for child processes")
    for child in main_process.children(recursive=True):
        rss, pss, uss, shared = log_memory_usage_for_process(child)
        total_rss += rss
        total_pss += pss
        total_uss += uss
        total_shared += shared

    # 将 PSS 转换为 GB 单位返回
    return total_pss / (1024 * 1024 * 1024)
