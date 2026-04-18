"""模型嵌入提取模块。

本模块提供从模型中提取嵌入向量的功能，是线性探针和 KNN 评估的上游步骤。
主要功能：遍历数据集，通过模型前向传播获取嵌入，支持嵌入量化以节省存储空间。

核心函数：
- get_embeddings: 从数据加载器中提取模型嵌入和标签

使用场景：
  在评估流程中，先调用 get_embeddings 获取训练/验证/测试集的嵌入，
  然后将嵌入传递给线性探针或 KNN 进行下游任务评估。
"""

import logging

import torch
from torch.utils.data import DataLoader

from olmoearth_pretrain.evals.embedding_transforms import quantize_embeddings
from olmoearth_pretrain.evals.eval_wrapper import EvalWrapper
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)


def get_embeddings(
    data_loader: DataLoader,
    model: EvalWrapper,
    is_train: bool = True,
    quantize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从模型中提取数据集的嵌入向量和标签。

    将模型设为评估模式，遍历数据加载器，对每个批次进行前向传播获取嵌入。
    支持自动混合精度 (bfloat16) 和可选的嵌入量化。

    Args:
        data_loader: 评估数据集的数据加载器
        model: EvalWrapper 包装的模型实例
        is_train: 是否为训练数据（影响某些模型的行为，如 AnySat 子采样）
        quantize: 是否将嵌入量化为 int8，用于存储效率测试

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - embeddings: 嵌入张量，形状为 (N, dim) 或 (N, H, W, dim)
            - labels: 标签张量，形状为 (N,) 或 (N, H, W)
            如果 quantize=True，嵌入为 int8 类型
    """
    embeddings_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    model.eval()  # 设置模型为评估模式
    device = model.device
    with torch.no_grad():  # 禁用梯度计算以节省内存
        for i, (masked_olmoearth_sample, label) in enumerate(data_loader):
            # 将样本数据移动到模型所在设备
            masked_olmoearth_sample_dict = masked_olmoearth_sample.as_dict()
            for key, val in masked_olmoearth_sample_dict.items():
                if key == "timestamps":
                    masked_olmoearth_sample_dict[key] = val.to(device=device)
                else:
                    masked_olmoearth_sample_dict[key] = val.to(
                        device=device,
                    )

            masked_olmoearth_sample = MaskedOlmoEarthSample.from_dict(
                masked_olmoearth_sample_dict
            )
            # 使用 bfloat16 自动混合精度进行前向传播
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                batch_embeddings, label = model(
                    masked_olmoearth_sample=masked_olmoearth_sample,
                    labels=label,
                    is_train=is_train,
                )

            # 将嵌入移回 CPU 以释放 GPU 内存
            embeddings_list.append(batch_embeddings.cpu())
            labels_list.append(label)
            logger.info("Processed batch %d", i)

    # 沿批次维度拼接所有嵌入和标签
    embeddings = torch.cat(embeddings_list, dim=0)  # (N, dim)
    labels = torch.cat(labels_list, dim=0)  # (N)

    # 如果请求量化，将嵌入从浮点型量化为 int8
    if quantize:
        logger.info(f"Quantizing embeddings from {embeddings.dtype} to int8")
        embeddings = quantize_embeddings(embeddings)

    return embeddings, labels
