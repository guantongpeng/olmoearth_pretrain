"""模型加载模块 - 从 Hugging Face Hub 或本地路径加载 OlmoEarth 预训练模型。

本模块支持在有/无 olmo-core 的环境下加载模型：
- 无 olmo-core: 推理模式（加载预训练模型进行推理）
- 有 olmo-core: 完整功能，包括训练

支持的模型包括：
- OlmoEarth-v1-Nano: 最小规模的模型
- OlmoEarth-v1-Tiny: 小规模模型
- OlmoEarth-v1-Base: 基础规模模型
- OlmoEarth-v1-Large: 大规模模型

权重转换说明（从分布式检查点转换为 pth 文件）：

    import json
    from pathlib import Path

    import torch

    from olmo_core.config import Config
    from olmo_core.distributed.checkpoint import load_model_and_optim_state

    checkpoint_path = Path("/weka/dfive-default/helios/checkpoints/joer/nano_lr0.001_wd0.002/step370000")
    with (checkpoint_path / "config.json").open() as f:
        config_dict = json.load(f)
        model_config = Config.from_dict(config_dict["model"])

    model = model_config.build()

    train_module_dir = checkpoint_path / "model_and_optim"
    load_model_and_optim_state(str(train_module_dir), model)
    torch.save(model.state_dict(), "OlmoEarth-v1-Nano.pth")
"""

import copy
import json
from enum import StrEnum
from os import PathLike

import torch
from huggingface_hub import hf_hub_download
from upath import UPath

from olmoearth_pretrain.config import Config

# 模型配置文件的固定文件名
CONFIG_FILENAME = "config.json"
# 模型权重文件的固定文件名
WEIGHTS_FILENAME = "weights.pth"


class ModelID(StrEnum):
    """OlmoEarth 预训练模型 ID 枚举。

    定义了所有可用的 OlmoEarth 预训练模型标识符，
    用于指定要加载的模型。每个枚举值对应 Hugging Face Hub 上的
    一个模型仓库（repo ID 格式为 "allenai/{模型名}"）。

    可用模型：
        - OLMOEARTH_V1_NANO: 最小规模模型，适合快速测试
        - OLMOEARTH_V1_TINY: 小规模模型
        - OLMOEARTH_V1_BASE: 基础规模模型
        - OLMOEARTH_V1_LARGE: 大规模模型
    """

    OLMOEARTH_V1_NANO = "OlmoEarth-v1-Nano"
    OLMOEARTH_V1_TINY = "OlmoEarth-v1-Tiny"
    OLMOEARTH_V1_BASE = "OlmoEarth-v1-Base"
    OLMOEARTH_V1_LARGE = "OlmoEarth-v1-Large"

    def repo_id(self) -> str:
        """返回此模型在 Hugging Face Hub 上的仓库 ID。

        Returns:
            Hugging Face 仓库 ID，格式为 "allenai/{模型名}"。
        """
        return f"allenai/{self.value}"


def load_model_from_id(model_id: ModelID, load_weights: bool = True) -> torch.nn.Module:
    """根据模型 ID 从 Hugging Face 初始化并加载模型。

    首先下载模型配置文件并构建模型结构，然后根据参数决定
    是否下载并加载预训练权重。

    Args:
        model_id: 要加载的模型 ID（ModelID 枚举值）。
        load_weights: 是否加载预训练权重。设为 False 可跳过从 Hugging Face
                      下载权重，模型参数保持随机初始化。注意：无论此参数如何设置，
                      config.json 都会从 Hugging Face 下载。默认为 True。

    Returns:
        加载了权重的 PyTorch 模型（torch.nn.Module）。
    """
    # 解析配置文件路径（必要时从 Hugging Face 下载）
    config_fpath = _resolve_artifact_path(model_id, CONFIG_FILENAME)
    # 从配置文件构建模型
    model = _load_model_from_config(config_fpath)

    if not load_weights:
        return model

    # 解析权重文件路径（必要时从 Hugging Face 下载）
    state_dict_fpath = _resolve_artifact_path(model_id, WEIGHTS_FILENAME)
    # 加载权重字典
    state_dict = _load_state_dict(state_dict_fpath)
    # 将权重加载到模型中
    model.load_state_dict(state_dict)
    return model


def load_model_from_path(
    model_path: PathLike | str, load_weights: bool = True
) -> torch.nn.Module:
    """从本地路径初始化并加载模型。

    首先从本地路径读取模型配置文件并构建模型结构，然后根据参数决定
    是否加载本地权重文件。

    Args:
        model_path: 模型文件所在的本地路径，该路径下应包含
                    config.json 和 weights.pth 文件。
        load_weights: 是否加载权重。设为 False 可跳过加载权重，
                      模型参数保持随机初始化。默认为 True。

    Returns:
        加载了权重的 PyTorch 模型（torch.nn.Module）。
    """
    # 解析配置文件路径（本地路径拼接）
    config_fpath = _resolve_artifact_path(model_path, CONFIG_FILENAME)
    # 从配置文件构建模型
    model = _load_model_from_config(config_fpath)

    if not load_weights:
        return model

    # 解析权重文件路径（本地路径拼接）
    state_dict_fpath = _resolve_artifact_path(model_path, WEIGHTS_FILENAME)
    # 加载权重字典
    state_dict = _load_state_dict(state_dict_fpath)
    # 将权重加载到模型中
    model.load_state_dict(state_dict)
    return model


def _resolve_artifact_path(
    model_id_or_path: ModelID | PathLike | str, filename: str
) -> UPath:
    """解析模型文件的路径，必要时从 Hugging Face 下载。

    根据输入类型自动判断来源：
    - 如果是 ModelID 枚举：从 Hugging Face Hub 下载文件并返回本地缓存路径
    - 如果是本地路径：直接拼接路径并返回

    Args:
        model_id_or_path: 模型 ID（从 HF 下载）或本地路径（直接读取）。
        filename: 要解析的文件名（如 config.json 或 weights.pth）。

    Returns:
        文件的 UPath 路径（统一路径对象，支持本地和云端路径）。
    """
    if isinstance(model_id_or_path, ModelID):
        # 从 Hugging Face Hub 下载文件并返回缓存路径
        return UPath(
            hf_hub_download(repo_id=model_id_or_path.repo_id(), filename=filename)  # nosec
        )
    # 本地路径：拼接基础路径和文件名
    base = UPath(model_id_or_path)
    return base / filename


def patch_legacy_encoder_config(config_dict: dict) -> dict:
    """修补旧版编码器配置，兼容不含 use_linear_patch_embed 字段的检查点。

    旧版检查点使用 Conv2d 进行 patch 投影，配置中没有 use_linear_patch_embed 键。
    如果不修补，这些检查点会错误地默认为 True（使用 Linear），导致权重加载失败。
    应在将原始配置字典传递给 Config.from_dict 之前调用此函数。

    Args:
        config_dict: 原始配置字典，通常从 JSON 文件加载。

    Returns:
        修补后的配置字典。如果需要修补，返回深拷贝；否则返回原始字典。
    """
    # 检查是否存在 encoder_config 且缺少 use_linear_patch_embed 字段
    enc = config_dict.get("model", {}).get("encoder_config", {})
    if isinstance(enc, dict) and "use_linear_patch_embed" not in enc:
        # 深拷贝避免修改原始字典，并显式设置为 False（使用 Conv2d）
        config_dict = copy.deepcopy(config_dict)
        config_dict["model"]["encoder_config"]["use_linear_patch_embed"] = False
    return config_dict


def _load_model_from_config(path: UPath) -> torch.nn.Module:
    """从配置文件路径加载模型。

    读取 JSON 配置文件，修补旧版编码器配置，然后通过 Config 系统
    构建模型实例。

    Args:
        path: 配置文件的 UPath 路径。

    Returns:
        构建好的 PyTorch 模型（torch.nn.Module）。
    """
    with path.open() as f:
        config_dict = json.load(f)
    # 修补旧版检查点的编码器配置
    config_dict = patch_legacy_encoder_config(config_dict)
    # 通过 Config 系统反序列化并构建模型
    model_config = Config.from_dict(config_dict["model"])
    return model_config.build()


def _load_state_dict(path: UPath) -> dict[str, torch.Tensor]:
    """从文件路径加载模型权重字典。

    使用 torch.load 加载权重，映射到 CPU 设备。

    Args:
        path: 权重文件的 UPath 路径。

    Returns:
        模型权重字典，键为参数名，值为对应的张量。
    """
    with path.open("rb") as f:
        # map_location="cpu" 确保权重先加载到 CPU，避免 GPU 内存问题
        state_dict = torch.load(f, map_location="cpu")
    return state_dict
