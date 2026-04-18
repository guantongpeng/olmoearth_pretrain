"""超参数扫描和训练模块的内部工具函数。

本模块提供超参数扫描过程中使用的工具函数和模拟训练组件，
包括令牌退出配置构建、模型尺寸参数表和模拟数据加载器/训练模块。

主要函数:
    build_token_exit_config(): 构建令牌退出配置字符串

主要类:
    MockOlmoEarthDataLoader: 满足抽象接口的最小模拟数据加载器
    MockLatentMIMTrainModule: 满足抽象接口的最小模拟训练模块

主要常量:
    EXIT_CONFIG_TYPES: 令牌退出配置类型列表
    MODEL_SIZE_ARGS: 各种模型尺寸的参数映射表
"""

from collections.abc import Iterable
from typing import Any

import torch
from olmo_core.data.data_loader import DataLoaderBase
from olmo_core.train.train_module import EvalBatchSpec, TrainModule

EXIT_CONFIG_TYPES = ["zero", "half", "full", "varied"]  # 令牌退出配置类型：零层/半层/全层/变化层


def build_token_exit_config(
    config_type: str, modality_names: list[str], encoder_depth: int
) -> str:
    """构建令牌退出配置的命令行参数字符串。

    根据配置类型，为每个模态设置令牌退出的编码器层数。

    Args:
        config_type: 配置类型，必须是 EXIT_CONFIG_TYPES 之一
            - "zero": 所有模态退出层数为0
            - "half": 所有模态退出层数为编码器深度的一半
            - "full": 所有模态退出层数为编码器完整深度
            - "varied": latlon 和 worldcover 退出为0，其余为完整深度
        modality_names: 模态名称列表
        encoder_depth: 编码器深度（层数）

    Returns:
        str: 令牌退出配置的命令行参数字符串

    Raises:
        ValueError: 如果 config_type 无效
    """
    if config_type not in EXIT_CONFIG_TYPES:
        raise ValueError(f"Invalid config type: {config_type}")
    if config_type == "zero":
        return " ".join(
            f"--train_module.token_exit_cfg.{modality_name}=0"
            for modality_name in modality_names
        )
    elif config_type == "half":
        return " ".join(
            f"--train_module.token_exit_cfg.{modality_name}={encoder_depth // 2}"
            for modality_name in modality_names
        )
    elif config_type == "full":
        return " ".join(
            f"--train_module.token_exit_cfg.{modality_name}={encoder_depth}"
            for modality_name in modality_names
        )
    elif config_type == "varied":
        varied_args = []
        for modality_name in modality_names:
            if modality_name not in ["latlon", "worldcover"]:
                varied_args.append(
                    f"--train_module.token_exit_cfg.{modality_name}={encoder_depth}"
                )
            else:
                varied_args.append(f"--train_module.token_exit_cfg.{modality_name}=0")
        return " ".join(varied_args)
    else:
        raise ValueError(f"Invalid config type: {config_type}")


MODEL_SIZE_ARGS = {
    # nano: 最小模型，用于快速测试
    "nano": {
        "decoder_depth": 4,
        "encoder_embedding_size": 128,
        "decoder_embedding_size": 128,
        "encoder_depth": 4,
        "encoder_num_heads": 8,
        "decoder_num_heads": 8,
        "mlp_ratio": 4.0,
    },
    # tiny: 小型模型
    "tiny": {
        "decoder_depth": 12,
        "encoder_embedding_size": 192,
        "decoder_embedding_size": 192,
        "encoder_depth": 12,
        "encoder_num_heads": 3,
        "decoder_num_heads": 3,
        "mlp_ratio": 4.0,
    },
    # tiny_more_heads: 小型模型，更多注意力头
    "tiny_more_heads": {
        "decoder_depth": 12,
        "encoder_embedding_size": 192,
        "decoder_embedding_size": 192,
        "encoder_depth": 12,
        "encoder_num_heads": 8,
        "decoder_num_heads": 8,
        "mlp_ratio": 4.0,
    },
    # base: 基础模型
    "base": {
        "decoder_depth": 12,
        "encoder_embedding_size": 768,
        "decoder_embedding_size": 768,
        "encoder_depth": 12,
        "encoder_num_heads": 12,
        "decoder_num_heads": 12,
        "mlp_ratio": 4.0,
    },
    # large: 大型模型
    "large": {
        "decoder_depth": 24,
        "encoder_embedding_size": 1024,
        "decoder_embedding_size": 1024,
        "encoder_depth": 24,
        "encoder_num_heads": 16,
        "decoder_num_heads": 16,
        "mlp_ratio": 4.0,
    },
    # giga: 超大模型
    "giga": {
        "decoder_depth": 40,
        "encoder_embedding_size": 1536,
        "decoder_embedding_size": 1536,
        "encoder_depth": 40,
        "encoder_num_heads": 24,
        "decoder_num_heads": 24,
        "mlp_ratio": 4.0,
    },
    "tiny_shallow_decoder": {
        "decoder_depth": 4,
        "encoder_embedding_size": 192,
        "decoder_embedding_size": 192,
        "encoder_depth": 12,
        "encoder_num_heads": 3,
        "decoder_num_heads": 3,
        "mlp_ratio": 4.0,
    },
    "base_shallow_decoder": {
        "decoder_depth": 4,
        "encoder_embedding_size": 768,
        "decoder_embedding_size": 768,
        "encoder_depth": 12,
        "encoder_num_heads": 12,
        "decoder_num_heads": 12,
        "mlp_ratio": 4.0,
    },
    "large_shallow_decoder": {
        "decoder_depth": 4,
        "encoder_embedding_size": 1024,
        "decoder_embedding_size": 1024,
        "encoder_depth": 24,
        "encoder_num_heads": 16,
        "decoder_num_heads": 16,
        "mlp_ratio": 4.0,
    },
    "giga_shallow_decoder": {
        "decoder_depth": 4,
        "encoder_embedding_size": 1536,
        "decoder_embedding_size": 1536,
        "encoder_depth": 40,
        "encoder_num_heads": 24,
        "decoder_num_heads": 24,
        "mlp_ratio": 4.0,
    },
    "tiny_super_shallow_decoder": {
        "decoder_depth": 2,
        "encoder_embedding_size": 192,
        "decoder_embedding_size": 192,
        "encoder_depth": 12,
        "encoder_num_heads": 3,
        "decoder_num_heads": 3,
        "mlp_ratio": 4.0,
    },
    "base_super_shallow_decoder": {
        "decoder_depth": 2,
        "encoder_embedding_size": 768,
        "decoder_embedding_size": 768,
        "encoder_depth": 12,
        "encoder_num_heads": 12,
        "decoder_num_heads": 12,
        "mlp_ratio": 4.0,
    },
    "large_super_shallow_decoder": {
        "decoder_depth": 2,
        "encoder_embedding_size": 1024,
        "decoder_embedding_size": 1024,
        "encoder_depth": 24,
        "encoder_num_heads": 16,
        "decoder_num_heads": 16,
        "mlp_ratio": 4.0,
    },
    "giga_super_shallow_decoder": {
        "decoder_depth": 2,
        "encoder_embedding_size": 1536,
        "decoder_embedding_size": 1536,
        "encoder_depth": 40,
        "encoder_num_heads": 24,
        "decoder_num_heads": 24,
        "mlp_ratio": 4.0,
    },
    "base_many_heads_shallow_decoder": {
        "decoder_depth": 4,
        "encoder_embedding_size": 768,
        "decoder_embedding_size": 768,
        "encoder_depth": 8,
        "encoder_num_heads": 16,
        "decoder_num_heads": 12,
        "mlp_ratio": 4.0,
    },
    "base_many_heads_shallower_decoder": {
        "decoder_depth": 2,
        "encoder_embedding_size": 768,
        "decoder_embedding_size": 768,
        "encoder_depth": 8,
        "encoder_num_heads": 16,
        "decoder_num_heads": 12,
        "mlp_ratio": 4.0,
    },
}


class MockOlmoEarthDataLoader(DataLoaderBase):
    """满足抽象接口的最小模拟 OlmoEarth 数据加载器。

    不产生任何数据，仅用于满足 Trainer 对 DataLoaderBase 接口的要求。
    在评估模式下使用，因为评估不需要实际的数据加载。

    关键属性:
        _seed: 随机种子
        _epoch: 当前轮次
        token_budget: 令牌预算
    """

    def __init__(self) -> None:
        """初始化模拟加载器，使用最小的单 rank 默认值。"""
        super().__init__(
            work_dir="./",
            global_batch_size=128,
            dp_world_size=1,
            dp_rank=0,
            fs_local_rank=0,
        )
        self._seed = 42
        self._epoch = 0
        self.token_budget: int | None = None

    def _iter_batches(self) -> Iterable[Any]:
        return iter(())

    def state_dict(self) -> dict[str, Any]:
        """返回模拟加载器的最小持久化状态。"""
        return {"seed": self._seed, "epoch": self._epoch}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # noqa: D401
        """模拟数据加载器的状态加载（从 state_dict 恢复种子和轮次）。"""
        self._seed = state_dict.get("seed", self._seed)
        self._epoch = state_dict.get("epoch", self._epoch)

    def reshuffle(
        self, epoch: int | None = None, in_memory: bool = False, **_: Any
    ) -> None:
        """记录提供的轮次，其他参数被忽略。"""
        if epoch is not None:
            self._epoch = epoch

    @property
    def total_batches(self) -> int:
        """报告零批次，因为模拟加载器不产生数据。"""
        return 0

    def get_mock_batch(self) -> None:
        """返回空批次，此存根不生成数据。"""
        return None


class MockLatentMIMTrainModule(TrainModule):
    """满足抽象接口的最小模拟训练模块，用于 LatentMIM 风格的配置。

    不执行实际训练，仅用于评估模式下满足 Trainer 对 TrainModule 接口的要求。

    关键属性:
        model: 使用 Identity 模型作为占位符
    """

    def __init__(self) -> None:
        """初始化模拟训练模块，使用 Identity 模型作为占位符。"""
        super().__init__()
        self.model = torch.nn.Identity()

    @property
    def eval_batch_spec(self) -> EvalBatchSpec:
        """返回简单的评估批次规格。"""
        return EvalBatchSpec(rank_batch_size=1)

    def state_dict(self, *, optim: bool | None = None) -> dict[str, Any]:
        """返回空的状态字典。"""
        del optim
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """忽略任何状态字典内容。"""
        del state_dict

    def train_batch(self, batch: dict[str, Any], dry_run: bool = False) -> None:
        """空操作的训练步骤。"""
        del batch, dry_run

    def eval_batch(self, batch: dict[str, Any], labels: Any | None = None) -> Any:
        """返回常量张量以满足接口期望。"""
        del batch, labels
        return torch.tensor(0.0)

    def optim_step(self) -> None:
        """空操作的优化器步骤。"""

    def zero_grads(self) -> None:
        """空操作的梯度重置。"""
