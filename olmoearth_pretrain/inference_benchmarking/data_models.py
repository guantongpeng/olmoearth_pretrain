"""推理吞吐量基准测试核心数据模型。

本模块定义了基准测试运行参数的数据模型，支持从环境变量、
运行名称和对象实例之间相互转换。

主要类:
    RunParams: 基准测试运行参数，包含模型配置、硬件配置和测试配置

使用场景:
    1. 创建 RunParams 实例指定测试参数
    2. 通过 to_env_vars() 导出为环境变量
    3. 通过 from_env_vars() 从环境变量恢复实例
    4. 通过 from_run_name() 从运行名称解析实例
"""

import os
import re
from dataclasses import dataclass

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.inference_benchmarking import constants


@dataclass
class RunParams(Config):
    """基准测试运行参数，定义单次吞吐量测试的所有配置。

    关键属性:
        model_size: 模型大小（nano, tiny, base, large）
        use_s1: 是否使用 Sentinel-1 数据
        use_s2: 是否使用 Sentinel-2 数据
        use_landsat: 是否使用 Landsat 数据
        image_size: 图像空间尺寸
        patch_size: ViT 补丁大小
        num_timesteps: 时间步数量
        batch_size: 批量大小
        gpu_type: GPU 类型（cuda 或 cpu）
        bf16: 是否使用 BF16 精度
        benchmark_interval_s: 基准测试间隔（秒）
        min_batches_per_interval: 每个间隔最少批次数
        profiler_enabled: 是否启用性能分析器
        wandb_enabled: 是否启用 W&B 日志

    使用场景:
        定义单次基准测试的参数配置，支持序列化为环境变量或运行名称。
    """

    # TODO: Add a named constant for the default model size
    model_size: str = "base"
    use_s1: bool = False
    use_s2: bool = True
    use_landsat: bool = False
    image_size: int = 64
    patch_size: int = 4
    num_timesteps: int = 12
    batch_size: int = 128
    gpu_type: str = "cuda"
    bf16: bool = True
    benchmark_interval_s: int = 180
    min_batches_per_interval: int = 10
    profiler_enabled: bool = False
    wandb_enabled: bool = True

    @property
    def run_name(self) -> str:
        """生成表示当前运行参数的字符串名称。

        格式: {model_size}_{gpu_type}[_bf16][_s1][_s2][_ls]_is{image_size}_ps{patch_size}_ts{num_timesteps}_bs{batch_size}

        Returns:
            str: 运行参数的字符串表示
        """
        return "_".join(
            [
                item
                for item in [
                    self.model_size,
                    self.gpu_type,
                    "bf16" if self.bf16 else None,
                    "s1" if self.use_s1 else None,
                    "s2" if self.use_s2 else None,
                    "ls" if self.use_landsat else None,
                    f"is{self.image_size}",
                    f"ps{self.patch_size}",
                    f"ts{self.num_timesteps}",
                    f"bs{self.batch_size}",
                ]
                if item is not None
            ]
        )

    def to_env_vars(self) -> dict[str, str]:
        """将运行参数导出为环境变量字典。

        对象可以随后从这些环境变量重新创建。

        Returns:
            dict[str, str]: 环境变量名到值的映射
        """
        keys = constants.PARAM_KEYS
        env_vars = {
            keys["model_size"]: self.model_size,
            keys["use_s1"]: str(int(self.use_s1)),
            keys["use_s2"]: str(int(self.use_s2)),
            keys["use_landsat"]: str(int(self.use_landsat)),
            keys["image_size"]: str(self.image_size),
            keys["patch_size"]: str(self.patch_size),
            keys["num_timesteps"]: str(self.num_timesteps),
            keys["batch_size"]: str(self.batch_size),
            keys["gpu_type"]: self.gpu_type,
            keys["bf16"]: str(int(self.bf16)),
            keys["benchmark_interval_s"]: str(self.benchmark_interval_s),
            keys["min_batches_per_interval"]: str(self.min_batches_per_interval),
            keys["name"]: self.run_name,
        }
        # 添加额外的参数
        env_vars["profiler_enabled"] = str(int(self.profiler_enabled))
        env_vars["wandb_enabled"] = str(int(self.wandb_enabled))
        return env_vars

    @staticmethod
    def from_env_vars() -> "RunParams":
        """从环境变量恢复 RunParams 实例。

        Returns:
            RunParams: 从环境变量构建的运行参数实例
        """
        keys = constants.PARAM_KEYS
        model_size = os.getenv(keys["model_size"], "Unknown")
        use_s1 = True if os.getenv(keys["use_s1"], "0") == "1" else False
        use_s2 = True if os.getenv(keys["use_s2"], "0") == "1" else False
        use_landsat = True if os.getenv(keys["use_landsat"], "0") == "1" else False
        image_size = int(os.getenv(keys["image_size"], "1"))
        patch_size = int(os.getenv(keys["patch_size"], "1"))
        num_timesteps = int(os.getenv(keys["num_timesteps"], "1"))
        batch_size = int(os.getenv(keys["batch_size"], "1"))
        gpu_type = os.getenv(keys["gpu_type"], "cpu")
        bf16 = True if os.getenv(keys["bf16"], "0") == "1" else False
        benchmark_interval_s = int(os.getenv(keys["benchmark_interval_s"], "180"))
        min_batches_per_interval = int(os.getenv(keys["min_batches_per_interval"], 10))
        profiler_enabled = True if os.getenv("profiler_enabled", "0") == "1" else False
        wandb_enabled = True if os.getenv("wandb_enabled", "0") == "1" else False

        return RunParams(
            model_size=model_size,
            use_s1=use_s1,
            use_s2=use_s2,
            use_landsat=use_landsat,
            image_size=image_size,
            patch_size=patch_size,
            num_timesteps=num_timesteps,
            batch_size=batch_size,
            gpu_type=gpu_type,
            bf16=bf16,
            benchmark_interval_s=benchmark_interval_s,
            min_batches_per_interval=min_batches_per_interval,
            profiler_enabled=profiler_enabled,
            wandb_enabled=wandb_enabled,
        )

    @staticmethod
    def from_run_name(name: str) -> "RunParams":
        """从先前的运行名称字符串恢复 RunParams 实例。

        解析运行名称中的各组件以重建参数。

        Args:
            name: 先前的运行名称字符串

        Returns:
            RunParams: 从运行名称解析的运行参数实例
        """
        split_name = name.split("_")
        model_size = split_name[0]
        gpu_type = split_name[1]
        use_s1 = "_s1_" in name
        use_s2 = "_s2_" in name
        use_landsat = "_ls_" in name
        bf16 = "_bf16_" in name
        profiler_enabled = "_prof_" in name or "_prof" in name
        wandb_enabled = "_wandb_" in name or "_wandb" in name

        # 使用默认值初始化
        image_size = 64
        patch_size = 4
        num_timesteps = 12
        batch_size = 128
        benchmark_interval_s = 180
        min_batches_per_interval = 10

        for item in split_name:
            if item.startswith("is"):
                image_size = int(item.replace("is", ""))
            if item.startswith("ps"):
                patch_size = int(item.replace("ps", ""))
            if item.startswith("ts"):
                num_timesteps = int(item.replace("ts", ""))

        # 修复批量大小解析
        batch_size_matches = re.findall(r"bs(\d+)", name)
        if batch_size_matches:
            batch_size = int(batch_size_matches[0])

        return RunParams(
            model_size=model_size,
            use_s1=use_s1,
            use_s2=use_s2,
            use_landsat=use_landsat,
            image_size=image_size,
            patch_size=patch_size,
            num_timesteps=num_timesteps,
            batch_size=batch_size,
            gpu_type=gpu_type,
            bf16=bf16,
            benchmark_interval_s=benchmark_interval_s,
            min_batches_per_interval=min_batches_per_interval,
            profiler_enabled=profiler_enabled,
            wandb_enabled=wandb_enabled,
        )
