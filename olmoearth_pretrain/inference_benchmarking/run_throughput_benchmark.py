"""推理吞吐量基准测试执行脚本。

本模块实现了模型推理吞吐量的完整基准测试流程，包括：
    - 模型构建和加载
    - 合成数据生成
    - 性能测试执行（含预热）
    - 指标收集和日志记录
    - 参数扫描支持

主要类:
    MinimalTrainer: 最小训练器，仅支持文件持久化
    OlmoEarth: OlmoEarth 模型封装，仅加载编码器
    ThroughputBenchmarkRunnerConfig: 基准测试运行器配置
    ThroughputBenchmarkRunner: 基准测试运行器

主要函数:
    build_default_model_config(): 构建默认模型配置
    calculate_num_token_embeddings(): 计算令牌嵌入数量
"""

import itertools
import os
import time
import uuid
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from olmo_core.io import copy_file, file_exists, join_path
from olmo_core.train.callbacks import ProfilerCallback, WandBCallback
from olmo_core.train.trainer import PathOrStr

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.constants import BASE_GSD, Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.inference_benchmarking import constants
from olmoearth_pretrain.inference_benchmarking.data_models import RunParams
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.flexi_vit import (
    Encoder,
    EncoderConfig,
    PredictorConfig,
    TokensAndMasks,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

NUM_S1_BANDS = Modality.SENTINEL1.num_bands  # Sentinel-1 波段数
NUM_S2_BANDS = Modality.SENTINEL2.num_bands  # Sentinel-2 波段数
NUM_LANDSAT_BANDS = Modality.LANDSAT.num_bands  # Landsat 波段数

NUM_SQUARE_KM_LAND_IN_WORLD = 149_000_000  # 全球陆地面积（平方公里）

logger = getLogger(__name__)


class MinimalTrainer:
    """最小训练器，仅实现 persist_working_file 方法。

    用于在不创建完整 Trainer 的情况下使用回调功能（如 W&B 和 Profiler）。

    关键属性:
        device: 计算设备
        work_dir: 工作目录
        save_folder: 保存目录
    """

    def __init__(
        self, device: torch.device, work_dir: Path, save_folder: Path | None = None
    ):
        """初始化最小训练器。

        Args:
            device: 计算设备（CPU 或 GPU）
            work_dir: 工作目录路径
            save_folder: 保存目录路径，默认与 work_dir 相同
        """
        self.device = device
        self.work_dir = work_dir  # Will be set later
        if save_folder is None:
            self.save_folder = work_dir
        else:
            self.save_folder = save_folder

    def persist_working_file(self, name: PathOrStr) -> PathOrStr:
        """持久化工作文件，将其从工作目录复制到保存目录。

        Args:
            name: 文件名或路径

        Returns:
            PathOrStr: 保存后的文件路径
        """
        if Path(name).is_relative_to(self.work_dir):
            name = Path(name).relative_to(self.work_dir)
        source = join_path(self.work_dir, name)
        target = join_path(self.save_folder, name)
        if source != target:
            copy_file(source, target)
        elif not file_exists(source):
            raise FileNotFoundError(source)
        return target


class OlmoEarth(torch.nn.Module):
    """OlmoEarth 模型封装，仅加载编码器部分。

    在吞吐量基准测试中，只使用编码器，因为解码器会影响
    内存和延迟估计的准确性。

    关键属性:
        model: 编码器模型实例
    """

    def __init__(self, model_config: Config) -> None:
        """从模型配置加载检查点，仅保留编码器。

        Args:
            model_config: 模型配置对象
        """
        super().__init__()

        # 只需要编码器，其余网络会影响内存和延迟估计
        model = model_config.build()

        model = getattr(model, "encoder")

        self.model: Encoder = model
        self.model.eval()  # 设置为评估模式
        # 可能需要添加一个标志来控制此行为
        self.model.apply_compile()  # 应用 torch.compile 优化

    def forward(
        self,
        x: MaskedOlmoEarthSample,
        patch_size: int,
        fast_pass: bool = True,
    ) -> TokensAndMasks:
        """前向传播，直接委托给编码器。

        Args:
            x: 掩码化的 OlmoEarth 样本
            patch_size: 补丁大小
            fast_pass: 是否使用快速通道

        Returns:
            TokensAndMasks: 编码器输出的令牌和掩码
        """
        return self.model.forward(
            x,
            patch_size=patch_size,
            fast_pass=fast_pass,
        )["tokens_and_masks"]


def build_default_model_config(
    run_params: RunParams, training_modalities: list[str]
) -> LatentMIMConfig:
    """基于模型大小构建默认模型配置。

    从 MODEL_SIZE_ARGS 中查找对应模型大小的参数，构建编码器和解码器配置。

    Args:
        run_params: 运行参数，包含 model_size
        training_modalities: 支持的模态名称列表

    Returns:
        LatentMIMConfig: 用于构建模型的 LatentMIM 配置
    """
    model_size = MODEL_SIZE_ARGS[run_params.model_size]
    encoder_config = EncoderConfig(
        embedding_size=int(model_size["encoder_embedding_size"]),
        num_heads=int(model_size["encoder_num_heads"]),
        depth=int(model_size["encoder_depth"]),
        mlp_ratio=float(model_size["mlp_ratio"]),
        supported_modality_names=training_modalities,
    )
    decoder_config = PredictorConfig(
        encoder_embedding_size=int(model_size["encoder_embedding_size"]),
        decoder_embedding_size=int(model_size["decoder_embedding_size"]),
        depth=int(model_size["decoder_depth"]),
        mlp_ratio=float(model_size["mlp_ratio"]),
        num_heads=int(model_size["decoder_num_heads"]),
        supported_modality_names=training_modalities,
        max_sequence_length=12,
    )
    return LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )


@dataclass
class ThroughputBenchmarkRunnerConfig(Config):
    """吞吐量基准测试运行器配置。

    关键属性:
        sweep_dict: 参数扫描字典，键为参数名，值为要扫描的值列表
        sweep_keys: 预定义扫描键列表（如 "batch", "image" 等）
        sweep_group_name: 扫描组名称（用于 W&B 日志）
        training_modalities: 训练模态列表
        work_dir: 工作目录
        default_run_params: 默认运行参数
        save_folder: 保存目录
        cross_product_sweep: 是否对所有参数取笛卡尔积
    """

    sweep_dict: dict[str, Any] | None = None  # 参数扫描字典
    sweep_keys: list[str] | None = None  # 预定义扫描键列表
    sweep_group_name: str | None = None  # 扫描组名称
    training_modalities: list[str] = field(
        default_factory=lambda: [
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
            Modality.LANDSAT.name,
        ]
    )
    work_dir: Path = Path("./test_work_dir")
    default_run_params: RunParams | None = None
    save_folder: Path | None = None
    cross_product_sweep: bool = False

    def build(
        self,
        model_config: Any | None = None,
    ) -> "ThroughputBenchmarkRunner":
        """构建吞吐量基准测试运行器。

        Args:
            model_config: 可选的预构建模型配置。如果提供，将用于所有基准测试运行，
                而不是从默认参数构建。

        Returns:
            ThroughputBenchmarkRunner: 构建好的基准测试运行器

        Raises:
            ValueError: 如果 sweep_dict 和 sweep_keys 都未设置或都设置了
        """
        if self.default_run_params is None:
            self.default_run_params = RunParams()

        if self.sweep_dict is None and self.sweep_keys is None:
            raise ValueError("Either sweep_dict or sweep_keys must be set")
        if self.sweep_dict is not None and self.sweep_keys is not None:
            raise ValueError("Only one of sweep_dict or sweep_keys can be set")

        # 如果需要，从 sweep_keys 构建 sweep_dict
        if self.sweep_dict is not None:
            sweep_dict = self.sweep_dict
        else:
            assert self.sweep_keys is not None  # validated above
            sweep_dict = {}
            for sweep_key in self.sweep_keys:
                sweep_dict[sweep_key] = constants.SWEEPS[sweep_key]

        return ThroughputBenchmarkRunner(
            default_run_params=self.default_run_params,
            sweep_group_name=self.sweep_group_name,
            training_modalities=self.training_modalities,
            work_dir=self.work_dir,
            save_folder=self.save_folder,
            sweep_dict=sweep_dict,
            cross_product_sweep=self.cross_product_sweep,
            model_config=model_config,
        )


def calculate_num_token_embeddings(t: torch.Tensor | None) -> int:
    """计算张量中表示的令牌嵌入总数。

    Args:
        t: 令牌嵌入张量，形状为 (B, H, W, T, BandSets, D)，
            或 None 表示该模态不存在

    Returns:
        int: 令牌嵌入数量 = B * H * W * T * BandSets
    """
    if t is not None:
        batch_size, p_height, p_width, timestamps, bandsets, _ = tuple(t.shape)
        return batch_size * p_height * p_width * timestamps * bandsets

    return 0


class ThroughputBenchmarkRunner:
    """吞吐量基准测试运行器，执行模型推理性能测试。

    支持参数扫描和单次运行两种模式，收集吞吐量指标并记录到 W&B。

    关键属性:
        default_run_params: 默认运行参数
        sweep_group_name: 扫描组名称
        training_modalities: 训练模态列表
        work_dir: 工作目录
        save_folder: 保存目录
        sweep_dict: 参数扫描字典
        cross_product_sweep: 是否使用笛卡尔积扫描
        model_config: 预构建的模型配置
        sweep_name: 扫描名称（包含参数名和 UUID）
    """

    def __init__(
        self,
        default_run_params: RunParams,
        sweep_group_name: str | None,
        training_modalities: list[str],
        work_dir: Path,
        save_folder: Path | None = None,
        sweep_dict: dict[str, Any] = {},
        cross_product_sweep: bool = False,
        model_config: Any | None = None,
    ):
        """初始化吞吐量基准测试运行器。

        Args:
            default_run_params: 基准测试的默认参数
            sweep_group_name: 扫描组名称（用于日志记录）
            training_modalities: 使用的模态名称列表
            work_dir: 基准测试输出的工作目录
            save_folder: 可选的结果保存目录
            sweep_dict: 参数名到扫描值列表的映射字典
            cross_product_sweep: 如果为 True，扫描所有参数组合
            model_config: 可选的预构建模型配置。如果提供，将用于所有基准测试运行，
                而不是从运行参数构建。
        """
        self.default_run_params = default_run_params
        self.sweep_group_name = sweep_group_name
        self.training_modalities = training_modalities
        self.work_dir = work_dir
        self.work_dir.mkdir(exist_ok=True)
        self.save_folder = save_folder
        self.sweep_dict = sweep_dict
        self.cross_product_sweep = cross_product_sweep
        self.model_config = model_config
        uuid_str = str(uuid.uuid4())[:6]  # 生成短 UUID 用于唯一标识扫描
        self.sweep_name = "_".join(self.sweep_dict.keys()) + "-" + uuid_str

    def build_model(self, run_params: RunParams) -> OlmoEarth:
        """根据运行参数构建模型。

        如果提供了预构建的 model_config 则直接使用，否则使用
        build_default_model_config() 创建模型配置。

        Args:
            run_params: 运行参数

        Returns:
            OlmoEarth: 构建好的 OlmoEarth 模型封装
        """
        if self.model_config is not None:
            model_config = self.model_config
        else:
            model_config = build_default_model_config(
                run_params, self.training_modalities
            )
        return OlmoEarth(model_config=model_config)

    def build_sweep_run_params(self) -> list[RunParams]:
        """根据扫描字典构建运行参数列表。

        支持两种模式：
            - cross_product_sweep=True: 对所有参数取笛卡尔积
            - cross_product_sweep=False: 逐参数独立扫描
        最后追加默认运行参数。

        Returns:
            list[RunParams]: 运行参数列表
        """
        run_params_list: list[RunParams] = []
        if self.cross_product_sweep:
            # 对扫描字典取笛卡尔积
            sweep_dict_keys = list(self.sweep_dict.keys())
            # 对扫描字典的每种不同组合构建运行参数
            for combination in itertools.product(
                *[self.sweep_dict[key] for key in sweep_dict_keys]
            ):
                run_params_list.append(
                    self.default_run_params.replace(
                        **dict(zip(sweep_dict_keys, combination))
                    )
                )
        else:
            for key, value in self.sweep_dict.items():
                for v in value:
                    # 将扫描参数与默认参数合并
                    run_params_list.append(self.default_run_params.replace(**{key: v}))
        # 添加默认运行参数
        run_params_list.append(self.default_run_params)
        return run_params_list

    def run_benchmarking_sweep(self, run_params_list: list[RunParams]) -> None:
        """对运行参数列表执行基准测试扫描。

        Args:
            run_params_list: 运行参数列表
        """
        for run_params in run_params_list:
            try:
                logger.info(f"Running benchmarking for {run_params}")
                self.run_benchmarking(run_params)
            except Exception as e:
                logger.error(f"Error running benchmarking for {run_params}: {e}")
                continue

    def run_benchmarking(self, run_params: RunParams) -> None:
        """执行单次基准测试。

        核心逻辑:
            1. 构建模型并移至 GPU
            2. 生成合成数据（随机张量）
            3. 执行预热（5 次前向传播）
            4. 运行基准测试循环，记录令牌处理率和时间
            5. 计算并记录各项指标到 W&B

        需要模型封装、W&B 指标实例和运行参数。

        Args:
            run_params: 运行参数
        """
        model = self.build_model(run_params)
        if torch.cuda.is_available() and run_params.gpu_type == "cuda":
            logger.info("Model loaded and on gpu")
            model.to(run_params.gpu_type)
        device = next(model.parameters()).device
        batch_size = run_params.batch_size
        idx = 0

        if run_params.bf16:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        callbacks = []
        # 初始化回调（性能分析器和 W&B）
        if run_params.profiler_enabled:
            profiler = ProfilerCallback(
                skip_first=0,  # Don't skip any steps
                wait=0,  # Start profiling immediately
                warmup=5,  # Warmup for 5 steps (matches your current warmup)
                active=5,  # Profile for 5 steps
                repeat=1,  # Only one cycle
            )

            profiler.trainer = MinimalTrainer(device, self.work_dir)

            callbacks.append(profiler)

        if run_params.wandb_enabled:
            project = os.getenv(constants.PARAM_KEYS["project"], constants.PROJECT_NAME)
            owner = os.getenv(constants.PARAM_KEYS["owner"], constants.ENTITY_NAME)
            name = run_params.run_name
            name = f"{run_params.run_name}-{self.sweep_name}"
            group = self.sweep_group_name
            wandb_callback = WandBCallback(
                project=project,
                entity=owner,
                name=name,
                group=group,
                config=run_params.as_dict(),
            )
            wandb_callback.trainer = MinimalTrainer(device, self.work_dir)
            callbacks.append(wandb_callback)

        for callback in callbacks:
            callback.pre_train()

        if run_params.use_s1:
            # dims: (B, H, W, T, len(S1_BANDS)]
            s1_tensor = torch.rand(
                batch_size,
                run_params.image_size,
                run_params.image_size,
                run_params.num_timesteps,
                NUM_S1_BANDS,
                device=device,
                dtype=dtype,
            )
        else:
            s1_tensor = None

        if run_params.use_s2:
            # dims: (B, H, W, T, len(S2_BANDS)]
            s2_tensor = torch.rand(
                batch_size,
                run_params.image_size,
                run_params.image_size,
                run_params.num_timesteps,
                NUM_S2_BANDS,
                device=device,
                dtype=dtype,
            )
        else:
            s2_tensor = None

        if run_params.use_landsat:
            # dims: (B, H, W, T, len(LANDSAT_bands))
            landsat_tensor = torch.rand(
                batch_size,
                run_params.image_size,
                run_params.image_size,
                run_params.num_timesteps,
                NUM_LANDSAT_BANDS,
                device=device,
                dtype=dtype,
            )
        else:
            landsat_tensor = None

        latlon = torch.rand(batch_size, 2, device=device, dtype=dtype)  # dims: (B, 2)
        timestamps = torch.ones(
            batch_size, run_params.num_timesteps, 3, dtype=torch.int32, device=device
        )  # dims: (B, T, D=3)

        def maybe_make_mask(maybe_t: torch.Tensor | None) -> torch.Tensor | None:
            if maybe_t is not None:
                return (
                    torch.ones(
                        maybe_t.shape,
                        dtype=dtype,
                        device=device,
                    )
                    * MaskValue.ONLINE_ENCODER.value
                )
            return None

        masked_sample = MaskedOlmoEarthSample(
            timestamps=timestamps,
            sentinel2_l2a=s2_tensor,
            sentinel2_l2a_mask=maybe_make_mask(s2_tensor),
            sentinel1=s1_tensor,
            sentinel1_mask=maybe_make_mask(s1_tensor),
            landsat=landsat_tensor,
            landsat_mask=maybe_make_mask(landsat_tensor),
            latlon=latlon,
            latlon_mask=maybe_make_mask(latlon),
        )

        tokens_processed_per_batch: list[int] = []
        time_taken_per_batch: list[float] = []
        # 记录数据准备完成，开始预热
        logger.info("Data prepared, starting warmup")
        torch.cuda.set_sync_debug_mode("warn")
        # 运行 5 次前向传播作为预热
        oom_occurred = False
        for _ in range(5):
            try:
                with torch.inference_mode():
                    if run_params.bf16:
                        with torch.amp.autocast(
                            device_type=device.type, dtype=torch.bfloat16
                        ):
                            results = model.forward(
                                masked_sample, patch_size=run_params.patch_size
                            )
                    else:
                        results = model.forward(
                            masked_sample, patch_size=run_params.patch_size
                        )
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA OOM during warmup: {e}")
                oom_occurred = True
                break

        if oom_occurred:
            logger.info("CUDA OOM occurred during warmup, skipping benchmark")
            # 记录 OOM 状态到 W&B
            metrics_oom_occurred: dict[str, Any] = {
                constants.OOM_OCCURRED_METRIC: 1,
            }
            for callback in callbacks:
                callback.log_metrics(step=0, metrics=metrics_oom_occurred)
            for callback in callbacks:
                callback.post_train()
            return

        logger.info("Warmup complete, starting benchmark")
        # TODO: 使用 CUDA 事件计时
        if device.type == "cuda":
            torch.cuda.synchronize()
        overall_start_time = time.monotonic()
        interval_start_time = time.monotonic()
        while (
            time.monotonic() - interval_start_time
        ) < run_params.benchmark_interval_s or len(
            tokens_processed_per_batch
        ) < run_params.min_batches_per_interval:
            batch_start = time.monotonic()

            with torch.inference_mode():
                if run_params.bf16:
                    with torch.amp.autocast(
                        device_type=device.type, dtype=torch.bfloat16
                    ):
                        results = model.forward(
                            masked_sample,
                            patch_size=run_params.patch_size,
                        )
                else:
                    results = model.forward(
                        masked_sample, patch_size=run_params.patch_size
                    )
            # 在大循环外单独计时每个批次
            time_taken_per_batch.append(time.monotonic() - batch_start)

            # 对每次前向传播调用性能分析器步骤
            for callback in callbacks:
                callback.pre_load_batch()

            num_s1_tokens = calculate_num_token_embeddings(results.sentinel1)
            num_s2_tokens = calculate_num_token_embeddings(results.sentinel2_l2a)
            num_landsat_tokens = calculate_num_token_embeddings(results.landsat)
            tokens_processed_per_batch.append(
                num_s1_tokens + num_s2_tokens + num_landsat_tokens
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        overall_time_taken = time.monotonic() - overall_start_time
        logger.info(
            f"Overall time taken: {overall_time_taken} sum of time taken per batch: {sum(time_taken_per_batch)} num batches: {len(time_taken_per_batch)}"
        )
        metrics_to_submit: dict[str, Any] = {
            constants.PER_BATCH_TOKEN_RATE_METRIC: wandb.Histogram(
                np.array(
                    [
                        tokens_processed_per_batch,
                        time_taken_per_batch,
                    ]
                )
            ),
            constants.MEAN_BATCH_TOKEN_RATE_METRIC: sum(tokens_processed_per_batch)
            / overall_time_taken,
            constants.MEAN_BATCH_TIME_METRIC: overall_time_taken
            / len(time_taken_per_batch),
            constants.NUM_TOKENS_PER_BATCH_METRIC: sum(tokens_processed_per_batch)
            / len(tokens_processed_per_batch),
        }
        num_batches = len(time_taken_per_batch)
        num_centroids = num_batches * batch_size
        centroids_per_second = num_centroids / overall_time_taken
        tile_km2 = (
            run_params.image_size * BASE_GSD / 1000.0
        ) ** 2  # 米转公里，然后平方
        area_processed_km2 = batch_size * tile_km2 * num_batches
        square_km_per_second = area_processed_km2 / overall_time_taken
        metrics_to_submit[constants.SQUARE_KM_PER_SECOND_METRIC] = square_km_per_second
        metrics_to_submit[constants.PIXELS_PER_SECOND_METRIC] = centroids_per_second
        try:
            gpu_name = torch.cuda.get_device_name(device)
            metrics_to_submit[constants.GPU_NAME_METRIC] = gpu_name
        except Exception as e:
            logger.error(f"Error getting GPU name: {e}")

        logger.info(f"Metrics for {batch_size} were: {metrics_to_submit}")
        # TODO: 如果不同配置是不同运行，如何连续执行它们？
        for callback in callbacks:
            callback.log_metrics(step=idx, metrics=metrics_to_submit)
        for callback in callbacks:
            callback.post_train()

    def run(self) -> None:
        """执行完整的吞吐量基准测试流程。"""
        run_params_list = self.build_sweep_run_params()
        logger.info(
            f"Running {len(run_params_list)} benchmarking runs sweeping over {self.sweep_dict}"
        )
        self.run_benchmarking_sweep(run_params_list)
