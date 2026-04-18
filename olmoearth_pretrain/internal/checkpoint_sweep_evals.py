"""多检查点评估模块，在单个 W&B 运行中记录多个检查点的评估指标。

每个检查点的评估指标以其训练步数作为 x 轴记录，
可以在训练过程中可视化评估性能的变化。

推荐使用方式（通过 full_eval_sweep.py）:
    python -m olmoearth_pretrain.internal.full_eval_sweep \
        --checkpoint_dir=/weka/.../checkpoints/henryh/my_run \
        --cluster=ai2/saturn-cirrascale \
        --module_path=scripts/my_train.py

直接本地使用:
    TRAIN_SCRIPT_PATH=scripts/my_train.py \
    CHECKPOINT_DIR=/weka/.../checkpoints/henryh/my_run \
    torchrun olmoearth_pretrain/internal/checkpoint_sweep_evals.py \
        evaluate my_run_sweep local

Beaker 启动:
    TRAIN_SCRIPT_PATH=scripts/my_train.py \
    CHECKPOINT_DIR=/weka/.../checkpoints/henryh/my_run \
    python3 olmoearth_pretrain/internal/checkpoint_sweep_evals.py \
        launch_evaluate my_run_sweep ai2/saturn-cirrascale

主要函数:
    discover_checkpoints(): 发现检查点目录中的 step{N}/ 子目录
    evaluate_checkpoints(): 评估所有检查点并记录指标
    build_trainer_config(): 构建检查点扫描的训练器配置
"""

import gc
import logging
import os
import re
import sys
import time
from typing import cast

import torch
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.distributed.utils import get_rank
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.train.callbacks import (
    BeakerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.common import Duration
from olmo_core.train.config import TrainerConfig
from olmo_core.utils import get_default_device, prepare_cli_environment, seed_all

from olmoearth_pretrain.internal.all_evals import (
    EMBED_DIAG_TASKS,
    EVAL_TASKS,
    load_user_module,
)
from olmoearth_pretrain.internal.constants import EVAL_WANDB_PROJECT, WANDB_ENTITY
from olmoearth_pretrain.internal.experiment import (
    CommonComponents,
    OlmoEarthEvaluateConfig,
    SubCmd,
    build_evaluate_config,
    launch,
)
from olmoearth_pretrain.internal.utils import (
    MockLatentMIMTrainModule,
    MockOlmoEarthDataLoader,
)
from olmoearth_pretrain.train.callbacks import (
    DownstreamEvaluatorCallbackConfig,
    OlmoEarthWandBCallback,
)
from olmoearth_pretrain.train.callbacks.evaluator_callback import (
    DownstreamEvaluatorCallback,
)

logger = logging.getLogger(__name__)


def discover_checkpoints(
    checkpoint_dir: str, steps: list[int] | None = None
) -> list[tuple[int, str]]:
    """在 checkpoint_dir 中查找 step{N}/ 格式的目录，按步数排序。

    Args:
        checkpoint_dir: 检查点根目录路径
        steps: 可选的步数过滤列表，仅返回指定步数的检查点

    Returns:
        list[tuple[int, str]]: (步数, 目录路径) 元组列表，按步数升序排列
    """
    step_dirs = []
    for entry in os.listdir(checkpoint_dir):
        match = re.match(r"^step(\d+)$", entry)
        if match:
            step_num = int(match.group(1))
            full_path = os.path.join(checkpoint_dir, entry)
            if os.path.isdir(full_path):
                if steps is None or step_num in steps:
                    step_dirs.append((step_num, full_path))
    step_dirs.sort()
    return step_dirs


def evaluate_checkpoints(
    config: OlmoEarthEvaluateConfig,
    checkpoint_dir: str,
    steps: list[int] | None = None,
) -> None:
    """评估 checkpoint_dir 中的所有检查点，将指标记录到单个 W&B 运行。

    核心逻辑:
        1. 发现所有检查点目录
        2. 构建模型和数据加载器
        3. 初始化 W&B 回调，设置 checkpoint_step 为 x 轴
        4. 逐个加载检查点权重并运行评估
        5. 记录验证/测试指标和嵌入诊断

    Args:
        config: OlmoEarth 评估配置
        checkpoint_dir: 检查点根目录路径
        steps: 可选的步数过滤列表
    """
    seed_all(config.init_seed)  # 设置随机种子

    checkpoints = discover_checkpoints(checkpoint_dir, steps=steps)
    if not checkpoints:
        raise ValueError(f"No step directories found in {checkpoint_dir}")
    logger.info(f"Found {len(checkpoints)} checkpoints: {[s for s, _ in checkpoints]}")

    # 构建模型
    model = config.model.build()
    device = get_default_device()
    model = model.to(device)
    data_loader = MockOlmoEarthDataLoader()

    # 构建训练模块（如果可用，需要正确的模型架构初始化）
    if config.train_module is not None:
        train_module = config.train_module.build(model)
        data_loader.min_patch_size = model.encoder.min_patch_size
        data_loader.max_patch_size = model.encoder.max_patch_size
    else:
        train_module = MockLatentMIMTrainModule()
    train_module.model = model

    # 构建训练器（连接回调，包括评估器和 W&B）
    trainer = config.trainer.build(train_module, data_loader)

    config_dict = config.as_config_dict()
    wandb_callback = cast(OlmoEarthWandBCallback, trainer.callbacks["wandb"])
    wandb_callback.config = config_dict
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # 初始化 W&B（不运行评估或启动训练循环）
    wandb_callback.pre_train()

    # 告诉 W&B 使用 checkpoint_step 作为评估指标的 x 轴
    if wandb_callback.enabled and get_rank() == 0:
        wandb_callback.wandb.define_metric("checkpoint_step")
        wandb_callback.wandb.define_metric("eval/*", step_metric="checkpoint_step")
        wandb_callback.wandb.define_metric("eval/test/*", step_metric="checkpoint_step")
        wandb_callback.wandb.define_metric("eval_time/*", step_metric="checkpoint_step")
        wandb_callback.wandb.define_metric(
            "eval_embed_diagnostics/*", step_metric="checkpoint_step"
        )

    # 获取评估回调（包含构建好的评估器对象）
    eval_callback = trainer.callbacks.get("downstream_evaluator")
    if not isinstance(eval_callback, DownstreamEvaluatorCallback):
        raise ValueError("downstream_evaluator callback not found or disabled")

    for step_num, step_path in checkpoints:
        logger.info(f"=== Evaluating checkpoint step {step_num}: {step_path} ===")

        # 从分布式检查点加载模型权重
        train_module_dir = os.path.join(step_path, "model_and_optim")
        load_model_and_optim_state(train_module_dir, model)
        model.to(device)

        # 运行所有评估器并收集此检查点的指标
        metrics: dict[str, float | int] = {"checkpoint_step": step_num}

        for evaluator in eval_callback.evaluators:
            if not eval_callback._check_supported_modalities(evaluator):
                logger.info(
                    f"  Skipping {evaluator.evaluation_name} (unsupported modalities)"
                )
                continue
            if not eval_callback._check_input_requirements(evaluator):
                logger.info(
                    f"  Skipping {evaluator.evaluation_name} (input requirements)"
                )
                continue

            start_time = time.monotonic()
            result = evaluator.val()
            eval_time = time.monotonic() - start_time

            val_result = result.val_result
            test_result = result.test_result

            if val_result is not None:
                metrics[f"eval/{evaluator.evaluation_name}"] = val_result.primary
                for k, v in val_result.metrics.items():
                    metrics[f"eval/{evaluator.evaluation_name}/{k}"] = v

            if eval_callback.run_on_test and test_result is not None:
                metrics[f"eval/test/{evaluator.evaluation_name}"] = test_result.primary
                for k, v in test_result.metrics.items():
                    metrics[f"eval/test/{evaluator.evaluation_name}/{k}"] = v

            if result.embedding_diagnostics:
                for k, v in result.embedding_diagnostics.items():
                    metrics[
                        f"eval_embed_diagnostics/{evaluator.evaluation_name}/{k}"
                    ] = v

            metrics[f"eval_time/{evaluator.evaluation_name}"] = eval_time

            logger.info(
                f"  {evaluator.evaluation_name}: "
                f"val={val_result.primary if val_result else 'N/A'}, "
                f"test={test_result.primary if test_result else 'N/A'} "
                f"({eval_time:.1f}s)"
            )

        # 在一次调用中记录此检查点的所有指标
        if wandb_callback.enabled and get_rank() == 0:
            wandb_callback.wandb.log(metrics)
            logger.info(f"Logged {len(metrics)} metrics for step {step_num}")

        gc.collect()  # 显式垃圾回收
        torch.cuda.empty_cache()  # 清空 CUDA 缓存

    if wandb_callback.enabled and get_rank() == 0:
        wandb_callback.wandb.finish()
    logger.info("Checkpoint sweep evaluation complete.")


def _get_eval_tasks() -> dict:
    """根据 EMBEDDING_DIAGNOSTICS_ONLY 环境变量选择任务集。

    Returns:
        dict: 如果设置了 EMBEDDING_DIAGNOSTICS_ONLY 则返回嵌入诊断任务，否则返回完整评估任务
    """
    if os.environ.get("EMBEDDING_DIAGNOSTICS_ONLY"):
        return EMBED_DIAG_TASKS
    return EVAL_TASKS


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """为检查点扫描构建训练器配置（无训练、无自动评估）。

    配置包括: W&B 回调、GPU 内存监控、配置保存、下游评估器、
    垃圾回收和 Beaker 回调。

    Args:
        common: 通用组件

    Returns:
        TrainerConfig: 训练器配置
    """
    checkpointer_config = CheckpointerConfig(work_dir=common.save_folder)
    wandb_callback = OlmoEarthWandBCallback(
        name=common.run_name,
        project=EVAL_WANDB_PROJECT,
        entity=WANDB_ENTITY,
        enabled=True,
        upload_dataset_distribution_pre_train=False,
        upload_modality_data_band_distribution_pre_train=False,
    )
    trainer_config = (
        TrainerConfig(
            work_dir=common.save_folder,
            save_folder=common.save_folder,
            cancel_check_interval=1,
            metrics_collect_interval=10,
            max_duration=Duration.epochs(300),
            checkpointer=checkpointer_config,
        )
        .with_callback("wandb", wandb_callback)
        .with_callback("gpu_memory_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=_get_eval_tasks(),
                eval_on_startup=False,
                cancel_after_first_eval=False,
                run_on_test=True,
            ),
        )
        .with_callback("garbage_collector", GarbageCollectorCallback(gc_interval=1))
        .with_callback("beaker", BeakerCallback())
    )
    return trainer_config


def parse_steps(steps_str: str | None) -> list[int] | None:
    """解析逗号分隔的步数字符串（如 '5000,10000,15000'）。

    Args:
        steps_str: 逗号分隔的步数字符串，None 表示不过滤

    Returns:
        list[int] | None: 步数列表，输入为 None 时返回 None
    """
    if steps_str is None:
        return None
    return [int(s.strip()) for s in steps_str.split(",") if s.strip()]


if __name__ == "__main__":
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR")
    if checkpoint_dir is None:
        raise ValueError("CHECKPOINT_DIR environment variable must be set")

    module_path = os.environ.get("TRAIN_SCRIPT_PATH")
    if module_path is None:
        raise ValueError("TRAIN_SCRIPT_PATH environment variable must be set")

    # 可选：仅评估特定步数（逗号分隔）
    steps = parse_steps(os.environ.get("CHECKPOINT_STEPS"))

    user_mod = load_user_module(module_path)

    try:
        build_common_components = user_mod.build_common_components
    except AttributeError:
        from olmoearth_pretrain.internal.common import build_common_components

    try:
        build_train_module_config = user_mod.build_train_module_config
    except AttributeError:
        build_train_module_config = None

    try:
        build_model_config = user_mod.build_model_config
    except AttributeError:
        raise AttributeError(
            f"Module at {module_path} has no 'build_model_config'. "
            f"Point --module_path to the size-specific script "
            f"(e.g. scripts/official/base.py instead of scripts/official/script.py)."
        )

    usage = (
        f"Usage: CHECKPOINT_DIR=... TRAIN_SCRIPT_PATH=... "
        f"[CHECKPOINT_STEPS=5000,10000,...] "
        f"python3/torchrun {sys.argv[0]} "
        f"evaluate|launch_evaluate|dry_run_evaluate RUN_NAME CLUSTER [OVERRIDES...]"
    )
    if len(sys.argv) < 4:
        print(usage)
        sys.exit(1)

    script, cmd, run_name, cluster, *overrides = sys.argv
    common = build_common_components(script, cmd, run_name, cluster, overrides)

    cmd = SubCmd(cmd)

    config = build_evaluate_config(
        common=common,
        model_config_builder=build_model_config,
        trainer_config_builder=build_trainer_config,
        overrides=overrides,
        train_module_config_builder=build_train_module_config,
    )

    if cmd == SubCmd.launch_evaluate:
        prepare_cli_environment()
        launch(config)
    elif cmd == SubCmd.evaluate:
        prepare_training_environment()
        try:
            evaluate_checkpoints(config, checkpoint_dir, steps=steps)
        finally:
            teardown_training_environment()
    elif cmd == SubCmd.dry_run_evaluate:
        prepare_cli_environment()
        logger.info(config)
        checkpoints = discover_checkpoints(checkpoint_dir, steps=steps)
        logger.info(
            f"Would evaluate {len(checkpoints)} checkpoints: "
            f"{[s for s, _ in checkpoints]}"
        )
    else:
        raise ValueError(
            f"Unsupported command: {cmd}. "
            f"Use evaluate, launch_evaluate, or dry_run_evaluate."
        )
