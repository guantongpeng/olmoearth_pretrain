"""OlmoEarth Pretrain 项目内部常量定义。

本模块定义了评估和训练流程中使用的全局常量，包括：
    - EVAL_WANDB_PROJECT: 评估任务的 Weights & Biases 项目名称
    - WANDB_ENTITY: Weights & Biases 实体（组织）名称
    - EVAL_LAUNCH_PATH: 评估启动脚本的模块路径
    - CHECKPOINT_SWEEP_LAUNCH_PATH: 检查点扫描评估启动脚本的模块路径
"""

EVAL_WANDB_PROJECT = "olmoearth_pretrain_knn_and_lp_evals"  # 评估任务的 W&B 项目名称
WANDB_ENTITY = "eai-ai2"  # nosec - W&B 实体名称
EVAL_LAUNCH_PATH = "olmoearth_pretrain/internal/all_evals.py"  # 评估启动脚本路径
CHECKPOINT_SWEEP_LAUNCH_PATH = "olmoearth_pretrain/internal/checkpoint_sweep_evals.py"  # 检查点扫描评估脚本路径
