"""推理吞吐量基准测试共享常量。

本模块定义了基准测试使用的各种常量，包括：
    - Beaker 集群配置（预算、工作区、GPU 到集群映射）
    - GCP 配置（待实现）
    - Weights & Biases 配置（项目名和实体名）
    - 指标名称常量
    - 参数键映射
    - 预定义的参数扫描字典
"""

# BEAKER 平台配置
BEAKER_BUDGET = "ai2/es-platform"  # Beaker 预算
BEAKER_WORKSPACE = "ai2/earth-systems"  # Beaker 工作区
WEKA_BUCKET = "dfive-default"  # WEKA 存储桶
BEAKER_TASK_PRIORITY = "normal"  # Beaker 任务优先级
BEAKER_GPU_TO_CLUSTER_MAP = {  # GPU 类型到 Beaker 集群的映射
    "H100": [  # H100 GPU 对应的集群
        "ai2/jupiter-cirrascale-2",
    ],
    "A100": [  # A100 GPU 对应的集群
        "ai2/saturn-cirrascale",
    ],
    "L40S": [  # L40S GPU 对应的集群
        "ai2/neptune-cirrascale",
    ],
}

# GCP 平台配置
# TODO: 添加 GCP 项目、存储桶、GCR 镜像、GPU 到实例类型映射

# Weights & Biases 配置
PROJECT_NAME = "inference-throughput-no-mask"
ENTITY_NAME = "eai-ai2"

ARTIFACTS_DIR = "/artifacts"  # 产物目录

# 指标名称常量
PER_BATCH_TOKEN_RATE_METRIC = "per_batch_token_rate"  # nosec - 每批次令牌处理率
MEAN_BATCH_TOKEN_RATE_METRIC = "mean_batch_token_rate"  # nosec - 平均批次令牌处理率
MEAN_BATCH_TIME_METRIC = "mean_batch_time"  # nosec - 平均批次处理时间
NUM_TOKENS_PER_BATCH_METRIC = "num_tokens_per_batch"  # nosec - 每批次令牌数
SQUARE_KM_PER_SECOND_METRIC = "square_km_per_second"  # nosec - 每秒处理平方公里数
PIXELS_PER_SECOND_METRIC = "pixels_per_second"  # nosec - 每秒处理像素数
OOM_OCCURRED_METRIC = "oom_occurred"  # nosec - 是否发生 OOM（1=发生, 0=未发生）
GPU_NAME_METRIC = "gpu_name"  # nosec - GPU 名称

PARAM_KEYS = dict(  # 参数名到环境变量名的映射
    model_size="MODEL_SIZE",  # 模型大小
    checkpoint_path="CHECKPOINT_PATH",  # 检查点路径
    use_s1="USE_S1",  # 是否使用 Sentinel-1
    use_s2="USE_S2",  # 是否使用 Sentinel-2
    use_landsat="USE_LANDSAT",  # 是否使用 Landsat
    image_size="IMAGE_SIZE",  # 图像大小
    patch_size="PATCH_SIZE",  # 补丁大小
    num_timesteps="NUM_TIMESTEPS",  # 时间步数
    batch_size="BATCH_SIZE",  # 批量大小
    batch_sizes="BATCH_SIZES",  # 批量大小列表
    gpu_type="GPU_TYPE",  # GPU 类型
    bf16="BF16",  # 是否使用 BF16
    benchmark_interval_s="BENCHMARK_INTERVAL_S",  # 基准测试间隔（秒）
    min_batches_per_interval="MIN_BATCHES_PER_INTERVAL",  # 每个间隔最小批次数
    project="PROJECT",  # W&B 项目
    owner="OWNER",  # W&B 所有者
    name="NAME",  # 运行名称
)


# 预定义的参数扫描字典，用于不同维度的基准测试
sweep_batch_sizes = {  # 批量大小扫描
    "batch_size": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
}
sweep_image_sizes = {  # 图像大小扫描
    "image_size": [1, 2, 4, 8, 16, 32, 64, 128],
}
sweep_patch_sizes = {"patch_size": [1, 2, 4, 8]}  # 补丁大小扫描
sweep_num_timesteps = {"num_timesteps": [1, 2, 4, 6, 8, 12]}  # 时间步数扫描
sweep_use_s1 = {"use_s1": [True, False]}  # Sentinel-1 使用扫描
sweep_use_s2 = {"use_s2": [True, False]}  # Sentinel-2 使用扫描
sweep_use_landsat = {"use_landsat": [True, False]}  # Landsat 使用扫描
sweep_bf16 = {"bf16": [True, False]}  # BF16 精度扫描
sweep_model_size = {"model_size": ["nano", "tiny", "base", "large"]}  # 模型大小扫描


SWEEPS = {  # 预定义的扫描组合
    "batch": sweep_batch_sizes,  # 仅扫描批量大小
    "image": sweep_image_sizes,  # 仅扫描图像大小
    "patch": sweep_patch_sizes,  # 仅扫描补丁大小
    "time": sweep_num_timesteps,  # 仅扫描时间步数
    "use_s1": sweep_use_s1,  # 仅扫描 Sentinel-1 使用
    "use_s2": sweep_use_s2,  # 仅扫描 Sentinel-2 使用
    "use_landsat": sweep_use_landsat,  # 仅扫描 Landsat 使用
    "bf16": sweep_bf16,  # 仅扫描 BF16 精度
    "model_size": sweep_model_size,  # 仅扫描模型大小
    "all": sweep_batch_sizes  # 扫描所有参数组合
    | sweep_image_sizes
    | sweep_patch_sizes
    | sweep_num_timesteps
    | sweep_use_s1
    | sweep_use_s2
    | sweep_use_landsat
    | sweep_bf16
    | sweep_model_size,
}
