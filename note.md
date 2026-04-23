# OlmoEarth Pretrain 项目详细分析

## 1. 项目概述

**OlmoEarth Pretrain** 是由 Allen Institute for AI (AI2) 开发的**多模态时空地球观测基础模型**预训练框架。项目实现了基于 Latent Masked Image Modeling (Latent MIM) 的自监督预训练范式，采用编码器-解码器架构配合 EMA 目标编码器（类似 I-JEPA / data2vec 风格）。

- **包名**: `olmoearth-pretrain` v0.1.0
- **Python**: >=3.11, <3.14
- **PyTorch**: >=2.7, <2.8
- **训练框架**: 基于 AI2 的 `olmo-core` (v2.3.0)
- **许可证**: OlmoEarth Artifact License
- **HuggingFace 模型**: `allenai/OlmoEarth-v1-{Nano,Tiny,Base,Large}`

---

## 2. 目录结构

```
olmoearth_pretrain/
├── .cursor/rules/                    # Cursor IDE 规则
├── .github/workflows/                # CI/CD (tests, lint, publish)
├── assets/                           # Logo、模型图、数据地图
├── data/                             # 数据配置和映射
│   ├── eurocrops_hcat3_mapping.json
│   ├── rslearn_dataset_configs/      # 各数据源 JSON 配置
│   └── norm_configs/                 # 归一化配置 (predefined.json, computed.json, minmax_stats.json)
├── docs/                             # 文档
├── helios/                           # 向后兼容的旧包名 shim
├── olmoearth_pretrain/               # **主代码包**
│   ├── __init__.py
│   ├── _compat.py                    # 废弃类/函数别名
│   ├── config.py                     # 双模式配置系统
│   ├── datatypes.py                  # 核心数据结构
│   ├── decorators.py                 # @experimental 装饰器
│   ├── model_loader.py               # HuggingFace 模型加载
│   ├── types.py                      # 类型别名
│   ├── data/                         # 数据加载与处理
│   │   ├── constants.py              # 模态规格、BandSet、分辨率常量
│   │   ├── dataset.py                # OlmoEarthDataset (H5 PyTorch Dataset)
│   │   ├── dataloader.py             # OlmoEarthDataLoader
│   │   ├── collate.py                # 批次整理
│   │   ├── concat.py                 # 数据集拼接
│   │   ├── normalize.py              # 归一化器
│   │   ├── transform.py              # 数据增强
│   │   └── utils.py, visualize.py
│   ├── dataset/                      # H5 转换与解析
│   │   ├── convert_to_h5py.py
│   │   └── parse.py, sample.py, utils.py
│   ├── dataset_creation/             # 预训练数据集创建工具
│   │   ├── create_windows/           # 空间窗口生成
│   │   ├── rslearn_to_olmoearth/     # 各数据源转换器
│   │   ├── openstreetmap/            # OSM 瓦片处理
│   │   └── sentinel2_l1c/            # S2 L1C 数据管线
│   ├── nn/                           # 神经网络模块
│   │   ├── flexi_vit.py              # **核心 FlexiViT 编码器** (2168 行)
│   │   ├── flexi_patch_embed.py      # 灵活 Patch 嵌入
│   │   ├── attention.py              # Transformer 注意力块 (支持 Flash Attention)
│   │   ├── encodings.py              # 位置编码 (sin/cos, 分辨率感知, 月份)
│   │   ├── tokenization.py           # 自定义波段分组/分词配置
│   │   ├── latent_mim.py             # Latent MIM 模型
│   │   ├── mae.py                    # MAE 模型
│   │   ├── galileo.py                # Galileo 模型
│   │   ├── st_model.py               # 时空模型基类
│   │   ├── pooling.py                # Token 池化策略
│   │   └── utils.py                  # 分布式混入、序列长度工具
│   ├── train/                        # 训练基础设施
│   │   ├── masking.py                # 掩码策略 (2103 行)
│   │   ├── loss.py                   # 损失函数 (1122 行)
│   │   ├── utils.py                  # 训练工具
│   │   ├── callbacks/                # 训练回调
│   │   │   ├── evaluator_callback.py # 训练中下游评估
│   │   │   ├── speed_monitor.py      # 吞吐量监控
│   │   │   └── wandb.py             # W&B 日志回调
│   │   └── train_module/             # 训练模块
│   │       ├── train_module.py       # 基类 OlmoEarthTrainModule
│   │       ├── latent_mim.py         # Latent MIM 训练模块
│   │       ├── contrastive_latentmim.py  # 对比 + Latent MIM
│   │       ├── mae.py                # MAE 训练模块
│   │       └── galileo.py            # Galileo 训练模块
│   ├── evals/                        # 评估框架
│   │   ├── eval_wrapper.py           # 统一评估包装器
│   │   ├── embeddings.py, embedding_diagnostics.py
│   │   ├── linear_probe.py, knn.py, metrics.py
│   │   ├── datasets/                 # 评估数据集实现
│   │   ├── finetune/                 # 微调评估
│   │   ├── models/                   # 基线模型包装器 (10+ 模型)
│   │   └── studio_ingest/            # Studio 数据摄取
│   ├── inference_benchmarking/       # 推理吞吐量基准
│   └── internal/                     # AI2 内部实验编排
│       ├── experiment.py             # 训练实验主入口
│       └── common.py                 # 共享实验组件构建器
├── scripts/                          # 训练和实验脚本
│   ├── official/                     # 官方模型训练脚本
│   │   ├── script.py                 # 共享构建函数
│   │   ├── base.py, large.py, tiny.py, nano.py
│   │   ├── base_launch.sh, large_launch.sh, etc.
│   │   └── ablations/               # 消融实验
│   ├── vnext/                        # 下一版本实验
│   └── archived/                     # 历史实验
├── tests/                            # 测试套件
├── pyproject.toml                    # 项目配置和依赖
└── uv.lock                           # uv 锁文件
```

---

## 3. 模型架构

### 3.1 顶层架构: Latent MIM

**文件**: `olmoearth_pretrain/nn/latent_mim.py`

Latent MIM 遵循**联合嵌入预测架构 (JEPA)** 模式:

```
LatentMIM
  ├── encoder (Encoder)          -- 在线编码器，处理掩码输入
  ├── target_encoder             -- EMA 编码器 (encoder 的深拷贝)，处理完整未掩码输入
  ├── decoder (Predictor)        -- 从掩码位置预测目标编码器输出
  └── reconstructor (可选)       -- 可选 MAE 风格像素重建
```

**前向传播流程**:
1. 在线编码器处理掩码输入 → `latent` tokens
2. 解码器接收 latent tokens 并预测掩码位置 → `decoded` tokens
3. 目标编码器 (`no_grad`, EMA 更新) 处理未掩码输入 → `target_output` tokens
4. 在 DECODER 掩码位置计算 `decoded` 与 `target_output` 之间的损失
5. 可选: 重建器提供 MAE 风格像素重建

### 3.2 编码器 (FlexiViT)

**文件**: `olmoearth_pretrain/nn/flexi_vit.py` (类 `Encoder`, 约 2168 行)

`Encoder` 继承 `FlexiVitBase`，是一个**多模态 Vision Transformer**:

- **多模态 Patch 嵌入** (`MultiModalPatchEmbeddings`): 每个模态、每个 BandSet 有独立的 FlexiPatchEmbed 模块
- **复合位置编码** (`CompositeEncodings`): 4 种加性编码，各占嵌入维度的 1/4:
  - **通道/模态嵌入**: 可学习 (或零初始化) 的每 BandSet 嵌入
  - **时间位置编码**: 1D 正弦编码 (时间维度)
  - **月份编码**: 正弦月份编码 (12 个月)
  - **空间编码**: 2D 正弦编码，支持分辨率缩放
- **Transformer 块**: 标准 pre-norm Transformer 块，含自注意力
- **ProjectAndAggregate**: 投影和池化 tokens 用于对比损失
- **Register tokens**: 可选的 DINOv2 风格 register tokens
- **Band dropout**: 训练时随机置零波段，强制跨光谱学习
- **输出嵌入投影器**: 可选投影到不同嵌入大小

编码器前向传播:
1. 对输入数据应用多模态 Patch 嵌入
2. 添加复合编码 (通道 + 时间 + 月份 + 空间)
3. 将所有模态的 tokens 展平为单一序列
4. 通过 Transformer 块
5. 可选: 为可变长度 Flash Attention 打包 tokens
6. 拆分回每模态 token 表示

### 3.3 解码器/预测器

`Predictor` (同样在 `flexi_vit.py`) 是一个浅层 Transformer:
- 接收编码器输出 tokens (在 decoder-only 位置插入 mask tokens)
- 使用 mask tokens 与可见 (编码器输出) tokens 之间的交叉注意力
- 预测掩码位置的目标编码器表示

### 3.4 重建器

`Reconstructor` (在 `flexi_vit.py`) 可选地添加 MAE 风格像素重建:
- 包含一个解码器 + 每模态 `FlexiPatchReconstruction` 模块
- 从编码器输出重建原始像素值

### 3.5 注意力机制

**文件**: `olmoearth_pretrain/nn/attention.py`

- **自注意力** 和 **交叉注意力**
- **Flash Attention** (通过 `flash_attn` 包)，支持可变长度
- **PyTorch SDPA** (`F.scaled_dot_product_attention`)
- **QK 归一化** (可选 LayerNorm)
- **LayerScale** (DINOv2 风格可学习逐通道缩放)
- **DropPath** (随机深度正则化)
- **MLP**: GELU 激活，可配置隐藏维度比

### 3.6 FlexiPatchEmbed

**文件**: `olmoearth_pretrain/nn/flexi_patch_embed.py`

灵活 Patch 嵌入支持:
- **nn.Linear 投影** (默认，通过 cuBLAS GEMM 更快)
- **nn.Conv2d 投影** (用于旧检查点)
- **多分辨率支持**: 运行时可插值到不同 patch 大小
- **多时态支持**: 处理带时间维度的 5D 张量
- 每模态 `image_tile_size_factor` 用于不同空间分辨率

### 3.7 模型尺寸变体

| 变体 | 编码器嵌入 | 编码器深度 | 编码器头数 | 解码器深度 | 解码器嵌入 | 解码器头数 | MLP 比 |
|------|-----------|-----------|-----------|-----------|-----------|-----------|--------|
| nano | 128 | 4 | 8 | 4 | 128 | 8 | 4.0 |
| tiny | 192 | 12 | 3 | 12 | 192 | 3 | 4.0 |
| base | 768 | 12 | 12 | 12 | 768 | 12 | 4.0 |
| large | 1024 | 24 | 16 | 24 | 1024 | 16 | 4.0 |
| giga | 1536 | 40 | 24 | 40 | 1536 | 24 | 4.0 |

浅层解码器变体也存在 (base/large 的 decoder_depth=4)。

---

## 4. 数据模态与分词

### 4.1 支持的模态

**文件**: `olmoearth_pretrain/data/constants.py`

系统支持 **16 种数据模态**，每种定义为 `ModalitySpec`:

| 模态 | 波段 | 分辨率因子 | 时态 | 空间 |
|------|------|-----------|------|------|
| sentinel2_l2a | 10 波段 (B02-B12 不含 B10)，3 个 BandSet (10m, 20m, 60m) | 16 | 是 | 是 |
| sentinel1 | 2 波段 (vv, vh) | 16 | 是 | 是 |
| landsat | 11 波段，2 个 BandSet (15m, 30m) | 16 | 是 | 是 |
| worldcover | 1 波段 | 16 | 否 | 是 |
| openstreetmap_raster | 30 波段 (建筑、道路、河流等) | 16 | 否 | 是 |
| srtm | 1 波段 (高程) | 16 | 否 | 是 |
| cdl | 1 波段 (作物分类) | 16 | 否 | 是 |
| worldpop | 1 波段 (人口) | 16 | 否 | 是 |
| worldcereal | 8 波段 (作物类型分类) | 16 | 否 | 是 |
| wri_canopy_height_map | 1 波段 (冠层高度) | 16 | 否 | 是 |
| era5_10 | 6 波段 (气候: 温度、露点、气压、风速、降水) | 16 | 是 | 否 (仅时间) |
| gse | 64 波段 (Google 卫星嵌入) | 16 | 否 | 是 |
| naip_10 | 4 波段 (R,G,B,IR) | 16 | 否 | 是 |
| eurocrops | 1 波段 | 16 | 否 | 是 |
| ndvi | 1 波段 (由 S2 B04/B08 计算) | 16 | 是 | 是 |
| latlon | 2 值 (纬度, 经度) | 0 (非空间) | 否 | 否 |

**关键常量**: `BASE_GSD=10` (米/像素), `IMAGE_TILE_SIZE=256`, `MAX_SEQUENCE_LENGTH=12` (时间步), `MISSING_VALUE=-99999`

### 4.2 分词配置

**文件**: `olmoearth_pretrain/nn/tokenization.py`

`TokenizationConfig` 允许覆盖每模态的默认波段分组。每个 BandSet 成为一个独立 token。例如 Sentinel-2 L2A 有 3 个默认 BandSet:
- 10m 波段: [B02, B03, B04, B08] → 1 token
- 20m 波段: [B05, B06, B07, B8A, B11, B12] → 1 token
- 60m 波段: [B01, B09] → 1 token

自定义分词可以覆盖此设置，例如让每个波段成为独立 token。

---

## 5. 模型 Embedding 实现详解

### 5.1 整体架构

Embedding 管线遵循以下流程：

```
输入数据 → Patch Embedding (per-modality) → Composite Positional Encodings → Transformer Blocks
                |                                       |
                |                                       +-- Channel Embedding (模态/波段标识)
                |                                       +-- Temporal Position Embedding (1D sincos)
                |                                       +-- Month Embedding (sincos)
                |                                       +-- Spatial Embedding (2D sincos + GSD)
                |
                +-- 空间模态: FlexiPatchEmbed (nn.Linear 或 Conv2d)
                +-- 非空间模态: nn.Linear
                +-- Band Grouping via TokenizationConfig
                +-- 可选 Band Dropout
```

最终每个 token 的表示为：

```
final_token = patch_embed(pixel_data) + channel_embed + time_embed + month_embed + spatial_embed
```

### 5.2 Patch Embedding — `flexi_patch_embed.py`

#### `FlexiPatchEmbed` (`flexi_patch_embed.py:53-285`)

将 2D 图像划分为 patch 并投影到嵌入空间，核心特性是**支持推理时动态调整 patch 大小**。

**初始化** (`__init__`):
- `base_patch_size = base_patch_size_at_16 * modality_spec.image_tile_size_factor` — 权重存储的基准大小
- 两种投影后端：
  - `nn.Linear`（默认）：将 patch 展平为 `(P_h * P_w * C)` 维向量后线性投影，利用 cuBLAS GEMM 在 TensorCore 上高效运行
  - `nn.Conv2d`（兼容旧检查点）

**前向传播** (`forward`, line 231):
1. 输入 `[B, H, W, C]` 或 `[B, H, W, T, C]` 重排为 `[B*T, C, H, W]`
2. 若请求的 `patch_size` 与 `base_patch_size` 不同，**插值调整输入图像大小**使其能被 base_patch_size 整除（而非插值权重）
3. 通过 `nn.Linear` 或 `nn.Conv2d` 投影
4. 应用 LayerNorm

**关键设计**：灵活 patch 大小的实现方式是调整输入图像分辨率而非调整权重，输出形状为 `[B, H/P, W/P, D]` 或 `[B, H/P, W/P, T, D]`。

#### `FlexiPatchReconstruction` (`flexi_patch_embed.py:288-430`)

解码器端的逆操作，使用 `nn.ConvTranspose2d` 将 token 重建回像素空间，同样支持灵活 patch 大小。

### 5.3 多模态 Patch Embedding 管理 — `flexi_vit.py:270-548`

#### `MultiModalPatchEmbeddings`

为每种模态的每个波段组（bandset）创建独立的嵌入模块。

**初始化** (`__init__`, line 288):
- 遍历 `supported_modality_names`，为每个模态调用 `_get_patch_embedding_module_for_modality()`
- 空间模态 → 每个波段组一个 `FlexiPatchEmbed`
- 非空间模态 → 每个波段组一个 `nn.Linear`
- 为每个波段组注册索引 buffer（`register_buffer`），用于从输入数据中选择对应波段通道

**前向传播** (`apply_embedding_to_modality`, line 402):
1. 根据波段组索引从输入数据中 `torch.index_select` 选择对应波段
2. 训练时可选应用 **Band Dropout**：以 `band_dropout_rate` 概率随机将某些波段通道置零，确保每样本至少保留 1 个波段
3. 通过对应的嵌入模块投影
4. 输出形状：空间模态 `[B, H/P, W/P, T, b_s, D]`，非空间模态 `[B, T, b_s, D]`（`b_s` = 波段组数）

### 5.4 波段分组/分词配置 — `tokenization.py`

#### `ModalityTokenization` (line 36)

定义单个模态的波段分组策略。`band_groups: list[list[str]]` 指定波段如何分组，每组成为一个 token。例如 Sentinel-2 的 12 个波段可以：
- 全部作为一组 → 1 个 token
- 每个波段一组 → 12 个 token
- 按光谱范围分组 → 如可见光、近红外、短波红外

#### `TokenizationConfig` (line 107)

全局分词配置，`overrides` 字典允许对特定模态覆盖默认分组。未被覆盖的模态使用 `ModalitySpec.bandsets_as_indices()` 的默认配置。

### 5.5 位置编码函数 — `encodings.py`

四个纯函数，全部**确定性、无学习参数**：

| 函数 | 行号 | 用途 | 输出形状 |
|------|------|------|----------|
| `get_1d_sincos_pos_encoding` | 17 | 时间步位置编码 | `(L, D)` |
| `get_2d_sincos_pos_encoding` | 43 | 2D 空间编码 | `(H*W, D)` |
| `get_2d_sincos_pos_encoding_with_resolution` | 67 | **带 GSD 的 2D 编码** | `(n, H*W, D)` |
| `get_month_encoding_table` | 119 | 月份编码表 | `(12, D)` |

**分辨率感知编码**（`get_2d_sincos_pos_encoding_with_resolution`）是遥感场景的关键设计：将空间坐标乘以 GSD（地面采样距离），使编码反映**实际地面距离**而非像素索引，从而正确处理不同空间分辨率的输入。

```python
# encodings.py:104 — 坐标乘以分辨率
grid = torch.einsum("chw,n->cnhw", grid, res)
```

### 5.6 复合位置编码 — `CompositeEncodings` (`flexi_vit.py:792-1050`)

这是 Embedding 管线的**核心模块**，将四种编码组合后加到 patch embedding 上。

#### 维度分配策略

嵌入维度 `D` 平均分为 4 份，每份 `n = D/4`：

| 维度范围 | 编码类型 | 是否可学习 | 适用条件 |
|----------|----------|-----------|----------|
| `[0, n)` | Channel Embedding | 可学习（默认）或冻结零初始化 | 所有模态 |
| `[n, 2n)` | Temporal Position | 冻结 (sincos) | 多时相模态 |
| `[2n, 3n)` | Month Embedding | 冻结 (sincos) | 多时相模态 |
| `[3n, 4n)` | Spatial Embedding | 冻结 (sincos + GSD) | 空间模态 |

#### 初始化 (`__init__`, line 808)

```python
# 时间位置编码：1D sincos，冻结
self.pos_embed = nn.Parameter(
    get_1d_sincos_pos_encoding(torch.arange(max_sequence_length), n),
    requires_grad=False,
)

# 月份编码：sincos 表，冻结
month_tab = get_month_encoding_table(n)
self.month_embed = nn.Embedding.from_pretrained(month_tab, freeze=True)

# 通道编码：每个模态每个波段组一个向量，可学习
self.per_modality_channel_embeddings = nn.ParameterDict()
for modality in supported_modalities:
    num_bandsets = tokenization_config.get_num_bandsets(modality.name)
    shape = (num_bandsets, n)
    channel_embeddings = nn.Parameter(torch.zeros(shape), requires_grad=True)
```

#### 编码应用 (`_apply_encodings_per_modality`, line 896)

创建一个与 token 同形状的零张量，将四种编码分别写入对应维度切片，最后加到 patch embedding 上：

```python
modality_embed = torch.zeros(modality_tokens.shape, device=device)

# 1. Channel Embedding → dims [0:n]
channel_embed = self.per_modality_channel_embeddings[modality.name]  # (b_s, n)
modality_embed[..., :n] += channel_embed

# 2. Temporal Position → dims [n:2n]（仅多时相模态）
time_embed = self.pos_embed[:t]  # (t, n)
modality_embed[..., n : n*2] += time_embed

# 3. Month Embedding → dims [2n:3n]（仅多时相模态）
months = timestamps[:, :, 1]
month_embed = self.month_embed(months)  # (B, T, n)
modality_embed[..., n*2 : n*3] += month_embed

# 4. Spatial Embedding → dims [3n:4n]（仅空间模态）
gsd_ratio = input_res * patch_size / BASE_GSD
spatial_embed = get_2d_sincos_pos_encoding_with_resolution(...)  # (B, H*W, n)
modality_embed[..., n*3 : n*4] += spatial_embed

return modality_tokens + modality_embed
```

#### GSD 比率计算

```python
@staticmethod
def calculate_gsd_ratio(input_res: float, patch_size: int) -> float:
    return input_res * patch_size / BASE_GSD
```

这确保了不同分辨率输入的空间编码按实际地面距离缩放，例如 10m 分辨率下 `patch_size=16` 的空间范围与 0.625m 分辨率下 `patch_size=256` 的空间范围会得到不同的 GSD 比率。

### 5.7 Encoder 中的完整流程 — `flexi_vit.py:1314-1998`

`Encoder.forward()` 的 Embedding 阶段：

1. **Patch Embedding**：`patch_embeddings.forward(x, patch_size)` → 将原始数据转为 per-modality token
2. **Composite Encodings**：`composite_encodings.forward(tokens, timestamps, patch_size, input_res)` → 添加四种位置/模态编码
3. **Collapse & Combine**：`collapse_and_combine_hwtc(tokens_dict)` → 将所有模态的 token 拼接为单一序列送入 Transformer
4. Transformer 处理后，`split_and_expand_per_modality()` 将输出拆分回 per-modality 形状

### 5.8 设计要点总结

1. **四等分维度分配**：Channel/Temporal/Month/Spatial 各占 D/4，互不干扰，无需学习如何融合不同类型的位置信息
2. **冻结 vs 可学习**：空间、时间、月份编码全部冻结（sincos），仅通道编码（模态/波段标识）可学习，让模型学习不同模态/波段组之间的关系
3. **分辨率感知**：空间编码乘以 GSD 比率，使同一地理位置不同分辨率的输入获得一致的空间编码
4. **灵活 Patch 大小**：FlexiViT 设计允许推理时改变 patch 大小，通过插值输入图像而非权重实现
5. **波段分组**：`TokenizationConfig` 允许实验不同的波段分组策略（全波段一个 token vs 逐波段 token vs 光谱分组），无需修改模型架构
6. **Band Dropout**：训练时随机丢弃波段通道，迫使模型学习跨光谱的鲁棒表示

---

## 6. 数据管线

### 6.1 数据集

**文件**: `olmoearth_pretrain/data/dataset.py`

`OlmoEarthDataset` 是一个 PyTorch `Dataset`:
- 读取预处理的 **HDF5 文件** (每样本 1 个文件)
- 支持**远程存储** (GCS 等) 通过 `UPath`，可选本地缓存
- 处理**缺失模态**和**缺失时间步** (用 MISSING_VALUE 填充)
- 从 Sentinel-2 B04 (红) 和 B08 (近红外) 波段即时计算 **NDVI**
- 应用**归一化** (预定义 min-max 或计算均值-标准差)
- 支持**子集化** (矩形裁剪或 CutMix)，基于 token 预算
- 过滤没有时空变化训练模态的样本
- 支持 `dataset_percentage` 进行部分数据集使用
- 读取节流 (`samples_per_sec`) 用于远程存储

### 6.2 数据归一化

**文件**: `olmoearth_pretrain/data/normalize.py`

两种策略:
- **PREDEFINED**: 从 `predefined.json` 进行 min-max 归一化
- **COMPUTED**: 从 `computed.json` 进行均值-标准差归一化，通过 `(x - (mean - 2*std)) / (4*std)` 映射到 [0,1]

### 6.3 DataLoader

**文件**: `olmoearth_pretrain/data/dataloader.py`

`OlmoEarthDataLoader` 扩展 `olmo_core` 的 `DataLoaderBase`:
- **动态 patch 大小**: 每批次从 `[min_patch_size, max_patch_size]` 随机采样
- **动态空间分辨率**: 每批次采样 `sampled_hw_p` (高度/宽度 patch 数)
- **Token 预算**: 约束每个实例跨所有模态和时间步的总 token 数
- **DataLoader 工作进程中的掩码**: 卸载到 CPU 工作进程以提高 GPU 利用率
- 支持**单掩码视图**或**双掩码视图** (用于对比训练)
- DDP 感知，基于 rank 的数据切片
- 持久工作进程，spawn/forkserver 多进程上下文

### 6.4 整理

**文件**: `olmoearth_pretrain/data/collate.py`

两种整理策略:
- `collate_single_masked_batched`: 堆叠样本，应用变换 + 掩码
- `collate_double_masked_batched`: 堆叠样本，应用两个独立掩码策略 (用于对比 Latent MIM)

### 6.5 数据增强

**文件**: `olmoearth_pretrain/data/transform.py`

- `FlipAndRotateSpace`: 应用 D4 对称群的 8 种变换之一 (恒等、3 种旋转、4 种翻转) 到空间模态
- `Mixup`: Mixup 增强，Beta 分布采样，在旋转的批次视图间插值

---

## 7. 损失函数在训练中的使用详解

### 7.1 损失函数体系总览

所有损失函数继承自抽象基类 `Loss`（`loss.py:52`），通过 `LOSS_REGISTRY` 注册表管理，由 `LossConfig` 配置并构建。

```
Loss (ABC)
├── 判别损失（对比学习核心）
│   ├── AllDiscriminationLoss         # 全批次判别
│   ├── ModalityAllDiscriminationLoss # 逐模态全批次判别
│   ├── PatchDiscriminationLoss       # 逐样本判别（LatentMIM 默认）
│   ├── ModalityPatchDiscriminationLoss          # 逐模态逐样本判别
│   ├── ModalityPatchDiscriminationMaskedNegatives # 带掩码负样本的判别
│   ├── ModalityPatchDiscriminationLossVec       # 全向量化判别（无 for 循环）
│   └── AdjustedPatchDiscriminationLoss          # 高斯加权判别
├── 重建损失
│   ├── L1Loss         # L1/MAE
│   ├── L2Loss         # L2/MSE
│   ├── MAELoss        # 掩码自编码重建
│   └── CrossEntropyLoss
├── 对比损失
│   └── InfoNCELoss    # 实例级对比
└── 正则化
    └── KoLeoLoss      # 微分熵正则化（防特征坍缩）
```

### 7.2 判别损失的核心计算模式

所有判别损失共享相同的计算范式：

```python
# 1. 展平 token 并筛选 DECODER 掩码位置
pred = all_preds[all_masks == MaskValue.DECODER.value]   # 解码器预测
target = all_targets[all_masks == MaskValue.DECODER.value] # 目标编码器输出

# 2. 可选：批次统计标准化
if pred2unit:
    pred = (pred - pred.mean()) / (pred.std() + 1e-4)

# 3. L2 归一化
pred = F.normalize(pred, p=2, dim=-1)
target = F.normalize(target, p=2, dim=-1)

# 4. 计算相似度矩阵
scores = torch.einsum("npd,nqd->npq", pred, target) / tau

# 5. 以对角线为正样本的交叉熵
labels = torch.arange(nt)  # 对角线标签
loss = F.cross_entropy(scores, labels) * (tau * 2)
```

**关键语义**：每个解码 token 的预测应该与对应位置的目标编码器输出最相似（正样本），与其余位置不相似（负样本）。`tau` 控制对比的集中度，`tau * 2` 是缩放约定。

### 7.3 各判别损失的区别

| 损失 | 注册键 | 负样本范围 | 特点 |
|------|--------|-----------|------|
| `AllDiscriminationLoss` | `all_discrimination` | 整个批次所有样本 | Galileo 论文原始损失，内存开销大 |
| `PatchDiscriminationLoss` | `patch_discrimination` | 同一样本内 | LatentMIM 默认，内存高效 |
| `ModalityPatchDiscriminationLoss` | `modality_patch_discrimination` | 同一样本内，逐模态独立 | 支持 `modality_weights` |
| `ModalityPatchDiscriminationMaskedNegatives` | `modality_patch_discrimination_masked_negatives` | 同上，但屏蔽同目标负样本 | 对地图类模态有用 |
| `ModalityPatchDiscriminationLossVec` | `modality_patch_discrimination_vec` | 同一样本内，逐模态 | 全向量化，无 for 循环 |
| `AdjustedPatchDiscriminationLoss` | `adjusted_patch_discrimination` | 同一样本内 | 高斯加权，关注困难负样本 |

**`ModalityPatchDiscriminationMaskedNegatives` 的特殊处理**：对分类地图模态（如 worldcover、cdl），许多空间位置的目标嵌入完全相同。该损失在计算相似度矩阵后，将目标间余弦相似度 > `same_target_threshold`（默认 0.999）的非对角线位置设为 `-inf`，避免这些"同目标"负样本干扰训练。

**`AdjustedPatchDiscriminationLoss` 的高斯加权**：对负样本的相似度分数施加高斯权重 `N(mu, sigma)`，使接近正样本相似度的负样本获得更高权重，引导模型关注"困难"负样本。

### 7.4 LatentMIM 训练中的损失使用

**文件**: `train/train_module/latent_mim.py`

#### 损失组合

```python
# 初始化时构建各损失
self.base_loss = loss_config.build()          # 基础判别损失（默认 PatchDiscrimination）
self.mae_loss = mae_loss_config.build()       # 可选 MAE 重建损失
self.regularizer = regularizer_config.build() # 可选正则化（如 KoLeo）
```

#### 前向传播中的损失计算 (`model_forward`, line 275)

```python
# 1. 在线编码器 + 解码器前向传播
latent, decoded, _, reconstructed, extra_metrics = self.model(batch, patch_size)

# 2. 目标编码器前向传播（no_grad，EMA 更新）
with torch.no_grad():
    output_dict = self.model.target_encoder.forward(batch.unmask(), patch_size, ...)
    target_output, _, _ = unpack_encoder_output(output_dict)

# 3. 基础判别损失：decoded vs target_output
loss = self.loss_fn(decoded, target_output)  # self.base_loss.compute(decoded, target_output)

# 4. 可选 MAE 重建损失：reconstructed vs batch
if self.mae_loss is not None:
    loss += self.mae_loss.compute(reconstructed, batch)
```

#### 训练循环中的完整损失 (`train_batch`, line 193)

```python
for microbatch in microbatches:
    # 前向传播 + 基础损失 + 可选 MAE 损失
    loss, latent, decoded, target_output = self.model_forward(batch, patch_size, ...)

    # 可选正则化（如 KoLeo，作用于在线编码器输出 latent）
    reg_term = self.compute_regularization(latent)
    if reg_term is not None:
        loss = loss + reg_term

    # 微批次缩放
    loss = loss / num_microbatches
    loss.backward()
```

**总损失公式**：

```
total_loss = base_loss(decoded, target_output) + [mae_loss(reconstructed, batch)] + [regularizer(latent)]
```

### 7.5 ContrastiveLatentMIM 训练中的损失使用

**文件**: `train/train_module/contrastive_latentmim.py`

这是官方训练使用的训练模块，支持双视角数据。

#### 损失组合

```python
self.base_loss = loss_config.build()              # 基础判别损失
self.mae_loss = mae_loss_config.build()           # 可选 MAE 重建损失
self.regularizer = regularizer_config.build()     # 可选正则化
self.contrastive_loss = contrastive_config.build() # 可选对比损失（如 InfoNCE）
```

#### 前向传播中的损失计算 (`model_forward`, line 321)

```python
# 1. 在线编码器 + 解码器前向传播（返回池化表示用于对比学习）
latent, decoded, latent_projected_and_pooled, reconstructed, extra_metrics = self.model(batch, patch_size)

# 2. 目标编码器前向传播
with torch.no_grad():
    target_output = self.model.target_encoder.forward(batch.unmask(), ...)

# 3. 基础判别损失
loss = self.loss_fn(decoded, target_output)

# 4. 可选 MAE 重建损失
if self.mae_loss is not None and reconstructed is not None:
    loss += self.mae_loss.compute(reconstructed, batch)

return loss, latent, decoded, target_output, latent_projected_and_pooled
```

#### 训练循环中的完整损失 (`train_batch`, line 210)

```python
# 两个视角各自独立前向传播
loss_a, latent_a, decoded_a, target_output_a, pooled_a = self.model_forward(batch_a, ...)
loss_b, latent_b, decoded_b, target_output_b, pooled_b = self.model_forward(batch_b, ...)

# 基础损失：两视角平均
loss = (loss_a + loss_b) / 2

# 可选正则化：两视角平均
reg_term_a = self.compute_regularization(pooled_a)
reg_term_b = self.compute_regularization(pooled_b)
if reg_term_a is not None:
    loss = loss + (reg_term_a + reg_term_b) / 2

# 可选对比损失：连接两视角的池化表示
if self.contrastive_loss is not None:
    contrastive_loss = self.contrastive_loss.compute(pooled_a, pooled_b)
    loss += contrastive_loss

# 微批次缩放
loss = loss / num_microbatches
loss.backward()
```

**总损失公式**：

```
total_loss = (base_loss_a + base_loss_b) / 2
           + [mae_loss_a + mae_loss_b] / 2
           + [regularizer_a + regularizer_b] / 2
           + [contrastive_loss(pooled_a, pooled_b)]
```

### 7.6 InfoNCE 对比损失的具体使用

`InfoNCELoss` 在 `ContrastiveLatentMIM` 中作为 `contrastive_loss` 使用，输入是两个视角的**池化后表示**（`latent_projected_and_pooled`），而非逐 token 表示。

```python
# InfoNCELoss.compute (loss.py:1113)
predictions = F.normalize(predictions, p=2, dim=-1)  # L2 归一化
targets = F.normalize(targets, p=2, dim=-1)
logits = predictions @ targets.transpose(-2, -1)      # 相似度矩阵 [B, B]
labels = torch.arange(len(predictions))                # 对角线为正样本
return weight * F.cross_entropy(logits / tau, labels)  # 交叉熵
```

**语义**：同一样本的两个视角应互相匹配（正样本），不同样本间应不匹配（负样本）。这是实例级对比学习，与 patch 级判别损失互补。

### 7.7 KoLeo 正则化的具体使用

`KoLeoLoss` 在训练模块中作为 `regularizer` 使用，作用于**在线编码器输出**（`latent`），而非解码器输出。

```python
# compute_regularization (train_module.py:611)
def compute_regularization(self, latent: TokensAndMasks) -> torch.Tensor | None:
    regularizer = getattr(self, "regularizer", None)
    if regularizer is None:
        return None
    return regularizer.compute(latent, None)  # targets=None
```

```python
# KoLeoLoss.compute (loss.py:1195)
# mode="instance": 先池化到实例级，再计算最近邻
# mode="patch":   直接在 patch 级计算最近邻
online_encodings = F.normalize(online_encodings, p=2, dim=-1)
idx_of_nn = pairwise_nearest_neighbours(online_encodings)
distances_to_nn = pdist(online_encodings, online_encodings[idx_of_nn])
return weight * -torch.log(distances_to_nn + eps).mean()
```

**语义**：惩罚批次中任意嵌入到其最近邻的距离过小，鼓励特征在超球面上均匀分布，防止特征坍缩（所有嵌入趋同）。

### 7.8 MAE 重建损失的具体使用

`MAELoss` 作为可选的 `mae_loss` 使用，输入是**重建器输出**（`reconstructed`）和**原始数据**（`batch`）。

```python
# MAELoss.compute (loss.py:1031)
# 1. 展平时空数据：[B, H, W, T, C] → [B, H*W*T*C]
data, masks = self._flatten_spatiotemporal_data(predictions)
labels, label_masks = self._flatten_spatiotemporal_data(valid_targets)

# 2. 选择计算位置
if self.only_decode:
    decode = label_masks == MaskValue.DECODER.value  # 仅解码位置
else:
    decode = label_masks != MaskValue.MISSING.value  # 所有非缺失位置

# 3. 像素级损失（如 MSE）
return weight * loss(data * decode, labels * decode) / count_nonzero(decode)
```

### 7.9 官方训练配置中的损失使用

**文件**: `scripts/official/script.py`

```python
# 基础判别损失
loss_config = LossConfig(loss_config={
    "type": "modality_patch_discrimination",
    "tau": 0.1,
})

# 对比损失
contrastive_config = LossConfig(loss_config={
    "type": "InfoNCE",
    "tau": 0.1,
    "weight": 0.1,  # 对比损失权重
})

# 正则化
regularizer_config = LossConfig(loss_config={
    "type": "KoLeo",
    "weight": 0.1,
})
```

**实际总损失**：

```
total_loss = ModalityPatchDisc(decoded, target, tau=0.1)
           + 0.1 * InfoNCE(pooled_a, pooled_b, tau=0.1)
           + 0.1 * KoLeo(latent)
```

### 7.10 损失计算中的关键数据流

```
MaskedOlmoEarthSample
    │
    ├─[在线编码器]──→ latent (TokensAndMasks) ──→ [正则化] KoLeo(latent)
    │       │
    │       └─[解码器]──→ decoded (TokensAndMasks) ─┐
    │                                               │
    ├─[目标编码器, no_grad]──→ target_output ────────┤
    │                                               │
    │                   [基础判别损失] base_loss(decoded, target_output)
    │
    ├─[重建器]──→ reconstructed ──→ [MAE损失] mae_loss(reconstructed, batch)
    │
    └─[池化]──→ pooled ──→ [对比损失] InfoNCE(pooled_a, pooled_b)
```

**掩码值在损失中的作用**：只有 `mask == MaskValue.DECODER` 的 token 参与损失计算。这些是掩码策略标记为"需要预测"的位置，解码器尝试预测它们的目标编码器表示。

---

## 8. 掩码策略实现详解

**文件**: `olmoearth_pretrain/train/masking.py` (约 2300 行)

### 8.1 掩码值类型 (MaskValue)

| 值 | 整数 | 含义 |
|---|---|---|
| `ONLINE_ENCODER` | 0 | 送入在线编码器（梯度流通） |
| `TARGET_ENCODER_ONLY` | 1 | 仅送入目标编码器（生成训练目标） |
| `DECODER` | 2 | 送入解码器（需从编码表示重建） |
| `MISSING` | 3 | 缺失数据（不参与任何计算） |

掩码张量的最后一维等于**波段组（bandset）数量**而非原始波段数。例如 Sentinel-2 有 10 个波段但 3 个 bandset，掩码最后一维为 3。

### 8.2 基类 MaskingStrategy (line 63)

**核心辅助方法**：

#### `_create_random_mask` (line ~220)

创建随机掩码的核心工具方法：

1. 计算掩码形状：将数据形状最后一维替换为 `num_bandsets`
2. 若模态为空间模态，将 H/W 除以 `patch_size` 得到 patch 粒度的掩码
3. 计算总 token 数，按 `encode_ratio` / `decode_ratio` / 剩余分配
4. 拼接三段掩码值：`[ONLINE_ENCODER * n_enc, DECODER * n_dec, TARGET_ENCODER_ONLY * n_tgt]`
5. 随机打乱：空间/多时相模态每个样本独立打乱，静态模态批次共享
6. 重塑为掩码形状，若为空间模态则上采样回像素级：`"b h w -> b (h hp) (w wp)"`
7. 返回与输入数据同空间维度但最后一维为 `num_bandsets` 的掩码

#### `fill_mask_with_missing_values` (line 184)

在已有掩码基础上，将数据中实际缺失（`== MISSING_VALUE`）的位置标记为 `MaskValue.MISSING`。对每个波段组，若任一波段缺失则整个波段组标记为缺失。

### 8.3 RandomMaskingStrategy (line 1309, 注册键 `"random"`)

每个 token 独立随机分配到 ONLINE_ENCODER / DECODER / TARGET_ENCODER_ONLY，无结构约束。

```python
# apply_mask 流程：
for modality in batch:
    mask = self._create_random_mask(modality, shape, patch_size, device)
    mask = self.fill_mask_with_missing_values(instance, mask, modality)
```

**掩码形状**：
- 空间模态 `(B, H, W, C)` → 掩码 `(B, H, W, num_bandsets)`
- 多时相模态 `(B, H, W, T, C)` → 掩码 `(B, H, W, T, num_bandsets)`
- 静态模态 `(B, C)` → 掩码 `(B, num_bandsets)`

### 8.4 SpaceMaskingStrategy (line 507, 注册键 `"space"`)

整个空间 patch 共享相同掩码值，**所有空间模态共享同一空间掩码模式**。

```python
# apply_mask 流程：
# 1. 从第一个空间模态创建 patch 级空间掩码 (B, h_p, w_p)
patch_mask = self._create_patch_spatial_mask(modality, shape, patch_size, device)

# 2. 对每个空间模态，将 patch 掩码上采样到像素级
for modality in batch:
    if spatial:
        mask = self._resize_spatial_mask_for_modality(patch_mask, modality, patch_size)
        # 若有时间维度，重复到时间维度和波段组维度
        mask = repeat(mask, "... -> ... t b_s", t=T, b_s=num_bandsets)
    else:
        mask = self._create_random_mask(...)  # 非空间模态回退到随机
```

**关键设计**：跨模态空间一致性——同一空间位置的 Sentinel-2、Sentinel-1、Landsat 等共享掩码角色。

### 8.5 TimeMaskingStrategy (line 365, 注册键 `"time"`)

整个时间步共享相同掩码值，**所有多时相模态共享同一时间掩码**。

```python
# apply_mask 流程：
# 1. 获取有数据的时间步列表
timesteps_with_data = batch.get_timesteps_with_at_least_one_modality()

# 2. 创建时间掩码 (B, T)：每个时间步整体分配角色
temporal_mask = self._create_temporal_mask(shape, timesteps_with_data, device)
# 要求 present_t >= 3

# 3. 重复到空间和波段组维度
mask = repeat(temporal_mask, "b t -> b h w t b_s", h=H, w=W, b_s=num_bandsets)
```

**关键约束**：要求至少 3 个有效时间步，否则回退到随机掩码。

### 8.6 SpaceTimeMaskingStrategy (line 687, 注册键 `"space_time"`)

50/50 随机选择空间掩码或时间掩码：

```python
if random() < 0.5 or valid_time < 3:
    return self.space_strategy.apply_mask(batch, patch_size)
else:
    return self.time_strategy.apply_mask(batch, patch_size)
```

### 8.7 ModalityCrossMaskingStrategy (line 758, **官方训练默认**)

这是最核心的掩码策略，在基础策略之上添加**跨模态**层，强制模型用一种模态的信息重建另一种模态。

#### 初始化参数

| 参数 | 默认 | 含义 |
|------|------|------|
| `strategy` | 必需 | 基础掩码策略（如 SpaceMaskingStrategy） |
| `encode_ratio` | 0.5 | 编码 token 比例 |
| `decode_ratio` | 0.5 | 解码 token 比例 |
| `allow_encoding_decoding_same_bandset` | False | 同一波段组能否同时被编码和解码 |
| `min_encoded_bandsets` | None | 最小编码波段组数 |
| `max_encoded_bandsets` | None | 最大编码波段组数 |
| `only_decode_modalities` | [] | 仅解码、永不编码的模态列表 |

#### apply_mask 四步流程

```python
# 步骤 1：应用基础策略（如空间掩码）
masked_sample = self.strategy.apply_mask(batch, patch_size)

# 步骤 2：确定每个样本中哪些 (模态, 波段组) 有数据
present_modalities_bandsets = self.get_sample_present_modalities_bandsets(masked_sample)
# 返回: [[(sentinel2_l2a, 0), (sentinel2_l2a, 1), (srtm, 0), ...], ...]  # 每样本一个列表

# 步骤 3：随机选择哪些波段组用于编码、哪些用于解码
encoded_decoded_bandsets = self.select_encoded_decoded_bandsets(present_modalities_bandsets)
# 返回: [(encoded_set, decoded_set), ...]  # 每样本一个元组

# 步骤 4：应用跨模态规则修改掩码
masked_sample = self.apply_bandset_mask_rules(masked_sample, encoded_decoded_bandsets, ...)
```

#### select_encoded_decoded_bandsets 选择逻辑

```
若只有 1 个波段组有数据 → 编码它，不解码
若恰好 2 个波段组 → 编码第 1 个，解码第 2 个
若 3+ 个波段组：
  1. 从可编码集合中排除 only_decode_modalities
  2. 随机选择 [min_encoded, max_encoded] 个波段组用于编码
  3. 若 allow_encoding_decoding_same_bandset=False：
     解码集合 = 全部有数据集合 - 编码集合
  4. 若 True：解码集合独立随机选择
```

#### apply_bandset_mask_rules 规则执行

对每个模态的每个波段组：

```python
if bandset not in encoded_set:
    # 未被选为编码 → 将 ONLINE_ENCODER 降级为 TARGET_ENCODER_ONLY
    mask[mask == ONLINE_ENCODER] = TARGET_ENCODER_ONLY

if bandset not in decoded_set:
    # 未被选为解码 → 将 DECODER 降级为 TARGET_ENCODER_ONLY
    mask[mask == DECODER] = TARGET_ENCODER_ONLY
```

**效果**：在线编码器只看到被选为编码的波段组，解码器只尝试重建被选为解码的波段组，迫使模型学习跨模态预测。

#### 注册的子类

| 注册键 | 基础策略 | 特点 |
|--------|---------|------|
| `modality_cross_space` | SpaceMaskingStrategy | 非空间模态掩码由跨模态选择完全控制 |
| `modality_cross_time` | SpaceMaskingStrategy | 同上（名称有误导性） |
| `modality_cross_random` | RandomMaskingStrategy | 基础为随机掩码 |
| `modality_cross_space_time` | 50/50 选择 space 或 time | 组合策略 |

### 8.8 其他掩码策略

| 注册键 | 类 | 描述 |
|--------|-----|------|
| `random_space` | RandomSpaceMaskingStrategy | 50/50 随机选择 random 或 space |
| `random_increasing` | RandomIncreasingMaskingStrategy | 课程学习：线性增加掩码比例 |
| `random_range` | RandomRangeMaskingStrategy | 每批次随机采样 encode_ratio 范围 |
| `random_with_decode` | RandomWithDecodeMaskingStrategy | 先全部标记 DECODER，再分离编码/解码波段组 |
| `selectable_modality` | SelectableModalityMaskingStrategy | 随机后额外完全掩码部分模态 |

### 8.9 官方训练的掩码配置

```python
# scripts/official/script.py
masking_config = MaskingConfig(strategy_config={
    "type": "modality_cross_random",
    "encode_ratio": 0.5,
    "decode_ratio": 0.5,
})
```

使用 `modality_cross_random`：先对每个 token 随机分配角色，再随机选择约一半波段组用于编码、另一半用于解码，强制跨模态预测。

---

## 9. 原始数据到模型可用数据完整处理流程

### 9.1 总览

```
HDF5 文件 (磁盘/GCS)
  → OlmoEarthDataset.__getitem__()     # 读取、归一化、子集化
    → OlmoEarthDataLoader              # 批次化、动态 patch_size/hw_p
      → collate 函数                    # 堆叠、增强、掩码
        → MaskedOlmoEarthSample         # 模型输入
          → Encoder.forward()           # Patch Embed + 编码 + Transformer
```

### 9.2 HDF5 文件结构

每个 HDF5 文件对应一个空间窗口（由 `create_windows` 创建），包含：

- 各模态数据：`sentinel2_l2a`, `sentinel1`, `landsat`, `srtm`, `worldcover` 等
- 每个模态的形状：空间模态 `(H, W, T, C)`，非空间模态 `(T, C)`
- 时间戳信息：`timestamps` 字段，格式为 `(year, month, day)`
- 缺失值用 `MISSING_VALUE = -99999` 填充

### 9.3 OlmoEarthDataset.__getitem__() — 数据读取与预处理

**文件**: `data/dataset.py`，`__getitem__` 在 line 977

每个 worker 进程独立调用 `__getitem__`，处理单个样本：

```
步骤 1: 索引映射 (line 996)
  - 若有 sample_indices，映射到过滤后的索引
  - sample_indices 已移除无有效训练模态的样本
  - 可选 dataset_percentage 子采样

步骤 2: 读取 HDF5 文件 (line 1000)
  - h5_file_path = self._get_h5_file_path(index)
  - read_h5_file() (line 880): 打开 HDF5，读取 training_modalities 对应的键 + timestamps
  - 读取 missing_timesteps_mask 组
  - 支持原子重命名缓存（并发安全）
  - 通过 UPath 支持本地/远程(GCS)存储
  - 读取节流 (samples_per_sec)

步骤 3: 裁剪时间戳到有效范围 (line 1005)
  - _crop_timestamps_and_masks() (line 944)
  - 找到所有模态中最早和最晚的有效时间步
  - 裁剪 timestamps 和 missing_timesteps_masks 到该范围

步骤 4: 填充时间戳到 max_sequence_length (line 1010)
  - _pad_timestamps(): 使用边缘填充（重复最后一个时间步）
  - 返回填充前的原始长度

步骤 5: 填充缺失模态和时间步 (line 1012)
  - fill_sample_with_missing_values() (line 695)
  - 完全缺失的模态: 用 MISSING_VALUE (-99999) 填充预期形状 (line 816)
  - 部分缺失的时间步: 创建全 MISSING_VALUE 数组，将有效数据复制到 missing_timestep_mask 指示的位置
  - 返回 missing_modalities 列表

步骤 6: 子集化 (基于 token 预算) (line 1017)
  两种策略:
  (a) subset_sample_default (line 168) — 矩形裁剪:
    - _get_max_t_within_token_budget() (line 64): 遍历所有模态，计算静态 token 数和每时间步 token 数
    - max_t = min(floor(remaining / time_multiply), sample.time)
    - 随机选择有效时间起始位置
    - 随机选择空间起始 (start_h, start_w)
    - 对每个模态按 is_spacetime_varying/is_space_only_varying/is_time_only_varying/is_static 裁剪
    - 使用 image_tile_size_factor 处理不同分辨率
  (b) subset_sample_cutmix (line 252) — 非连续 patch 采样:
    - 同样的时间子集化
    - 随机选择 sampled_hw_p 个非连续 patch 索引（无替换）
    - 用 np.meshgrid + 高级索引选择非连续空间区域

步骤 7: 即时计算 NDVI (line 1041)
  - 仅当 sentinel2_l2a 存在且非完全缺失时计算
  - _compute_ndvi() (line 660):
    red = s2_data[..., B04_index]    # B04 = 红波段
    nir = s2_data[..., B08_index]    # B08 = 近红外波段
    ndvi = (nir - red) / (nir + red) # 安全除法（|denom| < 1e-10 时设为 0）
    ndvi[missing] = MISSING_VALUE
  - 从**未归一化**的 S2 数据计算
  - 输出形状 [H, W, T, 1]

步骤 8: 归一化 (line 1050)
  - 跳过 timestamps 和完全缺失的模态
  - 记录缺失值位置 → 归一化 → 恢复 MISSING_VALUE
  - normalize_image() (line 642): 优先尝试 COMPUTED，失败回退 PREDEFINED
  - PREDEFINED (normalize.py line 98):
    x_norm = (x - min) / (max - min)
    参数来自 data/norm_configs/predefined.json，逐波段 min/max
  - COMPUTED (normalize.py line 125):
    min = mean - std_multiplier * std  (默认 std_multiplier=2)
    max = mean + std_multiplier * std
    x_norm = (x - min) / (max - min)
    覆盖 ~95% 数据范围
    参数来自 data/norm_configs/computed.json，逐波段 mean/std
```

### 9.4 OlmoEarthDataLoader — 批次化与动态参数

**文件**: `data/dataloader.py`

```
步骤 1: 动态采样训练参数 (每批次, line 684)
  - patch_size: 从 patch_size_array = arange(min, max+1) 均匀采样
    官方配置: patch_size ∈ [1, 8]
  - sampled_hw_p: 从 hw_p_to_sample_array 采样
    过滤: 仅保留 hw_p <= IMAGE_TILE_SIZE / patch_size 的候选
    例如 patch_size=8 时, max hw_p = 256/8 = 32
  - 同一批次所有样本共享相同 patch_size 和 sampled_hw_p

步骤 2: Worker 端迭代 (line 738)
  - 每个 worker 独立迭代:
    global_indices → _get_local_instance_indices() → _get_batch_item_params_iterator()
  - 对每个样本调用 _get_dataset_item(idx, patch_size, sampled_hw_p)
  - 批次化: iter_batched(instance_iterator, rank_batch_size, drop_last)

步骤 3: Collator 选择 (line 871)
  - num_masked_views == 1 → collate_single_masked_batched
  - num_masked_views == 2 → collate_double_masked_batched
  - 掩码在 collator 中执行（批次级向量化）

步骤 4: DDP 感知
  - 基于 rank 的数据切片: _get_local_instance_indices()
  - 每个 rank 处理不同的数据子集

步骤 5: 持久工作进程
  - spawn/forkserver 多进程上下文
  - 避免每 epoch 重新初始化
```

### 9.5 Collate 函数 — 批次整理

**文件**: `data/collate.py`

#### collate_olmoearth_pretrain (line 24)

将多个 `OlmoEarthSample` 堆叠为批次张量：

```python
# 对每个模态字段
stacked = torch.stack([sample.modality for sample in batch], dim=0)  # (B, ...)
# None 字段保持 None
```

#### collate_single_masked_batched (line 66)

单掩码视图（LatentMIM 训练使用）：

```python
# 1. 堆叠为批次
patch_size, stacked_sample = collate_olmoearth_pretrain(batch)

# 2. 批次级数据增强
if transform is not None:
    stacked_sample = transform.apply(stacked_sample)

# 3. 批次级掩码
masked_sample = masking_strategy.apply_mask(stacked_sample, patch_size)

return (patch_size, masked_sample)
```

#### collate_double_masked_batched (line 97)

双掩码视图（ContrastiveLatentMIM 训练使用）：

```python
# 1. 堆叠为批次
patch_size, stacked_sample = collate_olmoearth_pretrain(batch)

# 2. 批次级数据增强（共享增强后的数据）
if transform is not None:
    stacked_sample = transform.apply(stacked_sample)

# 3. 两个独立掩码策略
masked_sample_a = masking_strategy.apply_mask(stacked_sample, patch_size)
masked_sample_b = strategy_b.apply_mask(stacked_sample, patch_size)

return (patch_size, masked_sample_a, masked_sample_b)
```

**关键设计**：增强和掩码在批次级别执行（向量化），而非样本级别，提高效率。两个掩码视图共享增强后的数据但使用独立掩码。

### 9.6 数据增强

**文件**: `data/transform.py`

#### FlipAndRotateSpace (line 64)

应用 D4 对称群的 8 种变换之一到所有空间模态：

```python
self.transformations = [
    no_transform,        # 恒等
    rotate_90,           # 90° 旋转
    rotate_180,          # 180° 旋转
    rotate_270,          # 270° 旋转
    hflip,               # 水平翻转
    vflip,               # 垂直翻转
    hflip_rotate_90,     # 水平翻转后旋转 90°
    vflip_rotate_90,     # 垂直翻转后旋转 90°
]
```

- 每个批次随机选择一种变换 (`random.choice`)
- **所有空间模态同步应用**同一变换（保持空间对齐）
- 使用 torchvision `F.rotate`, `F.hflip`, `F.vflip`
- 数据重排为 `(B, T, C, H, W)` 应用变换后恢复
- timestamps 不变

#### Mixup (line 161)

Beta 分布采样的 Mixup 增强：

```python
other_microbatch = batch.rotate()  # [B1, B2, B3] → [B2, B3, B1]
lam = Beta(alpha, alpha).sample()  # 采样混合系数
new = (1 - lam) * batch + lam * other_batch
```

- 配对方式：将批次旋转一个位置（样本 i+1 成为样本 i 的配对）
- 凸组合：`new = (1-lam) * sample + lam * other_sample`
- timestamps 来自权重较大的样本

### 9.7 数据张量形状总结

| 阶段 | 空间模态形状 | 非空间模态形状 |
|------|-------------|---------------|
| HDF5 原始 | `(H, W, T, C)` | `(T, C)` |
| Dataset 输出 | `(H, W, T, C)` | `(T, C)` |
| 批次化后 | `(B, H, W, T, C)` | `(B, T, C)` |
| 掩码后数据 | `(B, H, W, T, C)` | `(B, T, C)` |
| 掩码张量 | `(B, H, W, T, num_bandsets)` | `(B, num_bandsets)` |
| Patch Embed 后 | `(B, H/P, W/P, T, b_s, D)` | `(B, T, b_s, D)` |

---

## 10. 模型训练全流程详解

### 10.1 ContrastiveLatentMIM 训练全流程（官方默认）

以下以 `ContrastiveLatentMIMTrainModule.train_batch()` 为例，这是官方 Phase 2 训练使用的训练模块。

### 10.2 训练步完整流程

```
┌─────────────────────────────────────────────────────────────────┐
│ 步骤 0: EMA 更新目标编码器                                        │
│   cur_ema = start_ema + (step / max_steps) * (end_ema - start_ema) │
│   for each param:                                                │
│     target_param = cur_ema * target_param + (1 - cur_ema) * online_param │
│   (当 start_ema == end_ema == 1.0 时跳过)                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 步骤 1: 接收批次数据                                              │
│   batch = (patch_size, MaskedOlmoEarthSample_a, MaskedOlmoEarthSample_b) │
│   两个视角来自同一批数据但使用独立掩码策略                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 步骤 2: 拆分为微批次                                              │
│   microbatches_a = split_masked_batch(batch_a, rank_microbatch_size) │
│   microbatches_b = split_masked_batch(batch_b, rank_microbatch_size) │
│   每个微批次独立前向/反向，梯度累积                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 步骤 3: 微批次循环 (for each microbatch_idx)                     │
│                                                                  │
│   3a. 进入微批次上下文                                            │
│     FSDP: 仅在最后微批次触发梯度同步                               │
│     DDP: 非最后微批次用 no_sync() 延迟 all-reduce                 │
│                                                                  │
│   3b. 移动数据到设备                                              │
│     masked_batch_a = microbatch_a.to_device(device)               │
│     masked_batch_b = microbatch_b.to_device(device)               │
│                                                                  │
│   3c. 视角 A 前向传播 (model_forward)                             │
│   3d. 视角 B 前向传播 (model_forward)                             │
│   3e. 损失计算与反向传播                                          │
│   3f. 梯度累积                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 步骤 4: 优化器步进 (在所有微批次完成后)                            │
│   - 梯度裁剪: clip_grad_norm_(max_grad_norm)                     │
│   - optimizer.step()                                             │
│   - scheduler.step()                                             │
└─────────────────────────────────────────────────────────────────┘
```

### 10.3 model_forward 详细流程

每个视角独立执行以下流程：

```
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 1: 在线编码器前向传播                                        │
│   output_dict = self.model.encoder(masked_batch, patch_size)     │
│                                                                  │
│   Encoder.forward() (flexi_vit.py:1923) 内部流程:                │
│                                                                  │
│   1. Patch Embedding (MultiModalPatchEmbeddings.forward):        │
│      - 对每个模态的每个波段组:                                     │
│        a. torch.index_select 选择波段通道 (via register_buffer)  │
│        b. 可选 Band Dropout (训练时随机置零部分波段)               │
│        c. FlexiPatchEmbed: 展平 patch → nn.Linear 投影 → LayerNorm│
│      - 输出: tokens [B, H/P, W/P, T, b_s, D] per modality       │
│      - 输出: masks  [B, H/P, W/P, T, b_s] per modality          │
│                                                                  │
│   2. Composite Encodings (CompositeEncodings.forward):           │
│      - Channel Embedding (可学习, dims [0:D/4])                   │
│      - Temporal Position (冻结 sincos, dims [D/4:D/2])           │
│      - Month Embedding (冻结 sincos, dims [D/2:3D/4])            │
│      - Spatial Embedding (冻结 sincos+GSD, dims [3D/4:D])        │
│      - tokens = patch_embed + channel + temporal + month + spatial│
│                                                                  │
│   3. Collapse & Combine (collapse_and_combine_hwtc):             │
│      - 将所有模态 token 拼接为单一序列                             │
│      - 形状: [B, total_tokens, D]                                │
│                                                                  │
│   4. 移除掩码 token (MAE 效率优化, line 1507):                    │
│      - 排序: DECODER(2) 前, TARGET(1) 中, ENCODER(0) 后          │
│      - 仅保留 mask == ONLINE_ENCODER 的 token                     │
│      - 截断到 max(visible_count)                                 │
│      - 效率提升: 如 75% 掩码率下约 4x 加速                        │
│                                                                  │
│   5. 可选 Pack tokens (Flash Attention varlen 模式)               │
│      - 计算累积序列长度 cu_seqlens                                │
│      - 打包变长序列为连续内存                                      │
│                                                                  │
│   6. 可选添加 Register Tokens                                     │
│                                                                  │
│   7. Transformer Blocks (L 层 pre-norm):                         │
│      x = x + DropPath(LayerScale(Attention(LayerNorm(x))))       │
│      x = x + DropPath(LayerScale(MLP(LayerNorm(x))))             │
│      支持 Flash Attention / PyTorch SDPA / QK-Norm               │
│                                                                  │
│   8. 可选移除 Register Tokens                                     │
│   9. 可选 Unpack tokens (Flash Attention)                        │
│   10. LayerNorm                                                  │
│   11. 恢复掩码位置 (用 0 填充)                                    │
│   12. 拆分回 per-modality: split_and_expand_per_modality()        │
│                                                                  │
│   13. ProjectAndAggregate (对比学习用):                            │
│      - 投影到对比学习空间                                         │
│      - 池化: 对非掩码 token 取均值                                │
│      - 输出: latent_projected_and_pooled [B, D_proj]              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 2: 可选重建器前向传播                                        │
│   if reconstructor:                                              │
│     reconstructed = reconstructor(latent, timestamps, patch_size) │
│     - 从编码器输出重建原始像素值                                   │
│     - 使用 FlexiPatchReconstruction (ConvTranspose2d)             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 3: 解码器前向传播                                            │
│   decoded = decoder(latent, timestamps, patch_size, **kwargs)    │
│                                                                  │
│   Predictor.forward() (flexi_vit.py:2384) 内部流程:               │
│                                                                  │
│   1. 输入归一化 + 编码器→解码器维度映射:                           │
│      x = encoder_to_decoder_embed(LayerNorm(x))                  │
│                                                                  │
│   2. 在 DECODER 位置插入可学习的 mask_token:                      │
│      x = where(mask == DECODER, mask_token, x)                   │
│      (ONLINE_ENCODER 位置保留编码器输出)                          │
│                                                                  │
│   3. 添加解码器位置编码                                           │
│                                                                  │
│   4. 拼接所有模态 → split_x_y() 分离:                            │
│      - tokens_to_decode (Q): DECODER 位置的 token                │
│      - unmasked_tokens (K,V): ONLINE_ENCODER 位置的 token         │
│      - 排序: DECODER(2) 前, TARGET(1) 中, ENCODER(0) 后          │
│                                                                  │
│   5. 交叉注意力 Transformer 块:                                  │
│      tokens_to_decode = Block(                                   │
│        x=tokens_to_decode,         # Q: 需要预测的 token          │
│        y=unmasked_tokens,          # K,V: 编码器可见 token        │
│      )                                                            │
│                                                                  │
│   6. 合并解码和未掩码 token → 拆分回 per-modality                 │
│                                                                  │
│   7. 逐波段组输出投影:                                            │
│      output = to_output_embed(LayerNorm(per_bandset_data))        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 4: 目标编码器前向传播 (no_grad)                              │
│   with torch.no_grad():                                          │
│     target_output = model.target_encoder(batch.unmask(), ...)    │
│                                                                  │
│   - unmask(): 将所有非 MISSING token 设为 ONLINE_ENCODER           │
│   - 目标编码器看到完整未掩码数据                                   │
│   - 参数不计算梯度，仅通过 EMA 更新                                │
│   - 禁用 band dropout，始终看到完整光谱                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 5: 损失计算                                                 │
│                                                                  │
│   # 基础判别损失                                                  │
│   loss_a = base_loss.compute(decoded_a, target_output_a)         │
│   loss_b = base_loss.compute(decoded_b, target_output_b)         │
│   loss = (loss_a + loss_b) / 2                                   │
│                                                                  │
│   # 可选正则化 (如 KoLeo)                                         │
│   reg_a = regularizer.compute(pooled_a, None)                    │
│   reg_b = regularizer.compute(pooled_b, None)                    │
│   loss += (reg_a + reg_b) / 2                                    │
│                                                                  │
│   # 可选对比损失 (如 InfoNCE)                                     │
│   con_loss = contrastive_loss.compute(pooled_a, pooled_b)        │
│   loss += con_loss                                               │
│                                                                  │
│   # 可选 MAE 重建损失                                             │
│   loss += mae_loss.compute(reconstructed, batch)                 │
│                                                                  │
│   # 微批次缩放                                                    │
│   loss = loss / num_microbatches                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 6: 反向传播                                                  │
│   loss.backward()                                                │
│   - NaN/Inf 检测: 若检测到异常值，跳过该微批次                     │
│   - 梯度在微批次间累积                                            │
└─────────────────────────────────────────────────────────────────┘
```

### 10.4 LatentMIM.forward() 模型前向传播

```python
# nn/latent_mim.py:80
def forward(self, x: MaskedOlmoEarthSample, patch_size: int):
    # 步骤 1: 在线编码器
    output_dict = self.encoder(x, patch_size=patch_size)
    latent, latent_projected_and_pooled, decoder_kwargs = unpack_encoder_output(output_dict)

    # 步骤 2: 可选重建器
    reconstructed = None
    if self.reconstructor:
        reconstructed = self.reconstructor(latent, x.timestamps, patch_size)

    # 步骤 3: 解码器
    decoded = self.decoder(latent, timestamps=x.timestamps, patch_size=patch_size, **decoder_kwargs)

    return (latent, decoded, latent_projected_and_pooled, reconstructed, extra_metrics)
```

### 10.5 目标编码器的 unmask() 操作

目标编码器接收 `batch.unmask()` (`datatypes.py:725`)：

```python
def unmask(self):
    updates = {}
    for name in _MASKED_SAMPLE_MASK_FIELDS:
        val = getattr(self, name)
        if val is not None:
            # MISSING 保持不变，其余全部设为 ONLINE_ENCODER (0)
            updates[name] = val * (val == MaskValue.MISSING.value)
    return self._replace(**updates)
```

具体效果：
- `DECODER (2)` → `ONLINE_ENCODER (0)`：原本需要预测的 token 变为可见
- `TARGET_ENCODER_ONLY (1)` → `ONLINE_ENCODER (0)`：原本仅目标编码器可见的 token 也变为可见
- `MISSING (3)` → `MISSING (3)`：缺失数据保持缺失

这确保目标编码器看到**尽可能完整的数据**，为解码器提供高质量训练目标。目标编码器还禁用了 band dropout，始终看到完整光谱信息。

### 10.6 官方训练完整配置

```python
# 模型
model = LatentMIM(encoder=Encoder(base), decoder=Predictor(shallow), reconstructor=None)

# 训练模块
train_module = ContrastiveLatentMIMTrainModule(
    model=model,
    optim_config=AdamW(lr=1e-4, weight_decay=0.02),
    scheduler=CosWithWarmup(warmup_steps=8000),
    loss_config=ModalityPatchDisc(tau=0.1),
    contrastive_config=InfoNCE(tau=0.1, weight=0.1),
    regularizer_config=KoLeo(weight=0.1),
    masking_config=ModalityCrossRandom(encode_ratio=0.5, decode_ratio=0.5),
    ema_decay=(1.0, 1.0),  # 实际无 EMA
    max_grad_norm=1.0,
    rank_microbatch_size=32,
)

# 数据
dataloader = OlmoEarthDataLoader(
    dataset=OlmoEarthDataset(hdf5_files),
    global_batch_size=512,
    token_budget=2250,
    min_patch_size=1, max_patch_size=8,
    min_hw_p=1, max_hw_p=12,
    transform=FlipAndRotateSpace(),
    collate_fn=collate_double_masked_batched,
)

# 训练
trainer = Trainer(
    train_module=train_module,
    dataloader=dataloader,
    epochs=300,
    dp_config=FSDP(param_dtype=bfloat16, reduce_dtype=float32),
)
```

---

## 11. 训练系统

### 11.1 掩码策略

**文件**: `olmoearth_pretrain/train/masking.py` (2103 行)

掩码值 (`MaskValue` 枚举):

| MaskValue | 含义 |
|-----------|------|
| `ONLINE_ENCODER (0)` | Token 输入到在线/主编码器 |
| `TARGET_ENCODER_ONLY (1)` | Token 仅输入到目标编码器 (EMA 编码器)，不输入在线编码器 |
| `DECODER (2)` | Token 仅输入到解码器 (需要预测) |
| `MISSING (3)` | Token 为缺失数据 |

注册的掩码策略 (`MASKING_REGISTRY`):

| 注册键 | 类 | 策略 |
|--------|-----|------|
| `random` | `RandomMaskingStrategy` | 随机分配每个 token 到编码器/解码器/目标 |
| `time` | `TimeMaskingStrategy` | 整个时间步共享相同掩码 (时间掩码) |
| `space` | `SpaceMaskingStrategy` | 整个空间 patch 共享相同掩码 |
| `space_time` | `SpaceTimeMaskingStrategy` | 50/50 随机选择空间或时间掩码 |
| `random_space` | `RandomSpaceMaskingStrategy` | 50/50 随机选择随机或空间掩码 |
| `modality_cross_random` | `ModalityCrossMaskingStrategy` | 为编码和解码选择独立 BandSet (跨模态预测) |

每种策略由 `encode_ratio` 和 `decode_ratio` 参数化 (剩余部分分配给 `TARGET_ENCODER_ONLY`)。`fill_mask_with_missing_values()` 方法将任何带有 `MISSING_VALUE` 哨兵值的 token 标记为 `MaskValue.MISSING`。

### 11.2 损失函数

**文件**: `olmoearth_pretrain/train/loss.py` (1122 行)

所有损失继承自抽象 `Loss` 基类，通过 `class_registry` 注册在 `LOSS_REGISTRY` 中:

| 注册键 | 类名 | 描述 |
|--------|------|------|
| `all_discrimination` | `AllDiscriminationLoss` | 全批次对比损失，计算完整相似度矩阵 |
| `modality_all_discrimination` | `ModalityAllDiscriminationLoss` | 按模态独立计算的全批次对比损失 |
| `patch_discrimination` | `PatchDiscriminationLoss` | 每样本对比损失 (内存高效)，LatentMIM 默认损失 |
| `modality_patch_discrimination` | `ModalityPatchDiscriminationLoss` | 每样本、每模态 patch 判别，支持 `modality_weights` |
| `modality_patch_discrimination_masked_negatives` | `ModalityPatchDiscriminationMaskedNegatives` | 同上但掩码掉相同目标的负样本 (用于分类地图模态) |
| `modality_patch_discrimination_vec` | `ModalityPatchDiscriminationLossVec` | 完全并行化 (无 for 循环) 版本 |
| `adjusted_patch_discrimination` | `AdjustedPatchDiscriminationLoss` | 高斯加权负样本分数 (NeurIPS 2023 I-JEPA 论文) |
| `l1` | `L1Loss` | L1/MAE 损失 |
| `l2` | `L2Loss` | L2/MSE 损失 |
| `mae` | `MAELoss` | 掩码自编码重建损失 |
| `cross_entropy` | `CrossEntropyLoss` | 交叉熵损失 |
| `InfoNCE` | `InfoNCELoss` | InfoNCE 对比损失 (实例级对比学习) |
| `KoLeo` | `KoLeoLoss` | Kozachenko-Leonenko 微分熵正则化器 (来自 DINOv2) |

**对比损失的共享模式**:
- 用 `F.normalize(p=2)` 归一化预测和目标
- 通过 `torch.einsum("npd,nqd->npq", pred, target) / tau` 计算相似度
- 以对角线标签为正样本的交叉熵
- 乘以 `tau * 2` (缩放约定)
- 可选 `pred2unit` 使用批次统计量标准化预测

### 11.3 训练模块

#### 基类: OlmoEarthTrainModule

**文件**: `olmoearth_pretrain/train/train_module/train_module.py`

- **模型设置**: 日志参数计数，应用 `torch.compile()`，应用 FSDP 或 DDP
- **优化器**: 从 `OptimConfig` 构建 (通常 AdamW)
- **梯度处理**: `max_grad_norm` 裁剪 (支持 DTensor)，`SkipStepOptimizer` 集成
- **学习率调度**: 可选 `Scheduler` (通常 `CosWithWarmup`)，每步调整
- **微批次**: `_train_microbatch_context()` 管理 FSDP/DDP 梯度同步
- **AMP**: `autocast_precision` 混合精度
- **EMA 目标编码器**: `target = ema * target + (1-ema) * online`，EMA 值从 `start_ema` 调度到 `end_ema`
- **检查点**: 使用 `torch.distributed.checkpoint` 进行 FSDP 兼容检查点

#### LatentMIM 训练模块

**文件**: `olmoearth_pretrain/train/train_module/latent_mim.py`

- 单视图训练，`(patch_size, MaskedOlmoEarthSample)` 批次格式
- `model_forward()`: 在线编码器 → 解码器; 目标编码器 (no_grad) 处理未掩码输入
- 损失: `base_loss(decoded, target_output)` + 可选 `mae_loss` + 正则化
- 每批次开始时 EMA 更新目标编码器

#### ContrastiveLatentMIM 训练模块

**文件**: `olmoearth_pretrain/train/train_module/contrastive_latentmim.py`

- **双视图训练**，`(patch_size, MaskedOlmoEarthSample_a, MaskedOlmoEarthSample_b)` 批次格式
- 同一在线编码器独立处理两个视图
- 目标编码器为两个视图生成目标
- 损失: `(loss_a + loss_b) / 2` + 可选 `contrastive_loss(pooled_a, pooled_b)` + 正则化
- `contrastive_config` 启用两个视图池化嵌入之间的 InfoNCE
- `reinit_targets` 选项随机重新初始化目标编码器

#### Galileo 训练模块

**文件**: `olmoearth_pretrain/train/train_module/galileo.py`

- **双视图、双解码器**训练
- 编码器 → decoder_a 处理视图 A，编码器 → decoder_b 处理视图 B
- 损失: `(loss_a + loss_b) / 2` + 可选 `contrastive_loss` + 可选 `mae_loss` + 正则化
- 每视图独立的掩码策略和损失配置

### 11.4 通用训练循环模式

所有训练模块的通用训练循环:
1. 可选更新目标编码器 EMA
2. 设置模型为训练模式
3. 将批次拆分为微批次
4. 对每个微批次:
   - 进入微批次上下文 (处理 FSDP/DDP 同步)
   - 移动到设备
   - 前向传播 (模型 + 目标编码器)
   - 计算损失 + 正则化 + 对比
   - 按 `1/num_microbatches` 缩放
   - 反向传播
5. 记录指标 (带 NaN/Inf 安全检查)
6. 删除中间张量以释放内存

---

## 12. 分布式训练

### 12.1 数据并行

两种模式通过 `DataParallelConfig`:

- **FSDP** (`DataParallelType.fsdp`): 全分片数据并行，`MixedPrecisionPolicy`，应用于所有模型组件
- **DDP** (`DataParallelType.ddp`): `DistributedDataParallel`，`find_unused_parameters=True` (处理缺失模态)

两者使用 `build_world_mesh()` 和 `get_dp_mesh()` 构建设备网格。

### 12.2 微批次处理

`_train_microbatch_context()` 上下文管理器:
- FSDP: 在最终微批次调用 `set_is_last_backward(True)`
- DDP: 非最终微批次用 `model.no_sync()` 延迟梯度 all-reduce

### 12.3 梯度范数裁剪

`_clip_grad_norm()` 处理 DTensor (分布式张量) 梯度，调用 `full_tensor()` 在 rank 间归约后裁剪。

### 12.4 检查点

使用 `torch.distributed.checkpoint` 进行 FSDP 兼容分布式检查点:
- `StateDictOptions(flatten_optimizer_state_dict=True, cpu_offload=True)` 用于保存
- `StateDictOptions(flatten_optimizer_state_dict=True, strict=True)` 用于加载
- 向后兼容检查: 验证模型键与检查点键匹配

---

## 13. 评估框架

### 13.1 评估包装器

**文件**: `olmoearth_pretrain/evals/eval_wrapper.py`

`EvalWrapper` 提供统一接口评估任何模型，包括基线模型。

### 13.2 评估任务

默认评估任务 (来自官方脚本):
- **m-eurosat**: 多标签分类 (EuroSAT)，每 4000 步线性探测
- **m-so2sat**: 多标签分类 (So2Sat)，每 20000 步线性探测
- **mados**: 语义分割 (MADOS)，每 4000 步线性探测
- **pastis**: 作物分割 (PASTIS)，每 20000 步线性探测

### 13.3 评估方法

- **Linear Probe**: 冻结嵌入上的简单线性分类器
- **Attention-pooling Probe**: `AttnPoolLinearProbe` 用于分割任务
- **KNN**: K 近邻评估，可配置 k (默认 20)，bootstrap 不确定性估计
- **Finetune**: 完整微调评估

### 13.4 评估数据集

已实现: BreizhCrops, Floods, GeoBench, MADOS, PASTIS, pretrain_subset, rslearn 数据集

### 13.5 基线模型

已实现基线包装器: AnySat, Clay, CROMA, DINOv3, Galileo, Panopticon, Presto, PrithviV2, Satlas, TerraMind, Tessera

---

## 14. 回调系统

### 14.1 速度监控

**文件**: `olmoearth_pretrain/train/callbacks/speed_monitor.py`

跟踪每步吞吐量指标:
- 每秒 token 数 (编码、解码、目标编码器)
- 每秒批次数
- 数据加载时间占步时间百分比
- 模型持续时间占步时间百分比

### 14.2 W&B 日志

**文件**: `olmoearth_pretrain/train/callbacks/wandb.py`

- 训练开始时上传训练数据集的地理分布图
- 可选上传每模态波段分布直方图
- 支持 `restart_on_same_run` 恢复 W&B 运行

### 14.3 下游评估器

**文件**: `olmoearth_pretrain/train/callbacks/evaluator_callback.py`

训练中定期运行评估:
- **评估模式**: KNN, Linear Probe, Finetune, Embedding Diagnostics
- **任务**: 通过 `DownstreamTaskConfig` 配置
- **嵌入管线**: 获取嵌入 → 可选量化 → 可选 PCA 降维 → 评估函数
- **Bootstrap**: 可选不确定性估计

---

## 15. 官方训练配置

**文件**: `scripts/official/script.py` 和 `scripts/official/base.py`

默认 Phase 2 训练配置:

| 配置项 | 值 |
|--------|-----|
| 模型 | LatentMIM + FlexiHelios (base_shallow_decoder) |
| 训练模块 | ContrastiveLatentMIMTrainModuleConfig |
| 优化器 | AdamW (lr=1e-4, weight_decay=0.02) |
| 调度器 | CosWithWarmup (warmup_steps=8000) |
| 损失 | modality_patch_discrimination (tau=0.1) + InfoNCE (weight=0.1) |
| 掩码 | modality_cross_random (encode_ratio=0.5, decode_ratio=0.5) |
| 训练模态 | S2 L2A, S1, Landsat, WorldCover, SRTM, OSM, CHM, CDL, WorldCereal |
| 全局批次大小 | 512 |
| 每 rank 微批次大小 | 32 |
| Token 预算 | 2250 每实例 |
| Patch 大小 | 1-8 (随机) |
| 空间维度 | 1-12 patches |
| FSDP | bfloat16 参数, float32 归约 |
| EMA 衰减 | (1.0, 1.0) -- 实际无 EMA |
| 梯度裁剪 | max_grad_norm=1.0 |
| 训练时长 | 300 epochs |
| 检查点 | 每 5000 步永久，每 250 步临时 |
| W&B 项目 | "2025_10_02_phase2" |

---

## 16. 配置系统

**文件**: `olmoearth_pretrain/config.py`

双模式配置系统:
- **有 olmo-core**: 使用 `olmo_core.config.Config`，支持 OmegaConf/YAML/CLI 覆盖
- **无 olmo-core** (仅推理): 使用 `_StandaloneConfig`，JSON 序列化/反序列化，`_CLASS_` 字段多态

所有模型配置 (`EncoderConfig`, `PredictorConfig`, `LatentMIMConfig` 等) 都是继承自 `Config` 的数据类，实现 `build()` 和 `validate()` 方法。

---

## 17. 数据集创建管线

**目录**: `olmoearth_pretrain/dataset_creation/`

将各种来源的原始数据转换为 HDF5 文件:
- `rslearn_to_olmoearth/`: 每个模态的转换器 (sentinel2, sentinel1, landsat, srtm, openstreetmap, cdl, worldcereal, era5, naip, gse, worldpop, wri_canopy_height_map, eurocrops 等)
- `create_windows/`: 窗口/Patch 创建策略 (随机、从经纬度列表)
- `openstreetmap/`: OSM 采样 (Go + Python)
- `sentinel2_l1c/`: Sentinel-2 L1C 特定处理

---

## 18. 核心数据类型

**文件**: `olmoearth_pretrain/datatypes.py`

三个核心 NamedTuple:
- **`OlmoEarthSample`**: 原始数据样本，每模态可选字段
- **`MaskedOlmoEarthSample`**: 数据 + 每模态掩码 (MaskValue 枚举)
- **`TokensAndMasks`**: 嵌入 tokens + 掩码 (编码器/解码器输出)

---

## 19. 推理与模型加载

**文件**: `olmoearth_pretrain/model_loader.py`

支持仅推理模式 (无需 `olmo-core`):
- `load_model_from_id(model_id)`: 从 HuggingFace Hub 加载模型
- `load_model_from_path(path)`: 从本地路径加载模型
- 加载流程: `config.json` → `Config.from_dict()` → `model_config.build()` → `model.load_state_dict()`
- 通过 `patch_legacy_encoder_config()` 兼容旧检查点

---

## 20. 依赖关系

### 核心依赖 (仅推理)

| 包 | 用途 |
|----|------|
| `torch>=2.7,<2.8` | 深度学习框架 |
| `numpy>=1.26.4` | 数值计算 |
| `einops>=0.7.0` | 张量重排 |
| `huggingface_hub` | HuggingFace 模型下载 |
| `universal-pathlib>=0.2.5` | 云存储路径支持 |

### 训练依赖

| 包 | 用途 |
|----|------|
| `ai2-olmo-core==2.3.0` | AI2 训练框架 (FSDP, 检查点, 优化器, 调度器) |
| `albumentations>=1.4.10` | 图像增强 |
| `hdf5plugin>=6.0.0` | HDF5 压缩编解码器 |
| `pandas>=2.2` | 数据操作 |
| `rasterio>=1.4.3` | 地理空间栅格 I/O |
| `wandb>=0.19.0` | 实验跟踪 |
| `gcsfs>=2025.9.0` | Google Cloud Storage 文件系统 |

### 评估依赖

| 包 | 用途 |
|----|------|
| `geobench` | GeoBench 基准 |
| `claymodel` | Clay 模型 (基线对比) |
| `pytorch-lightning==2.5.5` | 评估微调训练框架 |
| `rslearn>=0.0.26` | AI2 遥感学习框架 |
| `terratorch==1.1` | TerraTorch 基线模型 |
| `scikit-learn>=1.7.2` | ML 工具 |
| `torchmetrics==1.7.1` | 指标计算 |

---

## 21. 关键架构决策

1. **双模式配置系统**: 有/无 `olmo-core` 均可工作，实现最小依赖的仅推理安装
2. **多模态设计**: `OlmoEarthSample` 支持 15+ 模态，每模态独立缺失值处理
3. **FlexiViT 架构**: 分辨率感知位置编码，灵活 patch 大小 (1-8)，每 BandSet 可配置分词
4. **Latent MIM 训练**: 学生编码器 + 解码器 + EMA 目标编码器，patch 判别损失 + 可选对比和 MAE 损失
5. **丰富的掩码策略**: 随机、跨模态随机、band dropout、每模态掩码，注册模式可扩展
6. **分布式训练**: 基于 `olmo-core` 的 FSDP 和 DDP 支持，微批次梯度累积，混合精度，分布式检查点
7. **向后兼容**: `helios` 包名通过元路径查找器导入 shim 保留，废弃类/函数别名
8. **全面评估框架**: 支持与 10+ 基线 EO 模型在多个基准数据集上对比

---

## 22. 训练数据流总览

```
HDF5 文件 → OlmoEarthDataset (读取, 归一化, 子集化)
  → OlmoEarthDataLoader (批次化, 动态 patch_size/hw_p, 掩码)
    → collate_single/double_masked_batched (变换, 应用掩码)
      → LatentMIMTrainModule.train_batch()
        → LatentMIM.forward()
          → Encoder(掩码输入) → latent tokens
          → Decoder(latent, mask_tokens) → decoded tokens
          → TargetEncoder(未掩码输入) → target tokens [no_grad, EMA]
        → Loss(decoded, target) 在 DECODER 掩码位置
        → backward + optimizer step
        → EMA 更新目标编码器
```
