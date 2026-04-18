# Large 模型训练全流程推演

以 `scripts/official/large.py` 为入口，完整推演从启动到每一步训练迭代的全流程。

---

## 1. 入口与配置构建

### 1.1 入口 (`large.py:58`)

```python
if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
    )
```

与 nano 使用相同的 `script.py` 中的 builder 函数，仅 `build_model_config` 不同。

### 1.2 通用组件 (`script.py:53` → `build_common_components`)

```python
CommonComponents(
    run_name="...",
    save_folder="...",
    training_modalities=[     # 9 个训练模态（与 nano 完全相同）
        "sentinel2_l2a",      # Sentinel-2 L2A (10波段, 3 bandset)
        "sentinel1",          # Sentinel-1 (2波段, 1 bandset)
        "landsat",            # Landsat (11波段, 2 bandset)
        "worldcover",         # WorldCover (1波段, 1 bandset) [仅解码]
        "srtm",               # SRTM 高程 (1波段, 1 bandset) [仅解码]
        "openstreetmap_raster", # OSM (30波段, 1 bandset) [仅解码]
        "wri_canopy_height_map", # 冠层高度 (1波段, 1 bandset) [仅解码]
        "cdl",                # 作物分类 (1波段, 1 bandset) [仅解码]
        "worldcereal",        # 作物类型 (8波段, 1 bandset) [仅解码]
    ],
    tokenization_config=None,
)
```

**模态分类**：
- **可编码+可解码** (3个): sentinel2_l2a, sentinel1, landsat
- **仅解码** (6个): worldcover, srtm, openstreetmap_raster, wri_canopy_height_map, cdl, worldcereal

---

## 2. 模型配置

### 2.1 模型尺寸 (`internal/utils.py:160`)

```python
MODEL_SIZE_ARGS["large_shallow_decoder"] = {
    "encoder_embedding_size": 1024,   # 编码器嵌入维度 D=1024
    "encoder_depth": 24,              # 编码器 Transformer 层数
    "encoder_num_heads": 16,          # 编码器注意力头数
    "decoder_embedding_size": 1024,   # 解码器嵌入维度
    "decoder_depth": 4,               # 解码器 Transformer 层数 (浅层!)
    "decoder_num_heads": 16,          # 解码器注意力头数
    "mlp_ratio": 4.0,               # MLP 隐藏层倍率
}
```

**与 nano 的关键区别**：
| 参数 | Nano | Large |
|------|------|-------|
| 编码器嵌入维度 | 128 | **1024** (8x) |
| 编码器层数 | 4 | **24** (6x) |
| 编码器头数 | 8 | **16** (2x) |
| 每头维度 | 16 | **64** (4x) |
| 编码器 MLP 维度 | 512 | **4096** (8x) |
| 解码器层数 | 4 | **4** (相同) |
| 解码器嵌入维度 | 128 | **1024** (8x) |

### 2.2 编码器配置 (`large.py:31`)

```python
EncoderConfig(
    embedding_size=1024,             # D = 1024
    num_heads=16,                    # 16 个注意力头，每头 1024/16=64 维
    depth=24,                        # 24 层 Transformer
    mlp_ratio=4.0,                   # MLP 隐藏层 = 1024 * 4 = 4096
    supported_modality_names=[上述 9 个模态],
    max_patch_size=8,
    drop_path=0.1,                   # DropPath 概率
    max_sequence_length=12,
    use_linear_patch_embed=False,    # 使用 Conv2d patch embedding
)
```

**编码器内部组件**：
- `MultiModalPatchEmbeddings`: 9 个模态，共 11 个 bandset，每个 bandset 一个 FlexiPatchEmbed
  - 每个 FlexiPatchEmbed: Conv2d 投影，输入通道数 = bandset 波段数，输出 = 1024
  - 例如 sentinel2_l2a bandset 0: Conv2d(4, 1024, kernel_size=patch_size)
  - 例如 sentinel2_l2a bandset 1: Conv2d(6, 1024, kernel_size=patch_size)
  - 例如 openstreetmap_raster: Conv2d(30, 1024, kernel_size=patch_size)
- `CompositeEncodings`: 4 种编码各占 D/4 = 256 维
  - Channel Embedding: 可学习, 11 个 bandset × 256 维 = 2816 个可学习参数
  - Temporal Position: 冻结 sincos, 256 维, 最大 12 个时间步
  - Month Embedding: 冻结 sincos, 256 维, 12 个月
  - Spatial Embedding: 冻结 sincos+GSD, 256 维
- 24 个 Transformer Block: pre-norm, 支持 Flash Attention
  - 每个 Block: MultiHeadAttention(16 heads, 64 dim/head) + MLP(1024→4096→1024)
  - DropPath: 线性从 0 增长到 0.1 (depth=24 时, 每层递增约 0.0042)
  - LayerScale: 可学习逐通道缩放 (DINOv2 风格)
- `ProjectAndAggregate`: 投影+池化用于对比学习

### 2.3 解码器配置 (`large.py:42`)

```python
PredictorConfig(
    encoder_embedding_size=1024,
    decoder_embedding_size=1024,
    depth=4,                         # 浅层解码器 (仅 4 层)
    mlp_ratio=4.0,                   # MLP 隐藏层 = 1024 * 4 = 4096
    num_heads=16,                    # 16 个注意力头，每头 64 维
    supported_modality_names=[上述 9 个模态],
    max_sequence_length=12,
)
```

**解码器内部组件**：
- `encoder_to_decoder_embed`: Linear(1024→1024) 维度映射
- `mask_token`: 可学习掩码 token (1024 维)
- `CompositeEncodings`: 同编码器结构，各占 256 维
- 4 个 Transformer Block: 交叉注意力 (Q=decoder tokens, K/V=encoder visible tokens)
  - 每个 Block: MultiHeadAttention(16 heads, 64 dim/head) + MLP(1024→4096→1024)
- `to_output_embed`: 逐 bandset 输出投影 Linear(1024→1024) + LayerNorm

**浅层解码器设计**：编码器 24 层负责学习高质量表示，解码器仅 4 层做轻量预测。这大幅减少计算量（解码器仅占总计算约 14%），同时保持预训练质量。

### 2.4 模型组装 (`large.py:51`)

```python
LatentMIMConfig(
    encoder_config=encoder_config,
    decoder_config=decoder_config,
    # reconstructor=None  (未配置，无 MAE 重建)
)
```

**LatentMIM 模型**：
- `encoder`: 在线编码器 (24 层 Transformer, D=1024)
- `decoder`: 解码器 (4 层交叉注意力, D=1024)
- `target_encoder`: encoder 的深拷贝, requires_grad=False, 禁用 band dropout
- `reconstructor`: None (未启用)

---

## 3. 训练模块配置

### 3.1 优化器 (`script.py:108`)

```python
AdamWConfig(lr=1e-4, weight_decay=0.02, fused=False)
```

### 3.2 学习率调度 (`script.py:125`)

```python
CosWithWarmup(warmup_steps=8000)
```

余弦退火 + 8000 步线性预热。

### 3.3 掩码策略 (`script.py:78`)

```python
MaskingConfig(
    strategy_config={
        "type": "modality_cross_random",
        "encode_ratio": 0.5,
        "decode_ratio": 0.5,
        "allow_encoding_decoding_same_bandset": True,
        "only_decode_modalities": [
            "worldcover", "srtm", "openstreetmap_raster",
            "wri_canopy_height_map", "cdl", "worldcereal",
        ],
    },
)
```

**掩码策略详解**：
1. **基础层**: RandomMaskingStrategy — 每个 token 独立随机分配 ONLINE_ENCODER(50%) / DECODER(50%) / TARGET_ENCODER_ONLY(0%)
2. **跨模态层**: ModalityCrossMaskingStrategy
   - 随机选择部分 bandset 用于编码，部分用于解码
   - `allow_encoding_decoding_same_bandset=True`: 同一 bandset 可同时被编码和解码
   - `only_decode_modalities`: 6 个地图类模态**永不编码**，仅作为解码目标
   - 效果: 迫使模型从 S2/S1/Landsat 的编码信息预测 worldcover/srtm/OSM 等的表示

### 3.4 损失函数 (`script.py:111`)

```python
# 基础判别损失
loss_config = LossConfig(loss_config={
    "type": "modality_patch_discrimination_new",  # → ModalityPatchDiscriminationLoss
    "tau": 0.1,
})

# 对比损失
contrastive_config = LossConfig(loss_config={
    "type": "InfoNCE",
    "weight": 0.1,
})
```

**损失组成**：
- `ModalityPatchDiscriminationLoss(tau=0.1)`: 逐模态逐样本判别
  - 对每个模态独立计算: L2 归一化 → 余弦相似度矩阵 → 交叉熵 (对角线正样本)
  - 缩放: loss * (tau * 2) = loss * 0.2
- `InfoNCE(weight=0.1)`: 两视角池化表示间的实例级对比损失
  - L2 归一化 → 相似度矩阵 [B, B] → 交叉熵 (对角线正样本)
- 无 MAE 重建损失 (mae_loss_config=None)
- 无正则化 (regularizer_config=None)

### 3.5 其他训练参数

```python
ContrastiveLatentMIMTrainModuleConfig(
    rank_microbatch_size=32,         # 每 rank 微批次大小
    max_grad_norm=1.0,               # 梯度裁剪范数
    ema_decay=(1.0, 1.0),           # EMA 衰减: start=1.0, end=1.0 → 不更新目标编码器
    token_exit_cfg={modality: 0 for modality in training_modalities},
    dp_config=DataParallelConfig(
        name=DataParallelType.fsdp,
        param_dtype=DType.bfloat16,  # 参数 bfloat16
        reduce_dtype=DType.float32,  # 归约 float32
    ),
)
```

**EMA 衰减 (1.0, 1.0) 的含义**：目标编码器参数不通过 EMA 更新，保持初始随机权重。目标编码器提供随机投影目标，类似 BYOL 的初始阶段。

---

## 4. 数据配置

### 4.1 数据集 (`script.py:164`)

```python
OlmoEarthDatasetConfig(
    h5py_dir="/weka/.../h5py_data_w_missing_timesteps_zstd_3_128_x_4/...",
    training_modalities=[上述 9 个模态],
)
```

HDF5 文件路径指向预处理的遥感数据集，包含 1,138,828 个样本。

### 4.2 DataLoader (`script.py:147`)

```python
OlmoEarthDataLoaderConfig(
    num_workers=12,
    global_batch_size=512,
    token_budget=2250,
    prefetch_factor=2,
    sampled_hw_p_list=[1,2,...,12],
    min_patch_size=1,
    max_patch_size=8,
    seed=3622,
    num_masked_views=2,              # 双掩码视图
    masking_config=get_masking_config(common),
    tokenization_config=None,
)
```

---

## 5. 训练器配置

### 5.1 训练时长与检查点 (`script.py:172`)

```python
max_duration = Duration.epochs(300)
permanent_save_interval = 5000
ephemeral_save_interval = 250
```

### 5.2 回调

| 回调 | 作用 |
|------|------|
| `OlmoEarthWandBCallback` | W&B 日志 (project="2025_10_02_phase2") |
| `OlmoEarthSpeedMonitorCallback` | 吞吐量监控 (tokens/sec, batches/sec) |
| `GPUMemoryMonitorCallback` | GPU 内存监控 |
| `DownstreamEvaluatorCallback` | 训练中下游评估 |
| `CheckpointerCallback` | 检查点保存 |
| `GarbageCollectorCallback` | 定期 GC |
| `BeakerCallback` | Beaker 集群集成 |

### 5.3 下游评估任务

| 任务 | 评估间隔 | 池化方式 | 备注 |
|------|---------|---------|------|
| m-eurosat | 每 4000 步 | MEAN | 多标签分类 |
| m-so2sat | 每 20000 步 | MEAN | 多标签分类 |
| mados | 每 4000 步 | MEAN | 语义分割, 50 epoch |
| pastis | 每 20000 步 | MEAN | 作物分割, 仅 S2, 50 epoch |

---

## 6. 训练迭代全流程

### 6.1 训练步概览

```
每个训练步:
  1. DataLoader 生成批次 (patch_size, MaskedOlmoEarthSample_a, MaskedOlmoEarthSample_b)
  2. EMA 更新目标编码器 (ema=1.0 → 跳过)
  3. 拆分为微批次 (每微批次 32 样本)
  4. 对每个微批次:
     a. 视角 A 前向传播 → loss_a
     b. 视角 B 前向传播 → loss_b
     c. loss = (loss_a + loss_b) / 2
     d. 可选对比损失: loss += 0.1 * InfoNCE(pooled_a, pooled_b)
     e. loss /= num_microbatches
     f. loss.backward()
  5. 梯度裁剪 (max_norm=1.0)
  6. optimizer.step()
  7. scheduler.step()
```

### 6.2 数据加载详细流程

```
┌─ DataLoader Worker (12 个并行) ─────────────────────────────────┐
│                                                                  │
│ 1. 采样批次参数 (每批次一次):                                     │
│    patch_size ~ Uniform({1, 2, 3, 4, 5, 6, 7, 8})              │
│    sampled_hw_p ~ Uniform({1..12} ∩ {x | x ≤ 256/patch_size})  │
│                                                                  │
│ 2. 对批次中每个样本:                                             │
│    a. OlmoEarthDataset.__getitem__(idx):                         │
│       - 读取 HDF5 文件                                           │
│       - 裁剪时间戳到有效范围                                      │
│       - 填充到 max_sequence_length=12                             │
│       - 填充缺失模态/时间步 (MISSING_VALUE=-99999)                │
│       - 子集化: 矩形裁剪到 token_budget=2250 内                   │
│       - 计算 NDVI = (B08-B04)/(B08+B04)                         │
│       - 归一化: COMPUTED (mean±2std → [0,1]) 或 PREDEFINED       │
│    b. 返回 (patch_size, OlmoEarthSample)                         │
│                                                                  │
│ 3. Collate (批次级):                                             │
│    a. 堆叠: torch.stack([sample1, sample2, ...], dim=0)         │
│    b. 增强: FlipAndRotateSpace (D8 群随机变换)                   │
│    c. 掩码 A: modality_cross_random.apply_mask(batch, patch_size)│
│    d. 掩码 B: modality_cross_random.apply_mask(batch, patch_size)│
│       (同一增强后数据, 独立掩码)                                  │
│                                                                  │
│ 4. 返回 (patch_size, MaskedOlmoEarthSample_a, MaskedOlmoEarthSample_b)│
└──────────────────────────────────────────────────────────────────┘
```

### 6.3 掩码策略执行详细流程

```
ModalityCrossRandomMaskingStrategy.apply_mask(batch, patch_size):

步骤 1: 基础随机掩码
  对每个模态的每个 token:
    - 50% 概率 → ONLINE_ENCODER (0)
    - 50% 概率 → DECODER (2)
    - 0%  概率 → TARGET_ENCODER_ONLY (1)

步骤 2: 确定存在的 (模态, 波段组) 对
  对每个样本, 列出有非缺失数据的 bandset:
  例如: [(sentinel2_l2a,0), (sentinel2_l2a,1), (sentinel2_l2a,2),
         (sentinel1,0), (landsat,0), (landsat,1),
         (worldcover,0), (srtm,0), (cdl,0)]
  (共 9 个 bandset, 其中 6 个来自 only_decode_modalities)

步骤 3: 选择编码/解码 bandset
  可编码集合 = 全部 - only_decode_modalities
             = {sentinel2_l2a[0,1,2], sentinel1[0], landsat[0,1]}  (6 个)
  随机选择 num_to_encode 个用于编码
  解码集合 = 全部有数据集合 (allow_encoding_decoding_same_bandset=True)

步骤 4: 应用跨模态规则
  对每个 bandset:
    if bandset NOT in encoded_set:
      将所有 ONLINE_ENCODER → TARGET_ENCODER_ONLY  (在线编码器看不到)
    if bandset NOT in decoded_set:
      将所有 DECODER → TARGET_ENCODER_ONLY  (解码器不尝试重建)

  对 only_decode_modalities 中的 bandset:
    强制所有非缺失 token → DECODER  (完全由解码器预测)

步骤 5: 标记缺失数据
  数据 == MISSING_VALUE 的位置 → MaskValue.MISSING
```

### 6.4 模型前向传播详细流程 (每个视角)

```
输入: MaskedOlmoEarthSample (9 个模态, 各带掩码)

┌─ LatentMIM.forward() ──────────────────────────────────────────┐
│                                                                  │
│ ┌─ 在线编码器 Encoder.forward() ──────────────────────────────┐ │
│ │                                                              │ │
│ │ 1. Patch Embedding (MultiModalPatchEmbeddings.forward):      │ │
│ │    对每个模态的每个 bandset:                                   │ │
│ │    ┌─ sentinel2_l2a bandset 0 (B02,B03,B04,B08): ─────────┐ │ │
│ │    │  inp = index_select(data, bandset_0_indices)           │ │ │
│ │    │  可选 Band Dropout (训练时随机置零)                      │ │ │
│ │    │  tokens = FlexiPatchEmbed(inp)  # Conv2d 投影          │ │ │
│ │    │  → [B, H/P, W/P, T, 1, 1024]                          │ │ │
│ │    └────────────────────────────────────────────────────────┘ │ │
│ │    同理对 bandset 1 (B05,B06,B07,B8A,B11,B12) 和             │ │
│ │              bandset 2 (B01,B09)                              │ │
│ │    同理对 sentinel1, landsat, worldcover, srtm, OSM, ...     │ │
│ │                                                              │ │
│ │ 2. Composite Encodings:                                      │ │
│ │    对每个模态的每个 token:                                     │ │
│ │    encoding = zeros_like(token)                               │ │
│ │    encoding[..., 0:256]   += channel_embed[modality][bandset]│ │
│ │    encoding[..., 256:512] += sincos_pos_embed[timestep]      │ │
│ │    encoding[..., 512:768] += month_embed[month]              │ │
│ │    encoding[..., 768:1024]+= sincos_2d_embed[h,w]*gsd_ratio │ │
│ │    token = token + encoding                                  │ │
│ │                                                              │ │
│ │ 3. Collapse: 拼接所有模态 → [B, total_tokens, 1024]          │ │
│ │                                                              │ │
│ │ 4. 移除掩码 token:                                           │ │
│ │    仅保留 mask==ONLINE_ENCODER 的 token                       │ │
│ │    (DECODER/TARGET/MISSING 不进入 Transformer)               │ │
│ │    效率提升: 约 50% 掩码率下 ~2x 加速                         │ │
│ │                                                              │ │
│ │ 5. 24 层 Transformer Block:                                  │ │
│ │    for i, blk in enumerate(self.blocks):                     │ │
│ │      drop_path_rate = 0.1 * i / 23  # 线性递增              │ │
│ │      x = x + DropPath(LayerScale(Attention(LayerNorm(x))))   │ │
│ │      x = x + DropPath(LayerScale(MLP(LayerNorm(x))))         │ │
│ │    Attention: 16 heads, 64 dim/head                          │ │
│ │    MLP: 1024 → 4096 → 1024                                  │ │
│ │                                                              │ │
│ │ 6. LayerNorm                                                 │ │
│ │ 7. 恢复掩码位置 (0 填充)                                      │ │
│ │ 8. 拆分回 per-modality: TokensAndMasks                       │ │
│ │ 9. ProjectAndAggregate → pooled [B, 1024]                    │ │
│ └──────────────────────────────────────────────────────────────┘ │
│                                                                  │
│ 输出: latent (TokensAndMasks), pooled (Tensor [B, 1024])         │
│                                                                  │
│ ┌─ 解码器 Predictor.forward() ────────────────────────────────┐ │
│ │                                                              │ │
│ │ 1. LayerNorm + Linear(1024→1024) 维度映射                    │ │
│ │ 2. 在 DECODER 位置插入 mask_token (可学习, 1024 维)           │ │
│ │ 3. 添加 Composite Encodings (同编码器结构, 各 256 维)         │ │
│ │ 4. 拼接所有模态 → split_x_y():                               │ │
│ │    tokens_to_decode (Q): DECODER 位置的 mask_token           │ │
│ │    unmasked_tokens (K,V): ONLINE_ENCODER 位置的编码器输出     │ │
│ │ 5. 4 层交叉注意力 Transformer:                                │ │
│ │    Q = tokens_to_decode, K/V = unmasked_tokens               │ │
│ │    Attention: 16 heads, 64 dim/head                          │ │
│ │    MLP: 1024 → 4096 → 1024                                  │ │
│ │ 6. 合并 → 拆分回 per-modality                                │ │
│ │ 7. 逐 bandset 输出投影: Linear(1024→1024) + LayerNorm        │ │
│ └──────────────────────────────────────────────────────────────┘ │
│                                                                  │
│ 输出: decoded (TokensAndMasks)                                   │
└──────────────────────────────────────────────────────────────────┘

┌─ 目标编码器 (no_grad) ─────────────────────────────────────────┐
│ target_output = target_encoder(batch.unmask(), patch_size)       │
│                                                                  │
│ unmask(): 将 DECODER→ONLINE_ENCODER, TARGET→ONLINE_ENCODER     │
│          MISSING 保持不变                                         │
│ → 目标编码器看到尽可能完整的数据                                   │
│ → 禁用 band dropout, 始终看到完整光谱                            │
│ → 参数不计算梯度 (ema_decay=1.0, 不更新)                         │
│ → 同样 24 层 Transformer, D=1024                                │
└──────────────────────────────────────────────────────────────────┘
```

### 6.5 损失计算详细流程

```
┌─ 视角 A 损失 ──────────────────────────────────────────────────┐
│                                                                  │
│ base_loss_a = ModalityPatchDiscriminationLoss.compute(           │
│     predictions=decoded_a,                                       │
│     targets=target_output_a                                      │
│ )                                                                │
│                                                                  │
│ 对每个模态独立计算:                                               │
│   1. 展平 token, 筛选 mask==DECODER 的位置                       │
│      decoded_a: 每个解码 token 是 1024 维向量                    │
│      target_output_a: 每个目标 token 是 1024 维向量              │
│   2. L2 归一化: pred = F.normalize(pred, p=2, dim=-1)           │
│   3. 逐样本计算:                                                 │
│      score = einsum("npd,nqd->npq", pred, target) / 0.1        │
│      labels = arange(num_decoder_tokens)                         │
│      loss = cross_entropy(score, labels) * 0.2                   │
│   4. 跨样本平均                                                  │
│   5. 累加所有模态的损失                                          │
└──────────────────────────────────────────────────────────────────┘

同理计算 base_loss_b

┌─ 总损失 ───────────────────────────────────────────────────────┐
│                                                                  │
│ # 基础判别损失: 两视角平均                                        │
│ loss = (base_loss_a + base_loss_b) / 2                           │
│                                                                  │
│ # 对比损失: 两视角池化表示间的 InfoNCE                            │
│ pooled_a = F.normalize(pooled_a, p=2, dim=-1)  # [B, 1024]     │
│ pooled_b = F.normalize(pooled_b, p=2, dim=-1)  # [B, 1024]     │
│ logits = pooled_a @ pooled_b.T / tau                             │
│ labels = arange(B)  # 对角线为正样本                             │
│ contrastive_loss = 0.1 * cross_entropy(logits, labels)           │
│ loss += contrastive_loss                                         │
│                                                                  │
│ # 无正则化 (regularizer_config=None)                              │
│ # 无 MAE 重建 (mae_loss_config=None)                             │
│                                                                  │
│ # 微批次缩放                                                     │
│ loss = loss / num_microbatches                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 6.6 反向传播与参数更新

```
1. loss.backward()
   - 梯度流经: 解码器(4层) → 编码器(24层)
   - 目标编码器: no_grad, 不产生梯度
   - FSDP: bfloat16 参数, float32 归约

2. 梯度裁剪
   clip_grad_norm_(model.parameters(), max_norm=1.0)

3. optimizer.step()
   AdamW(lr=1e-4, weight_decay=0.02)
   - 更新编码器(24层)和解码器(4层)参数
   - 目标编码器不更新 (ema=1.0)

4. scheduler.step()
   CosWithWarmup: 余弦退火 + 8000 步预热
```

---

## 7. 完整训练数据流图

```
┌──────────────────────────────────────────────────────────────────────┐
│                        HDF5 文件 (磁盘/Weka)                         │
│                    1,138,828 个样本, 9 个模态                        │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│               OlmoEarthDataset.__getitem__()                         │
│  读取 → 裁剪时间 → 填充缺失 → 子集化(token_budget=2250)             │
│  → NDVI计算 → 归一化(COMPUTED/PREDEFINED)                           │
│  输出: OlmoEarthSample (numpy arrays)                                │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│               OlmoEarthDataLoader (12 workers)                       │
│  动态采样: patch_size∈[1,8], hw_p∈[1,12]                            │
│  Collate: stack → FlipAndRotateSpace → 双掩码                        │
│  输出: (patch_size, MaskedOlmoEarthSample_a, MaskedOlmoEarthSample_b)│
└──────────┬───────────────────────────────────────┬───────────────────┘
           │                                           │
           ▼                                           ▼
┌──────────────────────┐                  ┌──────────────────────┐
│   视角 A 前向传播     │                  │   视角 B 前向传播     │
│                      │                  │                      │
│ Encoder(masked_a)    │                  │ Encoder(masked_b)    │
│  24层, D=1024       │                  │  24层, D=1024       │
│   → latent_a         │                  │   → latent_b         │
│   → pooled_a         │                  │   → pooled_b         │
│ Decoder(latent_a)    │                  │ Decoder(latent_b)    │
│  4层, D=1024        │                  │  4层, D=1024        │
│   → decoded_a        │                  │   → decoded_b        │
│                      │                  │                      │
│ TargetEncoder(unmask)│                  │ TargetEncoder(unmask)│
│  24层, D=1024       │                  │  24层, D=1024       │
│   → target_a         │                  │   → target_b         │
│                      │                  │                      │
│ loss_a = PatchDisc(  │                  │ loss_b = PatchDisc(  │
│   decoded_a,         │                  │   decoded_b,         │
│   target_a)          │                  │   target_b)          │
└──────────┬───────────┘                  └──────────┬───────────┘
           │                                           │
           └──────────────┬────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        损失计算                                      │
│                                                                      │
│  loss = (loss_a + loss_b) / 2                                       │
│       + 0.1 * InfoNCE(pooled_a, pooled_b)                           │
│                                                                      │
│  loss /= num_microbatches                                            │
│  loss.backward()                                                     │
│  clip_grad_norm_(1.0)                                                │
│  AdamW.step()                                                        │
│  CosWithWarmup.step()                                                │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 8. 关键数值总结

### 模型参数

| 组件 | 嵌入维度 | 层数 | 头数 | 每头维度 | MLP 维度 |
|------|---------|------|------|---------|---------|
| 在线编码器 | 1024 | 24 | 16 | 64 | 4096 |
| 解码器 | 1024 | 4 | 16 | 64 | 4096 |
| 目标编码器 | 1024 | 24 | 16 | 64 | 4096 |

### 编码维度分配 (D=1024)

| 编码类型 | 维度范围 | 大小 | 可学习 |
|----------|---------|------|--------|
| Channel Embedding | [0, 256) | 256 | 是 |
| Temporal Position | [256, 512) | 256 | 否 (sincos) |
| Month Embedding | [512, 768) | 256 | 否 (sincos) |
| Spatial Embedding | [768, 1024) | 256 | 否 (sincos+GSD) |

### 训练超参数

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW (lr=1e-4, wd=0.02) |
| 调度器 | CosWithWarmup (warmup=8000) |
| 全局批次大小 | 512 |
| 微批次大小 | 32 |
| Token 预算 | 2250 |
| Patch 大小范围 | [1, 8] |
| 空间 patch 数范围 | [1, 12] |
| 掩码策略 | modality_cross_random (encode=0.5, decode=0.5) |
| 基础损失 | ModalityPatchDisc (tau=0.1) |
| 对比损失 | InfoNCE (weight=0.1) |
| EMA 衰减 | (1.0, 1.0) → 不更新 |
| 梯度裁剪 | max_norm=1.0 |
| 训练时长 | 300 epochs |
| FSDP | bfloat16 参数, float32 归约 |

### 模态信息

| 模态 | 波段数 | BandSet 数 | 角色 | 空间 | 时态 |
|------|--------|-----------|------|------|------|
| sentinel2_l2a | 10 | 3 (10m/20m/60m) | 编码+解码 | 是 | 是 |
| sentinel1 | 2 | 1 | 编码+解码 | 是 | 是 |
| landsat | 11 | 2 (15m/30m) | 编码+解码 | 是 | 是 |
| worldcover | 1 | 1 | 仅解码 | 是 | 否 |
| srtm | 1 | 1 | 仅解码 | 是 | 否 |
| openstreetmap_raster | 30 | 1 | 仅解码 | 是 | 否 |
| wri_canopy_height_map | 1 | 1 | 仅解码 | 是 | 否 |
| cdl | 1 | 1 | 仅解码 | 是 | 否 |
| worldcereal | 8 | 1 | 仅解码 | 是 | 否 |

---

## 9. Large vs Nano 对比

| 方面 | Nano | Large |
|------|------|-------|
| 编码器维度 | 128 | 1024 |
| 编码器层数 | 4 | 24 |
| 解码器层数 | 4 | 4 |
| 注意力头数 | 8 (16/head) | 16 (64/head) |
| MLP 维度 | 512 | 4096 |
| 编码维度分配 | 32+32+32+32 | 256+256+256+256 |
| Channel Embed 参数 | 11×32=352 | 11×256=2816 |
| DropPath 范围 | 0→0.1 (4层) | 0→0.1 (24层, 更细粒度) |
| Patch Embed | Conv2d | Conv2d |
| 训练模块 | ContrastiveLatentMIM | ContrastiveLatentMIM (相同) |
| 掩码策略 | modality_cross_random (相同) | modality_cross_random (相同) |
| 损失函数 | PatchDisc + InfoNCE (相同) | PatchDisc + InfoNCE (相同) |
| 数据配置 | 完全相同 | 完全相同 |
| 训练器配置 | 完全相同 | 完全相同 |

**核心区别仅在模型规模**：Large 用 24 层 1024 维编码器替换了 Nano 的 4 层 128 维编码器，解码器维度同步扩大但保持 4 层浅层设计。训练管线（数据、掩码、损失、优化）完全一致。

