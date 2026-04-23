# OlmoEarth Pretrain - 训练过程中的Tensor全流程推理

## 概述

本文档详细描述OlmoEarth预训练过程中**tensor从数据加载到梯度计算**的完整流程，以Latent MIM为主要例子。

---

## 1. 数据加载阶段 (DataLoader)

### 1.1 初始数据形状

**来源**: `olmoearth_pretrain/data/dataset.py` - `OlmoEarthDataset`

```
数据存储格式: H5 文件
  ├─ S2 (Sentinel-2)       : 11 bands (4 个时间步)      -> shape: (T=4, H=W, 11 channels)
  ├─ VV/VH (SAR)           : 2 bands  (4 个时间步)      -> shape: (T=4, H=W, 2 channels)  
  ├─ DEM (数字高程)         : 1 band   (单个时间步)     -> shape: (1, H=W, 1 channel)
  ├─ Agera5 (气象)         : 4 bands  (12 个时间步)     -> shape: (T=12, H=W, 4 channels)
  └─ SRTM 斜率/方向等      : 多个衍生层
```

**加载过程**:
```python
# 来自 OlmoEarthDataset.__getitem__()
batch = dataset[idx]  # 获取单个地点的数据

# batch 结构: OlmoEarthSample
batch.data = {
    'S2_L1C': torch.FloatTensor([T, H, W, 11]),      # Sentinel-2 多光谱数据
    'VV': torch.FloatTensor([T, H, W, 2]),           # SAR VV极化
    'VH': torch.FloatTensor([T, H, W, 2]),           # SAR VH极化
    'DEM': torch.FloatTensor([1, H, W, 1]),          # 高程
    'Agera5': torch.FloatTensor([T, H, W, 4]),       # 气象变量
    # ... 其他模态
}
batch.metadata = {...}  # 地理信息、时间戳等
```

### 1.2 数据增强 (Transform)

**文件**: `olmoearth_pretrain/data/transform.py`

```python
# TransformConfig.build() -> Transform 实例
transform = Transform(transform_type="flip_and_rotate")

# 对每个样本应用增强
augmented_batch = transform(batch)  # 就地操作或返回新的 OlmoEarthSample

# 增强操作:
#   - Random Flip (水平/垂直)
#   - Random Rotation (0°, 90°, 180°, 270°)
#   - 保留时间维度和所有通道

# 输出形状不变
augmented_batch.data['S2_L1C'].shape  # 仍为 [T, H, W, 11]
```

### 1.3 批次收集和掩码应用 (Collate + Masking)

**文件**: `olmoearth_pretrain/data/collate.py`

```python
# DataLoader 调用 collate_fn，通常为 collate_single_masked_batched()

def collate_single_masked_batched(batch_samples, masking_strategy, transform, ...):
    """
    输入: 
      batch_samples: List[OlmoEarthSample]  (batch_size 个样本)
    """
    
    # Step 1: 应用数据增强到每个样本
    transformed = [transform(s) for s in batch_samples]
    
    # Step 2: 按模态和 patch 大小重新组织数据
    # 将样本中的多个模态数据标准化为统一的 patch grid
    
    # Step 3: 应用掩码策略
    masked_batch = masking_strategy.apply(transformed)
    
    # 输出: MaskedOlmoEarthSample
    return MaskedOlmoEarthSample(
        data={...},
        masks={
            'encoder_mask': torch.BoolTensor,   # 编码器可见的 token
            'decoder_mask': torch.BoolTensor,   # 解码器可见的 token
            'masked_positions': torch.BoolTensor # 需要预测的位置
        },
        patch_size=P
    )
```

**掩码类型**:
```
样本中的每个 token 被分为三类:

1. Encoder-only tokens   (可见于编码器)
2. Decoder-visible       (解码器可见，用于补充信息)
3. Masked tokens         (掩码，需要通过潜在表示预测)

掩码策略 (MaskingStrategy):
  - Random:              随机掩盖 mask_ratio % 的 token
  - BlockWise:           按块掩盖，保留空间连贯性
  - TemporalCausal:      沿时间轴因果掩盖
```

---

## 2. Tensor 维度变化汇总表

### 2.1 数据加载到掩码前

| 阶段 | 关键tensor | 形状 | 描述 |
|------|----------|------|------|
| **原始数据** | `S2_L1C` | `[4, H, W, 11]` | T时间步，空间分辨率H×W，11个波段 |
| | `VV/VH` | `[4, H, W, 2]` | SAR 数据，4个时间步 |
| **批次前** | `OlmoEarthSample` | 多个tensor的集合 | 单个地点的多模态数据 |
| **批次收集** | `batch` | `List[OlmoEarthSample]` | batch_size 个样本，未对齐 |
| **标准化** | `standardized` | 统一为 patch grid | 按 patch_size 重新组织空间维度 |
| **应用掩码** | `MaskedOlmoEarthSample` | 附加 mask tensors | 标记编码器/解码器/掩码位置 |

### 2.2 基本 Patch Embedding 维度

```
假设 patch_size = 16 (常用配置)

原始图像:  [4, 512, 512, 11]  (H=W=512, T=4)
          ↓ 按 patch_size 分割空间维度
Patches:   [4, 32, 32, 11*256]  (32 = 512/16, 256 = 16*16)
          ↓ Flatten patches
Tokens:    [4*32*32, 11*256]  = [4096, 2816]  # 每个 token 是 patch 的展平表示
          ↓ 线性投影 (patch embedding)
Embedded:  [4096, D_embed]  (D_embed 通常 768/1024)
```

---

## 3. 模型前向传播 (Latent MIM)

**文件**: `olmoearth_pretrain/train/train_module/latent_mim.py`

### 3.1 模型前向传播的关键函数

```python
# OlmoEarthTrainModule.train_batch() -> LatentMIMTrainModule.model_forward()

def model_forward(batch: MaskedOlmoEarthSample, patch_size: int, ...):
    """
    输入:
      batch: MaskedOlmoEarthSample (已掩码的多模态样本)
      patch_size: int (patch 分割大小)
    """
    
    # === PHASE 1: 在线编码器前向传播 ===
    # 来自 olmoearth_pretrain/nn/latent_mim.py
    latent, decoded, _, reconstructed, extra_metrics = model(batch, patch_size)
    
    # latent:       TokensAndMasks  # 编码器输出潜在表示
    # decoded:      TokensAndMasks  # 解码器的重建输出
    # reconstructed: ...             # MAE 风格的像素级重建 (可选)
    
    # === PHASE 2: 目标编码器前向传播（无梯度） ===
    with torch.no_grad():
        # 对完整未掩码数据运行目标编码器
        output_dict = model.target_encoder.forward(
            batch.unmask(),  # 移除掩码，使用完整数据
            patch_size=patch_size,
            token_exit_cfg=token_exit_cfg
        )
        target_output, _, _ = unpack_encoder_output(output_dict)
    
    # === PHASE 3: 损失计算 ===
    # 判别损失: 解码器输出 vs 目标编码器输出
    loss = loss_fn(decoded, target_output)  # PatchDiscrimination 损失
    
    # 可选: MAE 重建损失
    if mae_loss is not None:
        loss += mae_loss.compute(reconstructed, batch)
    
    return loss, latent, decoded, target_output
```

### 3.2 在线编码器内部流程

**文件**: `olmoearth_pretrain/nn/latent_mim.py` - `LatentMIM` class

```python
class LatentMIM:
    def forward(self, batch: MaskedOlmoEarthSample, patch_size: int):
        
        # Step 1: Patch Embedding + Token 创建
        # ========================================
        # 输入: MaskedOlmoEarthSample 
        #   batch.data['S2_L1C']: [B, T, H, W, 11]
        #   batch.data['VV']:     [B, T, H, W, 2]
        #   其他模态...
        
        # 输出: tokens_and_masks
        #   tokens: [B, N_tokens, D]  # N_tokens = sum(patch_grid_size)
        #   encoder_mask: [B, N_tokens]
        #   decoder_mask: [B, N_tokens]
        
        tokens_and_masks = self.encoder(batch, patch_size)
        
        # Step 2: 编码器处理可见 token
        # ==============================
        # 仅对编码器掩码为 True 的 token 输入
        encoder_output = self.encoder.forward_masked(
            tokens_and_masks,
            mask=tokens_and_masks.encoder_mask
        )
        # encoder_output: [B, N_visible, D]
        
        # Step 3: 解码器重建掩码 token 的潜在表示
        # =========================================
        # 将编码器输出和掩码位置传递给解码器
        decoded_latent = self.decoder(
            encoder_output,
            mask_positions=tokens_and_masks.masked_positions,
            full_sequence_length=N_tokens
        )
        # decoded_latent: [B, N_masked, D]  # 仅掩码位置
        
        # Step 4: 可选的 MAE 重建损失计算
        # ==================================
        if self.mae_head is not None:
            reconstructed = self.mae_head(encoder_output)
            # reconstructed: [B, H_reconstructed, W_reconstructed, channels]
        
        return encoder_output, decoded_latent, ..., reconstructed, metrics
```

### 3.3 目标编码器 (EMA 更新)

```python
# model.target_encoder 是 model.encoder 的 EMA 副本

# EMA 更新 (每个训练步骤):
ema_decay_current = start_ema + (end_ema - start_ema) * progress
# EMA 衰减范围: (0.996, 1.0)，通常不更新非常快

for (name, p_online), (name, p_target) in zip(encoder.params, target_encoder.params):
    p_target.data = ema_decay_current * p_target.data + (1 - ema_decay_current) * p_online.data
```

---

## 4. FlexiViT 编码器内部细节

**文件**: `olmoearth_pretrain/nn/flexi_vit.py` (2168 行)

### 4.1 Patch Embedding 阶段

```python
class FlexiViT(nn.Module):
    def forward(self, batch: MaskedOlmoEarthSample, patch_size: int):
        """
        输入: 
          batch: 包含多模态数据的 MaskedOlmoEarthSample
          patch_size: 空间分割粒度
        """
        
        # Step 1: 多模态 Token 创建
        tokens_and_masks = self._create_tokens_and_masks(batch, patch_size)
        
        # tokens_and_masks.tokens: [B, N, D]
        #   N = num_patches_per_time * T + additional_tokens
        #   D = embedding_dim (通常 768)
        
        # Step 2: 位置编码
        # 支持多种位置编码方案:
        #   - Sinusoidal (sin/cos)
        #   - Resolution-aware (分辨率感知)
        #   - Temporal (时间感知)
        #   - Monthly (月份编码)
        
        pos_embed = self._get_position_embeddings(tokens_and_masks, patch_size)
        # pos_embed: [B, N, D]
        
        tokens = tokens_and_masks.tokens + pos_embed  # [B, N, D]
        
        # Step 3: Transformer 编码器块
        for layer_idx, transformer_block in enumerate(self.transformer_blocks):
            # Attention + MLP
            tokens = transformer_block(tokens)  # [B, N, D]
            
            # 可选: Token 退出 (early exit)
            if layer_idx in self.token_exit_cfg:
                # 在某些层提前退出特定模态的 token
                pass
        
        return tokens, masks, ...
```

### 4.2 关键维度

```
典型 FlexiViT Base 配置:
  - Hidden size (D):        768
  - Num layers:             12
  - Num attention heads:     12
  - MLP hidden:             3072
  
Tensor 形状变化:

Input:       [B, T, H, W, C_modalities]  (多模态输入)
             ↓ 按 patch_size 分割空间维
Patches:     [B, T, num_patches_h, num_patches_w, patch_dim]
             ↓ Flatten + Linear projection
Tokens:      [B, N_tokens, 768]
             ↓ 每个 transformer 块
After Attn:  [B, N_tokens, 768]
After MLP:   [B, N_tokens, 768]
             ↓ 12 个块
Output:      [B, N_tokens, 768]
```

---

## 5. 损失函数计算

**文件**: `olmoearth_pretrain/train/loss.py` (1122 行)

### 5.1 PatchDiscrimination 损失

```python
class PatchDiscriminationLoss:
    """
    判别损失: 将解码器预测与目标编码器的表示匹配
    """
    
    def compute(self, decoded: TokensAndMasks, target: TokensAndMasks) -> torch.Tensor:
        """
        输入:
          decoded: 解码器的重建输出 [B, N_masked, D]
          target:  目标编码器的输出 [B, N_all, D]
        """
        
        # Step 1: 对齐: 从目标中提取掩码位置的表示
        target_masked = target.tokens[target.masks.masked_positions]  # [B, N_masked, D]
        
        # Step 2: 计算损失
        # 常用方法: L2 距离、余弦相似度等
        loss = F.mse_loss(decoded.tokens, target_masked)  # 标量
        
        # Step 3: 可选的归一化
        # (按 token 数或批次大小)
        loss = loss / num_masked_tokens  # 或其他归一化
        
        return loss  # torch.Tensor (scalar)
```

### 5.2 MAE 重建损失 (可选)

```python
class MAEReconstructionLoss:
    """
    在像素/重建空间计算重建损失
    """
    
    def compute(self, reconstructed, batch):
        """
        输入:
          reconstructed: 解码器的像素级输出 [B, H, W, C]
          batch: 原始数据 (用于提取目标)
        """
        
        # 构造目标 (掩码区域的原始像素值)
        target = batch.get_target_pixels(masked_only=True)
        
        loss = F.l1_loss(reconstructed, target)
        return loss
```

### 5.3 正则化项 (可选)

```python
class RegularizationLoss:
    """
    额外的正则化项 (L1/L2 范数、稀疏性约束等)
    """
    
    def compute(self, latent):
        # 例: L2 正则化
        reg_loss = torch.norm(latent.tokens, p=2)
        return reg_loss
```

---

## 6. 反向传播和梯度累积

**文件**: `olmoearth_pretrain/train/train_module/latent_mim.py` - `train_batch()`

### 6.1 微批次梯度累积

```python
def train_batch(batch, dry_run=False):
    """
    支持梯度累积用于更大的有效批次大小
    """
    
    # 将批次分割为微批次
    masked_microbatches = split_masked_batch(batch, rank_microbatch_size)
    num_microbatches = len(masked_microbatches)
    
    total_batch_loss = 0
    
    for microbatch_idx in range(num_microbatches):
        with self._train_microbatch_context(microbatch_idx, num_microbatches):
            microbatch = masked_microbatches[microbatch_idx]
            
            # 前向传播
            loss, latent, decoded, target_output = self.model_forward(
                microbatch, patch_size, token_exit_cfg
            )
            
            # 计算正则化损失
            reg_term = self.compute_regularization(latent)
            if reg_term is not None:
                loss = loss + reg_term
            
            # 损失缩放 (用于梯度累积)
            loss = loss / num_microbatches
            
            # 反向传播 (梯度累积)
            loss.backward()  # ∇L 累积
            
            total_batch_loss += loss.detach()
    
    # 梯度剪裁 (在所有微批次之后)
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    
    # 优化器步骤 (外部调用，通常在 trainer 中)
    # optimizer.step()
    # scheduler.step()
    # optimizer.zero_grad()
```

### 6.2 梯度流图

```
Loss (标量) 
    ↓ backward()
∇Loss/∇Loss = 1
    ↓ 链式法则
∇Loss/∇decoded_logits
    ↓
∇Loss/∇decoder_hidden
    ↓
∇Loss/∇encoder_hidden
    ↓
∇Loss/∇patch_embeddings
    ↓
∇Loss/∇online_encoder_weights
    ↓ 梯度剪裁 (clip_grad_norm_)
∇Loss_clipped
    ↓ 优化器步骤
online_encoder_weights ← weights - lr * ∇Loss_clipped
    ↓ EMA 更新
target_encoder_weights ← decay * target + (1 - decay) * online
```

---

## 7. 完整的 Tensor 流程示意

```
┌────────────────────────────────────────────────────────────────────┐
│                      数据加载阶段                                   │
├────────────────────────────────────────────────────────────────────┤
│  H5 文件 ──> OlmoEarthSample (单个地点)                             │
│             ├─ S2_L1C:    [T=4, H, W, 11]                          │
│             ├─ VV/VH:     [T=4, H, W, 2]                           │
│             └─ 其他模态...                                          │
│                  ↓ Transform (Flip, Rotate)                        │
│             Augmented OlmoEarthSample (形状不变)                    │
└────────────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────────────┐
│                      批次收集与掩码                                  │
├────────────────────────────────────────────────────────────────────┤
│  List[OlmoEarthSample] ──> 标准化为 patch grid                    │
│                              ↓ Collate                              │
│                        MaskedOlmoEarthSample                        │
│                        ├─ tokens:    [B, N, D]                     │
│                        ├─ encoder_mask: [B, N]                     │
│                        └─ masked_positions: [B, N]                 │
└────────────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────────────┐
│                    模型前向传播 (Latent MIM)                       │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────┐          ┌──────────────────────┐       │
│  │  在线编码器          │          │  目标编码器 (无梯度)  │       │
│  │  (Gradient enabled)  │          │  (EMA 更新)          │       │
│  └──────────────────────┘          └──────────────────────┘       │
│           │                                 │                      │
│      输入: tokens                      输入: full_data             │
│     (仅编码器掩码 True)                                             │
│           ↓                                 ↓                      │
│   FlexiViT Encoder                   FlexiViT Encoder              │
│    (12 层)                             (EMA 副本)                  │
│           ↓                                 ↓                      │
│   encoder_output [B, N_vis, D]    target_output [B, N, D]         │
│           ↓                                 ↓                      │
│   Decoder (重建掩码位置)              (no decoder)                 │
│           ↓                                                        │
│   decoded_latent [B, N_masked, D]         ↘                      │
│                                            ↙                      │
│                                  Loss 计算 (decoded vs target)     │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────────────┐
│                        损失计算                                     │
├────────────────────────────────────────────────────────────────────┤
│  loss = L2(decoded_latent, target_masked_latent)                   │
│  if mae_loss: loss += L1(reconstructed, target_pixels)             │
│  if reg: loss += reg_term                                          │
│                                                                     │
│  loss: torch.Tensor (scalar, 标量)                                 │
└────────────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────────────┐
│                   反向传播与梯度更新                                │
├────────────────────────────────────────────────────────────────────┤
│  loss.backward()  # 计算所有参数的梯度                             │
│     ↓                                                               │
│  clip_grad_norm_(model, max_grad_norm)  # 梯度剪裁                │
│     ↓                                                               │
│  optimizer.step()  # 更新在线编码器参数                           │
│     ↓                                                               │
│  scheduler.step()  # 更新学习率                                   │
│     ↓                                                               │
│  EMA 更新: target ← decay * target + (1-decay) * online            │
└────────────────────────────────────────────────────────────────────┘
```

---

## 8. 关键 Tensor 操作总结

| 操作 | 输入形状 | 输出形状 | 描述 |
|------|---------|---------|------|
| **数据加载** | H5 | `[B, T, H, W, C]` | 多模态数据加载 |
| **Transform** | `[B, T, H, W, C]` | `[B, T, H, W, C]` | Flip/Rotate |
| **Patch Embed** | `[B, T, H, W, C]` | `[B, N, D]` | 将图像划分为 patch 并嵌入 |
| **Pos Encoding** | - | `[B, N, D]` | 位置编码相加 |
| **Attention** | `[B, N, D]` | `[B, N, D]` | Multi-head 自注意力 |
| **MLP** | `[B, N, D]` | `[B, N, D]` | 前馈网络 |
| **Decoder** | `[B, N_vis, D]` | `[B, N_masked, D]` | 重建掩码位置 |
| **Loss** | `[B, N_masked, D]` x2 | `scalar` | MSE/L2 损失 |
| **Backward** | `scalar` | grad tensors | 反向传播 |
| **Optimizer** | grad tensors | updated params | 参数更新 |

---

## 9. 分布式训练考虑

### 9.1 数据并行 (DDP/FSDP)

```python
# FSDP 包装
model = FSDP(model, device_mesh=mesh, ...)

# 前向传播自动处理分布式
# 每个 rank 处理 batch_size/world_size 个样本

# 梯度同步发生在 loss.backward() 期间
# AllReduce 操作:
#   - 每个 rank 计算本地梯度
#   - AllReduce 同步梯度
#   - 所有 rank 获得平均梯度
```

### 9.2 分布式检查点

```python
# 使用 torch.distributed.checkpoint
# 支持梯度状态、优化器状态的分布式保存/加载

state_dict = get_state_dict(model, optimizer, ...)
dist_cp.save(state_dict, checkpoint_dir)

# 恢复
state_dict = dist_cp.load(checkpoint_dir, ...)
set_state_dict(model, optimizer, state_dict)
```

---

## 10. 性能监控关键指标

```python
# 监控的关键 metrics (来自 wandb callback)

# 损失相关
"train/loss": float                 # 基础损失
"train/mae_loss": float             # MAE 损失 (可选)
"train/reg_loss": float             # 正则化损失

# 梯度相关
"train/grad_norm": float            # 梯度范数
"train/grad_norm_clipped": bool     # 是否进行了梯度剪裁

# 数据相关
"train/num_masked_tokens": int      # 掩码 token 数量
"train/mask_ratio": float           # 掩码比例

# 计算效率
"train/throughput": float           # token/s
"train/memory_usage": float         # GPU 内存使用率

# 模型相关
"train/ema_decay": float            # 当前 EMA 衰减系数
"target_encoder/weight_diff": float # 在线与目标编码器权重差异
```

---

## 11. 调试 Tensor 流程的方法

### 11.1 添加日志

```python
# 在关键位置添加
logger.info(f"Batch shape: {batch.data['S2_L1C'].shape}")
logger.info(f"Tokens shape: {tokens_and_masks.tokens.shape}")
logger.info(f"Decoded shape: {decoded.tokens.shape}")
logger.info(f"Loss: {loss.item()}")
```

### 11.2 使用 Hook 监控梯度

```python
def register_grad_hooks(model):
    def hook_fn(name):
        def hook(grad):
            logger.info(f"{name} grad norm: {grad.norm().item()}")
            if torch.isnan(grad).any():
                logger.warning(f"NaN in {name} gradient!")
        return hook
    
    for name, param in model.named_parameters():
        param.register_hook(hook_fn(name))
```

### 11.3 Tensor 验证

```python
# 检查异常值
def check_tensor_health(tensor, name):
    if torch.isnan(tensor).any():
        logger.error(f"{name} contains NaN!")
    elif torch.isinf(tensor).any():
        logger.error(f"{name} contains Inf!")
    else:
        logger.info(f"{name}: min={tensor.min():.4f}, max={tensor.max():.4f}, mean={tensor.mean():.4f}")

check_tensor_health(tokens, "tokens")
check_tensor_health(loss, "loss")
```

---

## 12. 常见问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|--------|
| Loss 为 NaN | 梯度爆炸、数值不稳定 | 增加梯度剪裁、降低学习率、检查数据范围 |
| GPU OOM | 批次太大、tensor 维度过高 | 减小 batch_size、rank_microbatch_size |
| 训练缓慢 | 数据加载、GPU 利用率低 | 增加 num_workers、prefetch_factor |
| 梯度为 0 | 某些参数未被使用 | 检查 find_unused_parameters，或移除未使用的层 |
| Loss 不下降 | 学习率过小、掩码比例不当 | 调整学习率、mask_ratio |

---

## 参考代码文件

- **数据加载**: `olmoearth_pretrain/data/`
- **模型**: `olmoearth_pretrain/nn/`
- **训练**: `olmoearth_pretrain/train/`
- **训练脚本**: `scripts/official/`

---

## 附录: 常用配置参数

```yaml
# 数据相关
batch_size: 32
rank_microbatch_size: 4  # 梯度累积: 32/4=8 步
patch_size: 16
mask_ratio: 0.75

# 模型相关
hidden_size: 768
num_layers: 12
num_heads: 12

# 训练相关
learning_rate: 1e-4
max_grad_norm: 1.0
ema_decay: [0.996, 1.0]  # 线性增长

# 优化器
optimizer: AdamW
weight_decay: 0.1
warmup_steps: 10000
```

---

**文档最后更新**: 2026-04-21
