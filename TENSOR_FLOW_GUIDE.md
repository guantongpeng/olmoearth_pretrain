# OlmoEarth 训练过程中Tensor全流程推理 - 使用指南

## 📋 文档和工具概览

本目录包含三个主要的资源，用于理解和追踪OlmoEarth预训练过程中的tensor流动：

### 1. **TENSOR_FLOW_INFERENCE.md** - 完整的理论文档
- 📖 **类型**: 参考文档
- 🎯 **用途**: 理解tensor在训练各阶段的维度变化和计算流
- ✨ **包含内容**:
  - 数据加载阶段的tensor形状变化
  - 模型前向传播的详细步骤
  - FlexiViT编码器内部细节
  - 损失函数计算流程
  - 反向传播和梯度累积
  - 完整的tensor流程图
  - 分布式训练考虑事项
  - 常见问题和解决方案

### 2. **tensor_flow_debug.py** - 实用的调试工具
- 🔧 **类型**: Python工具类
- 🎯 **用途**: 实时监控和检查tensor的统计信息
- ✨ **主要功能**:
  - `TensorFlowMonitor` 类用于检查tensor的形状、数值范围、梯度信息
  - 自动检测NaN/Inf异常值
  - 记录各层梯度范数
  - 支持前向传播、损失计算、反向传播的追踪

### 3. **trace_full_training_step.py** - 完整的训练步骤追踪器
- 🎯 **类型**: 集成的追踪框架
- 🎯 **用途**: 在实际训练中追踪完整的训练步骤
- ✨ **主要功能**:
  - `TrainingStepTracer` 类支持一键追踪整个训练步骤
  - 按阶段记录tensor信息 (输入、掩码、前向传播、损失、反向传播、优化器更新、EMA)
  - 支持可配置的追踪频率
  - 日志输出到文件和控制台

---

## 🚀 快速开始

### 方案 A: 阅读理论文档 (推荐首选)

```bash
# 打开并阅读完整的tensor流程推理文档
cat TENSOR_FLOW_INFERENCE.md

# 或用您喜欢的编辑器打开
vim TENSOR_FLOW_INFERENCE.md
code TENSOR_FLOW_INFERENCE.md
```

**适合场景**: 想要深入理解OlmoEarth训练流程的各个阶段

---

### 方案 B: 在训练中使用调试工具

```python
# 在你的训练脚本中导入并使用
from tensor_flow_debug import TensorFlowMonitor

# 初始化监控器
monitor = TensorFlowMonitor(log_level=logging.INFO)

# 在关键位置检查tensor
for batch in data_loader:
    # 检查输入数据
    for name, data in batch.data.items():
        monitor.check_tensor(data, f"input_{name}")
    
    # 前向传播
    loss = model(batch)
    
    # 检查损失
    monitor.check_tensor(loss, "loss")
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    backward_stats = monitor.trace_backward_pass(loss, model)
```

**适合场景**: 在已有的训练脚本中添加tensor监控

---

### 方案 C: 使用完整的追踪框架

```python
# 在你的训练脚本中集成完整追踪器
from trace_full_training_step import TrainingStepTracer

# 创建追踪器 (每10步进行一次详细追踪)
tracer = TrainingStepTracer(
    trace_interval=10,
    log_dir=Path("./tensor_traces")
)

# 在训练循环中使用
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(data_loader):
        # 一键追踪整个训练步骤的所有9个阶段
        tracer.trace_training_step(train_module, batch, batch_idx)
```

**适合场景**: 想要在实际训练中进行完整的端到端调试

---

## 📊 Tensor流程快速参考

### 数据加载到模型输入

```
H5 文件 [4, H, W, 11] (S2数据)
   ↓
OlmoEarthSample (单个地点)
   ↓ Transform (Flip/Rotate)
Augmented Sample
   ↓ Collate + Masking
MaskedOlmoEarthSample [B, N_tokens, D]
```

### 模型前向传播

```
Input: [B, N_tokens, D] (masked)
   ↓
Online Encoder (12层)
   ↓
Encoder Output: [B, N_visible, D]
   ↓
Decoder (重建掩码位置)
   ↓
Decoded Output: [B, N_masked, D]
   ↓ (与Target Encoder输出对比)
Loss: scalar
```

### 完整的训练步骤 (9个阶段)

```
1. 输入数据检查
   ↓
2. 掩码策略分析
   ↓
3. 前向传播追踪
   ↓
4. 损失计算检查
   ↓
5. 正则化项
   ↓
6. 反向传播 (梯度计算)
   ↓
7. 梯度剪裁
   ↓
8. 优化器步骤 (参数更新)
   ↓
9. EMA目标编码器更新
```

---

## 🎯 常见使用场景

### 场景 1: 调试NaN/Inf问题

```python
monitor = TensorFlowMonitor()

# 逐步追踪
monitor.check_tensor(input_data, "input")      # ✓ OK
monitor.check_tensor(embedded, "embedding")    # ✓ OK
monitor.check_tensor(output, "output")         # ⚠️ NaN detected!

# 根据日志找到问题来源
# 输出会清楚地标记包含NaN的tensor
```

### 场景 2: 监控梯度爆炸/消失

```python
tracer = TrainingStepTracer(trace_interval=1)  # 每步追踪

for batch_idx, batch in enumerate(data_loader):
    if batch_idx >= 100:  # 关注100+步的训练
        tracer.trace_training_step(train_module, batch, batch_idx)
        
# 日志会显示:
# - 梯度范数大小
# - 参数更新幅度
# - 学习率变化
```

### 场景 3: 理解模型的tensor形状

```python
# 从TENSOR_FLOW_INFERENCE.md的表格 8 获取快速参考
# 或运行追踪看实际输出:

monitor.trace_forward_pass(model, batch, patch_size=16)
# 输出详细的shape信息

# 示例输出:
# encoder_output.tokens           | shape: [4, 1024, 768]  
# decoder_output.tokens           | shape: [4, 256, 768]
# target_encoder_output.tokens    | shape: [4, 1024, 768]
```

### 场景 4: 验证数据增强效果

```python
monitor = TensorFlowMonitor()

# 增强前后的数据统计
monitor.check_tensor(original_data, "before_augmentation")
monitor.check_tensor(augmented_data, "after_augmentation")

# 对比:
# before_augmentation  | mean: 0.1234 | std: 0.0567
# after_augmentation   | mean: 0.1200 | std: 0.0589
```

---

## 🔍 输出解读

### TensorFlowMonitor 输出示例

```
input.S2_L1C                       | shape: (4, 512, 512, 11)     | dtype: torch.float32 | mean:  0.1234 | std:  0.0567 | min: -0.1234 | max:  0.9876
encoder_output.tokens             | shape: (4, 1024, 768)        | dtype: torch.float32 | mean:  0.0123 | std:  0.0456 | min: -0.5678 | max:  0.5678 | grad_norm: 0.1234
loss                              | shape: ()                    | dtype: torch.float32 | mean:  1.2345 | std:  0.0000 | min:  1.2345 | max:  1.2345
```

**关键指标说明**:
- **shape**: tensor维度
- **mean/std**: 数值的平均值和标准差 (检查数值范围是否合理)
- **min/max**: 最小最大值 (检查是否有异常值)
- **grad_norm**: 梯度范数 (检查梯度大小)
- **⚠️**: 标记包含NaN或Inf的tensor

---

## 📝 集成到训练脚本的示例

```python
#!/usr/bin/env python3
"""
集成tensor追踪的训练脚本
"""

import torch
from pathlib import Path
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModule
from trace_full_training_step import TrainingStepTracer

def train():
    # 初始化模型、数据加载器、优化器等
    # ...
    
    # 创建追踪器
    # trace_interval=10 表示每10个batch进行一次详细追踪
    tracer = TrainingStepTracer(
        trace_interval=10,
        log_dir=Path("./debug_logs")
    )
    
    # 训练循环
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(data_loader):
            # 使用tracer进行追踪
            # (如果不满足trace_interval条件，将跳过详细追踪)
            tracer.trace_training_step(train_module, batch, batch_idx)
            
            # 日志将被保存到 ./debug_logs/tensor_trace.log
    
    print("✓ Training completed. Check ./debug_logs/tensor_trace.log for details.")

if __name__ == "__main__":
    train()
```

---

## 📚 参考资源

### 理论背景
- **TENSOR_FLOW_INFERENCE.md** - 完整的tensor流程推理文档
  - 第3-5节: 模型架构和前向传播
  - 第6-7节: 损失和反向传播
  - 第8节: 关键tensor操作总结

### 工具API
- **tensor_flow_debug.py**
  - `TensorFlowMonitor.check_tensor()` - 检查单个tensor
  - `TensorFlowMonitor.trace_forward_pass()` - 追踪前向传播
  - `TensorFlowMonitor.trace_backward_pass()` - 追踪反向传播

### 框架集成
- **trace_full_training_step.py**
  - `TrainingStepTracer` - 完整的训练步骤追踪
  - 支持9个阶段的详细日志

---

## 🛠️ 故障排除

### Q: 如何处理过多的日志输出？
**A**: 调整日志级别或增加 `trace_interval` 值
```python
tracer = TrainingStepTracer(
    trace_interval=100  # 只在第100、200、300...步追踪
)
```

### Q: 追踪对性能有影响吗？
**A**: 有。建议仅在调试时启用详细追踪。正常训练时设置 `trace_interval=0`。

### Q: 如何只追踪特定的层或tensor？
**A**: 修改 `tensor_flow_debug.py` 中的 `check_tensor()` 方法以添加过滤逻辑。

---

## 📞 支持

如需帮助，请查阅：
1. **TENSOR_FLOW_INFERENCE.md** 中的"常见问题与解决方案"部分
2. OlmoEarth官方文档: https://github.com/allenai/OlmoEarth
3. 训练日志文件 (通常位于 `./debug_logs/tensor_trace.log`)

---

**版本**: 1.0  
**最后更新**: 2026-04-21  
**作者**: OlmoEarth Debugging Suite
