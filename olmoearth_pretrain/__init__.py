"""OlmoEarth Pretrain 库的根包。

本包是 OlmoEarth 预训练框架的核心入口，提供了地球观测多模态数据的
预训练与推理功能。主要包含以下子模块：

- types: 类型别名定义，统一 numpy 数组和 torch 张量的类型标注
- decorators: 装饰器工具，用于标记实验性、已弃用或内部功能
- _compat: 向后兼容模块，支持从旧版 helios 命名空间平滑迁移
- config: 双模式配置系统，支持有/无 olmo-core 依赖的运行环境
- model_loader: 模型加载模块，从 Hugging Face Hub 或本地路径加载预训练模型
- datatypes: 核心数据类型定义，包含样本、掩码样本及编码器输出等数据结构
- data: 数据处理相关模块，包含数据常量、数据集定义等

使用场景：
    - 地球观测多模态大模型（如 Sentinel-1/2、WorldCover、NAIP 等）的预训练
    - 从 Hugging Face Hub 加载预训练权重进行推理
    - 自定义配置和数据流水线进行模型训练
"""
