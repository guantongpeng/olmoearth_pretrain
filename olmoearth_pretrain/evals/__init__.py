"""OlmoEarth Pretrain 评估模块。

本模块是 OlmoEarth 预训练模型的评估框架入口，提供以下核心功能：
- 线性探针 (Linear Probe) 和 KNN 评估
- 嵌入提取与诊断分析
- 多种下游任务的评估（分类、分割等）
- 多种基线模型的评估包装器
- 微调 (Finetune) 评估流程
"""
