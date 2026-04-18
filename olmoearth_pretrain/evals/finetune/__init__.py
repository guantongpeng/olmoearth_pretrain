"""OlmoEarth 微调评估子包。

本模块提供模型微调评估功能，在预训练模型上添加任务特定的分类/分割头，
然后训练并评估下游任务性能。

核心入口：
- run_finetune_eval: 主函数，执行完整的微调训练和评估流程
"""

from olmoearth_pretrain.evals.finetune.train import run_finetune_eval

__all__ = ["run_finetune_eval"]
