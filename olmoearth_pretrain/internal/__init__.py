"""OlmoEarth Pretrain 内部代码模块。

本模块包含训练和评估的内部工具，需要 olmo-core 依赖。
安装方式: pip install olmoearth-pretrain[training]

使用场景:
    - 实验配置和启动
    - 评估扫描和检查点扫描
    - H5 格式转换和归一化计算
"""

from olmoearth_pretrain.config import require_olmo_core

require_olmo_core("Internal training utilities")
