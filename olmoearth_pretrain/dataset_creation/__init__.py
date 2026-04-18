"""OlmoEarth Pretrain 数据集创建模块。

本模块包含从原始数据源创建 OlmoEarth Pretrain 训练数据集的代码，
包括窗口创建、数据格式转换和元数据生成等功能。

子模块:
    - create_windows/: 创建训练窗口（从经纬度列表或随机生成）
    - rslearn_to_olmoearth/: 从 rslearn 格式转换为 OlmoEarth 格式
    - scripts/: 数据集处理脚本
    - sentinel2_l1c/: Sentinel-2 L1C 数据处理
    - openstreetmap/: OpenStreetMap 数据处理
    - wri_canopy_height_map/: WRI 冠层高度图数据处理
"""
