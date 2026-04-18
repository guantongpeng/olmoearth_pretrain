"""OlmoEarth Pretrain 数据集解析与采样模块。

本模块负责解析 OlmoEarth Pretrain 数据集，并构建可用于训练的样本。

模块结构:
    - parse.py: 解析原始数据集中的 CSV 文件，识别各种模态可用的瓦片(tiles)位置。
    - sample.py: 跨模态综合解析信息，确定加载单个训练样本所需的数据。
    - utils.py: 提供数据集结构相关的工具函数和基类。
    - convert_to_h5py.py: 将 GeoTIFF 格式的数据集转换为 H5PY 格式以供训练使用。

使用场景:
    1. 解析阶段: 使用 parse_dataset() 解析数据集目录下的 CSV 元数据文件。
    2. 采样阶段: 使用. 使用 image_tiles_to_samples() 将解析结果转换为训练样本列表。
    3. 加载阶段: 使用 load_image_for_sample() 加载单个样本的图像数据。
    4. 转换阶段: 使用 ConvertToH5py 类将原始 GeoTIFF 数据转换为 H5 格式。
"""
