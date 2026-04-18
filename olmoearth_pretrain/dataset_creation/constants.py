"""OlmoEarth Pretrain 数据集创建相关常量。

本模块定义了数据集创建过程中使用的全局常量，包括：
    - WINDOW_RESOLUTIONS: 窗口分辨率列表（米/像素）
    - WINDOW_DURATION: 窗口时间跨度（14天）
    - WINDOW_SIZE: 窗口空间大小（256像素）
    - METADATA_COLUMNS: 模态元数据 CSV 的列名列表
    - GEOTIFF_BLOCK_SIZE: GeoTIFF 块大小
    - GEOTIFF_RASTER_FORMAT: GeoTIFF 栅格格式配置
"""

from datetime import timedelta

from rslearn.utils.raster_format import GeotiffRasterFormat

# 需要的分辨率列表
# 创建给定分辨率的窗口时，确保每个更粗的分辨率也有覆盖
WINDOW_RESOLUTIONS = [0.625, 10, 160]  # 单位: 米/像素

WINDOW_DURATION = timedelta(days=14)  # 窗口时间跨度: 14天
WINDOW_SIZE = 256  # 窗口空间大小: 256像素

# 每个模态元数据 CSV 中的列名
METADATA_COLUMNS = [
    "crs",
    "col",
    "row",
    "tile_time",
    "image_idx",
    "start_time",
    "end_time",
]

GEOTIFF_BLOCK_SIZE = 32  # GeoTIFF 块大小
GEOTIFF_RASTER_FORMAT = GeotiffRasterFormat(  # GeoTIFF 栅格格式配置，始终启用分块
    block_size=GEOTIFF_BLOCK_SIZE, always_enable_tiling=True
)
