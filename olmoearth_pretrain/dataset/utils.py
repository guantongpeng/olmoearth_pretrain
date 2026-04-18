"""OlmoEarth Pretrain 数据集结构相关的工具函数和基类。

本模块提供数据集目录结构、窗口元数据和文件命名等基础工具，
供 parse.py 和 convert_to_h5py.py 等模块使用。

主要类:
    WindowMetadata: rslearn 窗口元数据，包含 CRS、分辨率、行列号和时间

主要函数:
    get_modality_dir(): 获取模态数据的存储目录
    list_examples_for_modality(): 列出模态可用的样本 ID
    get_modality_fname(): 获取模态数据的文件名
"""

from datetime import datetime

from upath import UPath

from olmoearth_pretrain.data.constants import (
    BASE_RESOLUTION,
    ModalitySpec,
    TimeSpan,
)


class WindowMetadata:
    """rslearn 窗口元数据类，用于 OlmoEarth Pretrain 数据集。

    窗口名称指定了 CRS、列号、行号、分辨率和时间戳。
    这些信息也可以从 rslearn 窗口元数据中派生。

    关键属性:
        crs: UTM 坐标参考系统（如 EPSG:32610）
        resolution: 窗口所在网格的分辨率
        col: 瓦片在网格中的列号
        row: 瓦片在网格中的行号
        time: 瓦片使用的中心时间

    使用场景:
        在数据解析和转换过程中，用于标识和定位数据文件。
    """

    def __init__(
        self,
        crs: str,
        resolution: float,
        col: int,
        row: int,
        time: datetime,
    ):
        """创建新的 WindowMetadata 实例。

        Args:
            crs: 样本所在的 UTM 坐标参考系统
            resolution: 窗口所在网格的分辨率
            col: 瓦片在网格中的列号
            row: 瓦片在网格中的行号
            time: 瓦片使用的中心时间
        """
        self.crs = crs
        self.resolution = resolution
        self.col = col
        self.row = row
        self.time = time

    def get_window_name(self) -> str:
        """将元数据编码回窗口名称字符串。

        Returns:
            str: 格式为 "{crs}_{resolution}_{col}_{row}" 的窗口名称
        """
        return f"{self.crs}_{self.resolution}_{self.col}_{self.row}"

    def get_resolution_factor(self) -> int:
        """获取分辨率因子，即分辨率相对于 BASE_RESOLUTION 的倍数。

        参考 olmoearth_pretrain.data.constants 中的定义。

        Returns:
            int: 分辨率因子
        """
        return round(self.resolution / BASE_RESOLUTION)


def get_modality_dir(path: UPath, modality: ModalitySpec, time_span: TimeSpan) -> UPath:
    """获取指定模态数据的存储目录路径。

    Args:
        path: OlmoEarth Pretrain 数据集根路径
        modality: 模态规格
        time_span: 时间跨度，决定目录名后缀

    Returns:
        UPath: 模态数据存储目录的完整路径
    """
    suffix = time_span.get_suffix()
    dir_name = f"{modality.get_tile_resolution()}_{modality.name}{suffix}"
    return path / dir_name


def list_examples_for_modality(
    path: UPath, modality: ModalitySpec, time_span: TimeSpan
) -> list[str]:
    """列出指定模态可用的样本 ID。

    通过列出模态目录内容来确定可用的样本，不使用索引和元数据 CSV。

    Args:
        path: OlmoEarth Pretrain 数据集根路径
        modality: 要检查的模态
        time_span: 要检查的时间跨度

    Returns:
        list[str]: 样本 ID 列表
    """
    modality_dir = get_modality_dir(path, modality, time_span)
    if not modality_dir.exists():
        return []

    # 列出目录内容并去除扩展名，获取样本 ID
    example_ids = []
    for fname in modality_dir.iterdir():
        example_ids.append(fname.name.split(".")[0])
    return example_ids


def get_modality_fname(
    path: UPath,
    modality: ModalitySpec,
    time_span: TimeSpan,
    window_metadata: WindowMetadata,
    resolution: float,
    ext: str,
) -> UPath:
    """获取指定窗口和模态的数据存储文件名。

    文件名格式为: {modality_dir}/{crs}_{col}_{row}_{resolution}.{ext}

    Args:
        path: OlmoEarth Pretrain 数据集根路径
        modality: 模态规格
        time_span: 数据的时间跨度
        window_metadata: 从窗口名称提取的元数据
        resolution: 波段的分辨率，应为窗口分辨率乘以2的幂次
        ext: 文件扩展名，如 "tif" 或 "geojson"

    Returns:
        UPath: 数据存储的完整文件路径
    """
    modality_dir = get_modality_dir(path, modality, time_span)
    crs = window_metadata.crs
    col = window_metadata.col
    row = window_metadata.row
    fname = f"{crs}_{col}_{row}_{resolution}.{ext}"
    return modality_dir / fname
