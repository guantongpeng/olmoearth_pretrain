"""OlmoEarth Pretrain 数据集解析模块。

本模块负责解析 OlmoEarth Pretrain 数据集中的 CSV 元数据文件，
识别各种模态在哪些瓦片(tiles)位置上有可用数据。

主要类:
    ModalityImage: 模态图像信息，包含时间范围
    GridTile: 网格瓦片位置信息，包含 CRS、分辨率因子、行列号
    ModalityTile: 模态瓦片信息，包含网格位置、图像列表、波段集等

主要函数:
    parse_modality_csv(): 解析单个模态的 CSV 文件
    parse_dataset(): 解析整个数据集的所有模态

使用场景:
    作为数据加载流程的第一步，解析 CSV 元数据以确定数据的位置和可用性。
"""

import csv
import logging
from dataclasses import dataclass
from datetime import datetime

from upath import UPath

from olmoearth_pretrain.data.constants import (
    BASE_RESOLUTION,
    BandSet,
    Modality,
    ModalitySpec,
    TimeSpan,
)

from .utils import WindowMetadata, get_modality_fname

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModalityImage:
    """模态图像信息，描述瓦片中堆叠图像时间序列的单个图像。

    每个瓦片包含堆叠的图像时间序列，此数据类记录序列中
    每个图像的起始和结束时间。

    关键属性:
        start_time: 图像的起始时间
        end_time: 图像的结束时间
    """

    start_time: datetime
    end_time: datetime

    # 添加此方法以检查两个 ModalityImage 对象是否相同
    def __eq__(self, other: object) -> bool:
        """检查两个 ModalityImage 对象是否相同（起始和结束时间都相等）。

        Args:
            other: 另一个对象

        Returns:
            bool: 如果是 ModalityImage 且时间范围相同则返回 True
        """
        if not isinstance(other, ModalityImage):
            return False
        return self.start_time == other.start_time and self.end_time == other.end_time


@dataclass(frozen=True)
class GridTile:
    """网格瓦片位置信息，描述瓦片在特定分辨率网格上的位置。

    关键属性:
        crs: 坐标参考系统，例如 EPSG:32610
        resolution_factor: 相对于 BASE_RESOLUTION 的分辨率因子
        col: 基于分辨率因子定义的网格列号
        row: 基于分辨率因子定义的网格行号
    """


@dataclass
class ModalityTile:
    """模态瓦片信息，描述某个模态在特定瓦片位置的数据。

    关键属性:
        grid_tile: 网格瓦片位置信息
        images: 该瓦片上图像的时间序列列表
        center_time: 定义瓦片时间范围的中心时间
        band_sets: 波段集到对应文件路径的映射
        modality: 模态规格信息

    使用场景:
        在数据解析阶段由 parse_modality_csv() 创建，
        在采样阶段用于确定加载哪些数据。
    """

    grid_tile: GridTile
    images: list[ModalityImage]

    # The center time that defines the time ranges for this tile.
    center_time: datetime

    # The band sets along with the file containing them.
    band_sets: dict[BandSet, UPath]

    modality: ModalitySpec

    def get_flat_bands(self) -> list[str]:
        """获取所有波段名称的扁平列表。

        对应于将波段集合并为单个张量时的波段顺序。

        Returns:
            list[str]: 所有波段名称的列表，按波段集顺序排列
        """
        bands: list[str] = []
        for band_set in self.band_sets:
            bands.extend(band_set.bands)
        return bands


def parse_modality_csv(
    path: UPath, modality: ModalitySpec, time_span: TimeSpan, csv_path: UPath
) -> list[ModalityTile]:
    """解析单个模态和时间跨度的 CSV 文件。

    核心逻辑:
        1. 读取 CSV 文件，每行对应一个图像
        2. 根据 CRS、分辨率因子、行列号构建 GridTile
        3. 根据起始/结束时间构建 ModalityImage
        4. 将同一 GridTile 的图像聚合到同一个 ModalityTile
        5. 填充每个波段的文件路径

    Args:
        path: OlmoEarth Pretrain 数据集根路径
        modality: 要解析的模态规格
        time_span: 要解析的时间跨度
        csv_path: CSV 文件路径

    Returns:
        list[ModalityTile]: 解析得到的模态瓦片列表
    """
    # 首先获取瓦片及其中的图像
    # 然后填充波段集和图像路径
    modality_tiles: dict[GridTile, ModalityTile] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for csv_row in reader:
            grid_tile = GridTile(
                crs=csv_row["crs"],
                resolution_factor=modality.tile_resolution_factor,
                col=int(csv_row["col"]),
                row=int(csv_row["row"]),
            )
            image = ModalityImage(
                start_time=datetime.fromisoformat(csv_row["start_time"]),
                end_time=datetime.fromisoformat(csv_row["end_time"]),
            )
            image_idx = int(csv_row["image_idx"])
            if grid_tile not in modality_tiles:
                modality_tiles[grid_tile] = ModalityTile(
                    grid_tile=grid_tile,
                    images=[],
                    center_time=datetime.fromisoformat(csv_row["tile_time"]),
                    band_sets={},
                    modality=modality,
                )

            # 该图像应该出现在上述索引处，但索引应该在 CSV 中按顺序排列
            if image_idx != len(modality_tiles[grid_tile].images):
                # 这应该是一个错误，但发现原始 rslearn 数据集中
                # 有一两个瓦片实际有两个时间戳，这意味着 OlmoEarth Pretrain
                # 数据集在 CSV 中有两行记录，但实际只有一个文件
                # raise ValueError(
                #    "expected image index to be in increasing order and contiguous"
                # )
                continue
            modality_tiles[grid_tile].images.append(image)

    # 现在可以填充波段集
    # 同时检查图像列表中没有 None 值
    for tile in modality_tiles.values():
        grid_tile = tile.grid_tile
        window_metadata = WindowMetadata(
            crs=grid_tile.crs,
            resolution=BASE_RESOLUTION * grid_tile.resolution_factor,
            col=grid_tile.col,
            row=grid_tile.row,
            time=tile.center_time,
        )
        for band_set in modality.band_sets:
            fname = get_modality_fname(
                path,
                modality,
                time_span,
                window_metadata,
                band_set.get_resolution(),
                "tif",
            )
            tile.band_sets[band_set] = fname

    return list(modality_tiles.values())


def parse_dataset(
    path: UPath, supported_modalities: list[ModalitySpec] = Modality.values()
) -> dict[ModalitySpec, dict[TimeSpan, list[ModalityTile]]]:
    """解析 OlmoEarth Pretrain 数据集中各模态的瓦片信息。

    核心逻辑:
        1. 遍历所有模态（跳过标记为忽略和不支持的模态）
        2. 根据模态类型确定时间跨度（多时相用 YEAR，静态用 STATIC）
        3. 对每个时间跨度，解析对应的 CSV 文件获取 ModalityTile 列表

    Args:
        path: OlmoEarth Pretrain 数据集根路径
        supported_modalities: 需要支持的模态列表，默认为所有模态

    Returns:
        dict[ModalitySpec, dict[TimeSpan, list[ModalityTile]]]:
            模态 -> 时间跨度 -> 模态瓦片列表的嵌套映射
    """
    tiles: dict[ModalitySpec, dict[TimeSpan, list[ModalityTile]]] = {}

    for modality in Modality.values():
        if modality.ignore_when_parsing:
            continue
        if modality not in supported_modalities:
            logger.warning(
                f"ignoring modality {modality.name} not in supported_modalities"
            )
            continue

        if modality.is_multitemporal:
            # 需要分别加载年度和双周数据（当前仅使用年度数据）
            time_spans = [TimeSpan.YEAR]  # [TimeSpan.YEAR, TimeSpan.TWO_WEEK]
        else:
            # 只需要加载静态数据
            time_spans = [TimeSpan.STATIC]

        # 对每个可用的时间跨度，解析对应的 CSV 文件以获取 ModalityTile 列表
        tiles[modality] = {}
        for time_span in time_spans:
            # 从网格分辨率、模态和时间跨度重建 CSV 文件名
            tile_resolution = modality.get_tile_resolution()
            csv_fname = (
                path / f"{tile_resolution}_{modality.name}{time_span.get_suffix()}.csv"  # type: ignore
            )
            logger.debug(f"Parsing {modality.name} {time_span} {csv_fname}")
            tiles[modality][time_span] = parse_modality_csv(  # type: ignore
                path,
                modality,
                time_span,  # type: ignore
                csv_fname,
            )

    return tiles
