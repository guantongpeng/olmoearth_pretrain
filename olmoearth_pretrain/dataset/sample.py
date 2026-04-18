"""OlmoEarth Pretrain 训练样本构建模块。

本模块基于解析后的 CSV 信息，跨模态综合构建训练样本。
确定每个训练样本所需加载的数据，包括图像裁剪、重采样和坐标转换。

主要类:
    SampleInformation: 训练样本信息，包含网格位置、时间跨度和可用模态

主要函数:
    image_tiles_to_samples(): 将解析的模态瓦片信息转换为训练样本列表
    load_image_for_sample(): 加载样本对应的图像数据（支持裁剪和重采样）

使用场景:
    1. 在数据解析后，使用 image_tiles_to_samples() 构建样本列表。
    2. 在数据加载时，使用 load_image_for_sample() 读取图像数据。
"""

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterio
import rasterio.windows
from pyproj import Transformer

from olmoearth_pretrain.data.constants import (
    BASE_RESOLUTION,
    IMAGE_TILE_SIZE,
    PROJECTION_CRS,
    Modality,
    ModalitySpec,
    TimeSpan,
)

from .parse import GridTile, ModalityTile

logger = logging.getLogger(__name__)


@dataclass
class SampleInformation:
    """训练样本信息，对应数据集中的一个 GridTile。

    包含加载该瓦片位置所有模态数据所需的信息，以及从包含该瓦片的
    更粗分辨率瓦片中裁剪所需的信息。

    关键属性:
        grid_tile: 网格瓦片位置信息
        time_span: 时间跨度（YEAR 或 TWO_WEEK，不应为 STATIC）
        modalities: 该网格瓦片或更粗瓦片中可用的模态映射

    使用场景:
        由 image_tiles_to_samples() 创建，用于确定训练样本的数据加载方式。
    """

    grid_tile: GridTile

    # 该训练样本覆盖一年（TimeSpan.YEAR）还是两周（TimeSpan.TWO_WEEK）的时间段
    # 注意：time_span 不应为 TimeSpan.STATIC，因为训练样本总是与特定时间范围关联
    time_span: TimeSpan

    # 该网格瓦片或包含该瓦片的更粗瓦片上可用的模态
    # ModalityTile 的时间跨度应与样本的时间跨度匹配，或为 TimeSpan.STATIC
    modalities: dict[ModalitySpec, ModalityTile]

    def get_latlon(self) -> np.ndarray:
        """获取样本的经纬度坐标。

        核心逻辑:
            1. 根据网格瓦片的分辨率和行列号计算投影坐标
            2. 使用 pyproj 将投影坐标转换为经纬度

        Returns:
            np.ndarray: [纬度, 经度] 数组
        """
        # 获取投影坐标单位的坐标，然后转换为经纬度
        grid_resolution = self.grid_tile.resolution_factor * BASE_RESOLUTION
        x, y = (
            (self.grid_tile.col + 0.5) * grid_resolution * IMAGE_TILE_SIZE,
            (self.grid_tile.row + 0.5) * -grid_resolution * IMAGE_TILE_SIZE,
        )
        transformer = Transformer.from_crs(
            self.grid_tile.crs, PROJECTION_CRS, always_xy=True
        )
        lon, lat = transformer.transform(x, y)
        return np.array([lat, lon])

    def get_timestamps(self) -> dict[ModalitySpec, np.ndarray]:
        """获取样本中各多时相模态的时间戳。

        核心逻辑:
            对每个多时相模态，提取所有图像的起始时间，
            转换为 [日, 月-1, 年] 格式的数组。

        Returns:
            dict[ModalitySpec, np.ndarray]: 模态到时间戳数组的映射，
                每个时间戳数组的形状为 (T, 3)，列为 [day, month-1, year]
        """
        timestamps_dict: dict[ModalitySpec, np.ndarray] = {}

        for modality in self.modalities:
            if modality.is_multitemporal:
                sample_modality = self.modalities[modality]
                timestamps = [i.start_time for i in sample_modality.images]
                dt = pd.to_datetime(timestamps)
                # 将时间戳转换为 [日, 月-1, 年] 格式的数组，月从0开始
                timestamps_dict[modality] = np.array([dt.day, dt.month - 1, dt.year]).T

        return timestamps_dict


def image_tiles_to_samples(
    image_tiles: dict[ModalitySpec, dict[TimeSpan, list[ModalityTile]]],
    supported_modalities: list[ModalitySpec] = Modality.values(),
) -> list[SampleInformation]:
    """从解析的模态瓦片信息计算训练样本。

    核心逻辑:
        1. 将 (模态 -> 时间跨度 -> 瓦片列表) 转换为索引映射
        2. 枚举数据集中所有唯一的 (grid_tile, time_span) 元组
        3. 对每个唯一元组，查找所有可用模态（支持跨分辨率查找）
        4. 构建 SampleInformation 对象

    Args:
        image_tiles: 由 parse_dataset() 解析得到的数据集瓦片信息
        supported_modalities: 需要包含在样本中的模态列表，默认为所有模态

    Returns:
        list[SampleInformation]: 训练样本信息列表
    """
    # 将 (模态 -> 时间跨度 -> 瓦片列表) 转换为 (模态, 网格瓦片, 时间跨度) -> 瓦片 的索引映射
    image_tile_index: dict[tuple[ModalitySpec, GridTile, TimeSpan], ModalityTile] = {}
    for modality, modality_tiles in image_tiles.items():
        for time_span, time_span_tiles in modality_tiles.items():
            for tile in time_span_tiles:
                index_key = (modality, tile.grid_tile, time_span)
                image_tile_index[index_key] = tile

    # 枚举数据集中所有唯一的 (grid_tile, time_span) 元组
    # 每个元组标识一个训练样本
    # 此处忽略 STATIC 时间跨度，除非它在基础分辨率上，此时将其同时添加为
    # YEAR 和 TWO_WEEK（因为当前基础分辨率上的所有数据都是静态的）
    # （目的是避免仅因 WorldCover 可用就添加两周瓦片，
    #  而 Sentinel-2 等只有年度数据，但仍添加 NAIP 或 Maxar 瓦片）
    unique_image_tiles: set[tuple[GridTile, TimeSpan]] = set()
    for modality, grid_tile, time_span in image_tile_index.keys():
        if time_span == TimeSpan.STATIC:
            if grid_tile.resolution_factor > 1:
                logger.debug(
                    f"ignoring static tile {grid_tile.resolution_factor} "
                    f"because it is coarser than the base resolution for modality {modality.name}"
                )
                continue
            else:
                unique_image_tiles.add((grid_tile, TimeSpan.TWO_WEEK))  # type: ignore
                unique_image_tiles.add((grid_tile, TimeSpan.YEAR))  # type: ignore
        else:
            unique_image_tiles.add((grid_tile, time_span))  # type: ignore

    # 对每个 (grid_tile, time_span)，构建 Sample 对象
    # 如果不是所有模态都可用，则跳过
    samples: list[SampleInformation] = []
    for grid_tile, time_span in unique_image_tiles:
        sample = SampleInformation(
            grid_tile=grid_tile,
            time_span=time_span,
            modalities={},
        )

        # 逐个添加模态
        for modality in image_tiles.keys():
            if modality not in supported_modalities:
                logger.warning(
                    f"ignoring modality {modality.name} not in supported_modalities"
                )
                continue
            # 只使用分辨率等于或更粗的模态
            if modality.tile_resolution_factor < sample.grid_tile.resolution_factor:
                logger.debug(
                    f"ignoring modality {modality.name} with resolution factor "
                    f"{modality.tile_resolution_factor} because it is coarser than "
                    f"the sample grid tile resolution factor {sample.grid_tile.resolution_factor}"
                )
                continue

            downscale_factor = (
                modality.tile_resolution_factor // sample.grid_tile.resolution_factor
            )

            # 检查是否有该模态的可用图像瓦片
            # 如果模态是静态的，则使用 TimeSpan.STATIC 进行查找
            # 如果模态是多时相的，则使用样本的时间跨度进行查找
            lookup_time_span: TimeSpan
            if modality.is_multitemporal:
                lookup_time_span = sample.time_span  # type: ignore
            else:
                lookup_time_span = TimeSpan.STATIC  # type: ignore

            # 需要将网格瓦片下采样以进行查找
            modality_grid_tile = GridTile(
                crs=grid_tile.crs,
                resolution_factor=modality.tile_resolution_factor,
                col=grid_tile.col // downscale_factor,
                row=grid_tile.row // downscale_factor,
            )

            index_key = (modality, modality_grid_tile, lookup_time_span)
            if index_key not in image_tile_index:
                logger.debug(
                    f"ignoring modality {modality.name} because no tile found for index_key={index_key}"
                )
                continue
            image_tile = image_tile_index[index_key]

            # 找到瓦片，将其添加到样本的模态映射中
            # ModalityTile 对象包含加载图像所需的所有信息（可能需要裁剪）
            sample.modalities[modality] = image_tile

        samples.append(sample)
    return samples


def load_image_for_sample(
    image_tile: ModalityTile, sample: SampleInformation
) -> npt.NDArray:
    """加载样本对应的图像数据。

    核心逻辑:
        1. 如果图像瓦片和样本分辨率相同，加载整幅图像
        2. 如果图像瓦片分辨率更粗，加载与样本对齐的裁剪区域
        3. 对非空间模态（如 ERA5），读取完整瓦片并移除空间维度
        4. 根据需要进行上采样或下采样以匹配目标分辨率

    注意：样本不能比图像瓦片分辨率更粗，否则需要读取多个瓦片并下采样。

    Args:
        image_tile: 要加载的图像瓦片信息
        sample: 训练样本信息，用于确定加载整幅图像还是部分区域

    Returns:
        npt.NDArray: TCHW 格式的图像数据（时间维度在第一维）
    """
    # 计算图像瓦片比样本更粗（分辨率因子更大）的倍数
    factor = (
        image_tile.grid_tile.resolution_factor // sample.grid_tile.resolution_factor
    )
    # 逐个波段集读取模态图像
    # 目前将所有波段重采样到模态的网格分辨率
    band_set_images = []
    for band_set, fname in image_tile.band_sets.items():
        logger.debug(f"band_set={band_set}, fname={fname}")
        with fname.open("rb") as f:
            with rasterio.open(f) as raster:
                # 确定需要读取的瓦片部分，称为子瓦片(subtile)
                if raster.width != raster.height:
                    raise ValueError(
                        f"expected tile to be square but width={raster.width} != height={raster.height}"
                    )
                # 如果模态不随空间变化（如 ERA5），读取整个瓦片
                if not image_tile.modality.is_spatial:
                    logger.debug(
                        f"reading entire tile {fname} for modality {image_tile.modality.name}"
                    )
                    image: npt.NDArray = raster.read()
                    # 移除空间维度，因为不需要
                    image = image.reshape(-1, len(band_set.bands))
                    band_set_images.append(image)
                    continue

                # 假设所有瓦片覆盖与分辨率因子16瓦片相同的区域
                subtile_size = raster.width // factor
                col_offset = subtile_size * (sample.grid_tile.col % factor)
                row_offset = subtile_size * (sample.grid_tile.row % factor)

                # 执行窗口读取
                rasterio_window = rasterio.windows.Window(
                    col_off=col_offset,
                    row_off=row_offset,
                    width=subtile_size,
                    height=subtile_size,
                )
                logger.debug(f"reading window={rasterio_window} from {fname}")
                image: npt.NDArray = raster.read(window=rasterio_window)  # type: ignore
                logger.debug(f"image.shape={image.shape}")

                # 然后重采样到网格分辨率
                # 分辨率差异应始终是2的幂次
                # 如果因子小于1，则所需大小应乘以该因子
                # 如果瓦片大小更大，则保持该范围
                desired_subtile_size = int(
                    IMAGE_TILE_SIZE
                    * image_tile.modality.image_tile_size_factor
                    // factor
                )
                if desired_subtile_size < subtile_size:
                    # 需要下采样
                    # 这不常见，通常波段以图像瓦片分辨率或更低存储
                    # 但 OpenStreetMap 可能出现此情况
                    # 使用子采样而非平均，因为平均对 OSM 数据不正确
                    downscale_factor = subtile_size // desired_subtile_size
                    image = image[:, ::downscale_factor, ::downscale_factor]
                elif desired_subtile_size > subtile_size:
                    logger.debug(
                        f"desired_subtile_size={desired_subtile_size}, subtile_size={subtile_size}"
                    )
                    # 更常见的情况是需要上采样，因为某些波段以较低分辨率存储
                    # 例如 Sentinel-2 或 Landsat
                    upscale_factor = desired_subtile_size // subtile_size
                    image = image.repeat(repeats=upscale_factor, axis=1).repeat(
                        repeats=upscale_factor, axis=2
                    )

                # 分离时间和通道维度
                shape = (
                    -1,
                    len(band_set.bands),
                    desired_subtile_size,
                    desired_subtile_size,
                )
                image = image.reshape(shape)
                logger.debug(f"shape after scaling image.shape={image.shape}")
                band_set_images.append(image)

    return np.concatenate(band_set_images, axis=1)
