"""
OlmoEarth Pretrain 包中共享的常量定义模块。

本模块定义了数据管线中使用的核心常量和数据结构，包括：
- 分辨率相关常量（BASE_RESOLUTION、IMAGE_TILE_SIZE 等）
- BandSet：波段组，描述在同一分辨率下存储的一组波段
- TimeSpan：时间范围枚举，区分静态、年度和双周数据
- ModalitySpec：模态规格，描述一种数据模态的完整元信息（名称、分辨率、波段组、是否多时相等）
- Modality：模态枚举类，提供对所有预定义模态规格的便捷访问

注意：当前仅针对栅格数据（raster data）开发。
"""

from dataclasses import dataclass
from enum import Enum

# 基础分辨率（单位：度），即系统处理的最高分辨率。
# 所有其他分辨率均为此分辨率的某个 2 的幂次倍数（更粗糙）。
BASE_RESOLUTION = 0.625

# 默认图像瓦片尺寸（像素）。
# 某些图像如果以比网格基准分辨率更粗的分辨率存储，则尺寸可能更小。
IMAGE_TILE_SIZE = 256

# 投影坐标系，采用 WGS84 经纬度坐标系
PROJECTION_CRS = "EPSG:4326"

# 栅格数据的默认缺失值标识
MISSING_VALUE = -99999

# 默认最大时间序列长度（时间步数）
MAX_SEQUENCE_LENGTH = 12

# 输入数据的地面采样距离（Ground Sample Distance），单位为米
BASE_GSD = 10
# Sentinel-1 数据的默认无数据值
SENTINEL1_NODATA = -32768

# 年度数据的时间步数（12 个月）
YEAR_NUM_TIMESTEPS = 12


def get_resolution(resolution_factor: int) -> float | int:
    """根据分辨率因子计算实际分辨率值。

    实际分辨率 = BASE_RESOLUTION * resolution_factor。
    如果计算结果为整数，则返回 int 类型（兼容原始 OlmoEarth Pretrain 数据集中
    基于整数命名的文件）；否则返回 float。

    Args:
        resolution_factor: 分辨率因子，为 2 的幂次倍数。

    Returns:
        计算得到的分辨率值，整数或浮点数。
    """
    resolution = BASE_RESOLUTION * resolution_factor  # 基础分辨率乘以因子
    if float(int(resolution)) == resolution:  # 判断结果是否为整数
        return int(resolution)
    return resolution


@dataclass(frozen=True)
class BandSet:
    """波段组（BandSet），表示在同一分辨率下存储的一组波段。

    许多模态只有一个波段组，但某些模态的不同波段以不同分辨率存储，
    因此需要多个 BandSet。例如 Sentinel-2 有 10m、20m、60m 三组波段。

    属性:
        bands: 波段名称列表，如 ["R", "G", "B", "IR"]。
        resolution_factor: 分辨率因子，实际分辨率 = BASE_RESOLUTION * resolution_factor。
            若为 0，表示数据在空间上不变（如经纬度 latlons）。
    """

    # 波段名称列表
    bands: list[str]

    # 分辨率因子，实际分辨率 = BASE_RESOLUTION * resolution_factor
    # 若 resolution_factor == 0，表示该数据不随空间变化（如经纬度）
    resolution_factor: int

    def __hash__(self) -> int:
        """计算 BandSet 的哈希值，基于波段元组和分辨率因子。"""
        return hash((tuple(self.bands), self.resolution_factor))

    def get_resolution(self) -> float:
        """计算该波段组的实际分辨率。"""
        return get_resolution(self.resolution_factor)

    def get_expected_image_size(self, modality_resolution_factor: int) -> int:
        """获取包含这些波段的图像的预期尺寸。

        根据模态分辨率因子和该波段组的分辨率因子之间的比值，
        计算图像瓦片在空间维度上的像素尺寸。

        Args:
            modality_resolution_factor: 模态的分辨率因子。

        Returns:
            预期的图像尺寸（像素）。
        """
        # 图像瓦片尺寸除以分辨率因子的比值，得到该波段组的图像尺寸
        return IMAGE_TILE_SIZE // (self.resolution_factor // modality_resolution_factor)


class TimeSpan(str, Enum):
    """时间范围枚举，用于区分不同有效时间范围的数据。

    - STATIC: 静态数据，仅有一个数据点（非时间序列）
    - YEAR: 年度数据，每月一个数据点，共 12 个
    - TWO_WEEK: 双周数据，两周内每个时间点都有数据
    """

    # 静态数据，仅有一个数据点（非时间序列）
    STATIC = "static"

    # 年度数据，每月一个数据点
    YEAR = "year"

    # 双周数据，两周内每个时间点都有数据
    TWO_WEEK = "two_week"

    def get_suffix(self) -> str:
        """获取该时间范围在原始 OlmoEarth Pretrain 数据集中使用的文件名后缀。

        Returns:
            对应的文件名后缀字符串。STATIC 返回空字符串，YEAR 返回 "_monthly"，TWO_WEEK 返回 "_freq"。

        Raises:
            ValueError: 如果遇到无效的 TimeSpan 值。
        """
        if self == TimeSpan.STATIC:
            return ""  # 静态数据无后缀
        if self == TimeSpan.YEAR:
            return "_monthly"  # 年度数据后缀
        if self == TimeSpan.TWO_WEEK:
            return "_freq"  # 双周数据后缀
        raise ValueError("invalid TimeSpan")


@dataclass(frozen=True)
class ModalitySpec:
    """模态规格（ModalitySpec），描述一种数据模态的完整元信息。

    模态规格定义了数据模态的名称、分辨率、波段组、是否多时相等关键属性，
    是数据加载和处理管线的核心配置单元。

    属性:
        name: 模态名称，如 "sentinel2"、"naip"、"srtm" 等。
        tile_resolution_factor: 瓦片分辨率因子，描述该模态瓦片覆盖的地面面积
            相对于基准分辨率下 IMAGE_TILE_SIZE x IMAGE_TILE_SIZE 像素瓦片的倍数。
        band_sets: 波段组列表，即 tokenization 的基本单元。
        is_multitemporal: 是否为多时相数据（有时间维度）。
        ignore_when_parsing: 解析 CSV 文件时是否忽略该模态（派生模态如 NDVI 不从文件加载）。
        image_tile_size_factor: 图像瓦片尺寸因子，描述该模态的图像瓦片尺寸
            相对于基准瓦片尺寸的倍数（默认为 1）。负值表示缩小。
    """

    name: str
    tile_resolution_factor: int
    band_sets: list[BandSet]
    is_multitemporal: bool
    ignore_when_parsing: bool  # 若为 True，则不从 CSV 文件解析，也不从文件加载
    image_tile_size_factor: int = 1

    def __hash__(self) -> int:
        """计算模态的哈希值，基于模态名称。"""
        return hash(self.name)

    def get_tile_resolution(self) -> float:
        """计算瓦片的实际分辨率。"""
        return get_resolution(self.tile_resolution_factor)

    def bandsets_as_indices(self) -> list[list[int]]:
        """将波段组转换为通道索引列表。

        返回每个波段组对应的通道索引范围，例如 [[0, 1, 2, 3], [4, 5, 6, 7, 8, 9]]。

        Returns:
            波段组索引列表，每个子列表包含该波段组内波段的通道索引。
        """
        indices = []
        offset = 0  # 通道偏移量，累计已处理的波段数
        for band_set in self.band_sets:
            num_bands = len(band_set.bands)
            indices.append(list(range(offset, offset + num_bands)))
            offset += num_bands  # 更新偏移量
        return indices

    @property
    def band_order(self) -> list[str]:
        """获取所有波段的有序列表（按波段组顺序展平）。"""
        return sum((list(band_set.bands) for band_set in self.band_sets), [])

    @property
    def num_band_sets(self) -> int:
        """获取波段组的数量。"""
        return len(self.band_sets)

    @property
    def num_bands(self) -> int:
        """获取通道总数（所有波段组中波段数的总和）。"""
        return sum(len(band_set.bands) for band_set in self.band_sets)

    def get_expected_tile_size(self) -> int:
        """获取该模态的预期瓦片尺寸。

        当 image_tile_size_factor 为负数时，瓦片尺寸缩小；
        为正数时，瓦片尺寸放大。

        Returns:
            预期的瓦片边长（像素）。
        """
        if self.image_tile_size_factor < 0:
            # 负值表示缩小：瓦片尺寸 = 基准尺寸 / |因子|
            return IMAGE_TILE_SIZE // abs(self.image_tile_size_factor)
        else:
            # 正值表示放大：瓦片尺寸 = 基准尺寸 * 因子
            return IMAGE_TILE_SIZE * self.image_tile_size_factor

    @property
    def is_spatial(self) -> bool:
        """判断该模态是否具有空间数据（空间分辨率 > 0 且瓦片尺寸 > 1）。"""
        # 瓦片尺寸必须大于 1 才具有空间变化数据
        return self.get_tile_resolution() > 0 and self.get_expected_tile_size() > 1

    @property
    def is_spacetime_varying(self) -> bool:
        """判断该模态是否在空间和时间上均有变化。"""
        return self.is_spatial and self.is_multitemporal

    @property
    def is_space_only_varying(self) -> bool:
        """判断该模态是否仅在空间上变化（时间不变）。"""
        return self.is_spatial and not self.is_multitemporal

    @property
    def is_time_only_varying(self) -> bool:
        """判断该模态是否仅在时间上变化（空间不变）。"""
        return not self.is_spatial and self.is_multitemporal

    @property
    def is_static_in_space_and_time(self) -> bool:
        """判断该模态在空间和时间上均无变化。"""
        return not self.is_spatial and not self.is_multitemporal


class Modality:
    """模态枚举类，提供对所有预定义 ModalitySpec 的类属性访问。

    通过 Modality.SENTINEL2 等方式可直接获取对应模态的 ModalitySpec 实例。
    支持的模态包括：
    - NAIP: 美国高分辨率航拍影像（1m/pixel，4波段 RGB+IR）
    - NAIP_10: 覆盖更大范围的 NAIP 数据（tile_resolution_factor=16）
    - SENTINEL1: Sentinel-1 SAR 数据（VV/VH 极化，多时相）
    - SENTINEL2: Sentinel-2 MSI 数据（13 波段，分三组分辨率，多时相）
    - SENTINEL2_L2A: Sentinel-2 L2A 大气校正数据（12 波段，多时相）
    - LANDSAT: Landsat 数据（11 波段，分两组分辨率，多时相）
    - WORLDCOVER: ESA WorldCover 土地覆盖分类（单波段，静态）
    - WORLDCEREAL: WorldCereal 作物分类（8 波段，静态）
    - SRTM: SRTM 数字高程模型（单波段，静态）
    - OPENSTREETMAP: OpenStreetMap 矢量数据（30 波段，静态，不直接解析）
    - OPENSTREETMAP_RASTER: OpenStreetMap 栅格化数据（30 波段，静态）
    - ERA5: ERA5 再分析气象数据（6 波段，多时相，不直接解析）
    - ERA5_10: ERA5 降采样版本（6 波段，多时相）
    - LATLON: 经纬度坐标（2 波段，静态，不直接解析）
    - GSE: GSE 数据（64 波段，静态）
    - CDL: 美国作物数据层（单波段，静态）
    - WORLDPOP: 世界人口密度（单波段，静态）
    - WRI_CANOPY_HEIGHT_MAP: WRI 冠层高度图（单波段，静态）
    - NDVI: 归一化植被指数（单波段，多时相，由 S2 L2A 计算派生）
    - EUROCROPS: 欧洲作物分类（单波段，静态）
    """

    # NAIP 航拍影像，1m/pixel，4 波段（R, G, B, IR），静态
    NAIP = ModalitySpec(
        name="naip",
        tile_resolution_factor=1,
        band_sets=[BandSet(["R", "G", "B", "IR"], 1)],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )

    # NAIP_10 覆盖与 10m/pixel 分辨率下 IMAGE_TILE_SIZE x IMAGE_TILE_SIZE 瓦片相同范围的 NAIP 数据，
    # 但仍以 NAIP 原始分辨率存储。image_tile_size_factor=4 以降低实际图像尺寸方便训练。
    NAIP_10 = ModalitySpec(
        name="naip_10",
        tile_resolution_factor=16,
        band_sets=[BandSet(["R", "G", "B", "IR"], 1)],
        is_multitemporal=False,
        ignore_when_parsing=False,
        # 设为 4x (2.5 m/pixel) 以使训练更可行，得到 512x512 的 NAIP 图像
        # 而非 2048x2048（后者会拖慢数据加载）
        image_tile_size_factor=4,
    )

    # Sentinel-1 SAR 数据，VV/VH 极化，16x 分辨率因子，多时相
    SENTINEL1 = ModalitySpec(
        name="sentinel1",
        tile_resolution_factor=16,
        band_sets=[BandSet(["vv", "vh"], 16)],
        is_multitemporal=True,
        ignore_when_parsing=False,
    )

    # Sentinel-2 MSI 数据，13 波段分三组分辨率，多时相
    SENTINEL2 = ModalitySpec(
        name="sentinel2",
        tile_resolution_factor=16,
        band_sets=[
            # 10 m/pixel 波段（B02 蓝、B03 绿、B04 红、B08 近红外）
            BandSet(["B02", "B03", "B04", "B08"], 16),
            # 20 m/pixel 波段（植被红边、短波红外等）
            BandSet(["B05", "B06", "B07", "B8A", "B11", "B12"], 32),
            # 60 m/pixel 波段，实际存储为 40 m/pixel
            BandSet(["B01", "B09", "B10"], 64),
        ],
        is_multitemporal=True,
        ignore_when_parsing=False,
    )

    # Sentinel-2 L2A 大气校正数据，12 波段分三组分辨率，多时相
    SENTINEL2_L2A = ModalitySpec(
        name="sentinel2_l2a",
        tile_resolution_factor=16,
        band_sets=[
            # 10 m/pixel 波段
            BandSet(["B02", "B03", "B04", "B08"], 16),
            # 20 m/pixel 波段
            BandSet(["B05", "B06", "B07", "B8A", "B11", "B12"], 32),
            # 60 m/pixel 波段，实际存储为 40 m/pixel
            BandSet(["B01", "B09"], 64),
        ],
        is_multitemporal=True,
        ignore_when_parsing=False,
    )

    # Landsat 数据，11 波段分两组分辨率，多时相
    LANDSAT = ModalitySpec(
        name="landsat",
        tile_resolution_factor=16,
        band_sets=[
            # 15 m/pixel 全色波段，实际存储为 10 m/pixel
            BandSet(["B8"], 16),
            # 30 m/pixel 多光谱波段，实际存储为 20 m/pixel
            BandSet(["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"], 32),
        ],
        is_multitemporal=True,
        ignore_when_parsing=False,
    )

    # ESA WorldCover 土地覆盖分类，单波段，静态
    WORLDCOVER = ModalitySpec(
        name="worldcover",
        tile_resolution_factor=16,
        band_sets=[BandSet(["B1"], 16)],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )

    # WorldCereal 作物分类，8 波段，静态
    WORLDCEREAL = ModalitySpec(
        name="worldcereal",
        tile_resolution_factor=16,
        band_sets=[
            BandSet(
                [
                    "tc-annual-temporarycrops-classification",
                    "tc-maize-main-irrigation-classification",
                    "tc-maize-main-maize-classification",
                    "tc-maize-second-irrigation-classification",
                    "tc-maize-second-maize-classification",
                    "tc-springcereals-springcereals-classification",
                    "tc-wintercereals-irrigation-classification",
                    "tc-wintercereals-wintercereals-classification",
                ],
                16,
            )
        ],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )

    # SRTM 数字高程模型，单波段，静态
    SRTM = ModalitySpec(
        name="srtm",
        tile_resolution_factor=16,
        band_sets=[BandSet(["srtm"], 16)],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )

    # OpenStreetMap 矢量数据，30 波段，静态，不直接从文件解析
    OPENSTREETMAP = ModalitySpec(
        name="openstreetmap",
        tile_resolution_factor=16,
        band_sets=[
            BandSet(
                [
                    "aerialway_pylon",
                    "aerodrome",
                    "airstrip",
                    "amenity_fuel",
                    "building",
                    "chimney",
                    "communications_tower",
                    "crane",
                    "flagpole",
                    "fountain",
                    "generator_wind",
                    "helipad",
                    "highway",
                    "leisure",
                    "lighthouse",
                    "obelisk",
                    "observatory",
                    "parking",
                    "petroleum_well",
                    "power_plant",
                    "power_substation",
                    "power_tower",
                    "river",
                    "runway",
                    "satellite_dish",
                    "silo",
                    "storage_tank",
                    "taxiway",
                    "water_tower",
                    "works",
                ],
                1,
            )
        ],
        is_multitemporal=False,
        ignore_when_parsing=True,  # 矢量版 OSM 不直接解析加载
    )

    # OpenStreetMap 栅格化数据，30 波段，分辨率因子 4，静态
    OPENSTREETMAP_RASTER = ModalitySpec(
        name="openstreetmap_raster",
        tile_resolution_factor=16,
        band_sets=[
            BandSet(
                [
                    "aerialway_pylon",
                    "aerodrome",
                    "airstrip",
                    "amenity_fuel",
                    "building",
                    "chimney",
                    "communications_tower",
                    "crane",
                    "flagpole",
                    "fountain",
                    "generator_wind",
                    "helipad",
                    "highway",
                    "leisure",
                    "lighthouse",
                    "obelisk",
                    "observatory",
                    "parking",
                    "petroleum_well",
                    "power_plant",
                    "power_substation",
                    "power_tower",
                    "river",
                    "runway",
                    "satellite_dish",
                    "silo",
                    "storage_tank",
                    "taxiway",
                    "water_tower",
                    "works",
                ],
                4,
            )
        ],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )

    # ERA5 再分析气象数据，6 波段（温度、露点、气压、风速、降水），多时相，不直接解析
    ERA5 = ModalitySpec(
        name="era5",
        # 9 km/pixel 波段，实际存储为 150 m/pixel
        tile_resolution_factor=256,
        band_sets=[
            BandSet(
                [
                    "2m-temperature",
                    "2m-dewpoint-temperature",
                    "surface-pressure",
                    "10m-u-component-of-wind",
                    "10m-v-component-of-wind",
                    "total-precipitation",
                ],
                256,
            ),
        ],
        is_multitemporal=True,
        ignore_when_parsing=True,  # ERA5 原始版本不直接解析
    )

    # ERA5 降采样版本，6 波段，存储为 2.56 km/pixel，多时相
    ERA5_10 = ModalitySpec(
        name="era5_10",
        # 9 km/pixel 波段，实际存储为 2.56 km/pixel
        tile_resolution_factor=16,
        band_sets=[
            BandSet(
                [
                    "2m-temperature",
                    "2m-dewpoint-temperature",
                    "surface-pressure",
                    "10m-u-component-of-wind",
                    "10m-v-component-of-wind",
                    "total-precipitation",
                ],
                4096,
            ),
        ],
        is_multitemporal=True,
        ignore_when_parsing=False,
        image_tile_size_factor=-256,  # 负值表示瓦片尺寸缩小
    )

    # 经纬度坐标，2 波段，静态，不直接解析
    LATLON = ModalitySpec(
        name="latlon",
        tile_resolution_factor=0,  # 分辨率因子为 0，表示空间不变
        band_sets=[BandSet(["lat", "lon"], 0)],
        is_multitemporal=False,
        ignore_when_parsing=True,  # 经纬度不从文件解析
    )

    # GSE 数据，64 波段，静态
    GSE = ModalitySpec(
        name="gse",
        tile_resolution_factor=16,
        band_sets=[
            BandSet(
                [f"A{idx:02d}" for idx in range(64)],  # A00~A63 共 64 个波段
                16,
            ),
        ],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )

    # 美国作物数据层（CDL），单波段，静态
    CDL = ModalitySpec(
        name="cdl",
        tile_resolution_factor=16,
        band_sets=[BandSet(["cdl"], 16)],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )

    # 世界人口密度，单波段，静态
    WORLDPOP = ModalitySpec(
        name="worldpop",
        tile_resolution_factor=16,
        band_sets=[BandSet(["B1"], 16)],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )

    # WRI 冠层高度图，单波段，静态
    WRI_CANOPY_HEIGHT_MAP = ModalitySpec(
        name="wri_canopy_height_map",
        tile_resolution_factor=16,
        band_sets=[BandSet(["B1"], 16)],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )

    # 归一化植被指数（NDVI），单波段，多时相，由 S2 L2A 计算派生而非从文件加载
    NDVI = ModalitySpec(
        name="ndvi",
        tile_resolution_factor=16,
        band_sets=[BandSet(["ndvi"], 16)],
        is_multitemporal=True,
        ignore_when_parsing=True,  # 由 S2 L2A 计算派生，不从文件加载
    )

    # 欧洲作物分类，单波段，静态
    EUROCROPS = ModalitySpec(
        name="eurocrops",
        tile_resolution_factor=16,
        band_sets=[BandSet(["B1"], 16)],
        is_multitemporal=False,
        ignore_when_parsing=False,
    )

    @classmethod
    def get(self, name: str) -> ModalitySpec:
        """根据名称获取对应的 ModalitySpec 实例。

        通过模态名称（不区分大小写，自动转大写）获取类属性。

        Args:
            name: 模态名称字符串，如 "sentinel2"。

        Returns:
            对应的 ModalitySpec 实例。

        Raises:
            AssertionError: 如果名称与 ModalitySpec 内部名称不匹配。
        """
        modality = getattr(Modality, name.upper())  # 通过类属性名（大写）获取
        assert modality.name == name  # 确保名称一致
        return modality

    @classmethod
    def values(self) -> list[ModalitySpec]:
        """获取所有预定义的 ModalitySpec 列表。

        Returns:
            包含所有 ModalitySpec 实例的列表。
        """
        modalities = []
        for k in dir(Modality):
            modality = getattr(Modality, k)
            if not isinstance(modality, ModalitySpec):  # 仅筛选 ModalitySpec 类型
                continue
            modalities.append(modality)
        return modalities

    @classmethod
    def names(self) -> list[str]:
        """获取所有模态名称列表。

        Returns:
            所有模态名称字符串的列表。
        """
        return [modality.name for modality in self.values()]


# 经纬度和时间戳的字段名称常量
LATLON = ["lat", "lon"]
TIMESTAMPS = ["day", "month", "year"]


def get_modality_specs_from_names(names: list[str]) -> list[ModalitySpec]:
    """根据模态名称列表获取对应的 ModalitySpec 列表。

    Args:
        names: 模态名称字符串列表。

    Returns:
        对应的 ModalitySpec 实例列表。
    """
    return [Modality.get(name) for name in names]
