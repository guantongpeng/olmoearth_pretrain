"""
数据模块的工具函数集合。

本模块提供数据处理和可视化相关的工具函数，包括：
- to_cartesian: 将经纬度坐标转换为笛卡尔坐标
- convert_to_db: 将 Sentinel-1 数据转换为分贝值（dB）
- update_streaming_stats: 流式更新均值和方差（用于在线统计计算）
- plot_latlon_distribution: 绘制数据集的地理分布图
- plot_modality_data_distribution: 绘制模态数据的直方图分布
"""

import math

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np


def to_cartesian(lat: float, lon: float) -> np.ndarray:
    """将经纬度坐标转换为三维笛卡尔坐标（单位球面上的点）。

    转换公式：
        x = cos(lat) * cos(lon)
        y = cos(lat) * sin(lon)
        z = sin(lat)

    Args:
        lat: 纬度（度）。
        lon: 经度（度）。

    Returns:
        包含 (x, y, z) 笛卡尔坐标的 numpy 数组。
    """

    def validate_lat_lon(lat: float, lon: float) -> None:
        """验证经纬度范围的合法性（EPSG:4326 坐标系）。

        Args:
            lat: 纬度（度），范围 [-90, 90]。
            lon: 经度（度），范围 [-180, 180]。
        """
        assert -90 <= lat <= 90, (
            f"lat out of range ({lat}). Make sure you are in EPSG:4326"
        )
        assert -180 <= lon <= 180, (
            f"lon out of range ({lon}). Make sure you are in EPSG:4326"
        )

    def convert_to_radians(lat: float, lon: float) -> tuple:
        """将经纬度从度转换为弧度。

        Args:
            lat: 纬度（度）。
            lon: 经度（度）。

        Returns:
            (纬度弧度, 经度弧度) 元组。
        """
        return lat * math.pi / 180, lon * math.pi / 180

    def compute_cartesian(lat: float, lon: float) -> tuple:
        """从弧度制的经纬度计算笛卡尔坐标。

        Args:
            lat: 纬度（弧度）。
            lon: 经度（弧度）。

        Returns:
            (x, y, z) 笛卡尔坐标元组。
        """
        x = math.cos(lat) * math.cos(lon)
        y = math.cos(lat) * math.sin(lon)
        z = math.sin(lat)

        return x, y, z

    validate_lat_lon(lat, lon)  # 验证经纬度范围
    lat, lon = convert_to_radians(lat, lon)  # 转换为弧度
    x, y, z = compute_cartesian(lat, lon)  # 计算笛卡尔坐标

    return np.array([x, y, z])


# 根据 Earth Engine 文档，Sentinel-1 数据需要使用 10*log10(x) 转换为 dB
# 参考：https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD#description
def convert_to_db(data: np.ndarray) -> np.ndarray:
    """将数据转换为分贝值（dB）。

    使用公式：dB = 10 * log10(data)，用于 Sentinel-1 SAR 数据的转换。

    Args:
        data: 待转换的数据数组。

    Returns:
        转换为 dB 的数据数组。
    """
    # 将数据裁剪到最小值 1e-10 以避免 log(0)
    data = np.clip(data, 1e-10, None)
    result = 10 * np.log10(data)
    return result


def update_streaming_stats(
    current_count: int,
    current_mean: float,
    current_var: float,
    modality_band_data: np.ndarray,
) -> tuple[int, float, float]:
    """流式更新均值和方差，用于在线统计计算。

    当数据量过大无法一次性加载时，使用流式更新逐步计算均值和方差。
    参考：https://www.geeksforgeeks.org/expression-for-mean-and-variance-in-a-running-stream/

    Args:
        current_count: 当前的数据点计数。
        current_mean: 当前的均值。
        current_var: 当前的方差。
        modality_band_data: 新批次的数据。

    Returns:
        更新后的 (count, mean, variance) 元组。
    """
    band_data_count = np.prod(modality_band_data.shape)  # 新批次的数据点数

    # 使用流式公式更新均值和方差
    new_count = current_count + band_data_count
    new_mean = (
        current_mean
        + (modality_band_data.mean() - current_mean) * band_data_count / new_count
    )
    new_var = (
        current_var
        + ((modality_band_data - current_mean) * (modality_band_data - new_mean)).sum()
    )

    return new_count, new_mean, new_var


def plot_latlon_distribution(latlons: np.ndarray, title: str) -> plt.Figure:
    """绘制数据集的地理分布图。

    在全球地图上以散点图形式展示样本的经纬度分布。

    Args:
        latlons: 形状为 (N, 2) 的数组，每行为 [纬度, 经度]。
        title: 图表标题。

    Returns:
        matplotlib Figure 对象。
    """
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())  # 使用 PlateCarree 投影

    # 添加地图特征
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, alpha=0.1)
    ax.add_feature(cfeature.OCEAN, alpha=0.1)

    # 绘制数据点散点图
    ax.scatter(
        latlons[:, 1],  # 经度
        latlons[:, 0],  # 纬度
        transform=ccrs.PlateCarree(),
        alpha=0.5,
        s=0.01,
    )

    ax.set_global()  # 显示全球范围
    ax.gridlines()
    ax.set_title(title)
    return fig


def plot_modality_data_distribution(modality: str, modality_data: dict) -> plt.Figure:
    """绘制模态数据的直方图分布。

    为每个波段绘制一个直方图子图。

    Args:
        modality: 模态名称。
        modality_data: 字典，键为波段名，值为该波段的数据数组。

    Returns:
        matplotlib Figure 对象。
    """
    fig, axes = plt.subplots(
        len(modality_data), 1, figsize=(10, 5 * len(modality_data))
    )
    if len(modality_data) == 1:
        axes = [axes]  # 单波段时确保 axes 为列表
    for ax, (band, values) in zip(axes, modality_data.items()):
        ax.hist(values, bins=50, alpha=0.75)  # 绘制直方图
        ax.set_title(f"{modality} - {band}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig
