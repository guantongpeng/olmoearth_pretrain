"""
OlmoEarth Pretrain 数据集样本的可视化模块。

本模块提供了将 OlmoEarthDataset 中的样本可视化的功能，包括：
- 在网格中展示各模态各波段的数据
- 经纬度坐标在全球地图上的标注
- WorldCover 土地覆盖分类的离散色彩映射和图例
- Sentinel-1 数据的分贝转换
- 结果保存为 PNG 图片
"""

import logging
import os
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from einops import rearrange
from matplotlib.figure import Figure
from upath import UPath

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import GetItemArgs, OlmoEarthDataset
from olmoearth_pretrain.data.utils import convert_to_db

logger = logging.getLogger(__name__)

# WorldCover 土地覆盖分类的图例映射：类别值 -> (颜色, 标签)
WORLDCOVER_LEGEND = {
    10: ("#006400", "Tree cover"),                      # 林地覆盖
    20: ("#ffbb22", "Shrubland"),                       # 灌木地
    30: ("#ffff4c", "Grassland"),                       # 草地
    40: ("#f096ff", "Cropland"),                        # 农田
    50: ("#fa0000", "Built-up"),                        # 建筑用地
    60: ("#b4b4b4", "Bare / sparse vegetation"),        # 裸地/稀疏植被
    70: ("#f0f0f0", "Snow and ice"),                    # 雪和冰
    80: ("#0064c8", "Permanent water bodies"),          # 永久水体
    90: ("#0096a0", "Herbaceous wetland"),              # 草本湿地
    95: ("#00cf75", "Mangroves"),                       # 红树林
    100: ("#fae6a0", "Moss and lichen"),                # 苔藓和地衣
}


def visualize_sample(
    dataset: OlmoEarthDataset,
    sample_index: int,
    out_dir: str | Path | UPath,
) -> Figure:
    """可视化 OlmoEarth 数据集中的样本，以网格格式展示。

    布局规则：
    - 每个模态占一行
    - 该模态的每个波段占该行的列
    - LATLON 模态在地图上标注坐标点
    - WORLDCOVER 模态使用离散色彩映射并显示图例
    - Sentinel-1 模态先转换为 dB 再显示
    结果保存为 PNG 文件。

    Args:
        dataset: OlmoEarthDataset 实例。
        sample_index: 样本索引。
        out_dir: 输出目录路径。

    Returns:
        matplotlib Figure 对象。
    """
    wc_classes_sorted = sorted(WORLDCOVER_LEGEND.keys())  # [10, 20, 30, ...]
    wc_colors = [WORLDCOVER_LEGEND[val][0] for val in wc_classes_sorted]
    # 构建离散色彩映射和对应的归一化边界
    wc_cmap = mcolors.ListedColormap(wc_colors)
    wc_bounds = wc_classes_sorted + [
        wc_classes_sorted[-1] + 1
    ]  # 边界范围，如最后一个为 100 则上界为 101
    wc_norm = mcolors.BoundaryNorm(wc_bounds, wc_cmap.N)
    logger.info(f"Visualizing sample index: {sample_index}")

    # 从数据集获取样本
    args = GetItemArgs(
        idx=sample_index,
        patch_size=1,
        sampled_hw_p=256,
    )
    _, sample = dataset[args]
    modalities = sample.modalities
    if not modalities:
        logger.warning("No modalities found in this sample.")
        return
    total_rows = len(modalities)
    # 至少 1 列，同时处理模态中最大的波段数
    max_bands = 1
    for modality_name in modalities:
        modality_spec = Modality.get(modality_name)
        if modality_spec != Modality.LATLON:
            max_bands = max(max_bands, len(modality_spec.band_order))

    # 创建子图网格，多一行用于地图
    fig, axes = plt.subplots(
        nrows=total_rows + 1,
        ncols=max_bands,
        figsize=(5 * max_bands, 5 * total_rows),
        squeeze=False,
    )

    # 加载经纬度数据
    assert sample.latlon is not None
    latlon_data = sample.latlon
    lat = float(latlon_data[0])
    lon = float(latlon_data[1])

    # 移除原始 Axes 并替换为 Cartopy 投影的 Axes
    fig.delaxes(axes[0, 0])

    axes[0, 0] = fig.add_subplot(
        total_rows,
        max_bands,
        1,
        projection=ccrs.PlateCarree(),  # PlateCarree 投影（等经纬度投影）
    )
    ax_map = axes[0, 0]

    # 添加地图特征
    ax_map.add_feature(cfeature.COASTLINE)
    ax_map.add_feature(cfeature.LAND, alpha=0.1)
    ax_map.add_feature(cfeature.OCEAN, alpha=0.1)
    # 在地图上标注样本位置
    ax_map.scatter(lon, lat, transform=ccrs.PlateCarree(), c="red", s=60)

    ax_map.set_global()
    ax_map.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    ax_map.gridlines()
    ax_map.set_title(f"{Modality.LATLON.name.upper()} (Lat: {lat:.2f}, Lon: {lon:.2f})")

    # 隐藏地图行中未使用的列
    for empty_col in range(max_bands - 1, 0, -1):
        axes[0, empty_col].axis("off")
    sample_dict = sample.as_dict()
    for row_idx, (modality_name, modality_data) in enumerate(
        sample_dict.items(), start=1
    ):
        assert modality_data is not None
        if modality_name == "timestamps" or modality_name == Modality.LATLON.name:
            continue  # 跳过时间戳和经纬度
        logger.info(f"Plotting modality: {modality_name}")
        modality_spec = Modality.get(modality_name)
        # Sentinel-1 数据先转换为 dB
        if modality_spec == Modality.SENTINEL1:
            modality_data = convert_to_db(modality_data)
        logger.info(f"Modality data shape (loaded): {modality_data.shape}")

        # 如果有时序维度 [H, W, T, C]，取第一个时间步
        if modality_spec.is_spatial:
            modality_data = modality_data[:, :, 0]
            logger.info(
                f"Modality data shape after first time step: {modality_data.shape}"
            )

        # 重排为 [C, H, W] 以适配 imshow
        modality_data = rearrange(modality_data, "h w c -> c h w")
        logger.info(f"Modality data shape after rearranging: {modality_data.shape}")

        for band_i, band_name in enumerate(modality_spec.band_order):
            ax = axes[row_idx, band_i]
            channel_data = modality_data[band_i]  # 形状 [H, W]

            # WorldCover 使用离散色彩映射和图例
            if modality_spec == Modality.WORLDCOVER:
                _ = ax.imshow(channel_data, cmap=wc_cmap, norm=wc_norm)
                ax.set_title(f"{modality_spec.name.upper()} - {band_name}")

                # 创建图例色块
                patches = []
                for val in wc_classes_sorted:
                    color_hex, label_txt = WORLDCOVER_LEGEND[val]
                    patch = mpatches.Patch(color=color_hex, label=label_txt)
                    patches.append(patch)

                # 将图例放置在图像右侧
                ax.legend(
                    handles=patches,
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    borderaxespad=0.0,
                    fontsize=8,
                )
                ax.axis("off")

            else:
                # 其他模态使用 viridis 色彩映射
                _ = ax.imshow(channel_data, cmap="viridis")
                ax.set_title(f"{modality_spec.name.upper()} — {band_name}")
                ax.axis("off")

        # 隐藏波段数不足 max_bands 的未使用列
        used_cols = len(modality_spec.band_order)
        for empty_col in range(used_cols, max_bands):
            axes[row_idx, empty_col].axis("off")

    plt.tight_layout()
    fig.subplots_adjust(
        wspace=0.3, hspace=0.3, left=0.05, right=0.95, top=0.95, bottom=0.05
    )
    # 保存图片
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"sample_{sample_index}.png")
    fig.savefig(out_path)
    logger.info(f"Saved visualization to {out_path}")
    logger.info(f"type(fig): {type(fig)}")
    return fig
