r"""H5 格式数据集转换启动脚本。

本脚本用于将原始数据集转换为 H5PY 格式，支持通过命令行参数
覆盖支持的模态名称列表和压缩设置。

使用方式:
    python run_h5_conversion.py --tile-path=TILE_PATH --supported-modality-names="\[sentinel2_l2a,sentinel1,worldcover\]" --compression=zstd --compression_opts=3 --tile_size=128

模块功能:
    1. 构建默认的 H5 转换配置
    2. 解析命令行覆盖参数
    3. 执行 H5 格式转换
"""

import logging
import sys
from collections.abc import Callable
from typing import Any

from olmo_core.utils import prepare_cli_environment

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.dataset.convert_to_h5py import ConvertToH5pyConfig

logger = logging.getLogger(__name__)


def build_default_config() -> ConvertToH5pyConfig:
    """构建默认的 H5 转换配置。

    默认支持的模态包括: Sentinel-2 L2A, Sentinel-1, Landsat,
    WorldCover, OpenStreetMap, WorldCereal, SRTM, ERA5-10, NAIP-10。

    Returns:
        ConvertToH5pyConfig: 默认的转换配置
    """
    return ConvertToH5pyConfig(
        tile_path="",
        supported_modality_names=[
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
            Modality.LANDSAT.name,
            Modality.WORLDCOVER.name,
            Modality.OPENSTREETMAP_RASTER.name,
            Modality.WORLDCEREAL.name,
            Modality.SRTM.name,
            Modality.ERA5_10.name,
            Modality.NAIP_10.name,
        ],
        multiprocessed_h5_creation=True,
    )


def main(config_builder: Callable = build_default_config, *args: Any) -> None:
    """解析命令行参数、构建配置并执行 H5 转换。

    Args:
        config_builder: 配置构建函数，默认使用 build_default_config
        *args: 额外参数（未使用）
    """
    prepare_cli_environment()

    script, *overrides = sys.argv  # 获取脚本路径和命令行覆盖参数

    # 从参数创建配置对象
    default_config = config_builder()
    config = default_config.merge(overrides)
    logger.info(f"Configuration overrides: {overrides}")
    logger.info(f"Configuration loaded: {config}")

    # 构建并运行转换器
    converter = config.build()
    logger.info("Starting H5 conversion process...")
    converter.run()
    logger.info("H5 conversion process finished.")


if __name__ == "__main__":
    main()
