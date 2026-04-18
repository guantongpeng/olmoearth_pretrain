"""随机位置创建训练窗口。

本脚本在全球范围内随机选择位置创建训练窗口，
适用于大规模数据集的均匀采样。
"""

import argparse
import random

from upath import UPath

from .util import create_windows_with_highres_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create windows with random location for data ingestion",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        help="Dataset path",
        required=True,
    )
    parser.add_argument(
        "--count",
        type=int,
        help="Number of windows to create",
        required=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of worker processes",
        default=32,
    )
    args = parser.parse_args()

    lonlats: list[tuple[float, float]] = []
    for _ in range(args.count):
        # -175 to 175 so we don't have to worry about issues at antimeridian.
        lon = random.random() * 350 - 175
        # -80 to 80 since poles are not in UTM.
        lat = random.random() * 160 - 80
        lonlats.append((lon, lat))

    create_windows_with_highres_time(
        UPath(args.ds_path), lonlats, force_lowres_prob=0.25, workers=args.workers
    )
