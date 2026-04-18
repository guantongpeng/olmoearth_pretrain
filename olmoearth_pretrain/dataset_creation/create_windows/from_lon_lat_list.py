"""从经纬度列表创建训练窗口。

本脚本根据提供的经纬度坐标列表，在数据集中创建对应的训练窗口。
适用于已知采样位置的场景，如特定城市的遥感数据采集。
"""

import argparse
import json

from upath import UPath

from .util import create_windows_with_highres_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create windows based on specified locations",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        help="Dataset path",
        required=True,
    )
    parser.add_argument(
        "--fname",
        type=str,
        help="JSON filename containing list of [lot, lat]",
        required=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of worker processes",
        default=32,
    )
    args = parser.parse_args()

    with open(args.fname) as f:
        lonlats = [(lon, lat) for lon, lat in json.load(f)]

    create_windows_with_highres_time(
        UPath(args.ds_path), lonlats, force_lowres_prob=0.25, workers=args.workers
    )
