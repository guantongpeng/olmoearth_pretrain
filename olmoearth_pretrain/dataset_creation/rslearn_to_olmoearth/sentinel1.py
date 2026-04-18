"""将摄取的 Sentinel-1 SAR 数据后处理到 OlmoEarth Pretrain 数据集。

本模块负责将 Sentinel-1 合成孔径雷达(SAR)数据从 rslearn 格式转换为 OlmoEarth Pretrain 格式。
Sentinel-1 提供 C 波段 SAR 数据，包含 VV 和 VH 极化信息。
"""

import argparse
import multiprocessing

import tqdm
from rslearn.dataset import Dataset, Window
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

from olmoearth_pretrain.data.constants import Modality

from .multitemporal_raster import convert_freq, convert_monthly

# rslearn layer for frequent data.
LAYER_FREQ = "sentinel1_freq"

# rslearn layer prefix for monthly data.
LAYER_MONTHLY = "sentinel1"


def convert_sentinel1(window: Window, olmoearth_path: UPath) -> None:
    """Add Landsat data for this window to the OlmoEarth Pretrain dataset.

    Args:
        window: the rslearn window to read data from.
        olmoearth_path: OlmoEarth Pretrain dataset path to write to.
    """
    try:
        convert_freq(
            window,
            olmoearth_path,
            LAYER_FREQ,
            Modality.SENTINEL1,
            missing_okay=True,
            unprepared_okay=True,
        )
    except Exception as e:
        print(
            f"warning: got error {e} while converting frequent data for window {window.name}"
        )

    try:
        convert_monthly(window, olmoearth_path, LAYER_MONTHLY, Modality.SENTINEL1)
    except Exception as e:
        print(
            f"warning: got error {e} while converting monthly data for window {window.name}"
        )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser(
        description="Post-process OlmoEarth Pretrain data",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        help="Source rslearn dataset path",
        required=True,
    )
    parser.add_argument(
        "--olmoearth_path",
        type=str,
        help="Destination OlmoEarth Pretrain dataset path",
        required=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of workers to use",
        default=32,
    )
    args = parser.parse_args()

    dataset = Dataset(UPath(args.ds_path))
    olmoearth_path = UPath(args.olmoearth_path)

    jobs = []
    for window in dataset.load_windows(
        workers=args.workers, show_progress=True, groups=["res_10"]
    ):
        jobs.append(
            dict(
                window=window,
                olmoearth_path=olmoearth_path,
            )
        )

    p = multiprocessing.Pool(args.workers)
    outputs = star_imap_unordered(p, convert_sentinel1, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
