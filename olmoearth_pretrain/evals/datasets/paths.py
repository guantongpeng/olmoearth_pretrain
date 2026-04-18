"""数据集路径配置模块，通过环境变量配置。

本模块定义了所有评估数据集的存储路径，优先使用环境变量，
回退到内部默认路径（仅内部用户可用）。

支持的数据集路径：
- GEOBENCH_DIR: GeoBench 数据集目录
- BREIZHCROPS_DIR: BreizhCrops 数据集目录
- MADOS_DIR: MADOS 数据集目录
- FLOODS_DIR: Sen1Floods11 数据集目录
- PASTIS_DIR: PASTIS-R 数据集目录
- PASTIS_DIR_ORIG: PASTIS-R 原始尺寸数据集目录
- PASTIS_DIR_PARTITION: PASTIS 分区信息目录
"""

import os

from upath import UPath

# Only available to internal users
_DEFAULTS = {
    "GEOBENCH_DIR": "/weka/dfive-default/presto-geobench/dataset/geobench",
    "BREIZHCROPS_DIR": "/weka/dfive-default/skylight/presto_eval_sets/breizhcrops",
    "MADOS_DIR": "/weka/dfive-default/presto_eval_sets/mados",
    "FLOODS_DIR": "/weka/dfive-default/presto_eval_sets/floods",
    "PASTIS_DIR": "/weka/dfive-default/presto_eval_sets/pastis_r",
    "PASTIS_DIR_ORIG": "/weka/dfive-default/presto_eval_sets/pastis_r_origsize",
    "PASTIS_DIR_PARTITION": "/weka/dfive-default/presto_eval_sets/pastis",
}

GEOBENCH_DIR = UPath(os.getenv("GEOBENCH_DIR", _DEFAULTS["GEOBENCH_DIR"]))
BREIZHCROPS_DIR = UPath(os.getenv("BREIZHCROPS_DIR", _DEFAULTS["BREIZHCROPS_DIR"]))
MADOS_DIR = UPath(os.getenv("MADOS_DIR", _DEFAULTS["MADOS_DIR"]))
FLOODS_DIR = UPath(os.getenv("FLOODS_DIR", _DEFAULTS["FLOODS_DIR"]))
PASTIS_DIR = UPath(os.getenv("PASTIS_DIR", _DEFAULTS["PASTIS_DIR"]))
PASTIS_DIR_ORIG = UPath(os.getenv("PASTIS_DIR_ORIG", _DEFAULTS["PASTIS_DIR_ORIG"]))
PASTIS_DIR_PARTITION = UPath(
    os.getenv("PASTIS_DIR_PARTITION", _DEFAULTS["PASTIS_DIR_PARTITION"])
)
