"""从 rslearn 数据集转换数据到 OlmoEarth Pretrain 数据集。

本模块包含将各种遥感数据源从 rslearn 格式转换为 OlmoEarth Pretrain
格式所需的转换逻辑。每种数据源（模态）有独立的转换模块。

子模块:
    - cdl: 农作物数据层(CDL)转换
    - era5/era5_10: ERA5 气象再分析数据转换
    - eurocrops: 欧洲农作物数据转换
    - gse: Google Static Earth 数据转换
    - landsat: Landsat 卫星数据转换
    - multitemporal_raster: 多时相栅格数据通用转换
    - naip/naip_10: NAIP 航拍影像转换
    - openstreetmap: OpenStreetMap 数据转换
    - rasterize_openstreetmap: OpenStreetMap 栅格化转换
    - sentinel1: Sentinel-1 SAR 数据转换
    - sentinel2/sentinel2_l2a: Sentinel-2 光学数据转换
    - srtm: SRTM 地形数据转换
    - worldcereal: WorldCereal 农作物分类转换
    - worldcover: ESA WorldCover 土地覆盖转换
    - worldpop: WorldPop 人口数据转换
    - wri_canopy_height_map: WRI 冠层高度图转换
"""
