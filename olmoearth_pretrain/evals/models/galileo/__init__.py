"""Galileo 地球观测基础模型集成。

本模块提供 Galileo 模型的评估集成，包括模型包装器和配置类。
Galileo 是一种用于地球观测的预训练基础模型。

主要组件：
- GalileoWrapper: Galileo 模型包装器，适配 OlmoEarth 评估接口
- GalileoConfig: Galileo 模型配置类
"""

from .single_file_galileo import GalileoConfig, GalileoWrapper

__all__ = ["GalileoWrapper", "GalileoConfig"]
