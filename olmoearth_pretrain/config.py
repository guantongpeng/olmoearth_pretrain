"""双模式配置系统模块 - 支持有/无 olmo-core 依赖的运行环境。

本模块提供了统一的 Config 基类，能够根据 olmo-core 是否安装自动切换实现：
- 已安装 olmo-core: 使用 olmo-core 的全功能 Config（支持 OmegaConf 合并、
  CLI 覆盖、YAML 加载等高级功能）
- 未安装 olmo-core: 使用最小化的独立 Config（_StandaloneConfig），仅支持
  从 JSON 反序列化和模型构建，适用于推理模式

关键导出：
    - Config: 统一的配置基类（自动选择实现）
    - OLMO_CORE_AVAILABLE: olmo-core 是否可用的布尔标志
    - require_olmo_core: 训练代码的守卫函数，缺少 olmo-core 时抛出 ImportError

使用方式：
    from olmoearth_pretrain.config import Config, OLMO_CORE_AVAILABLE, require_olmo_core

    @dataclass
    class MyConfig(Config):
        ...

对于训练代码，在模块级别添加守卫：
    from olmoearth_pretrain.config import require_olmo_core
    require_olmo_core("Training")
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, fields, is_dataclass
from importlib import import_module
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

# === olmo-core 可用性的单一事实来源 ===
# 尝试导入 olmo-core 的 Config，如果失败则标记为不可用
try:
    from olmo_core.config import Config as _OlmoCoreConfig

    OLMO_CORE_AVAILABLE = True  # olmo-core 已安装，可使用全功能配置
except ImportError:
    OLMO_CORE_AVAILABLE = False  # olmo-core 未安装，将使用独立配置
    _OlmoCoreConfig = None  # type: ignore[assignment, misc]


def require_olmo_core(operation: str = "This operation") -> None:
    """训练代码的守卫函数 - 当 olmo-core 不可用时抛出 ImportError。

    在训练模块的入口点调用此函数，提供清晰的错误信息，
    避免用户在推理模式下误用训练功能。

    Args:
        operation: 描述需要 olmo-core 的操作名称（用于错误消息中）。
                   默认值为 "This operation"。

    Raises:
        ImportError: 当 olmo-core 未安装时抛出，提示安装命令。

    Example:
        from olmoearth_pretrain.config import require_olmo_core
        require_olmo_core("Training")  # 如果 olmo-core 不可用则抛出异常
    """
    if not OLMO_CORE_AVAILABLE:
        raise ImportError(
            f"{operation} requires olmo-core. "
            "Install with: pip install olmoearth-pretrain[training]"
        )


# 绑定到 _StandaloneConfig 的类型变量，用于 from_dict 的返回类型推断
C = TypeVar("C", bound="_StandaloneConfig")


@dataclass
class _StandaloneConfig:
    """推理模式下的最小化配置基类（无 olmo-core 依赖）。

    当 olmo-core 未安装时，此类作为 Config 的替代实现，仅提供足够的功能
    从 JSON 反序列化模型配置并构建模型。有意不支持以下高级功能：
    - 基于 OmegaConf 的配置合并
    - 通过 dotlist 的 CLI 覆盖
    - YAML 加载
    - 超出 dataclass 基础验证之外的校验

    关键属性：
        CLASS_NAME_FIELD: 用于标识配置类名的特殊字段名，值为 "_CLASS_"

    如需完整功能，请安装 olmo-core。
    """

    # 用于序列化/反序列化时标识类名的特殊字段
    CLASS_NAME_FIELD = "_CLASS_"

    @classmethod
    def _resolve_class(cls, class_name: str) -> type:
        """将全限定类名解析为类对象。

        通过 import_module 动态导入模块并获取类对象，
        用于从序列化的字典中恢复嵌套配置实例。

        Args:
            class_name: 全限定类名（如 "olmoearth_pretrain.config.MyConfig"），
                        必须包含模块路径和类名。

        Returns:
            对应的类对象。

        Raises:
            ValueError: 当类名不是全限定名（不包含点号）时。
        """
        if "." not in class_name:
            raise ValueError(f"Class name must be fully qualified (got '{class_name}')")
        # 分离模块路径和类名
        *modules, cls_name = class_name.split(".")
        module_name = ".".join(modules)
        # 动态导入模块并获取类对象
        module = import_module(module_name)
        return getattr(module, cls_name)

    @classmethod
    def _clean_data(cls, data: Any) -> Any:
        """递归清理数据，将 _CLASS_ 字段解析为实际的配置实例。

        遍历字典、列表和元组，对包含 CLASS_NAME_FIELD 的字典
        解析其类名并实例化为对应的 dataclass 对象。

        Args:
            data: 待清理的数据，可以是字典、列表、元组或基本类型。

        Returns:
            清理后的数据，嵌套的 _CLASS_ 字典已被替换为对应的配置实例。
        """
        if isinstance(data, dict):
            # 检查字典是否代表一个配置类（包含 _CLASS_ 字段）
            class_name = data.get(cls.CLASS_NAME_FIELD)
            # 递归清理所有值，并排除 _CLASS_ 字段本身
            cleaned = {
                k: cls._clean_data(v)
                for k, v in data.items()
                if k != cls.CLASS_NAME_FIELD
            }

            if class_name is not None:
                # 解析类名并实例化
                resolved_cls = cls._resolve_class(class_name)
                if not is_dataclass(resolved_cls):
                    raise TypeError(f"Class '{class_name}' is not a dataclass")
                # 获取该 dataclass 的字段名集合
                field_names = {f.name for f in fields(resolved_cls)}
                # 仅保留该 dataclass 支持的字段，忽略多余字段
                valid_kwargs = {k: v for k, v in cleaned.items() if k in field_names}
                try:
                    return resolved_cls(**valid_kwargs)
                except TypeError as e:
                    raise TypeError(f"Failed to instantiate {class_name}: {e}") from e
            return cleaned

        elif isinstance(data, list | tuple):
            # 递归清理列表/元组中的每个元素，并保持原始类型
            cleaned_items = [cls._clean_data(item) for item in data]
            return type(data)(cleaned_items)

        else:
            # 基本类型直接返回
            return data

    @classmethod
    def from_dict(
        cls: type[C], data: dict[str, Any], overrides: list[str] | None = None
    ) -> C:
        """从字典反序列化配置，支持处理嵌套的 _CLASS_ 字段。

        先通过 _clean_data 递归解析嵌套的配置类，然后根据清理后
        的数据类型创建配置实例。

        Args:
            data: 配置的字典表示，可能包含 _CLASS_ 字段表示嵌套配置类。
            overrides: 配置覆盖列表（dotlist 格式），在独立模式下被忽略，
                       需要安装 olmo-core 才能使用此功能。默认为 None。

        Returns:
            配置类的实例。

        Raises:
            TypeError: 当清理后的数据类型不是字典时。

        Note:
            overrides 参数为了 API 兼容性而保留，但在独立模式下被忽略。
            安装 olmo-core 可获得完整的覆盖支持。
        """
        if overrides:
            # 独立模式不支持配置覆盖，发出警告提示安装 olmo-core
            warnings.warn(
                "Config overrides are not supported in standalone mode. "
                "Install olmo-core for full functionality.",
                UserWarning,
                stacklevel=2,
            )

        # 递归清理数据，解析嵌套的配置类
        cleaned = cls._clean_data(data)

        if isinstance(cleaned, cls):
            # 清理后已经是目标类的实例（_clean_data 中已实例化）
            return cleaned
        elif isinstance(cleaned, dict):
            # 清理后仍是字典，需要手动实例化
            # 获取该类的字段名集合
            field_names = {f.name for f in fields(cls)}
            # 仅保留该类支持的字段
            valid_kwargs = {k: v for k, v in cleaned.items() if k in field_names}
            return cls(**valid_kwargs)
        else:
            raise TypeError(f"Expected dict, got {type(cleaned)}")

    def as_dict(
        self,
        *,
        exclude_none: bool = False,
        exclude_private_fields: bool = False,
        include_class_name: bool = False,
        json_safe: bool = False,
        recurse: bool = True,
    ) -> dict[str, Any]:
        """将配置转换为字典。

        支持多种序列化选项，可选择排除空值、私有字段，
        添加类名标记，以及转换为 JSON 安全类型。

        Args:
            exclude_none: 是否排除值为 None 的字段。默认为 False。
            exclude_private_fields: 是否排除以 _ 开头的私有字段。默认为 False。
            include_class_name: 是否在字典中包含 _CLASS_ 字段，
                                值为全限定类名。默认为 False。
            json_safe: 是否将非 JSON 安全类型转换为字符串。默认为 False。
            recurse: 是否递归转换嵌套的 dataclass。默认为 True。

        Returns:
            此配置的字典表示。
        """

        def convert(obj: Any) -> Any:
            """递归转换对象为字典形式的内部函数。"""
            if is_dataclass(obj) and not isinstance(obj, type):
                # dataclass 实例：遍历字段，根据选项过滤并转换
                result = {}
                if include_class_name:
                    # 添加全限定类名字段，用于反序列化时恢复类型
                    result[self.CLASS_NAME_FIELD] = (
                        f"{obj.__class__.__module__}.{obj.__class__.__name__}"
                    )
                for field in fields(obj):
                    if exclude_private_fields and field.name.startswith("_"):
                        # 跳过私有字段
                        continue
                    value = getattr(obj, field.name)
                    if exclude_none and value is None:
                        # 跳过 None 值字段
                        continue
                    if recurse:
                        value = convert(value)
                    result[field.name] = value
                return result
            elif isinstance(obj, dict):
                # 字典类型：递归转换每个值
                return {k: convert(v) if recurse else v for k, v in obj.items()}
            elif isinstance(obj, list | tuple | set):
                # 列表/元组/集合类型：递归转换每个元素
                converted = [convert(item) if recurse else item for item in obj]
                if json_safe:
                    # JSON 安全模式下统一返回列表
                    return converted
                return type(obj)(converted)
            elif obj is None or isinstance(obj, float | int | bool | str):
                # 基本类型直接返回
                return obj
            elif json_safe:
                # JSON 安全模式下将其他类型转为字符串
                return str(obj)
            else:
                return obj

        return convert(self)

    def as_config_dict(self) -> dict[str, Any]:
        """转换为适合 JSON 序列化的字典。

        这是 as_dict() 的便捷封装，使用适合保存配置到 JSON 文件的参数：
        - 排除 None 值
        - 排除私有字段
        - 包含类名标记
        - 转换非 JSON 安全类型为字符串
        - 递归处理嵌套 dataclass

        Returns:
            适合 JSON 序列化的字典表示。
        """
        return self.as_dict(
            exclude_none=True,
            exclude_private_fields=True,
            include_class_name=True,
            json_safe=True,
            recurse=True,
        )

    def validate(self) -> None:
        """验证配置的有效性。子类可覆盖此方法以添加自定义验证逻辑。"""
        pass

    def build(self) -> Any:
        """构建此配置所代表的对象。

        子类必须实现此方法，根据配置参数创建并返回对应的对象实例。

        Raises:
            NotImplementedError: 始终抛出，除非子类覆盖了此方法。
        """
        raise NotImplementedError("Subclasses must implement build()")


# === 统一导出 ===
# 根据 olmo-core 是否可用，选择对应的 Config 实现
if OLMO_CORE_AVAILABLE:
    logger.debug("olmo-core is available")
    # olmo-core 已安装，使用其全功能 Config
    Config = _OlmoCoreConfig  # type: ignore[assignment,misc]
else:
    logger.debug("olmo-core is not available")
    # olmo-core 未安装，使用独立 Config（仅支持推理）
    Config = _StandaloneConfig
    # 首次导入模块时发出一次性警告，提示安装 olmo-core
    warnings.warn(
        "olmo-core not installed. Running in inference-only mode. "
        "For training: pip install olmoearth-pretrain[training]",
        UserWarning,
        stacklevel=2,
    )


__all__ = ["Config", "OLMO_CORE_AVAILABLE", "require_olmo_core"]
