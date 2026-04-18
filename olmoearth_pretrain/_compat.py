"""向后兼容模块 - 支持从旧版 helios 命名空间平滑迁移。

本模块提供了用于创建已弃用别名的工具函数，帮助用户从旧版 helios
命名空间迁移到新的 olmoearth_pretrain 命名空间。当用户使用旧名称
调用类或函数时，会发出 DeprecationWarning 警告，提示迁移到新名称。

主要功能：
- deprecated_class_alias: 为类创建弃用别名，实例化时发出警告
- deprecated_function_alias: 为函数创建弃用别名，调用时发出警告

使用场景：
    - 项目从 helios 重命名为 olmoearth_pretrain 后，保持旧 API 的兼容性
    - 在过渡期内允许旧代码继续运行，同时引导用户更新到新 API
"""

from __future__ import annotations

import functools
import types
import warnings
from collections.abc import Callable
from typing import Any, TypeVar, no_type_check

# 类类型变量，约束为 type 的子类型
T = TypeVar("T", bound=type)
# 函数类型变量，约束为可调用对象
F = TypeVar("F", bound=Callable[..., Any])


@no_type_check
def deprecated_class_alias(new_class: T, old_qualname: str) -> T:
    """为类创建一个弃用别名，在实例化时发出 DeprecationWarning 警告。

    通过动态创建 new_class 的子类来实现别名，该子类在实例化时
    会发出弃用警告。同时保留旧类的元数据（__name__/__module__），
    使其在错误信息和文档中显示为旧名称。

    对于 tuple 的子类（如 NamedTuple），会覆盖 __new__ 方法；
    对于其他类，会覆盖 __init__ 方法。

    Args:
        new_class: 应当使用的新类，弃用别名将指向此类。
        old_qualname: 旧类的限定路径（如 ``helios.foo.Bar``），
                      用于构建警告消息和设置元数据。

    Returns:
        new_class 的子类，在实例化时发出 DeprecationWarning，
        并携带旧类的元数据（__name__/__module__）。
    """
    # 从旧类限定路径中解析模块名和类名
    module_name, _, class_name = old_qualname.rpartition(".")
    if not class_name:
        class_name = old_qualname

    # 构建弃用警告消息，提示用户迁移到新类
    warning_message = (
        f"'{old_qualname}' is deprecated and will be removed in a future release. "
        f"Please update your code to use '{new_class.__module__}.{new_class.__name__}'."
    )

    # 动态创建 new_class 的子类作为别名
    alias: type = types.new_class(class_name, (new_class,))
    # 复制原始类的文档字符串
    alias.__doc__ = new_class.__doc__
    # 设置模块名为旧模块名，或将 olmoearth_pretrain 替换为 helios
    alias.__module__ = module_name or new_class.__module__.replace(
        "olmoearth_pretrain", "helios"
    )
    alias.__qualname__ = class_name
    # 标记弃用别名的目标类，便于运行时检查
    alias.__deprecated_target__ = new_class

    # 区分处理 tuple 子类（如 NamedTuple）和普通类
    if issubclass(new_class, tuple):
        # tuple 子类需要覆盖 __new__ 方法来发出警告
        original_new = new_class.__new__

        def __new__(cls, *args, **kwargs):  # type: ignore[override]
            """覆盖 __new__，在创建实例前发出弃用警告。"""
            warnings.warn(warning_message, DeprecationWarning, stacklevel=2)
            return original_new(cls, *args, **kwargs)

        alias.__new__ = __new__  # type: ignore[assignment]
    else:
        # 普通类覆盖 __init__ 方法来发出警告
        original_init = new_class.__init__

        def __init__(self, *args, **kwargs):  # type: ignore[override]
            """覆盖 __init__，在初始化实例前发出弃用警告。"""
            warnings.warn(warning_message, DeprecationWarning, stacklevel=2)
            original_init(self, *args, **kwargs)

        alias.__init__ = __init__  # type: ignore[assignment]

    return alias  # type: ignore[return-value]


@no_type_check
def deprecated_function_alias(new_function: F, old_qualname: str) -> F:
    """为函数创建一个弃用别名，在调用时发出 DeprecationWarning 警告。

    使用 functools.wraps 包装新函数，保留其签名和元数据，
    但在每次调用时先发出弃用警告，再调用新函数。

    Args:
        new_function: 应当使用的新函数，弃用别名将指向此函数。
        old_qualname: 旧函数的限定路径（如 ``helios.foo.bar``），
                      用于构建警告消息和设置元数据。

    Returns:
        新函数的包装器，在调用时发出 DeprecationWarning，
        并携带旧函数的元数据（__name__/__module__）。
    """
    # 从旧函数限定路径中解析模块名和函数名
    module_name, _, func_name = old_qualname.rpartition(".")
    if not func_name:
        func_name = old_qualname

    # 构建弃用警告消息，提示用户迁移到新函数
    warning_message = (
        f"'{old_qualname}' is deprecated and will be removed in a future release. "
        f"Please update your code to use '{new_function.__module__}.{new_function.__name__}'."
    )

    @functools.wraps(new_function)
    def wrapper(*args: Any, **kwargs: Any):
        """包装器函数，在调用新函数前发出弃用警告。"""
        warnings.warn(warning_message, DeprecationWarning, stacklevel=2)
        return new_function(*args, **kwargs)

    # 设置旧函数的元数据，使包装器在错误信息中显示为旧名称
    wrapper.__name__ = func_name
    wrapper.__qualname__ = func_name
    # 设置模块名为旧模块名，或将 olmoearth_pretrain 替换为 helios
    wrapper.__module__ = module_name or new_function.__module__.replace(
        "olmoearth_pretrain", "helios"
    )
    # 标记弃用别名的目标函数，便于运行时检查
    wrapper.__deprecated_target__ = new_function  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


__all__ = ["deprecated_class_alias", "deprecated_function_alias"]
