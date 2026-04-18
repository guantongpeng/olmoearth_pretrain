"""装饰器模块。

本模块提供了用于标记函数或类的装饰器工具，主要包括：
- experimental: 将函数或类标记为"实验性"功能

实验性功能可能尚未经过充分测试，可能随时变更或被移除，
通过装饰器在调用时发出 FutureWarning 警告，提醒使用者注意风险。

使用场景：
    - 新功能开发阶段，API 尚未稳定时使用 @experimental 标记
    - 提醒用户某些功能可能会在后续版本中变更或移除
"""

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar, cast

# 函数类型变量，用于保持装饰器对函数签名的类型推断
F = TypeVar("F", bound=Callable[..., Any])
# 类类型变量，用于保持装饰器对类的类型推断
C = TypeVar("C", bound=type)


def experimental(reason: str = "This is an experimental feature") -> Callable[[F], F]:
    """将函数或类标记为实验性功能的装饰器。

    实验性功能可能未经充分测试，可能随时变更且不另行通知，
    也可能在未来的版本中不再维护。被装饰的函数/类在调用时会发出
    FutureWarning 警告，并在文档字符串前添加 **EXPERIMENTAL** 标记。

    对于类，装饰器会包装其 __init__ 方法以在实例化时发出警告；
    对于函数，装饰器会包装函数本身以在调用时发出警告。

    Args:
        reason: 可选的解释说明，阐述为何该功能是实验性的或存在哪些限制。
                默认值为 "This is an experimental feature"。

    Returns:
        一个装饰器函数，该函数接收目标对象并返回包装后的对象，
        同时设置 __experimental__ 属性为 True。

    Example:
        >>> @experimental("This feature is still under development")
        >>> def my_function():
        ...     pass

        >>> @experimental()
        >>> class MyClass:
        ...     pass
    """

    def decorator(obj: F) -> F:
        """内部装饰器函数，对目标对象进行包装并添加实验性标记。"""
        # 添加标记属性，允许运行时检查对象是否为实验性功能
        setattr(obj, "__experimental__", True)

        # 构建警告消息，包含对象名称和可选的原因说明
        obj_name = getattr(obj, "__qualname__", getattr(obj, "__name__", str(obj)))
        msg = f"'{obj_name}' is experimental and may change or be removed in future versions."
        if reason:
            msg += f" {reason}"

        # 区分处理类和函数
        if isinstance(obj, type):
            # 对于类：包装 __init__ 方法，在实例化时发出警告
            original_init = obj.__init__  # type: ignore[misc]

            @functools.wraps(original_init)
            def wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
                """包装后的 __init__，在调用原始初始化前发出 FutureWarning。"""
                warnings.warn(msg, FutureWarning, stacklevel=2)
                return original_init(self, *args, **kwargs)

            obj.__init__ = wrapped_init  # type: ignore[method-assign, misc]

            # 在文档字符串前添加实验性标记
            if obj.__doc__:
                obj.__doc__ = f"**EXPERIMENTAL**: {msg}\n\n{obj.__doc__}"
            else:
                obj.__doc__ = f"**EXPERIMENTAL**: {msg}"
            return cast(F, obj)
        else:
            # 对于函数：包装函数本身，在调用时发出警告
            @functools.wraps(obj)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                """包装后的函数，在调用原始函数前发出 FutureWarning。"""
                warnings.warn(msg, FutureWarning, stacklevel=2)
                return obj(*args, **kwargs)

            # 在文档字符串前添加实验性标记
            if obj.__doc__:
                wrapper.__doc__ = f"**EXPERIMENTAL**: {msg}\n\n{obj.__doc__}"
            else:
                wrapper.__doc__ = f"**EXPERIMENTAL**: {msg}"

            # 确保包装后的函数也带有实验性标记属性
            setattr(wrapper, "__experimental__", True)
            return cast(F, wrapper)

    return decorator
