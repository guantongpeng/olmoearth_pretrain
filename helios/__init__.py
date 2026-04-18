"""helios 包的向后兼容性垫片（shim）。

helios 包已重命名为 olmoearth_pretrain。此模块提供向后兼容的导入支持，
使得使用 `import helios` 的旧代码仍然可以正常工作。

核心机制:
    - _HeliosAliasLoader: 自定义加载器，将 helios.* 模块映射到 olmoearth_pretrain.* 模块
    - _HeliosAliasFinder: 自定义元路径查找器，拦截 helios 的导入请求
    - __getattr__: 模块级别的属性访问，自动重定向到 olmoearth_pretrain

注意:
    此兼容性垫片将在未来版本中移除，请更新导入语句。
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types
import warnings
from collections.abc import Sequence
from typing import cast

_TARGET_PACKAGE = "olmoearth_pretrain"  # 目标包名
_DEPRECATION_MESSAGE = (  # 弃弃警告消息
    "The 'helios' package has been renamed to 'olmoearth_pretrain'. "
    "Please update your imports; this compatibility shim will be removed in a future release."
)

warnings.warn(_DEPRECATION_MESSAGE, FutureWarning, stacklevel=2)  # 发出弃用警告

_target_pkg = importlib.import_module(_TARGET_PACKAGE)  # 导入目标包

__all__: list[str] = list(getattr(_target_pkg, "__all__", []))
_doc: str | None = cast(str | None, getattr(_target_pkg, "__doc__", None))
_path: list[str] = list(getattr(_target_pkg, "__path__", []))
__package__ = __name__

globals()["__doc__"] = _doc
globals()["__path__"] = _path

globals().update(
    {name: getattr(_target_pkg, name) for name in getattr(_target_pkg, "__all__", [])}
)


class _HeliosAliasLoader(importlib.abc.Loader):
    """自定义加载器，将 helios.* 模块映射到 olmoearth_pretrain.* 模块。

    当导入 helios 的子模块时，此加载器实际导入对应的 olmoearth_pretrain 子模块。

    关键属性:
        _target_name: 对应的 olmoearth_pretrain 模块全名
    """

    def __init__(self, target_name: str) -> None:
        self._target_name = target_name

    def create_module(
        self, spec: importlib.machinery.ModuleSpec
    ) -> types.ModuleType | None:
        module = importlib.import_module(self._target_name)
        return module

    def exec_module(self, module: types.ModuleType) -> None:
        """执行模块（已被 create_module 初始化，无需额外操作）。"""
        sys.modules.setdefault(self._target_name, module)  # 确保模块在 sys.modules 中注册


class _HeliosAliasFinder(importlib.abc.MetaPathFinder):
    """自定义元路径查找器，将 helios 的导入请求映射到 olmoearth_pretrain。

    拦截 helios 的子模块导入，查找对应的 olmoearth_pretrain 子模块规范，
    并创建使用 _HeliosAliasLoader 的模块规范。
    """

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: types.ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        """查找模块规范，将 helios 子模块映射到 olmoearth_pretrain 子模块。

        Args:
            fullname: 完整的模块名
            path: 搜索路径
            target: 目标模块（未使用）

        Returns:
            ModuleSpec | None: 如果是 helios 子模块则返回对应的规范，否则返回 None
        """
        if fullname == __name__:
            return None  # 跳过顶级 helios 模块本身
        if not fullname.startswith(__name__ + "."):
            return None  # 只处理 helios 的子模块

        # 构建对应的 olmoearth_pretrain 模块名

        target_name = _TARGET_PACKAGE + fullname[len(__name__) :]
        target_spec = importlib.util.find_spec(target_name)
        if target_spec is None:
            return None

        spec = importlib.machinery.ModuleSpec(
            name=fullname,
            loader=_HeliosAliasLoader(target_name),
            is_package=target_spec.submodule_search_locations is not None,
        )
        spec.origin = target_spec.origin
        spec.submodule_search_locations = target_spec.submodule_search_locations
        return spec


if not any(isinstance(finder, _HeliosAliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _HeliosAliasFinder())  # 注册查找器到元路径的最前面


def __getattr__(name: str) -> object:
    """模块级别属性访问，自动重定向到 olmoearth_pretrain 并发出弃用警告。

    Args:
        name: 属性名

    Returns:
        object: olmoearth_pretrain 包中对应的属性
    """
    warnings.warn(_DEPRECATION_MESSAGE, FutureWarning, stacklevel=2)
    value = getattr(_target_pkg, name)
    if isinstance(value, types.ModuleType):
        sys.modules.setdefault(f"{__name__}.{name}", value)
    return value


def __dir__() -> list[str]:
    """返回模块的属性列表，合并 __all__ 和目标包的属性。"""
    return sorted(set(__all__) | set(vars(_target_pkg)))
