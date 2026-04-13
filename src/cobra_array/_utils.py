# -*- coding: utf-8 -*-
# Python version: 3.9
# @TianZhen

from __future__ import annotations
import array_api_compat as api
import warnings
from types import ModuleType
from typing import (Any, overload, Optional, Literal)

from .exceptions import UnsupportedNamespaceError

# Try to import `cobra_log.warning`.
try:
    from cobra_log import warning
    _WARN_AVAILABLE = True
except ImportError:
    _WARN_AVAILABLE = False


def warn(msg: str, /, category: Any, stack: int = 2):
    """Issue a warning message."""
    if _WARN_AVAILABLE:
        return warning(msg, stack=stack)
    return warnings.warn(msg, category=category, stacklevel=stack+1)


@overload
def array_namespace_alias(xp: object, /, *, raise_on_unsupported: Literal[True] = ...) -> str: ...
@overload
def array_namespace_alias(xp: object, /, *, raise_on_unsupported: Literal[False]) -> Optional[str]: ...


def array_namespace_alias(xp: object, /, *, raise_on_unsupported: bool = True) -> Optional[str]:
    """
    Get the alias of the `array namespace`.

    Parameters
    ----------
        xp : object
            The `array namespace`.

        raise_on_unsupported : bool, optional
            Whether to raise an error for unsupported array namespaces.

    Returns
    -------
        str
            The alias of the `array namespace`.
            - Including: `"NumPy"`, `"Cupy"`, `"PyTorch"`, `"NDONNX"`, `"Dask"`, `"JAX"`, `"sparse"` and `"array-api-strict"`.
        `None`
            If the input is not a supported `array namespace` and `raise_on_unsupported` is `False`.

    Raises
    ------
        UnsupportedNameSpaceError
            If the input object is not a supported `array namespace` and :param:`raise_on_unsupported` is `True`.
    """
    if type(xp) is ModuleType:
        if api.is_numpy_namespace(xp):
            return "NumPy"

        if api.is_cupy_namespace(xp):
            return "Cupy"

        if api.is_torch_namespace(xp):
            return "PyTorch"

        if api.is_ndonnx_namespace(xp):
            return "NDONNX"

        if api.is_dask_namespace(xp):
            return "Dask"

        if api.is_jax_namespace(xp):
            return "JAX"

        if api.is_pydata_sparse_namespace(xp):
            return "sparse"

        if api.is_array_api_strict_namespace(xp):
            return "array-api-strict"

    if raise_on_unsupported:
        raise UnsupportedNamespaceError(
            f"Got unsupported array namespace of type {type(xp)}."
        )


def is_array_namespace(obj: object) -> bool:
    """
    Returns `True` if input is a supported `array namespace`.
    """
    return array_namespace_alias(obj, raise_on_unsupported=False) is not None


def is_compat_namespace(xp: object) -> bool:
    """
    Returns `True` if input is a `compatibility namespace` wrapped by :class:`CompatNamespace`
    """
    return "(compat)" in getattr(xp, "__name__", "")


# def is_array_api_object(obj: object) -> bool:
#     """
#     Returns `True` if input is an `array API object` that belongs to a supported `array namespace`.
#     """
#     return api.is_array_api_obj(obj)
