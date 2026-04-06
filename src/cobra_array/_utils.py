# -*- coding: utf-8 -*-
# Python version: 3.9
# @TianZhen

from __future__ import annotations
import array_api_compat as api
import warnings
from types import ModuleType
from typing import Any

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


def array_namespace_alias(xp: object) -> str:
    """
    Get the alias of the `array namespace`.

    Parameters
    ----------
        xp : object
            The `array namespace`.

    Returns
    -------
        str
            The alias of the `array namespace`.
            - Including: `"NumPy"`, `"Cupy"`, `"PyTorch"`, `"NDONNX"`, `"Dask"`, `"JAX"`, `"sparse"` and `"array-api-strict"`.

    Raises
    ------
        UnsupportedNameSpaceError
            If the input object is not a supported namespace.
    """
    if isinstance(xp, ModuleType):
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

    raise UnsupportedNamespaceError(
        f"Got unsupported array namespace of type {type(xp)}."
    )


def is_array_namespace(obj: object) -> bool:
    """
    Returns `True` if input is a supported `array namespace`.
    """
    try:
        array_namespace_alias(obj)
        return True
    except UnsupportedNamespaceError:
        return False


def is_compat_namespace(xp: object) -> bool:
    """
    Returns `True` if input is a `compatibility namespace` wrapped by :class:`CompatNamespace`
    """
    return "(compat)" in getattr(xp, "__name__", "")
