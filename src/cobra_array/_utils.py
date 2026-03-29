# -*- coding: utf-8 -*-
# Python version: 3.9
# @TianZhen

from __future__ import annotations
import array_api_compat as api
from array_api_compat.common._typing import Namespace
import warnings
from typing import (Any, NamedTuple, TYPE_CHECKING)

from .exceptions import UnsupportedNameSpaceError

if TYPE_CHECKING:
    from .types import (DType, Device)

# Try to import `cobra_log.warning`.
try:
    from cobra_log import warning
    _WARN_AVAILABLE = True
except ImportError:
    _WARN_AVAILABLE = False


def warn(msg: str, /, category: Any, stack: int = 2):
    if _WARN_AVAILABLE:
        return warning(msg, stack=stack)
    return warnings.warn(msg, category=category, stacklevel=stack+1)


class ArraySpec(NamedTuple):
    """
    A named tuple to hold the specifications of an array.
    - `xp`: Namespace
    - `dtype`: DType
    - `device`: Device
    """
    xp: Namespace
    dtype: DType
    device: Device
# TODO


def array_namespace_alias(xp: object) -> str:
    """
    Get the alias of the `array namespace`.

    Parameters
    ----------
        xp : object
            The `array namespace` object.

    Returns
    -------
        str
            The alias of the `array namespace`.
            Including: `"NumPy"`, `"Cupy"`, `"PyTorch"`, `"NDONNX"`, `"Dask"`, `"JAX"`, `"sparse"` and `"array-api-strict"`.

    Raises
    ------
        UnsupportedNameSpaceError
            If the input object is not a supported `array namespace`.
    """
    if is_cobra_array_namespace(xp):
        return getattr(xp, "xp_name")

    if isinstance(xp, Namespace):
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

    raise UnsupportedNameSpaceError(f"Got unsupported array namespace of type {type(xp)}.")


def is_cobra_array_namespace(xp: object) -> bool:
    """
    Returns `True` if :param:`xp` is a `array namespace` wrapped by :class:`NameSpace`
    """
    return "cobra_array" in getattr(xp, __name__, "")


def is_array_namespace(obj: object) -> bool:
    """
    Check if the input object is a supported `array namespace`.

    Parameters
    ----------
        obj : object
            The object to be checked.

    Returns
    -------
        bool

    Raises
    ------
        Refer to :func:`array_namespace_alias` for possible exceptions.
    """
    try:
        array_namespace_alias(obj)
        return True
    except UnsupportedNameSpaceError:
        return False
