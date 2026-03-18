# -*- coding: utf-8 -*-
# Python version: 3.9
# @TianZhen

from __future__ import annotations
import array_api_compat as api
from array_api_compat.common._typing import Namespace
from typing import NamedTuple


class ArraySpec(NamedTuple):
    xp: Namespace
    dtype: object
    device: object


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
    """
    return isinstance(obj, Namespace) and (
        api.is_numpy_namespace(obj)
        or api.is_torch_namespace(obj)
        or api.is_cupy_namespace(obj)
        or api.is_jax_namespace(obj)
        or api.is_dask_namespace(obj)
        or api.is_ndonnx_namespace(obj)
        or api.is_pydata_sparse_namespace(obj)
    )
