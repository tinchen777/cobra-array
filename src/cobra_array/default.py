# src/cobra_array/default.py
# -*- coding: utf-8 -*-
# Python version: 3.9
# @TianZhen
"""
Default utilities for :pkg:`cobra_array`.

Attributes
----------
- :attr:`DEFAULT_DTYPE`: The default data type for arrays.
- :attr:`DEFAULT_DEVICE`: The default device for arrays.
- :attr:`DEFAULT_TORCH_NAMESPACE`: The default array namespace for PyTorch.
- :attr:`DEFAULT_NUMPY_NAMESPACE`: The default array namespace for NumPy.
Functions
---------
- :func:`default_array_spec`: Get the default array specification.
- :func:`as_default_array`: Convert an array-like object to an array in the default context.
"""

from __future__ import annotations

from .convert import (as_array, np, torch)
from ._utils import ArraySpec
from .exceptions import MissingDependencyError


# get defaults
DEFAULT_DTYPE = float
DEFAULT_DEVICE = "cpu"
DEFAULT_TORCH_NAMESPACE = torch
DEFAULT_NUMPY_NAMESPACE = np


def default_array_spec() -> ArraySpec:
    """
    Try to get a suitable `array namespace` from the available array libraries in order of `PyTorch` > `NumPy`, and return it along with the default `dtype` and `device`.

    Returns
    -------
        ArraySpec
            An :class:`ArraySpec` named tuple containing the default `xp`(`array namespace`), `dtype` and `device`.

    Raises
    ------
        MissingDependencyError
            If all default array libraries (`NumPy` and `PyTorch`) are missing.
    """
    default_xp = DEFAULT_TORCH_NAMESPACE or DEFAULT_NUMPY_NAMESPACE
    if default_xp is None:
        raise MissingDependencyError(
            "Missing all default array libraries (`PyTorch` > `NumPy`)."
        )
    return ArraySpec(default_xp, DEFAULT_DTYPE, DEFAULT_DEVICE)


def as_default_array(
    obj: object,
    /,
    unify_dtype: bool = True,
    unify_device: bool = True,
    copy: bool = False
):
    """
    Convert an array-like object to an array in default `array namespace` with the default `dtype` and `device` if specified.

    Parameters
    ----------
        obj : object
            The object to be converted to an array.

        unify_dtype : bool, default to `True`
            Whether to unify the `dtype` of the converted array to that of the default context.

        unify_device : bool, default to `True`
            Whether to unify the `device` of the converted array to that of the default context.

        copy : bool, default to `False`
            Whether to return a copy of the array if it is already in the default context namespace.

    Returns
    -------
        Any
        The converted array representation of the object in the default context `array namespace`, with the default `dtype` and `device` if specified.

    Raises
    ------
        Refer to :func:`convert.as_array`, :func:`default.default_array_spec` for possible exceptions.
    """
    arr_spec = default_array_spec()
    return as_array(
        obj, arr_spec.xp,
        arr_spec.dtype if unify_dtype else None,
        arr_spec.device if unify_device else None,
        copy
    )
