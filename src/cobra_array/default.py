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
"""

from __future__ import annotations
import array_api_compat as api
from array_api_compat.common._typing import Namespace
from typing import (Tuple, Any)

from .convert import (np, torch)
from .exceptions import MissingDependencyError


# get defaults
DEFAULT_DTYPE = float
DEFAULT_DEVICE = "cpu"
DEFAULT_TORCH_NAMESPACE = api.array_namespace(torch.empty(0)) if torch is not None else None
DEFAULT_NUMPY_NAMESPACE = api.array_namespace(np.empty(0)) if np is not None else None


def default_array_spec() -> Tuple[Namespace, Tuple[Any, Any]]:
    """
    Try to get a suitable `array namespace` from the available array libraries in order of `PyTorch` > `NumPy`, and return it along with the default `dtype` and `device`.

    Returns
    -------
        Tuple[Namespace, Tuple[DTypeT, DeviceT]]
            A tuple containing the default `array namespace` and a tuple of the default `dtype` and default `device`.

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
    return default_xp, (DEFAULT_DTYPE, DEFAULT_DEVICE)



