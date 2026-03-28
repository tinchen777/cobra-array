# src/cobra_array/array_api.py
# -*- coding: utf-8 -*-
# Python version: 3.9
# @TianZhen
"""
Array API utilities for :pkg:`cobra_array`.

Attributes
----------
- :attr:`torch_xp`: The `PyTorch` array namespace from :pkg:`array_api_compat` if `PyTorch` is available, otherwise `None`.
- :attr:`numpy_xp`: The `NumPy` array namespace from :pkg:`array_api_compat` if `NumPy` is available, otherwise `None`.
- :attr:`CUDA_AVAILABLE`: A boolean indicating whether CUDA is available for `PyTorch`.
- :attr:`TORCH_SUPPORTED_DEVICES`: The set of devices supported by `PyTorch`.
- :attr:`NUMPY_SUPPORTED_DEVICES`: The set of devices supported by `NumPy`.
Functions
---------
- :func:`resolve_device`: Get the device string from an object or a device specification string, and check if it is compatible with the specified `array namespace` if provided.
"""

from __future__ import annotations
from typing import (TYPE_CHECKING, Optional, Union)

from ._utils import array_namespace_alias
from .exceptions import (
    CUDAUnavailableError,
    DeviceNotSupportedError
)

if TYPE_CHECKING:
    from array_api_compat.common._typing import Namespace
    from .types import ArrayLibraryName


# === PyTorch ===
TORCH_SUPPORTED_DEVICES = {"cpu", "cuda", "xpu", "mkldnn", "opengl", "opencl", "ideep", "hip", "ve", "ort", "mlc", "xla", "lazy", "vulkan", "meta", "hpu"}
try:
    from array_api_compat import torch as torch_xp
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch_xp = None
    CUDA_AVAILABLE = False

# === NumPy ===
NUMPY_SUPPORTED_DEVICES = {"cpu"}
try:
    from array_api_compat import numpy as numpy_xp
except ImportError:
    numpy_xp = None


def resolve_device(
    obj: object,
    /, *,
    xp: Optional[Union[Namespace, ArrayLibraryName]] = None
) -> Optional[str]:
    """
    Get the device string from an object or a device specification string, and check if it is compatible with the specified `array namespace` if provided.

    Parameters
    ----------
        obj : object
            The input object or device specification string to extract the device information from.

        xp : Optional[Union[Namespace, ArrayLibraryName]], default is `None`
            The `array namespace` to check the device compatibility against.
            - `None`: No compatibility check will be performed.

    Returns
    -------
        str
            The device string extracted from :param:`obj`.
        None
            If :param:`obj` is `None`.

    Raises
    ------
        Refer to :func:`array_namespace_alias` for possible exceptions.

        DeviceNotSupportedError
            If the extracted device is not compatible with the specified `array namespace`.
        CUDAUnavailableError
            If a CUDA device is specified but CUDA is not available for `PyTorch`.
    """
    # source
    if obj is None:
        return None
    source = str(obj).lower()
    # check for namespace
    if ":" in source:
        s_type, s_index = source.split(":", 1)
    else:
        s_type, s_index = source, ""
    s_type = s_type.strip()
    s_index = s_index.strip()
    s_fmt = f"{s_type}:{s_index}" if s_index else s_type

    if xp is not None:
        xp_name = xp if isinstance(xp, str) else array_namespace_alias(xp)
        if xp_name in ("numpy", "NumPy"):
            # NumPy
            if s_type != "cpu":
                raise DeviceNotSupportedError(f"`NumPy` only support CPU device, got {s_fmt}.")
        elif xp_name in ("torch", "PyTorch"):
            # PyTorch
            if s_type == "cuda":
                # check if CUDA is available
                if not CUDA_AVAILABLE:
                    raise CUDAUnavailableError("`PyTorch` specified a CUDA device but CUDA is not available.")
            else:
                if s_type not in TORCH_SUPPORTED_DEVICES:
                    raise DeviceNotSupportedError(f"`PyTorch` supports devices {TORCH_SUPPORTED_DEVICES}, got {s_fmt}.")
    return s_fmt
