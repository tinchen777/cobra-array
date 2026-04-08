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

Examples
--------
- Basic usage:

```python
from cobra_array.array_api import resolve_device, torch_xp, numpy_xp

r = resolve_device("cpu")        # "cpu"
r = resolve_device("cuda:0")     # "cuda:0"

if numpy_xp is not None:
    r = resolve_device("cpu", xp="numpy")

if torch_xp is not None:
    r = resolve_device("cpu", xp="torch")

if torch_xp is not None:
    r = resolve_device("cpu", xp=torch_xp)
```
"""

from __future__ import annotations
from typing import (Any, TYPE_CHECKING, Optional, Union)

from ._utils import (array_namespace_alias, is_compat_namespace)
from .exceptions import (
    CUDAUnavailableError,
    DeviceNotSupportedError
)

if TYPE_CHECKING:
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
    xp: Optional[Union[Any, ArrayLibraryName]] = None
) -> Optional[str]:
    """
    Get the device string from an object or a device specification string, and check if it is compatible with the specified `array namespace` if provided.

    Parameters
    ----------
        obj : object
            The input object or device specification string to extract the device information from.

        xp : Optional[Union[Any, ArrayLibraryName]], default is `None`
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

    Examples
    --------
    Basic parsing and normalization:

    >>> resolve_device("cpu")
    'cpu'
    >>> resolve_device(" CUDA:0 ")
    'cuda:0'
    >>> resolve_device(None)
    None

    Namespace compatibility checks:

    >>> resolve_device("cpu")
    'cpu'
    >>> resolve_device("cpu", xp="numpy")
    'cpu'
    >>> resolve_device("cpu", xp="torch")
    'cpu'

    Unsupported device for NumPy:

    >>> resolve_device("cuda:0", xp="numpy")
    Traceback (most recent call last):
        ...
    cobra_array.exceptions.DeviceNotSupportedError: ...

    Unsupported device type for PyTorch:

    >>> resolve_device("quantum", xp="torch")
    Traceback (most recent call last):
        ...
    cobra_array.exceptions.DeviceNotSupportedError: ...

    CUDA path for PyTorch (works with or without CUDA runtime):

    >>> from cobra_array.array_api import CUDA_AVAILABLE
    >>> result = None
    >>> if not CUDA_AVAILABLE:
    ...     try:
    ...         resolve_device("cuda:0", xp="torch")
    ...     except CUDAUnavailableError:
    ...         result = "CUDAUnavailableError"
    ... else:
    ...     result = resolve_device("cuda:0", xp="torch")
    >>> result in {"cuda:0", "CUDAUnavailableError"}
    True
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
        if isinstance(xp, str):
            xp_name = xp
        elif is_compat_namespace(xp):
            xp_name = xp.xp_name
        else:
            xp_name = array_namespace_alias(xp)
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
