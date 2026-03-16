# src/cobra_array/convert.py
# -*- coding: utf-8 -*-
# Python version: 3.9
# @TianZhen
"""
Conversion utilities for :pkg:`cobra_array`.

Functions
---------
- :func:`to_numpy`: Convert an object to a `NumPy array`.
- :func:`to_tensor`: Convert an object to a `PyTorch tensor`.
- :func:`to_list`: Convert an object to a built-in `list`.
- :func:`as_array`: Convert an object to an array in the specified `array namespace`.
- :func:`as_array_if_like`: Convert an array-like object to an array in the specified `array namespace`, otherwise return the object itself.
"""

from __future__ import annotations
from collections import abc
import array_api_compat as api
from array_api_compat.common._typing import Namespace
import warnings
from typing import (Any, Optional, List, Iterable, Union, overload)

from .exceptions import (
    ParameterIgnoredWarning,
    MissingDependencyError,
    ConvertNoneTypeError,
    UnsupportedNameSpaceError,
    ArrayConversionError,
    NumPyConversionError,
    TorchConversionError,
    CUDAUnavailableError
)
from .types import (_T, _T_Str, DTypeT, DeviceT, ArrayLike, ArrayLibraryName)

__all__ = [
    "to_numpy",
    "to_tensor",
    "to_list",
    "as_array",
    "as_array_if_like"
]

# Try to import `torch`.
try:
    import torch
except ImportError:
    torch = None
# Try to import `numpy`.
try:
    import numpy as np
except ImportError:
    np = None
# Try to import `cobra_log.warning`.
try:
    from cobra_log import warning
    _WARN_AVAILABLE = True
except ImportError:
    _WARN_AVAILABLE = False


def warn(msg: str, category: Any, stack: int = 2):
    if _WARN_AVAILABLE:
        return warning(msg, stack=stack)
    return warnings.warn(msg, category=category, stacklevel=stack+1)


def to_numpy(
    obj: object,
    /,
    dtype: Optional[DTypeT] = None,
    copy: bool = True
):
    """
    Convert the given object to a `NumPy array` (i.e. :class:`np.ndarray` instance).

    Parameters
    ----------
        obj : object
            The object to be converted to a `NumPy array`.
            - _torch.Tensor_(need :pkg:`torch`): Converted to a `NumPy array` after detaching and moving to CPU;
            - _set_: Converted to a `NumPy array` containing the elements of the set (order is not guaranteed);
            - _others_: Converted to a `NumPy array` directly.

        dtype : Optional[T_DType], default to `None`
            The data type of the resulting `NumPy array`.
            - `None`: Use the default data type of the object.

        copy : bool, default to `True`
            Control whether to create a copy of the object when converting to a `NumPy array`.
            - `True`: Create a copy of the object;
            - `False`: A copy will only be made if necessary.

    Returns
    -------
        NDArray
            The converted `NumPy array` representation of the object.

    Raises
    ------
        MissingDependencyError
            If `NumPy` is not installed when calling this function.
        NumPyConversionError
            If an error occurs during conversion to a `NumPy array`.
    """
    if np is None:
        raise MissingDependencyError("Dependency `NumPy` is required for `to_numpy()`.")

    if torch is not None and isinstance(obj, torch.Tensor):
        # as torch.Tensor
        obj = obj.detach().cpu()
    elif isinstance(obj, abc.Set):
        # as set
        obj = list(obj)
    try:
        if copy:
            return np.array(obj, dtype=dtype, copy=True)
        return np.asarray(obj, dtype=dtype)
    except Exception as e:
        raise NumPyConversionError(
            "An error occurred during conversion to NumPy array."
        ) from e


def to_tensor(
    obj: object,
    /,
    dtype: Optional[DTypeT] = None,
    device: Optional[DeviceT] = None,
    copy: bool = True
):
    """
    Convert the given object to a `PyTorch tensor` (i.e. :class:`torch.Tensor` instance).

    Parameters
    ----------
        obj : Any
            The object to be converted to a `PyTorch tensor`.
            - _set_: Converted to a `PyTorch tensor` containing the elements of the set (order is not guaranteed);
            - `None`: Raises `ConvertNoneTypeError`;
            - _others_: Converted to a `PyTorch tensor` directly.

        dtype : Optional[T_DType], default to `None`
            The data type of the resulting `PyTorch tensor`.
            - `None`: Use the default data type of the object.

        device : Optional[T_Device], default to `None`
            The device on which the resulting `PyTorch tensor` will be allocated.
            - `None`: Use the default device (usually `"cpu"`).

        copy : bool, default to `True`
            Control whether to create a copy of the object when converting to a `PyTorch tensor`.
            - `True`: Create a copy of the object;
            - `False`: A copy will only be made if necessary.

    Returns
    -------
        torch.Tensor
            The converted `PyTorch tensor` representation of the object.

    Raises
    ------
        MissingDependencyError
            If `PyTorch` is not installed when calling this function.
        ConvertNoneTypeError
            If `None` is passed as the object to be converted.
        CUDAUnavailableError
            If a non-CPU device is specified but CUDA is not available.
        TorchConversionError
            If an error occurs during conversion to a `PyTorch tensor`.
    """
    if obj is None:
        raise ConvertNoneTypeError("Can not convert `NoneType` to a PyTorch tensor.")
    if torch is None:
        raise MissingDependencyError("Dependency `PyTorch` is required for `to_tensor()`.")
    if device is not None and device != "cpu" and not torch.cuda.is_available():
        raise CUDAUnavailableError(
            f"Parameter `device` of `to_tensor()` specifies a non-CPU device {device!r} but CUDA is not available."
        )

    if isinstance(obj, torch.Tensor):
        # as torch.Tensor
        return obj.to(dtype=dtype, device=device, copy=copy)

    if isinstance(obj, abc.Set):
        # as set
        obj = list(obj)
    try:
        if copy:
            return torch.tensor(obj, dtype=dtype, device=device)
        return torch.as_tensor(obj, dtype=dtype, device=device)  # type: ignore
    except Exception as e:
        raise TorchConversionError(
            "An error occurred during conversion to PyTorch tensor."
        ) from e


@overload
def to_list(obj: List[_T], /, copy: bool = ...) -> List[_T]: ...
@overload
def to_list(obj: _T_Str, /, copy: bool = ...) -> List[_T_Str]: ...
@overload
def to_list(obj: Iterable[_T], /, copy: bool = ...) -> List[Any]: ...
@overload
def to_list(obj: _T, /, copy: bool = ...) -> List[_T]: ...


def to_list(obj: object, /, copy: bool = True) -> List[Any]:
    """
    Convert the given object to a built-in `list`.

    Parameters
    ----------
        obj : Any
            The object to be converted to a `list`.
            - _torch.Tensor_(need :pkg:`torch`): Converted to a `list` after detaching and moving to CPU;
            - _np.ndarray_(need :pkg:`numpy`): Converted to a `list`;
            - _Iterable_: Converted to a `list` containing the elements of the iterable (order is preserved);
            - _scalar_(including _str_ and _bytes_): Converted to a `list` containing the scalar value as its single element;
            - `None`: Raises `ConvertNoneTypeError`.

        copy : bool, default to `True`
            Control whether to create a copy when :param:`obj` is already a `list`. Other types of :param:`obj` will always be converted to a new `list`.

    Returns
    -------
        List[Any]
            The converted `list` representation of the object.

    Raises
    ------
        ConvertNoneTypeError
            If `None` is passed as the object to be converted.
    """
    if obj is None:
        # as NoneType
        raise ConvertNoneTypeError("Can not convert `NoneType` to a built-in list.")

    if torch is not None and isinstance(obj, torch.Tensor):
        # as torch.Tensor
        return obj.detach().cpu().tolist()

    if np is not None and isinstance(obj, np.ndarray):
        # as np.ndarray
        return obj.tolist()

    if isinstance(obj, list):
        # as list
        return obj.copy() if copy else obj

    if isinstance(obj, abc.Iterable) and not isinstance(obj, (str, bytes)):
        # as iterable (not including str and bytes)
        return list(obj)
    # as scalar or others
    return [obj]


def as_array(
    obj: object,
    xp: Union[Namespace, ArrayLibraryName],
    /,
    dtype: Optional[DTypeT] = None,
    device: Optional[DeviceT] = None,
    copy: bool = False
) -> ArrayLike:
    """
    Convert the given object to an array in the specified `array namespace` (e.g., `NumPy` or `PyTorch`).

    Parameters
    ----------
        obj : object
            The object to be converted to an array.

        xp : Union[Namespace, ArrayLibraryName]
            The target `array namespace` or array library name for the conversion.
            - _ArrayLibraryName_ (`"numpy"` or `"torch"`): Converted to a `NumPy array` or `PyTorch tensor` respectively using the corresponding conversion functions;
            - _Namespace_: Converted to an array using the `asarray()` function provided by the namespace module, which must be compatible with the array API standard.

        dtype : Optional[DTypeT], default to `None`
            The data type of the resulting array.
            - `None`: Use the default data type of the object.

        device : Optional[DeviceT], default to `None`
            The device on which the resulting array will be allocated (only if `array namespace` supports it).

        copy : bool, default to `False`
            Control whether to create a copy of the object when converting to an array.

    Returns
    -------
        ArrayLike
            The converted array representation of the object in the specified `array namespace`.

    Raises
    ------
        Refer to :func:`to_numpy` and :func:`to_tensor` for possible exceptions.

        UnsupportedNameSpaceError
            If an unsupported `array namespace` is specified.
        ArrayConversionError
            If an error occurs during array conversion in the specified `array namespace`.
    """
    if getattr(xp, "__name__", None) is not None:
        if api.is_numpy_namespace(xp):
            xp = "numpy"
        elif api.is_torch_namespace(xp):
            xp = "torch"

    if isinstance(xp, str):
        if xp == "numpy":
            # as NumPy array
            if device is not None and device != "cpu":
                warn(
                    "NumPy array does not support setting a non-CPU device. "
                    f"Parameter `device` is ignored, got {device!r}.",
                    category=ParameterIgnoredWarning
                )
            return to_numpy(obj, dtype=dtype, copy=copy)
        if xp == "torch":
            # as PyTorch tensor
            return to_tensor(obj, dtype=dtype, device=device, copy=copy)
        # as unsupported array namespace name
        raise UnsupportedNameSpaceError(
            "Parameter `xp` of `as_array()` must be one of the supported "
            f"array namespace names ('numpy', 'torch'), got {xp!r}."
        )
    # as other namespace object
    try:
        return xp.asarray(obj, dtype=dtype, device=device, copy=copy)
    except AttributeError:
        raise UnsupportedNameSpaceError(
            f"Parameter `xp` of `as_array()` is not a supported array namespace: {xp!r}"
        ) from None
    except Exception as e:
        raise ArrayConversionError(
            f"An error occurred during array conversion in the specified array namespace {xp!r}"
        ) from e


def as_array_if_like(
    obj: object,
    xp: Union[Namespace, ArrayLibraryName],
    /,
    dtype: Optional[DTypeT] = None,
    device: Optional[DeviceT] = None,
    copy: bool = False
):
    """
    Convert an array-like object to an array in the specified `array namespace`, otherwise return the object itself.

    Returns
    -------
        ArrayLike
            If the object is array-like, the converted array representation of the object in the specified `array namespace`.
        object
            If the object is not array-like, the object itself is returned without conversion.

    Raises
    ------
        Refer to :func:`as_array` for possible exceptions.

    Notes
    -----
    - All parameters follow the usage conventions of :func:`as_array`.
    """
    if api.is_array_api_obj(obj):
        return as_array(obj, xp, dtype=dtype, device=device, copy=copy)
    return obj
