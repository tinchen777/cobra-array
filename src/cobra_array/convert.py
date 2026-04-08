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
- :func:`to_array_namespace` or :func:`to_xp`: Convert an array library name to a `array namespace` or return the `array namespace` directly if is a supported namespace.
- :func:`as_array`: Convert an object to an array in the specified `array namespace`.

Examples
--------
- Basic usage::

    import numpy as np
    from cobra_array.convert import to_numpy, to_tensor, to_list, to_xp, as_array

    r = to_numpy([1, 2, 3])            # array([1, 2, 3])
    r = to_list(np.asarray([1, 2, 3]))  # [1, 2, 3]
    r = to_xp("numpy")                # NumPy namespace
    r = as_array([1, 2, 3], "numpy")   # NumPy array

- When PyTorch is available, the same helpers can be used with the torch namespace::

    from cobra_array.convert import as_array

    r = as_array([1, 2, 3], "torch")   # PyTorch tensor
"""

from collections import abc
import array_api_compat as api

from .array_api import (numpy_xp, torch_xp, resolve_device)
from ._utils import (warn, is_array_namespace, is_compat_namespace)
from .exceptions import (
    ParameterIgnoredWarning,
    MissingDependencyError,
    ConvertNoneTypeError,
    UnsupportedNamespaceError,
    UnsupportedArrayLibraryNameError,
    ArrayConversionError,
    NumPyConversionError,
    TorchConversionError
)

__all__ = [
    "to_numpy",
    "to_tensor",
    "to_list",
    "to_xp",
    "to_array_namespace",
    "as_array"
]


def to_numpy(obj, /, *, dtype=None, copy=True):
    """
    Convert the given object to a `NumPy array` (i.e. :class:`np.ndarray` instance).

    Parameters
    ----------
        obj : object
            The object to be converted to a `NumPy array`.
            - _torch.Tensor_(need :pkg:`torch`): Converted to a `NumPy array` after detaching and moving to CPU;
            - _set_: Converted to a `NumPy array` containing the elements of the set (order is not guaranteed);
            - _others_: Converted to a `NumPy array` directly.

        dtype : Optional[DTypeT], default to `None`
            The data type of the resulting `NumPy array`.
            - `None`: Use the default data type of the object.

        copy : bool, default to `True`
            Control whether to create a copy of the object when converting to a `NumPy array`.
            - `True`: Create a copy of the object;
            - `False`: A copy will only be made if necessary.

    Returns
    -------
        NDArray[Any]
            The converted `NumPy array` representation of the object.

    Raises
    ------
        MissingDependencyError
            If `NumPy` is not installed when calling this function.
        NumPyConversionError
            If an error occurs during conversion to a `NumPy array`.
    """
    if numpy_xp is None:
        raise MissingDependencyError("Dependency `NumPy` is required for `to_numpy()`.")

    if torch_xp is not None and isinstance(obj, torch_xp.Tensor):
        # as torch.Tensor
        obj = obj.detach().cpu()
    elif isinstance(obj, abc.Set):
        # as set
        obj = list(obj)
    try:
        if copy:
            return numpy_xp.array(obj, dtype=dtype, copy=True)
        return numpy_xp.asarray(obj, dtype=dtype)
    except Exception as e:
        raise NumPyConversionError(
            "An error occurred during conversion to NumPy array."
        ) from e


def to_tensor(obj, /, *, dtype=None, device=None, copy=True):
    """
    Convert the given object to a `PyTorch tensor` (i.e. :class:`torch.Tensor` instance).

    Parameters
    ----------
        obj : Any
            The object to be converted to a `PyTorch tensor`.
            - _set_: Converted to a `PyTorch tensor` containing the elements of the set (order is not guaranteed);
            - `None`: Raises `ConvertNoneTypeError`;
            - _others_: Converted to a `PyTorch tensor` directly.

        dtype : Optional[DTypeT], default to `None`
            The data type of the resulting `PyTorch tensor`.
            - `None`: Use the default data type of the object.

        device : Optional[DeviceT], default to `None`
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
    if torch_xp is None:
        raise MissingDependencyError("Dependency `PyTorch` is required for `to_tensor()`.")
    device = resolve_device(device, xp="torch")

    if isinstance(obj, torch_xp.Tensor):
        # as torch.Tensor
        return obj.to(dtype=dtype, device=device, copy=copy)

    if isinstance(obj, abc.Set):
        # as set
        obj = list(obj)
    try:
        if copy:
            return torch_xp.tensor(obj, dtype=dtype, device=device)
        return torch_xp.as_tensor(obj, dtype=dtype, device=device)  # type: ignore
    except Exception as e:
        raise TorchConversionError(
            "An error occurred during conversion to PyTorch tensor."
        ) from e


def to_list(obj, /, *, copy=True):
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

    if torch_xp is not None and isinstance(obj, torch_xp.Tensor):
        # as torch.Tensor
        return obj.detach().cpu().tolist()

    if numpy_xp is not None and isinstance(obj, numpy_xp.ndarray):
        # as numpy.ndarray
        return obj.tolist()

    if isinstance(obj, list):
        # as list
        return obj.copy() if copy else obj

    if isinstance(obj, abc.Iterable) and not isinstance(obj, (str, bytes)):
        # as iterable (not including str and bytes)
        return list(obj)
    # as scalar or others
    return [obj]


def to_xp(obj, /):
    """
    Convert an array library name or `compatibility namespace` to a `array namespace` or return the `array namespace` directly if is a supported namespace.

    Parameters
    ----------
        obj : Union[Namespace, CompatNamespace, ArrayLibraryName]
            The `array namespace` or array library name.
            - _ArrayLibraryName_ (`"numpy"` or `"torch"`): Return the corresponding `array namespace` module;
            - _Namespace_: Return the namespace module directly.
            - _CompatNamespace_: Return the underlying `array namespace` module.

    Returns
    -------
        Namespace
            The `array namespace` module corresponding to the input.

    Raises
    ------
        MissingDependencyError
            If the required array library for the specified `array namespace` is not installed.
        UnsupportedArrayLibraryNameError
            If an unsupported `array namespace` name is specified.
        UnsupportedNamespaceError
            If the input is not a supported `array namespace` name or module.
    """
    if isinstance(obj, str):
        if obj == "numpy":
            if numpy_xp is None:
                raise MissingDependencyError("Dependency `NumPy` is required for using array namespace.")
            return numpy_xp

        if obj == "torch":
            if torch_xp is None:
                raise MissingDependencyError("Dependency `PyTorch` is required for using array namespace.")
            return torch_xp

        raise UnsupportedArrayLibraryNameError(
            "Parameter `obj` of `to_xp()` must be a supported array namespace "
            f"name ('numpy', 'torch'), got {obj!r}."
        )

    if is_array_namespace(obj):
        return obj  # type: ignore

    if is_compat_namespace(obj):
        return obj.xp  # type: ignore

    raise UnsupportedNamespaceError(
        f"Parameter `obj` of `to_xp()` is not a supported array namespace, got {obj!r}."
    )


to_array_namespace = to_xp  # alias of `to_xp()`


def as_array(obj, xp, /, *, dtype=None, device=None, copy=False, arraylike_only=False):
    """
    Convert the given object to an array in the specified `array namespace` (e.g., `NumPy` or `PyTorch`).

    Parameters
    ----------
        obj : object
            The object to be converted to an array.

        xp : Union[object, ArrayLibraryName]
            The target `array namespace` or array library name for the conversion.
            - _ArrayLibraryName_ (`"numpy"` or `"torch"`): Converted to a `NumPy array` or `PyTorch tensor` respectively using the corresponding conversion functions;
            - _Namespace_ or _CompatNamespace_: Converted to an array using the `asarray()` function provided by the namespace module, which must be compatible with the array API standard.

        dtype : Optional[DTypeT], default to `None`
            The data type of the resulting array.
            - `None`: Use the default data type of the object.

        device : Optional[DeviceT], default to `None`
            The device on which the resulting array will be allocated (only if `array namespace` supports it).

        copy : bool, default to `False`
            Control whether to create a copy of the object when converting to an array.

        arraylike_only : bool, default to `False`
            Whether to only convert array-like objects to arrays in the specified `array namespace`, and return the object itself if it is not array-like.

    Returns
    -------
        ArrayLike[Any]
            The converted array representation of the object in the specified `array namespace`.
        object
            If :param:`arraylike_only` is `True` and the object is not array-like.

    Raises
    ------
        Refer to :func:`convert.to_xp`, :func:`convert.to_numpy` and :func:`convert.to_tensor` for possible exceptions.

        AttributeError
            If an unsupported `array namespace` is specified.
        ArrayConversionError
            If an error occurs during array conversion in the specified `array namespace`.

    Examples
    --------
    Convert to NumPy namespace (when available):

    >>> from cobra_array.array_api import numpy_xp
    >>> if numpy_xp is not None:
    ...     as_array([1, 2, 3], "numpy")
    array([1, 2, 3])
    >>> if numpy_xp is not None:
    ...     as_array([1, 2, 3], numpy_xp)
    array([1, 2, 3])

    Convert to PyTorch namespace (when available):

    >>> from cobra_array.array_api import torch_xp
    >>> if torch_xp is not None:
    ...     as_array([1, 2, 3], "torch")
    tensor([1, 2, 3])
    >>> if torch_xp is not None:
    ...     as_array([1, 2, 3], torch_xp)
    tensor([1, 2, 3])

    Pass through non-array-like objects with `arraylike_only=True`:

    >>> marker = object()
    >>> as_array(marker, "numpy", arraylike_only=True) is marker
    True

    Unsupported namespace name raises an error:

    >>> as_array([1, 2, 3], "unknown")
    Traceback (most recent call last):
        ...
    cobra_array.exceptions.UnsupportedArrayLibraryNameError: ...
    """
    if arraylike_only and not api.is_array_api_obj(obj):
        return obj

    arr_xp = to_xp(xp)
    if api.is_numpy_namespace(arr_xp):
        # as NumPy array
        if device is not None and device != "cpu":
            warn(
                "NumPy array does not support setting a non-CPU device. "
                f"Parameter `device` is ignored, got {device!r}.",
                category=ParameterIgnoredWarning
            )
        return to_numpy(obj, dtype=dtype, copy=copy)
    if api.is_torch_namespace(arr_xp):
        # as PyTorch tensor
        return to_tensor(obj, dtype=dtype, device=device, copy=copy)
    # as other namespace object
    try:
        return arr_xp.asarray(obj, dtype=dtype, device=device, copy=copy)  # type: ignore
    except AttributeError:
        raise AttributeError(
            f"Parameter `xp` of `as_array()` is not a supported array namespace: {xp!r}"
        ) from None
    except Exception as e:
        raise ArrayConversionError(
            f"An error occurred during array conversion in the specified array namespace {xp!r}"
        ) from e
