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
- :attr:`TORCH_COMPAT_NAMESPACE`: The `compatibility namespace` for `PyTorch`.
- :attr:`NUMPY_COMPAT_NAMESPACE`: The `compatibility namespace` for `NumPy`.
Functions
---------
- :func:`default_spec`: Get the default array specification.
- :func:`as_default`: Convert an array-like object to a :class:`CompatArray` array in the default context.
"""

from __future__ import annotations
from typing import (Any, Literal, TYPE_CHECKING, NamedTuple, overload)

from .compat import (CompatNamespace, wrap_arraylike)
from .convert import as_array
from .array_api import (numpy_xp, torch_xp)
from .exceptions import MissingDependencyError

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from .compat import CompatArray
    from .types import (ArrayLike, DType, dtypeT, T)


class ArraySpec(NamedTuple):
    """
    A named tuple to hold the specifications of an array.
    - `cxp`: CompatNamespace
    - `dtype`: DType
    - `device`: Any
    """
    cxp: CompatNamespace
    dtype: DType
    device: Any

    @classmethod
    def create(cls, xp: object, dtype: Any, device: Any) -> ArraySpec:
        """Create an `ArraySpec` instance, convert the `xp` to a :class:`CompatNamespace` instance if it is not already one."""
        return cls(
            cxp=CompatNamespace(xp),
            dtype=dtype,
            device=device,
        )


# get defaults
DEFAULT_DTYPE = float
DEFAULT_DEVICE = "cpu"
NUMPY_COMPAT_NAMESPACE = CompatNamespace(numpy_xp) if numpy_xp is not None else None
TORCH_COMPAT_NAMESPACE = CompatNamespace(torch_xp) if torch_xp is not None else None


def default_spec() -> ArraySpec:
    """
    Try to get a suitable `compatibility namespace` from the available array libraries in order of `PyTorch` > `NumPy`, and return it along with the default `dtype` and `device`.

    Returns
    -------
        ArraySpec
            An :class:`ArraySpec` named tuple containing the default `cxp`(`compatibility namespace`), `dtype` and `device`.

    Raises
    ------
        MissingDependencyError
            If all default array libraries (`NumPy` and `PyTorch`) are missing.
    """
    default_xp = TORCH_COMPAT_NAMESPACE or NUMPY_COMPAT_NAMESPACE
    if default_xp is None:
        raise MissingDependencyError(
            "Missing all default array libraries (`PyTorch` > `NumPy`)."
        )
    return ArraySpec(default_xp, DEFAULT_DTYPE, DEFAULT_DEVICE)


@overload
def as_default(obj: NDArray[dtypeT], /, *, unify_dtype: Literal[False], unify_device: bool = ..., copy: bool = ..., arraylike_only: bool = ...) -> CompatArray[dtypeT, Literal["cpu"]]: ...
@overload
def as_default(obj: ArrayLike[dtypeT], /, *, unify_dtype: Literal[False], unify_device: bool = ..., copy: bool = ..., arraylike_only: bool = ...) -> CompatArray[dtypeT, Any]: ...
@overload
def as_default(obj: ArrayLike[Any], /, *, unify_dtype: Literal[True] = ..., unify_device: bool = ..., copy: bool = ..., arraylike_only: bool = ...) -> CompatArray[Any, Any]: ...
@overload
def as_default(obj: object, /, *, unify_dtype: bool = ..., unify_device: bool = ..., copy: bool = ..., arraylike_only: Literal[False] = ...) -> CompatArray[Any, Any]: ...
@overload
def as_default(obj: T, /, *, unify_dtype: bool = ..., unify_device: bool = ..., copy: bool = ..., arraylike_only: Literal[True]) -> T: ...


def as_default(
    obj: object,
    /, *,
    unify_dtype: bool = True,
    unify_device: bool = True,
    copy: bool = False,
    arraylike_only: bool = False
) -> Any:
    """
    Convert an array-like object to a :class:`CompatArray` array in default `compatibility namespace` with the default `dtype` and `device` if specified.

    Parameters
    ----------
        obj : object
            The object to be converted to a :class:`CompatArray` array.

        unify_dtype : bool, default to `True`
            Whether to unify the `dtype` of the converted array to that of the default context.

        unify_device : bool, default to `True`
            Whether to unify the `device` of the converted array to that of the default context.

        copy : bool, default to `False`
            Whether to return a copy of the array if it is already in the default context namespace.

        arraylike_only : bool, default to `False`
            Whether to only convert array-like objects to arrays in the default context namespace, and return the object itself if it is not array-like.

    Returns
    -------
        CompatArray[Any, Any]
            The converted array representation of the object in the default context `compatibility namespace`, with the default `dtype` and `device` if specified.
        object
            If :param:`arraylike_only` is `True` and the object is not array-like.

    Raises
    ------
        Refer to :func:`convert.as_array`, :func:`default.default_spec` for possible exceptions.
    """
    spec = default_spec()
    return wrap_arraylike(as_array(
        obj, spec.cxp,
        dtype=spec.dtype if unify_dtype else None,
        device=spec.device if unify_device else None,
        copy=copy,
        arraylike_only=arraylike_only
    ), xp=spec.cxp)
