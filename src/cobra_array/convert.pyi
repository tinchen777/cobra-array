# src/cobra_array/convert.py
# -*- coding: utf-8 -*-
# Python version: 3.9
# @TianZhen

from __future__ import annotations
from numpy.typing import NDArray
from torch import Tensor
from array_api_compat.common._typing import Namespace
from typing import (Any, Literal, Optional, List, Iterable, Union, overload)

from .types import (
    T, StringT,
    DTypeT, dtypeT, DType, Device,
    ArrayLike, ArrayLibraryName
)


# === to_numpy() ===
@overload
def to_numpy(
    obj: NDArray[dtypeT],
    /, *,
    dtype: None = ...,
    copy: bool = ...
) -> NDArray[dtypeT]: ...


@overload
def to_numpy(
    obj: ArrayLike[dtypeT],
    /, *,
    dtype: None = ...,
    copy: bool = ...
) -> NDArray[dtypeT]: ...


@overload
def to_numpy(
    obj: object,
    /, *,
    dtype: None = ...,
    copy: bool = ...
) -> NDArray[Any]: ...


@overload
def to_numpy(
    obj: object,
    /, *,
    dtype: DTypeT,
    copy: bool = ...
) -> NDArray[DTypeT]: ...


def to_numpy(
    obj: object,
    /, *,
    dtype: Optional[DType] = None,
    copy: bool = True
) -> NDArray[Any]: ...


# === to_tensor() ===
def to_tensor(
    obj: object,
    /, *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
    copy: bool = True
) -> Tensor: ...


# === to_list() ===
@overload
def to_list(obj: List[T], /, *, copy: bool = ...) -> List[T]: ...
@overload
def to_list(obj: StringT, /, *, copy: bool = ...) -> List[StringT]: ...
@overload
def to_list(obj: NDArray[dtypeT], /, *, copy: bool = ...) -> List[dtypeT]: ...
@overload
def to_list(obj: ArrayLike[dtypeT], /, *, copy: bool = ...) -> List[dtypeT]: ...
@overload
def to_list(obj: Iterable[T], /, *, copy: bool = ...) -> List[T]: ...
@overload
def to_list(obj: T, /, *, copy: bool = ...) -> List[T]: ...
@overload
def to_list(obj: object, /, *, copy: bool = ...) -> List[Any]: ...
def to_list(obj: object, /, *, copy: bool = True) -> List[Any]: ...


# === to_xp() ===
def to_xp(obj: object, /) -> Namespace: ...


# === as_array() ===
@overload
def as_array(
    obj: NDArray[dtypeT],
    xp: Literal["numpy"],
    /, *,
    dtype: None = ...,
    device: Optional[Device] = ...,
    copy: bool = ...
) -> NDArray[dtypeT]: ...


@overload
def as_array(
    obj: ArrayLike[dtypeT],
    xp: Literal["numpy"],
    /, *,
    dtype: None = ...,
    device: Optional[Device] = ...,
    copy: bool = ...
) -> NDArray[dtypeT]: ...


@overload
def as_array(
    obj: object,
    xp: Literal["numpy"],
    /, *,
    dtype: None = ...,
    device: Optional[Device] = ...,
    copy: bool = ...
) -> NDArray[Any]: ...


@overload
def as_array(
    obj: object,
    xp: Literal["numpy"],
    /, *,
    dtype: DTypeT,
    device: Optional[Device] = ...,
    copy: bool = ...
) -> NDArray[DTypeT]: ...


@overload
def as_array(
    obj: object,
    xp: Literal["torch"],
    /, *,
    dtype: Optional[DType] = ...,
    device: Optional[Device] = ...,
    copy: bool = ...
) -> Tensor: ...


@overload
def as_array(
    obj: NDArray[dtypeT],
    xp: Union[Namespace, ArrayLibraryName],
    /, *,
    dtype: None = ...,
    device: Optional[Device] = ...,
    copy: bool = ...
) -> ArrayLike[dtypeT]: ...


@overload
def as_array(
    obj: ArrayLike[dtypeT],
    xp: Union[Namespace, ArrayLibraryName],
    /, *,
    dtype: None = ...,
    device: Optional[Device] = ...,
    copy: bool = ...
) -> ArrayLike[dtypeT]: ...


@overload
def as_array(
    obj: object,
    xp: Union[Namespace, ArrayLibraryName],
    /, *,
    dtype: None = ...,
    device: Optional[Device] = ...,
    copy: bool = ...
) -> ArrayLike[Any]: ...


@overload
def as_array(
    obj: object,
    xp: Union[Namespace, ArrayLibraryName],
    /, *,
    dtype: DTypeT,
    device: Optional[Device] = ...,
    copy: bool = ...
) -> ArrayLike[DTypeT]: ...


def as_array(
    obj: object,
    xp: Union[Namespace, ArrayLibraryName],
    /, *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
    copy: bool = False
) -> ArrayLike[Any]: ...


# === as_array_if_like() ===
@overload
def as_array_if_like(
    obj: NDArray[dtypeT],
    xp: Literal["numpy"],
    /, *,
    dtype: None = ...,
    device: Optional[Device] = ...,
    copy: bool = ...
) -> NDArray[dtypeT]: ...


@overload
def as_array_if_like(
    obj: ArrayLike[dtypeT],
    xp: Literal["numpy"],
    /, *,
    dtype: None = ...,
    device: Optional[Device] = ...,
    copy: bool = ...
) -> NDArray[dtypeT]: ...


@overload
def as_array_if_like(
    obj: ArrayLike[Any],
    xp: Literal["numpy"],
    /, *,
    dtype: DTypeT,
    device: Optional[Device] = ...,
    copy: bool = ...
) -> NDArray[DTypeT]: ...


@overload
def as_array_if_like(
    obj: ArrayLike[Any],
    xp: Literal["torch"],
    /, *,
    dtype: Optional[DType] = ...,
    device: Optional[Device] = ...,
    copy: bool = ...
) -> Tensor: ...


@overload
def as_array_if_like(
    obj: ArrayLike[dtypeT],
    xp: Union[Namespace, ArrayLibraryName],
    /, *,
    dtype: None = ...,
    device: Optional[Device] = ...,
    copy: bool = ...
) -> ArrayLike[dtypeT]: ...


@overload
def as_array_if_like(
    obj: ArrayLike[Any],
    xp: Union[Namespace, ArrayLibraryName],
    /, *,
    dtype: DTypeT,
    device: Optional[Device] = ...,
    copy: bool = ...
) -> ArrayLike[DTypeT]: ...
@overload
def as_array_if_like(obj: T, xp: Any, /) -> T: ...


def as_array_if_like(
    obj: object,
    xp: Union[Namespace, ArrayLibraryName],
    /, *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
    copy: bool = False
) -> Any: ...
