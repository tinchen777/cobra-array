# src/cobra_array/types.py
"""
Type definitions for :pkg:`cobra_array`.
"""

from __future__ import annotations
from typing import (Protocol, TypeVar, Any, Literal, NamedTuple, Generic)


T = TypeVar("T")
StringT = TypeVar("StringT", str, bytes)


class DType(Protocol):
    ...


class Device(Protocol):
    ...


DTypeT = TypeVar("DTypeT", bound=Any)
DeviceT = TypeVar("DeviceT", bound=Any)

ArrayLibraryName = Literal["numpy", "torch"]


class ArrayLike(Protocol):
    dtype: Any
    shape: Any

    def __array__(self) -> Any: ...


ArrayT = TypeVar("ArrayT", bound=ArrayLike)


class UniqueAllResult(NamedTuple, Generic[T]):
    values: T
    indices: T
    inverse_indices: T
    counts: T


class UniqueCountsResult(NamedTuple, Generic[T]):
    values: T
    counts: T


class UniqueInverseResult(NamedTuple, Generic[T]):
    values: T
    inverse_indices: T


# UniqueAllResult = namedtuple("UniqueAllResult", ["values", "indices", "inverse_indices", "counts"])
# UniqueCountsResult = namedtuple("UniqueCountsResult", ["values", "counts"])
# UniqueInverseResult = namedtuple("UniqueInverseResult", ["values", "inverse_indices"])
