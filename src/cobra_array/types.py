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


DTypeT = TypeVar("DTypeT", bound=DType)
DeviceT = TypeVar("DeviceT", bound=Device)

ArrayLibraryName = Literal["numpy", "torch"]


class ArrayLike(Protocol[DTypeT]):
    dtype: DTypeT
    shape: Any

    def __array__(self) -> Any: ...


ArrayT = TypeVar("ArrayT", bound=ArrayLike[Any])


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
