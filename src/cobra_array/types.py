# src/cobra_array/types.py
"""
Type definitions for :pkg:`cobra_array`.
"""

from __future__ import annotations
from typing import (Protocol, TypeVar, Any, Literal, Generic)


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
