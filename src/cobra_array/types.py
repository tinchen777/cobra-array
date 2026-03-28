# src/cobra_array/types.py
"""
Type definitions for :pkg:`cobra_array`.
"""

from __future__ import annotations
import torch
from typing import (Union, Protocol, TypeVar, Any, Literal, NamedTuple, Generic, TYPE_CHECKING)

if TYPE_CHECKING:
    from .compat._array import CompatArray

Value = Union[int, float, complex, bool]

T = TypeVar("T")
StringT = TypeVar("StringT", str, bytes)
ValueT = TypeVar("ValueT", bound=Value)

DType = Any
# FIXME 无法推断
DeviceLiteral = Literal["cpu", "cuda", "xpu", "mkldnn", "opengl", "opencl", "ideep", "hip", "ve", "ort", "mlc", "xla", "lazy", "vulkan", "meta", "hpu"]
Device = Union[DeviceLiteral, torch.device, str]

dtypeT = TypeVar("dtypeT", bound=DType)
deviceT = TypeVar("deviceT", bound=Device)
DTypeT = TypeVar("DTypeT", bound=DType)
DeviceT = TypeVar("DeviceT", bound=Device)

DTypeT_co = TypeVar("DTypeT_co", bound=DType, covariant=True)
DeviceT_co = TypeVar("DeviceT_co", bound=Device, covariant=True)


class ArrayLike(Protocol[DTypeT_co]):
    @property
    def dtype(self) -> DTypeT_co: ...
    @property
    def shape(self) -> Any: ...
    def __array__(self) -> Any: ...


ArrayLibraryName = Literal["numpy", "torch"]

ArrayT = TypeVar("ArrayT", bound=ArrayLike[Any])
ArrayOrAny = Union[ArrayLike[Any], int, float, complex, bool]
ArrayOrScalar = Union[ArrayLike[Any], int, float, complex]
ArrayOrReal = Union[ArrayLike[Any], int, float]
ArrayOrIntLike = Union[ArrayLike[Any], int, bool]
ArrayOrbool = Union[ArrayLike[Any], bool]
ArrayOrInt = Union[ArrayLike[Any], int]


class UniqueAllResult(NamedTuple, Generic[DTypeT_co, DeviceT_co]):
    values: CompatArray[DTypeT_co, DeviceT_co]
    indices: CompatArray[int, DeviceT_co]
    inverse_indices: CompatArray[int, DeviceT_co]
    counts: CompatArray[int, DeviceT_co]


class UniqueCountsResult(NamedTuple, Generic[DTypeT_co, DeviceT_co]):
    values: CompatArray[DTypeT_co, DeviceT_co]
    counts: CompatArray[int, DeviceT_co]


class UniqueInverseResult(NamedTuple, Generic[DTypeT_co, DeviceT_co]):
    values: CompatArray[DTypeT_co, DeviceT_co]
    inverse_indices: CompatArray[int, DeviceT_co]
