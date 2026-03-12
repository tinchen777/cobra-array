# src/cobra_array/types.py
"""
Type definitions for :pkg:`cobra_array`.
"""

from __future__ import annotations
from typing import (TypeVar, Any, Literal)


_T = TypeVar("_T")
_T_Str = TypeVar("_T_Str", str, bytes)

DTypeT = TypeVar("DTypeT", bound=Any)
DeviceT = TypeVar("DeviceT", bound=Any)

ArrayLike = Any

ArrayLibraryName = Literal["numpy", "torch"]
