# src/cobra_array/result.py
# -*- coding: utf-8 -*-
# Python version: 3.9
# @TianZhen
"""
Result utilities for :pkg:`cobra_array`.

Classes
-------

"""

from __future__ import annotations
from typing import (Any, Optional, Set, List)

from .._core import array_spec
from ..convert import (to_numpy, to_tensor, to_list)
from ..types import (DeviceT)


class ArrayResult:
    """
    A lightweight container for array-like result with unified format
    conversion utilities.

    The result behaves similarly to a NumPy array while providing
    convenient conversion methods such as `to_numpy()`, `to_tensor()`,
    and `to_list()`.
    """
    def __init__(self, val: object, /, class_name: str = "", **kwargs: Any):
        self._val = val
        self._xp, (self._dtype, self._device) = array_spec(val)
        self._class_name = class_name
        self._attrs = kwargs

    def to_numpy(self, copy: bool = False):
        """
        Convert result value to a `NumPy array`.
        """
        return to_numpy(self._val, copy=copy)

    def to_tensor(self, device: Optional[DeviceT] = None, copy: bool = False):
        """
        Convert result value to a `PyTorch tensor`.
        """
        return to_tensor(self._val, device=device, copy=copy)

    def to_list(self, copy: bool = False):
        """
        Convert result value to a built-in `list`.
        """
        return to_list(self._val, copy=copy)

    def update_attr(self, **kwargs: Any):
        """
        Update additional attributes of the result.
        """
        self._attrs.update(kwargs)
        return self

    @property
    def val(self):
        """
        The original result value.
        """
        return self._val

    def __array__(self):
        """
        Allow implicit NumPy conversion.
        """
        return self._val

    def __getitem__(self, item: Any):
        self._xp.
        
        return self._val[item]

    def __len__(self):
        return len(self._val)

    def __getattr__(self, name: str):
        """
        Allow dynamic access to attributes stored in attrs.
        """
        if name in self._attrs:
            return self._attrs[name]
        raise AttributeError(f"{self._class_name} instance has no attribute {name!r}")

    def __repr__(self):
        shape = self._val.shape
        return f"ArrayResult(class={self._class_name}, shape={shape})"
