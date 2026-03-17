# src/cobra_array/result.py
# -*- coding: utf-8 -*-
# Python version: 3.9
# @TianZhen

from __future__ import annotations
import array_api_compat as api

from ._core import array_spec
from .convert import (to_numpy, to_tensor, to_list)
from .exceptions import NotArrayAPIObjectError


class CompatArray:
    def __init__(self, arr, /):
        self._arr = arr
        if not api.is_array_api_obj(arr):
            raise NotArrayAPIObjectError(
                f"`arr` of `CompatArray` must be an array API compatible array object, got {type(arr)}."
            )
        self._xp, (self._dtype, self._device) = array_spec(arr)

    def to_numpy(self, copy=False):
        return to_numpy(self._arr, copy=copy)

    def to_tensor(self, device=None, copy=False):
        return to_tensor(self._arr, device=device, copy=copy)

    def to_list(self, copy=False):
        return to_list(self._arr, copy=copy)

    @property
    def arr(self):
        return self._arr

    @property
    def xp(self):
        return self._xp

    def __array__(self):
        """Allow implicit NumPy conversion."""
        # FIXME
        return self._arr

    def __getattr__(self, name: str):
        try:
            attr = getattr(self._xp, name)
        except AttributeError:
            raise AttributeError(f"Namespace {self._xp.__name__} of CompatArray has no attribute {name}") from None

        if callable(attr):
            def wrapper(*args, **kwargs):
                return attr(self._arr, *args, **kwargs)
            return wrapper
        return attr

    def __len__(self):
        return len(self._arr)

    def __repr__(self):
        shape = self._arr.shape
        return f"ArrayResult(class={self._class_name}, shape={shape})"



def unwrap(x):
    if isinstance(x, CompatArray):
        return x._arr
    return x

def wrap(x):
    if api.is_array_api_obj(x):
        return CompatArray(x)
    return x
