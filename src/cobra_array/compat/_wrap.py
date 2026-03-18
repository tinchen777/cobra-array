# -*- coding: utf-8 -*-
# Python version: 3.9
# @TianZhen

from __future__ import annotations
import array_api_compat as api

from .._core import array_spec
from ..convert import (to_numpy, to_tensor, to_list, to_xp, as_array)
from ..exceptions import NotArrayAPIObjectError


class CompatArray:
    def __init__(self, data, /, xp=None):
        # checking
        if not api.is_array_api_obj(data):
            raise NotArrayAPIObjectError(
                f"`arr` of `CompatArray` must be an array API compatible array object, got {type(data)}."
            )

        if xp is not None:
            self._xp = to_xp(xp)
            self._data = as_array(data, xp=self._xp)  # type: ignore
        else:
            self._xp = array_spec(data).xp
            self._data = data

    def to_numpy(self, copy=False):
        return to_numpy(self._data, copy=copy)

    def to_tensor(self, device=None, copy=False):
        return to_tensor(self._data, device=device, copy=copy)

    def to_list(self, copy=False):
        return to_list(self._data, copy=copy)

    def astype(self):
        
        api.as
        
        
    
    @property
    def data(self):
        return self._data

    @property
    def xp(self):
        return self._xp

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def device(self):
        return api.device(self._data)

    @property
    def shape(self):
        return tuple(self._data.shape)

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def size(self):
        return self._xp.size(self._data)

    def __array__(self):
        """Allow implicit NumPy conversion."""
        return self.to_numpy()

    def __getattr__(self, name: str):
        try:
            attr = getattr(self._xp, name)
        except AttributeError:
            raise AttributeError(f"Namespace {self._xp.__name__} of CompatArray has no attribute {name}") from None

        if callable(attr):
            def wrapper(*args, **kwargs):
                return attr(self._data, *args, **kwargs)
            return wrapper
        return attr

    def __len__(self):
        shape = self.shape
        if len(shape) == 0:
            raise TypeError("`len()` of a 0-D compatible array.")
        return shape[0]

    # def __repr__(self):
    #     shape = self._data.shape
    #     return f"ArrayResult(class={self._class_name}, shape={shape})"



def unwrap(x):
    if isinstance(x, CompatArray):
        return x._data
    return x

def wrap(x):
    if api.is_array_api_obj(x):
        return CompatArray(x)
    return x
