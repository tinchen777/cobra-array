# -*- coding: utf-8 -*-
# Python version: 3.9
# @TianZhen

from __future__ import annotations
import array_api_compat as api

from .._core import array_spec
from .._utils import array_namespace_alias
from ..convert import (to_numpy, to_tensor, to_list, to_xp, as_array)
from ..exceptions import NotArrayAPIObjectError


class CompatArray:
    @classmethod
    def from_other(cls, obj, xp, /):
        _xp = to_xp(xp)
        return cls(as_array(obj, xp=_xp), xp=_xp)  # type: ignore

    def __init__(self, arr, /, **kwargs):
        # checking
        if not api.is_array_api_obj(arr):
            raise NotArrayAPIObjectError(
                f"Parameter `arr` of `CompatArray` must be an array API compatible array object, got {type(arr)}."
            )
        self._arr = arr

        if "xp" in kwargs:
            self._xp = kwargs["xp"]
        else:
            self._xp = array_spec(arr).xp
        self._name = array_namespace_alias(self._xp)

    def to_numpy(self, copy=False):
        return to_numpy(self._arr, copy=copy)

    def to_tensor(self, device=None, copy=False):
        return to_tensor(self._arr, device=device, copy=copy)

    def to_list(self, copy=False):
        return to_list(self._arr, copy=copy)

    
    
    
    
    
    
    # def astype(self):
        
    #     pass
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    @property
    def arr(self):
        return self._arr

    @property
    def xp(self):
        return self._xp

    @property
    def dtype(self):
        return self._arr.dtype

    @property
    def device(self):
        return api.device(self._arr)

    @property
    def shape(self):
        return tuple(self._arr.shape)

    @property
    def ndim(self):
        return self._arr.ndim

    # @property
    # def size(self):
    #     return self._xp.size(self._data)

    def __array__(self):
        """Allow implicit NumPy conversion."""
        return self.to_numpy()

    def __getattr__(self, name: str):
        try:
            attr = getattr(self._xp, name)
        except AttributeError:
            raise AttributeError(f"Namespace {self._xp.__name__} of CompatArray has no attribute {name}") from None

        if callable(attr) and not isinstance(attr, type):
            def wrapper(*args, **kwargs):
                return attr(self._arr, *args, **kwargs)
            return wrapper
        return attr

    def __len__(self):
        shape = self.shape
        if len(shape) == 0:
            raise TypeError("`len()` of a 0-D compatible array.")
        return shape[0]

    def __repr__(self):
        return f"{self._name}_Array({self._arr})"
    
    
    
    
    



def unwrap(x):
    if isinstance(x, CompatArray):
        return x._arr
    return x

def wrap(x):
    if api.is_array_api_obj(x):
        return CompatArray(x)
    return x
