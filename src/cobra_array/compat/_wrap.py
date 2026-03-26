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

    
    
    
    
    
    
    def astype(self):
        
        pass
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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

        if callable(attr):
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
        return f"{self._name}_array({self._arr})"
    
    
    
    
    
    +x: array.__pos__()

operator.pos(x)

operator.__pos__(x)

-x: array.__neg__()

operator.neg(x)

operator.__neg__(x)

x1 + x2: array.__add__()

operator.add(x1, x2)

operator.__add__(x1, x2)

x1 - x2: array.__sub__()

operator.sub(x1, x2)

operator.__sub__(x1, x2)

x1 * x2: array.__mul__()

operator.mul(x1, x2)

operator.__mul__(x1, x2)

x1 / x2: array.__truediv__()

operator.truediv(x1,x2)

operator.__truediv__(x1, x2)

x1 // x2: array.__floordiv__()

operator.floordiv(x1, x2)

operator.__floordiv__(x1, x2)

x1 % x2: array.__mod__()

operator.mod(x1, x2)

operator.__mod__(x1, x2)

x1 ** x2: array.__pow__()

operator.pow(x1, x2)

operator.__pow__(x1, x2)

x1 @ x2: array.__matmul__()


~x: array.__invert__()

operator.inv(x)

operator.invert(x)

operator.__inv__(x)

operator.__invert__(x)

x1 & x2: array.__and__()

operator.and(x1, x2)

operator.__and__(x1, x2)

x1 | x2: array.__or__()

operator.or(x1, x2)

operator.__or__(x1, x2)

x1 ^ x2: array.__xor__()

operator.xor(x1, x2)

operator.__xor__(x1, x2)

x1 << x2: array.__lshift__()

operator.lshift(x1, x2)

operator.__lshift__(x1, x2)

x1 >> x2: array.__rshift__()

operator.rshift(x1, x2)

operator.__rshift__(x1, x2)

x1 < x2: array.__lt__()

operator.lt(x1, x2)

operator.__lt__(x1, x2)

x1 <= x2: array.__le__()

operator.le(x1, x2)

operator.__le__(x1, x2)

x1 > x2: array.__gt__()

operator.gt(x1, x2)

operator.__gt__(x1, x2)

x1 >= x2: array.__ge__()

operator.ge(x1, x2)

operator.__ge__(x1, x2)

x1 == x2: array.__eq__()

operator.eq(x1, x2)

operator.__eq__(x1, x2)

x1 != x2: array.__ne__()

operator.ne(x1, x2)

operator.__ne__(x1, x2)







array.__abs__()

Calculates the absolute value for each element of an array instance.

array.__add__(other, /)

Calculates the sum for each element of an array instance with the respective element of the array other.

array.__and__(other, /)

Evaluates self_i & other_i for each element of an array instance with the respective element of the array other.

array.__array_namespace__(*, api_version=None)

Returns an object that has all the array API functions on it.

array.__bool__()

Converts a zero-dimensional array to a Python bool object.

array.__complex__()

Converts a zero-dimensional array to a Python complex object.

array.__dlpack__(*, stream=None, max_version=None, dl_device=None, copy=None)

Exports the array for consumption by from_dlpack() as a DLPack capsule.

array.__dlpack_device__()

Returns device type and device ID in DLPack format.

array.__eq__(other, /)

Computes the truth value of self_i == other_i for each element of an array instance with the respective element of the array other.

array.__float__()

Converts a zero-dimensional array to a Python float object.

array.__floordiv__(other, /)

Evaluates self_i // other_i for each element of an array instance with the respective element of the array other.

array.__ge__(other, /)

Computes the truth value of self_i >= other_i for each element of an array instance with the respective element of the array other.

array.__getitem__(key, /)

Returns self[key].

array.__gt__(other, /)

Computes the truth value of self_i > other_i for each element of an array instance with the respective element of the array other.

array.__index__()

Converts a zero-dimensional integer array to a Python int object.

array.__int__()

Converts a zero-dimensional array to a Python int object.

array.__invert__()

Evaluates ~self_i for each element of an array instance.

array.__le__(other, /)

Computes the truth value of self_i <= other_i for each element of an array instance with the respective element of the array other.

array.__lshift__(other, /)

Evaluates self_i << other_i for each element of an array instance with the respective element of the array other.

array.__lt__(other, /)

Computes the truth value of self_i < other_i for each element of an array instance with the respective element of the array other.

array.__matmul__(other, /)

Computes the matrix product.

array.__mod__(other, /)

Evaluates self_i % other_i for each element of an array instance with the respective element of the array other.

array.__mul__(other, /)

Calculates the product for each element of an array instance with the respective element of the array other.

array.__ne__(other, /)

Computes the truth value of self_i != other_i for each element of an array instance with the respective element of the array other.

array.__neg__()

Evaluates -self_i for each element of an array instance.

array.__or__(other, /)

Evaluates self_i | other_i for each element of an array instance with the respective element of the array other.

array.__pos__()

Evaluates +self_i for each element of an array instance.

array.__pow__(other, /)

Calculates an implementation-dependent approximation of exponentiation by raising each element (the base) of an array instance to the power of other_i (the exponent), where other_i is the corresponding element of the array other.

array.__rshift__(other, /)

Evaluates self_i >> other_i for each element of an array instance with the respective element of the array other.

array.__setitem__(key, value, /)

Sets self[key] to value.

array.__sub__(other, /)

Calculates the difference for each element of an array instance with the respective element of the array other.

array.__truediv__(other, /)

Evaluates self_i / other_i for each element of an array instance with the respective element of the array other.

array.__xor__(other, /)

Evaluates self_i ^ other_i for each element of an array instance with the respective element of the array other.

array.to_device(device, /, *, stream=None)

Copy the array from the device on which it currently resides to the specified device.


    
    



def unwrap(x):
    if isinstance(x, CompatArray):
        return x._arr
    return x

def wrap(x):
    if api.is_array_api_obj(x):
        return CompatArray(x)
    return x
