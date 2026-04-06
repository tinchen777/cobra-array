# -*- coding: utf-8 -*-
# Python version: 3.9
# @TianZhen

from __future__ import annotations
import array_api_compat as api
from collections import namedtuple

from ._base import Compat
from ..convert import (to_numpy, to_tensor, to_list, to_xp, as_array)
from ..exceptions import (NotArrayAPIObjectError, CompatArrayAttributeError)


UniqueResult = namedtuple("UniqueResult", ["values", "indices", "inverse_indices", "counts"])


class CompatArray(Compat):
    """
    A backend-agnostic array abstraction compliant with the [`Python Array API standard`](https://data-apis.org/array-api/2024.12/API_specification/index.html).

    :class:`CompatArray` provides a unified interface for numerical computation across multiple array backends (e.g., `NumPy`, `PyTorch`), strictly adhering to the `Python Array API standard`.
    Detailed documentation is provided for all supported operations to ensure consistent and predictable behavior.

    Notes
    -----
    - All operations follow the semantics defined by the `Python Array API standard`.
    - Methods correspond directly to standard functions, but are exposed in an object-oriented form.
    - All methods guarantee that any array-like objects in the returned value are automatically wrapped as :class:`CompatArray`. This applies recursively to arrays contained in Python containers (e.g., `tuple`, `list`, `dict`). Non-array objects remain unchanged.
    """
    _arr = None
    _cxp = None

    @classmethod
    def from_other(cls, obj, /, *, xp, copy=False):
        """
        Create a :class:`CompatArray` array from another object using the specified `array namespace`.

        Parameters
        ----------
            obj : object
                The object to be converted to a :class:`CompatArray` array.

            xp : Union[Any, ArrayLibraryName]
                The `array namespace` or `compatibility namespace` to use for conversion.

            copy : bool, default to `False`
                Whether to create a copy of the data during conversion via :func:`convert.as_array`.

        Raises
        ------
            Refer to :func:`convert.as_array` for possible exceptions.
        """
        _xp = to_xp(xp)
        return cls(
            as_array(unwrap(obj), _xp, copy=copy),
            xp=_xp
        )

    def __new__(cls, arr, /, *, copy=False, **kwargs):
        if isinstance(arr, CompatArray):
            # for `CompatArray` input
            return arr.copy() if copy else arr

        # for non-`CompatArray` input
        if not api.is_array_api_obj(arr):
            raise NotArrayAPIObjectError(
                f"Parameter `arr` of `CompatArray` must be an array API compatible array object, got {type(arr)}."
            )
        _cxp = to_cxp(kwargs.get("xp", api.array_namespace(arr)))
        _xp = _cxp.xp
        obj = super().__new__(cls, _xp)
        obj._arr = as_array(arr, _xp, copy=True) if copy else arr
        obj._cxp = _cxp

        return obj

    # === Conversion functions ===
    def to_numpy(self, *, copy=False):
        """
        Convert `self` to a `NumPy array`.
        See also :func:`convert.to_numpy`.
        """
        return to_numpy(self._arr, copy=copy)

    def to_tensor(self, *, device=None, copy=False):
        """
        Convert `self` to a `PyTorch tensor`.
        See also :func:`convert.to_tensor`.
        """
        return to_tensor(self._arr, device=device, copy=copy)

    def to_list(self, *, copy=False):
        """
        Convert `self` to a built-in `list`.
        See also :func:`convert.to_list`.
        """
        return to_list(self._arr, copy=copy)

    # === Device functions ===
    def to_device(self, device, /, *, stream=None):
        """
        Copy `self` from the device on which it currently resides to the specified `device`.

        Parameters
        ----------
            device : AnyDevice
                A device object or name.

            stream : Optional[Union[int, Any]], default to `None`
                Stream object to use during copy.
                In addition to the types supported in `array.__dlpack__`, implementations may choose to support any library-specific stream object with the caveat that any code using such an object would not be portable.

        Returns
        -------
            CompatArray
                An array with the same data and data type as `self` and located on the specified device.

        Notes
        -----
        - For `NumPy`, this function effectively does nothing since the only supported device is the `CPU`;
        - For `CuPy`, this method supports CuPy CUDA Device <cupy.cuda.Device> and Stream <cupy.cuda.Stream> objects.
        - For `PyTorch`, this is the same as `self.to(device)` <torch.Tensor.to> (the stream argument is not supported in PyTorch).
        """
        return api.to_device(self._arr, device, stream=stream)

    # === Manipulation functions ===
    def unstack(self, *, axis=0):
        """
        Splits `self` into a sequence of arrays along the given axis.

        Parameters
        ----------
            axis : int, default to `0`
                Axis along which the array will be split.
                A valid :param:`axis` must be on the interval `[-N, N)`, where `N` is the rank (number of dimensions) of `self`.
                If provided an :param:`axis` outside of the required interval, the function must raise an exception.

        Returns
        -------
            Tuple[CompatArray[TT, DT], ...]
                Tuple of slices along the given dimension.
                All the arrays have the same shape.
        """
        result = self._get_xp_attr("unstack")(self._arr, axis=axis)
        return tuple(CompatArray(arr, xp=self._xp) for arr in result)

    # === Searching functions ===
    def nonzero(self):
        """
        Returns the indices of `self` elements which are non-zero.
        - `self` must have a positive rank. If `self` is zero-dimensional, the function must raise an exception.

        Returns
        -------
            Tuple[CompatArray[int, DT], ...]
                A tuple of `k` arrays, one for each dimension of `self` and each of size `n` (where `n` is the total number of non-zero elements), containing the indices of the non-zero elements in that dimension.
                The indices must be returned in row-major, C-style order.
                The returned array must have the default array index data type.

        Notes
        -----
        - If `self` has a complex floating-point data type, non-zero elements are those elements having at least one component (real or imaginary) which is non-zero;
        - If `self` has a boolean data type, non-zero elements are those elements which are equal to `True`.
        """
        result = self._get_xp_attr("nonzero")(self._arr)
        return tuple(CompatArray(arr, xp=self._xp) for arr in result)

    # === Set functions ===
    def unique_all(self):
        """
        Returns the unique elements of `self`, the first occurring indices for each unique element in `self`, the indices from the set of unique elements that reconstruct `self`, and the corresponding counts for each unique element in `self`.
        - `self`:
            - more than one dimension: the function must flatten `self` and return the unique elements of the flattened `self`.

        Returns
        -------
            UniqueAllResult[CompatArray[TT, DT]]
                A namedtuple (`values`, `indices`, `inverse_indices`, `counts`):
                1. :attr:`values`: A one-dimensional array containing the unique elements of `self`. The array must have the same data type as `self`;
                2. :attr:`indices`: An array containing the indices (first occurrences) of a flattened `self` that result in :attr:`values`. The array must have the same shape as :attr:`values` and must have the default array index data type;
                3. :attr:`inverse_indices`: An array containing the indices of :attr:`values` that reconstruct `self`. The array must have the same shape as `self` and must have the default array index data type;
                4. :attr:`counts`: An array containing the number of times each unique element occurs in `self`. The order of the returned counts must match the order of :attr:`values`, such that a specific element in :attr:`counts` corresponds to the respective unique element in :attr:`values`. The returned array must have same shape as :attr:`values` and must have the default array index data type.
        """
        result = self._get_xp_attr("unique_all")(self._arr)
        return UniqueResult(
            values=CompatArray(result.values, xp=self._xp),
            indices=CompatArray(result.indices, xp=self._xp),
            inverse_indices=CompatArray(result.inverse_indices, xp=self._xp),
            counts=CompatArray(result.counts, xp=self._xp),
        )

    def unique_counts(self):
        """
        Returns the unique elements of `self` and the corresponding counts for each unique element in `self`.
        - `self`:
            - more than one dimension: the function must flatten `self` and return the unique elements of the flattened `self`.

        Returns
        -------
            UniqueCountsResult[CompatArray[TT, DT]]
                A namedtuple (`values`, `counts`):
                1. :attr:`values`: A one-dimensional array containing the unique elements of `self`. The array must have the same data type as `self`;
                2. :attr:`counts`: An array containing the number of times each unique element occurs in `self`. The order of the returned counts must match the order of :attr:`values`, such that a specific element in :attr:`counts` corresponds to the respective unique element in :attr:`values`. The returned array must have same shape as :attr:`values` and must have the default array index data type.
        """
        result = self._get_xp_attr("unique_counts")(self._arr)
        return UniqueResult(
            values=CompatArray(result.values, xp=self._xp),
            indices=None,
            inverse_indices=None,
            counts=CompatArray(result.counts, xp=self._xp),
        )

    def unique_inverse(self):
        """
        Returns the unique elements of `self` and the indices from the set of unique elements that reconstruct `self`.
        - `self`:
            - more than one dimension: the function must flatten `self` and return the unique elements of the flattened `self`.

        Returns
        -------
            UniqueInverseResult[TT, DT]
                A namedtuple (`values`, `inverse_indices`):
                1. :attr:`values`: A one-dimensional array containing the unique elements of `self`. The array must have the same data type as `self`;
                2. :attr:`inverse_indices`: An array containing the indices of :attr:`values` that reconstruct `self`. The array must have the same shape as `self` and must have the default array index data type.
        """
        result = self._get_xp_attr("unique_inverse")(self._arr)
        return UniqueResult(
            values=CompatArray(result.values, xp=self._xp),
            indices=None,
            inverse_indices=CompatArray(result.inverse_indices, xp=self._xp),
            counts=None,
        )

    # === Others ===
    def copy(self):
        """
        Return a copy of `self` via :func:`convert.as_array`.
        """
        return CompatArray.from_other(self._arr, xp=self._xp, copy=True)

    def _get_attr(self, name: str):
        """Try to get the attribute `name` from `self`."""
        try:
            return getattr(self._arr, name)
        except AttributeError:
            raise AttributeError(f"`CompatArray` `{self._xp_name}` has no attribute `{name}`.") from None

    def _get_cxp_attr(self, name: str):
        """Try to get the attribute `name` from the `compatibility namespace`."""
        try:
            return getattr(self._cxp, name)
        except AttributeError:
            raise AttributeError(f"Compatibility namespace `{self._xp_name}` of `{self.__class__.__name__}` has no attribute `{name}`.") from None

    @property
    def arr(self):
        """
        The backend-specific array instance managed by :class:`CompatArray`.
        """
        return self._arr

    @property
    def dtype(self):
        """
        Data type of the elements of `self`.
        """
        try:
            return self._get_xp_attr("dtype")(self._arr)
        except (AttributeError, TypeError):
            return self._get_attr("dtype")

    @property
    def device(self):
        """
        DeviceT on which `self` is stored.
        """
        return api.device(self._arr)

    @property
    def shape(self):
        """
        Dimensions of `self`.
        """
        try:
            result = self._get_xp_attr("shape")(self._arr)
        except (AttributeError, TypeError):
            result = self._get_attr("shape")
        return tuple(result)

    @property
    def ndim(self):
        """
        Number of `self` dimensions (axes).
        """
        try:
            return self._get_xp_attr("ndim")(self._arr)
        except (AttributeError, TypeError):
            return self._get_attr("ndim")

    @property
    def size(self):
        """
        Number of elements in `self`.
        """
        try:
            return self._get_xp_attr("size")(self._arr)
        except (AttributeError, TypeError):
            return self._get_attr("size")

    @property
    def T(self):
        """
        Transpose of `self`.
        - If `self` has fewer than two dimensions, an error should be raised.
        """
        try:
            result = self._get_xp_attr("T")(self._arr)
        except (AttributeError, TypeError):
            result = self._get_attr("T")
        return CompatArray(result, xp=self._xp)

    @property
    def mT(self):
        """
        Transpose of a matrix (or a stack of matrices).
        - If `self` has fewer than two dimensions, an error should be raised.
        """
        try:
            result = self._get_xp_attr("mT")(self._arr)
        except (AttributeError, TypeError):
            result = self._get_attr("mT")
        return CompatArray(result, xp=self._xp)

    def __array__(self):
        """Allow implicit NumPy conversion."""
        return self.to_numpy()

    def __getattr__(self, name: str):
        attr = self._get_cxp_attr(name)

        if callable(attr) and not isinstance(attr, type):
            def wrapper(*args, **kwargs):
                return attr(self._arr, *args, **kwargs)
            return wrapper
        raise CompatArrayAttributeError(f"`CompatArray` `{self._xp_name}` does not support attribute `{name}`.")

    def __len__(self):
        shape = self.shape
        if len(shape) == 0:
            raise TypeError("`len()` of a 0-D compatible array.")
        return shape[0]

    def __repr__(self):
        return f"{self._xp_name}_Array({self._arr})"

    def __abs__(self):
        """See also :func:`CompatArray.abs`."""
        return self.abs()

    def __add__(self, other, /):
        """See also :func:`CompatArray.add`."""
        return self.add(other)

    def __and__(self, other, /):
        """See also :func:`CompatArray.bitwise_and`."""
        return self.bitwise_and(other)

    def __bool__(self):
        """Converts `self` to a Python `bool` object."""
        return bool(self._arr)

    def __complex__(self):
        """Converts `self` to a Python `complex` object."""
        return complex(self._arr)  # type: ignore

    def __eq__(self, other, /):
        """See also :func:`CompatArray.equal`."""
        return self.equal(other)

    def __float__(self):
        """Converts `self` to a Python `float` object."""
        return float(self._arr)  # type: ignore

    def __floordiv__(self, other, /):
        """See also :func:`CompatArray.floor_divide`."""
        return self.floor_divide(other)

    def __ge__(self, other, /):
        """See also :func:`CompatArray.greater_equal`."""
        return self.greater_equal(other)

    def __getitem__(self, key, /):
        """Returns `self[key]`."""
        return self._arr[key]  # type: ignore

    def __gt__(self, other, /):
        """See also :func:`CompatArray.greater`."""
        return self.greater(other)

    def __index__(self):
        """ Converts `self` to a Python `int` object."""
        return int(self._arr)  # type: ignore

    def __int__(self):
        """ Converts `self` to a Python `int` object."""
        return int(self._arr)  # type: ignore

    def __invert__(self):
        """See also :func:`CompatArray.bitwise_invert`."""
        return self.bitwise_invert()

    def __le__(self, other, /):
        """See also :func:`CompatArray.less_equal`."""
        return self.less_equal(other)

    def __lshift__(self, other, /):
        """See also :func:`CompatArray.bitwise_left_shift`."""
        return self.bitwise_left_shift(other)

    def __lt__(self, other, /):
        """See also :func:`CompatArray.less`."""
        return self.less(other)

    def __matmul__(self, other, /):
        """See also :func:`CompatArray.matmul`."""
        return self.matmul(other)

    def __mod__(self, other, /):
        """See also :func:`CompatArray.remainder`."""
        return self.remainder(other)

    def __mul__(self, other, /):
        """See also :func:`CompatArray.multiply`."""
        return self.multiply(other)

    def __ne__(self, other, /):
        """See also :func:`CompatArray.not_equal`."""
        return self.not_equal(other)

    def __neg__(self):
        """See also :func:`CompatArray.negative`."""
        return self.negative()

    def __or__(self, other, /):
        """See also :func:`CompatArray.bitwise_or`."""
        return self.bitwise_or(other)

    def __pos__(self):
        """See also :func:`CompatArray.positive`."""
        return self.positive()

    def __pow__(self, other, /):
        """See also :func:`CompatArray.power`."""
        return self.power(other)

    def __rshift__(self, other, /):
        """See also :func:`CompatArray.bitwise_right_shift`."""
        return self.bitwise_right_shift(other)

    def __setitem__(self, key, value, /):
        """Sets `self[key]` to `value`."""
        self._arr[key] = value  # type: ignore

    def __sub__(self, other, /):
        """See also :func:`CompatArray.subtract`."""
        return self.subtract(other)

    def __truediv__(self, other, /):
        """See also :func:`CompatArray.divide`."""
        return self.divide(other)

    def __xor__(self, other, /):
        """See also :func:`CompatArray.bitwise_xor`."""
        return self.bitwise_xor(other)


def unwrap(obj):
    """
    Unwraps a :class:`CompatArray` array to get the backend-specific array instance, or returns the object itself if it is not a :class:`CompatArray` array.
    """
    return obj.arr if isinstance(obj, CompatArray) else obj


def wrap_arraylike(arr, xp=None):
    """
    Wraps an array-like object in a :class:`CompatArray` array if it is an array API object.
    """
    if api.is_array_api_obj(arr):
        if xp is None:
            return CompatArray(arr)
        return CompatArray(arr, xp=xp)
    return arr


def to_cxp(xp):
    """Convert an `array namespace` or `compatibility namespace` to a :class:`CompatNamespace` instance."""
    from ._namespace import CompatNamespace

    return CompatNamespace(xp)
