# -*- coding: utf-8 -*-
# Python version: 3.9
# @TianZhen

from __future__ import annotations

from ._base import Compat
from ._array import (CompatArray, wrap_arraylike)
from ..exceptions import NameSpaceAttributeError


class NameSpace(Compat):
    """
    A wrapper around an `array namespace` providing a unified, backend-agnostic functional interface.

    `NameSpace` exposes functions from the underlying `array namespace` (e.g., `NumPy`, `PyTorch`) while ensuring compliance with the `Python Array API standard`.
    It includes detailed documentation for
    functions that are not suitable for an object-oriented interface.

    All functions preserve the semantics of the underlying namespace, with additional guarantees on input and output handling.

    Notes
    -----
    - Functions correspond directly to those defined in the underlying `array namespace`, following the `Python Array API standard`.
    - This namespace complements :class:`CompatArray` by providing a functional interface for operations that are not naturally expressed as methods.
    - All functions guarantee that any array-like objects in the returned value are automatically wrapped as :class:`CompatArray`. This conversion is applied recursively to arrays contained in Python containers (e.g., `tuple`, `list`, `dict`). Non-array objects remain unchanged.
    """
    def __new__(cls, xp, /):
        if isinstance(xp, NameSpace):
            # for `NameSpace` input
            return xp
        # for `Namespace` input
        return super().__new__(cls, xp)

    # === Creation functions ===
    def meshgrid(self, *arrays, indexing):
        """
        Returns coordinate matrices from coordinate vectors.

        Parameters
        ----------
            arrays : ArrayLike[Any]
                An arbitrary number of one-dimensional arrays representing grid coordinates.
                Each array should have the same numeric data type.

            indexing : Literal["xy", "ij"], default to `"xy"`
                Cartesian `"xy"` or matrix `"ij"` indexing of output.
                If provided zero or one one-dimensional vector(s) (i.e., the zero- and one-dimensional cases, respectively), the indexing keyword has no effect and should be ignored.

        Returns
        -------
            List[CompatArray]
                List of `N` arrays, where `N` is the number of provided one-dimensional input arrays.
                Each returned array must have rank `N`.
                For `N` one-dimensional arrays having lengths `Ni = len(xi)`,
                - `matrix indexing ij`: Each returned array must have the shape `(N1, N2, N3, ..., Nn)`;
                - `Cartesian indexing xy`: Each returned array must have shape `(N2, N1, N3, ..., Nn)`.

                Accordingly, for the two-dimensional case with input one-dimensional arrays of length `M` and `N`,
                - `matrix indexing ij`: Each returned array must have shape `(M, N)`;
                - `Cartesian indexing xy`: Each returned array must have shape `(N, M)`.

                Similarly, for the three-dimensional case with input one-dimensional arrays of length `M`, `N`, and `P`,
                - `matrix indexing ij`: Each returned array must have shape `(M, N, P)`;
                - `Cartesian indexing xy`: Each returned array must have shape `(N, M, P)`.

                Each returned array should have the same data type as the input arrays.
        """
        result = self._get_xp_attr("meshgrid")(*arrays, indexing=indexing)
        return [CompatArray(arr, xp=self._xp) for arr in result]

    # === Data Type functions ===
    def can_cast(self, from_, to, /):
        """
        Determines whether an array can be cast to a different data type according to type promotion rules.

        Parameters
        ----------
            from_ : Union[DType, ArrayLike[Any]]
                Input array or data type.

            to : DType
                Output data type.

        Returns
        -------
            bool
                A boolean indicating whether the cast is possible.
        """
        return self._get_xp_attr("can_cast")(from_, to)

    def finfo(self, type_, /):
        """
        Machine limits for floating-point data types.

        Parameters
        ----------
            type_ : Union[DType, ArrayLike[Any]]
                The kind of floating-point data-type about which to get information.
                - _complex_: The information is about its component data type.

        Returns
        -------
            finfo_object
                An object having the following attributes:
                - `bits`: _int_
                Number of bits occupied by the real-valued floating-point data type.
                - `eps`: _float_
                Difference between `1.0` and the next smallest representable real-valued floating-point number larger than 1.0 according to the IEEE-754 standard.
                - `max`: _float_
                Largest representable real-valued number.
                - `min`: _float_
                Smallest representable real-valued number.
                - `smallest_normal`: _float_
                Smallest positive real-valued floating-point number with full precision.
                - `dtype`: _dtype_
                Real-valued floating-point data type.
        """
        return self._get_xp_attr("finfo")(type_)

    def iinfo(self, type_, /):
        """
        Machine limits for integer data types.

        Parameters
        ----------
            type_ : Union[DType, ArrayLike[Any]]
                The kind of integer data-type about which to get information.
                - _complex_: The information is about its component data type.

        Returns
        -------
            iinfo_object
                an object having the following attributes:
                - `bits`: _int_
                Number of bits occupied by the integer data type.
                - `max`: _int_
                Largest representable integer value.
                - `min`: _int_
                Smallest representable integer value.
                - `dtype`: _dtype_
                Integer data type.
        """
        return self._get_xp_attr("iinfo")(type_)

    def isdtype(self, dtype, kind) -> bool:
        """
        Returns a boolean indicating whether a provided :param:`dtype` is of a specified data type :param:`kind`.

        Parameters
        ----------
            dtype : DType
                The input dtype.

            kind : Union[str, DType, Tuple[Union[str, DType], ...]]
                The data type kind.
                If kind is a dtype, the function must return a boolean indicating whether the input dtype is equal to the dtype specified by kind.
                - _string_: The function must return a boolean indicating whether the input dtype is of a specified data type kind. The following dtype kinds must be supported:

                    - `bool`: boolean data types (e.g., bool);
                    - `signed integer`: signed integer data types (e.g., int8, int16, int32, int64);
                    - `unsigned integer`: unsigned integer data types (e.g., uint8, uint16, uint32, uint64);
                    - `integral`: integer data types. Shorthand for (`signed integer`, `unsigned integer`);
                    - `real floating`: real-valued floating-point data types (e.g., float32, float64);
                    - `complex floating`: complex floating-point data types (e.g., complex64, complex128);
                    - `numeric`: numeric data types. Shorthand for (`integral`, `real floating`, `complex floating`).
                - _tuple_: The tuple specifies a union of `dtypes` and/or `kinds`, and the function must return a boolean indicating whether the input :param:`dtype` is either equal to a specified dtype or belongs to at least one specified data type kind.

        Returns
        -------
            bool
                A boolean indicating whether the input dtype is of the specified data type kind.
        """
        return self._get_xp_attr("isdtype")(dtype, kind)

    def result_type(self, *arrays_and_dtypes):
        """
        Returns the `dtype` that results from applying type promotion rules (see Type Promotion Rules) to the arguments.

        Parameters
        ----------
            arrays_and_dtypes : Union[ArrayOrAny, DType]
                An arbitrary number of input arrays, scalars, and/or dtypes.

        Returns
        -------
            DType
                The dtype resulting from an operation involving the input arrays, scalars, and/or dtypes.
        """
        return self._get_xp_attr("result_type")(*arrays_and_dtypes)

    # === Manipulation functions ===
    def broadcast_arrays(self, *arrays):
        """
        Broadcasts one or more arrays against one another.

        Parameters
        ----------
            arrays : ArrayLike[Any]
                An arbitrary number of to-be broadcasted arrays.

        Returns
        -------
            List[CompatArray]
                A list of broadcasted :class:`CompatArray` arrays.
                Each array must have the same shape.
                Each array must have the same dtype as its corresponding input array.
        """
        result = self._get_xp_attr("broadcast_arrays")(*arrays)
        return [CompatArray(arr, xp=self._xp) for arr in result]

    # === Constants ===
    @property
    def e(self):
        return self._get_xp_attr("e")

    @property
    def pi(self):
        return self._get_xp_attr("pi")

    @property
    def inf(self):
        return self._get_xp_attr("inf")

    @property
    def nan(self):
        return self._get_xp_attr("nan")

    @property
    def newaxis(self):
        """An alias for None which is useful for indexing arrays."""
        return self._get_xp_attr("newaxis")

    # === Data type ===
    @property
    def int8(self):
        return self._get_xp_attr("int8")

    @property
    def int16(self):
        return self._get_xp_attr("int16")

    @property
    def int32(self):
        return self._get_xp_attr("int32")

    @property
    def int64(self):
        return self._get_xp_attr("int64")

    @property
    def uint8(self):
        return self._get_xp_attr("uint8")

    @property
    def uint16(self):
        return self._get_xp_attr("uint16")

    @property
    def uint32(self):
        return self._get_xp_attr("uint32")

    @property
    def uint64(self):
        return self._get_xp_attr("uint64")

    @property
    def float32(self):
        return self._get_xp_attr("float32")

    @property
    def float64(self):
        return self._get_xp_attr("float64")

    @property
    def complex64(self):
        return self._get_xp_attr("complex64")

    @property
    def complex128(self):
        return self._get_xp_attr("complex128")

    @property
    def bool(self):
        return self._get_xp_attr("bool")

    @property
    def __name__(self):
        return "(cobra_array)" + getattr(self._xp, "__name__", type(self._xp).__name__)

    def __getattr__(self, name: str):
        attr = self._get_xp_attr(name)

        if callable(attr):
            def wrapper(*args, **kwargs):
                return wrap_arraylike(attr(*args, **kwargs), xp=self._xp)
            return wrapper
        raise NameSpaceAttributeError(f"Namespace `{self._xp_name}` does not support attribute `{name}`.")
