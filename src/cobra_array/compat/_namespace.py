# -*- coding: utf-8 -*-
# Python version: 3.9
# @TianZhen

from __future__ import annotations

from ._base import Compat
from ._array import (CompatArray, wrap_arraylike, unwrap)
from ..exceptions import CompatNamespaceAttributeError


class CompatNamespace(Compat):
    """
    A wrapper around an `array namespace` that provides a compatibility layer for backend-agnostic array operations.

    :class:`CompatNamespace` exposes functions from the underlying `array namespace` (e.g., `NumPy`, `PyTorch`) while ensuring compliance with the `Python Array API standard`.
    It includes detailed documentation for functions that are not suitable for an object-oriented interface.

    All functions preserve the semantics of the underlying namespace, with additional guarantees on input and output handling.

    Notes
    -----
    - Functions correspond directly to those defined in the underlying `array namespace`, following the `Python Array API standard`.
    - This namespace complements :class:`CompatArray` by providing a functional interface for operations that are not naturally expressed as methods.
    - All functions guarantee that any array-like objects in the returned value are automatically wrapped as :class:`CompatArray`. This conversion is applied recursively to arrays contained in Python containers (e.g., `tuple`, `list`, `dict`). Non-array objects remain unchanged.

    Examples
    --------
    Create a compatibility namespace from a backend namespace:

    >>> import numpy as np
    >>> cxp = CompatNamespace(np)
    >>> cxp.xp_name
    'NumPy'

    Create arrays and call namespace functions:

    >>> a = cxp.asarray([1, 2, 3])
    >>> b = cxp.asarray([10, 20, 30])
    >>> a
    NumPy_Array([1 2 3])
    >>> cxp.add(a, b).to_list()
    [11, 22, 33]

    Missing attributes raise an exception:

    >>> cxp.this_attr_does_not_exist
    Traceback (most recent call last):
        ...
    AttributeError: ...
    """
    def __new__(cls, xp, /):
        if type(xp) is cls:
            # for `CompatNamespace` input
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
                List of `N` :class:`CompatArray` arrays, where `N` is the number of provided one-dimensional input arrays.
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
        result = self._get_xp_attr("meshgrid")(
            *[unwrap(arr) for arr in arrays],
            indexing=indexing
        )
        return [CompatArray(arr, xp=self) for arr in result]

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
        return self._get_xp_attr("can_cast")(unwrap(from_), to)

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
        return self._get_xp_attr("finfo")(unwrap(type_))

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
        return self._get_xp_attr("iinfo")(unwrap(type_))

    def isdtype(self, dtype, kind):
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
        return self._get_xp_attr("result_type")(*[unwrap(arr) for arr in arrays_and_dtypes])

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
        result = self._get_xp_attr("broadcast_arrays")(*[unwrap(arr) for arr in arrays])
        return [CompatArray(arr, xp=self) for arr in result]

    # === Linear Algebra Extension ===
    def vector_norm(self, x, /, *, axis=None, keepdims=False, ord=2):
        """
        Computes the vector norm of a vector (or batch of vectors) :param:`x`.

        Parameters
        ----------
            x : ArrayLike[Any]
                The input array. Should have a floating-point data type.

            axis : Optional[Union[int, Tuple[int, ...]]], default to `None`
                - _int_: :param:`axis` specifies the axis (dimension) along which to compute vector norms;
                - _tuple_: :param:`axis` specifies the axes (dimensions) along which to compute batched vector norms;
                - `None`: The vector norm must be computed over all array values (i.e., equivalent to computing the vector norm of a flattened array).

                Negative indices must be supported.

            keepdims : bool, default to `False`
                - `True`: The axes (dimensions) specified by :param:`axis` must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array (see Broadcasting);
                - `False`: The axes (dimensions) specified by :param:`axis` must not be included in the result.

            ord : Union[int, float, Literal['inf', '-inf']], default to `2`
                Order of the norm.
                The following mathematical norms must be supported:
                - `1`: L1-norm (Manhattan);
                - `2`: L2-norm (Euclidean);
                - `"inf"`: infinity norm;
                - _int_ or _float_ (>=1): p-norm.

                The following non-mathematical “norms” must be supported:
                - `0`: sum(a != 0);
                - `-1`: 1./sum(1./abs(a));
                - `-2`: 1./sqrt(sum(1./a**2));
                - `"-inf"`: min(abs(a));
                - _int_ or _float_ (<1): sum(abs(a)**ord)**(1./ord).

        Returns
        -------
            CompatArray
                A :class:`CompatArray` array containing the vector norms.
                - :param:`axis` is `None`: The returned array must be a zero-dimensional array containing a vector norm;
                - :param:`axis` is a scalar value (_int_ or _float_): The returned array must have a rank which is one less than the rank of :param:`x`;
                - :param:`axis` is _tuple_ (`n` elements): The returned array must have a rank which is `n` less than the rank of :param:`x`;

                - :param:`x` is real-valued data type: The returned array must have a real-valued floating-point data type determined by Type Promotion Rules;
                - :param:`x` is complex-valued data type: The returned array must have a real-valued floating-point data type whose precision matches the precision of :param:`x` (e.g., if :param:`x` is complex128, then the returned array must have a float64 data type).
        """
        if ord == "inf":
            ord = float("inf")
        elif ord == "-inf":
            ord = float("-inf")

        result = getattr(self.linalg, "vector_norm")(unwrap(x), axis=axis, keepdims=keepdims, ord=ord)
        return CompatArray(result, xp=self)

    def matrix_norm(self, x, /, *, keepdims=False, ord="fro"):
        """
        Computes the matrix norm of a matrix (or a stack of matrices) :param:`x`.

        Parameters
        ----------
            x : ArrayLike[Any]
                Input array having shape (..., `M`, `N`) and whose innermost two dimensions form `MxN` matrices. Should have a floating-point data type.

            keepdims : bool, default to `False`
                - `True`: The last two axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array (see Broadcasting);
                - `False`: The last two axes (dimensions) must not be included in the result.

            ord : Optional[Union[int, float, Literal['inf', '-inf', 'fro', 'nuc']]], default to `"fro"`
                order of the norm.
                The following mathematical norms must be supported:
                - `"fro"`: Frobenius norm;
                - `"nuc"`: nuclear norm;
                - `1`: max(sum(abs(x), axis=0)). The norm corresponds to the induced matrix norm where `p=1` (i.e., the maximum absolute value column sum);
                - `2`: largest singular value. The norm corresponds to the induced matrix norm where `p=inf` (i.e., the maximum absolute value row sum);
                - `"inf"`: max(sum(abs(x), axis=1)). The norm corresponds to the induced matrix norm where `p=2` (i.e., the largest singular value).

                The following non-mathematical “norms” must be supported:
                - `-1`: min(sum(abs(x), axis=0));
                - `-2`: smallest singular value;
                - `"-inf"`: min(sum(abs(x), axis=1)).

        Returns
        -------
            CompatArray
                A :class:`CompatArray` array containing the norms for each `MxN` matrix.
                - :param:`keepdims` is `False`: The returned array must have a rank which is two less than the rank of :param:`x`;

                - :param:`x` is real-valued data type: The returned array must have a real-valued floating-point data type determined by Type Promotion Rules;
                - :param:`x` is complex-valued data type: The returned array must have a real-valued floating-point data type whose precision matches the precision of :param:`x` (e.g., if :param:`x` is complex128, then the returned array must have a float64 data type).

        """
        if ord == "inf":
            ord = float("inf")
        elif ord == "-inf":
            ord = float("-inf")

        result = getattr(self.linalg, "matrix_norm")(unwrap(x), keepdims=keepdims, ord=ord)
        return CompatArray(result, xp=self)

    @property
    def linalg(self):
        """
        The `linalg` namespace for linear algebra functions.
        The following functions must be supported in the `linalg` namespace:
        - `cholesky`(x, /, *, upper=False): Returns the lower (upper) Cholesky decomposition of a complex Hermitian or real symmetric positive-definite matrix x.
        - `cross`(x1, x2, /, *, axis=-1): Returns the cross product of 3-element vectors.
        - `det`(x, /): Returns the determinant of a square matrix (or a stack of square matrices) x.
        - `diagonal`(x, /, *, offset=0): Returns the specified diagonals of a matrix (or a stack of matrices) x.
        - `eigh`(x, /): Returns an eigenvalue decomposition of a complex Hermitian or real symmetric matrix (or a stack of matrices) x.
        - `eigvalsh`(x, /): Returns the eigenvalues of a complex Hermitian or real symmetric matrix (or a stack of matrices) x.
        - `inv`(x, /): Returns the multiplicative inverse of a square matrix (or a stack of square matrices) x.
        - `matmul`(x1, x2, /): Alias for matmul().
        - `matrix_norm`(x, /, *, keepdims=False, ord='fro'): Computes the matrix norm of a matrix (or a stack of matrices) x.
        - `matrix_power`(x, n, /): Raises a square matrix (or a stack of square matrices) x to an integer power n.
        - `matrix_rank`(x, /, *, rtol=None): Returns the rank (i.e., number of non-zero singular values) of a matrix (or a stack of matrices).
        - `matrix_transpose`(x, /): Alias for matrix_transpose().
        - `outer`(x1, x2, /): Returns the outer product of two vectors x1 and x2.
        - `pinv`(x, /, *, rtol=None): Returns the (Moore-Penrose) pseudo-inverse of a matrix (or a stack of matrices) x.
        - `qr`(x, /, *, mode='reduced'): Returns the QR decomposition of a full column rank matrix (or a stack of matrices).
        - `slogdet`(x, /): Returns the sign and the natural logarithm of the absolute value of the determinant of a square matrix (or a stack of square matrices) x.
        - `solve`(x1, x2, /): Returns the solution of a square system of linear equations with a unique solution.
        - `svd`(x, /, *, full_matrices=True): Returns a singular value decomposition (SVD) of a matrix (or a stack of matrices) x.
        - `svdvals`(x, /): Returns the singular values of a matrix (or a stack of matrices) x.
        - `tensordot`(x1, x2, /, *, axes=2): Alias for tensordot().
        - `trace`(x, /, *, offset=0, dtype=None): Returns the sum along the specified diagonals of a matrix (or a stack of matrices) x.
        - `vecdot`(x1, x2, /, *, axis=-1): Alias for vecdot().
        - `vector_norm`(x, /, *, axis=None, keepdims=False, ord=2): Computes the vector norm of a vector (or batch of vectors) x.
        """
        return self._get_xp_attr("linalg")

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
        return "(compat)" + getattr(self._xp, "__name__", type(self._xp).__name__)

    def __getattr__(self, name):
        attr = self._get_xp_attr(name)

        if callable(attr):
            wrapped = _make_wrapper(self._xp_name, attr, self)
            self.__dict__[name] = wrapped
            return wrapped
        raise CompatNamespaceAttributeError(f"`CompatNamespace` `{self._xp_name}` does not support attribute `{name}`.")


def _make_wrapper(xp_name, attr, cxp, /):
    """Make a wrapper function for the attribute `name` of the `array namespace`."""
    wrap_ = lambda x: wrap_arraylike(x, xp=cxp)
    unwrap_ = unwrap

    def wrapper(*args, **kwargs):
        if xp_name == "NumPy":
            return wrap_(attr(*args, **kwargs))

        n = len(args)
        if n == 0 and not kwargs:
            return wrap_(attr())
        if n == 1 and not kwargs:
            return wrap_(attr(unwrap_(args[0])))
        if n == 2 and not kwargs:
            a0, a1 = args
            return wrap_(attr(unwrap_(a0), unwrap_(a1)))
        new_args = [unwrap_(a) for a in args]
        if kwargs:
            new_kwargs = {k: unwrap_(v) for k, v in kwargs.items()}
            return wrap_(attr(*new_args, **new_kwargs))
        return wrap_(attr(*new_args))
    wrapper.__name__ = getattr(attr, "__name__", "wrapper")
    return wrapper
