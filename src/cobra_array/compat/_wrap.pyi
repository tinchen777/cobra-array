# -*- coding: utf-8 -*-
# Python version: 3.9
# @TianZhen

from torch import Tensor
from numpy.typing import NDArray
from array_api_compat.common._typing import Namespace
from typing import (Union, List, Tuple, Optional, Any, Sequence, Generic)

from ..types import (ArrayT, DType, Device, ArrayLike, ArrayLibraryName)


class CompatArray(Generic[ArrayT]):
    @classmethod
    def from_other(cls, obj: object, xp: Union[Namespace, ArrayLibraryName], /) -> CompatArray[Any]: ...
    def __init__(self, arr: ArrayT, /, **kwargs): ...

    # === Conversion functions ===
    def to_numpy(self, *, copy: bool = False) -> NDArray[Any]:
        """
        Convert the array to a `NumPy array`.
        """
        ...

    def to_tensor(
        self, *,
        device: Optional[Device] = None,
        copy: bool = False
    ) -> Tensor:
        """
        Convert the array to a `PyTorch tensor`.
        """
        ...

    def to_list(self, *, copy: bool = False) -> List[Any]:
        """
        Convert the array to a built-in `list`.
        """
        ...

    # === Data type functions ===
    def astype(
        self,
        dtype: DType,
        /, *,
        copy: bool = True,
        device: Optional[Device] = None
    ) -> CompatArray[Any]:
        """
        Copies the `array` to a specified data type irrespective of Type Promotion Rules rules.

        Parameters
        ----------
            dtype : DType
                Desired data type.

            copy : bool, default to `True`
                Specifies whether to copy an array when the specified dtype matches the data type of the `array`.
                - `True`: A newly allocated array must always be returned (see Copy keyword argument behavior);
                - `False` and the specified dtype matches the data type of the input array: the input array must be returned; otherwise, a newly allocated array must be returned.

            device : Optional[Device], default to `None`
                Device on which to place the returned array.
                - `None`: The output array device must be inferred from x.

        Returns
        -------
            CompatArray[Any]
        """
        ...

    def broadcast_to(self, shape: Tuple[int, ...]) -> CompatArray[ArrayT]: ...

    # === Elementwise functions ===
    def abs(self) -> CompatArray[ArrayT]:
        """
        Calculates the absolute value for each element x_i of the `array`.
        """
        ...

    def acos(self) -> CompatArray[ArrayT]:
        """
        Calculates an implementation-dependent approximation of the principal value of the inverse cosine for each element x_i of the `array`.
        """
        ...

    def acosh(self) -> CompatArray[ArrayT]:
        """
        Calculates an implementation-dependent approximation to the inverse hyperbolic cosine for each element x_i of the `array`.
        """
        ...

    def add(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Calculates the sum for each element x_i of the `array` and the corresponding element y_i of the `other` array.
        """
        ...

    def asin(self) -> CompatArray[ArrayT]:
        """
        Calculates an implementation-dependent approximation of the principal value of the inverse sine for each element x_i of the `array`.
        """
        ...

    def asinh(self) -> CompatArray[ArrayT]:
        """
        Calculates an implementation-dependent approximation to the inverse hyperbolic sine for each element x_i of the `array`.
        """
        ...

    def atan(self) -> CompatArray[ArrayT]:
        """
        Calculates an implementation-dependent approximation of the principal value of the inverse tangent for each element x_i of the `array`.
        """
        ...

    def atan2(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Calculates an implementation-dependent approximation of the inverse tangent of the quotient x_i/y_i, having domain [-infinity, +infinity] x [-infinity, +infinity] (where the x notation denotes the set of ordered pairs of elements (x_i, y_i)) and codomain [-π, +π], for each pair of elements (x_i, y_i) of the `array` and `other` array, respectively.
        """
        ...

    def atanh(self) -> CompatArray[ArrayT]:
        """
        Calculates an implementation-dependent approximation to the inverse hyperbolic tangent for each element x_i of the `array`.
        """
        ...

    def bitwise_and(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Calculates the bitwise AND of each element x_i of the `array` and the corresponding element y_i of the `other` array.
        """
        ...

    def bitwise_left_shift(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Shifts the bits of each element x_i of the `array` to the left by appending y_i (i.e., the respective element in the `other` array) zeros to the right of x_i.
        """
        ...

    def bitwise_invert(self) -> CompatArray[ArrayT]:
        """
        Inverts (flips) each bit for each element x_i of the `array`.
        """
        ...

    def bitwise_or(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Computes the bitwise OR of the underlying binary representation of each element x_i of the `array` with the respective element y_i of the `other` array.
        """
        ...

    def bitwise_right_shift(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Shifts the bits of each element x_i of the `array` to the right according to the respective element y_i of the `other` array.
        """
        ...

    def bitwise_xor(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Computes the bitwise XOR of the underlying binary representation of each element x_i of the `array` with the respective element y_i of the `other` array.
        """
        ...

    def ceil(self) -> CompatArray[ArrayT]:
        """
        Rounds each element x_i of the `array` to the smallest (i.e., closest to -infinity) integer-valued number that is not less than x_i.
        """
        ...

    def clip(self, *, min: Optional[Any] = None, max: Optional[Any] = None) -> CompatArray[ArrayT]:
        """
        Clamps each element x_i of the `array` to the range [:param:`min`, :param:`max`].
        """
        ...

    def conj(self) -> CompatArray[ArrayT]:
        """
        Returns the complex conjugate for each element x_i of the `array`.
        """
        ...

    def copysign(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Composes a floating-point value with the magnitude of x_i and the sign of y_i for each element of the `array`.
        """
        ...

    def cos(self) -> CompatArray[ArrayT]:
        """
        Calculates an implementation-dependent approximation to the cosine for each element x_i of the `array`.
        """
        ...

    def cosh(self) -> CompatArray[ArrayT]:
        """
        Calculates an implementation-dependent approximation to the hyperbolic cosine for each element x_i in the input array x.
        """
        ...

    def divide(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Calculates the division of each element x_i of the `array` with the respective element y_i of the `other` array.
        """
        ...

    def equal(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Computes the truth value of x_i == y_i for each element x_i of the `array` with the respective element y_i of the `other` array.
        """
        ...

    def exp(self) -> CompatArray[ArrayT]:
        """
        Calculates an implementation-dependent approximation to the exponential function for each element x_i of the `array` (e raised to the power of x_i, where e is the base of the natural logarithm).
        """
        ...

    def expm1(self) -> CompatArray[ArrayT]:
        """
        Calculates an implementation-dependent approximation to exp(x)-1 for each element x_i of the `array`.
        """
        ...

    def floor(self) -> CompatArray[ArrayT]:
        """
        Rounds each element x_i of the `array` to the greatest (i.e., closest to +infinity) integer-valued number that is not greater than x_i.
        """
        ...

    def floor_divide(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Rounds the result of dividing each element x_i of the `array` by the respective element y_i of the `other` array to the greatest (i.e., closest to +infinity) integer-value number that is not greater than the division result.
        """
        ...

    def greater(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Computes the truth value of x_i > y_i for each element x_i of the `array` with the respective element y_i of the `other` array.
        """
        ...

    def greater_equal(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Computes the truth value of x_i >= y_i for each element x_i of the `array` with the respective element y_i of the `other` array.
        """
        ...

    def hypot(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Computes the square root of the sum of squares for each element x_i of the `array` with the respective element y_i of the `other` array.
        """
        ...

    def imag(self) -> CompatArray[ArrayT]:
        """
        Returns the imaginary component of a complex number for each element x_i of the `array`.
        """
        ...

    def isfinite(self) -> CompatArray[ArrayT]:
        """
        Tests each element x_i of the `array` to determine if finite.
        """
        ...

    def isinf(self) -> CompatArray[ArrayT]:
        """
        Tests each element x_i of the `array` to determine if equal to positive or negative infinity.
        """
        ...

    def isnan(self) -> CompatArray[ArrayT]:
        """
        Tests each element x_i of the `array` to determine whether the element is NaN.
        """
        ...

    def less(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Computes the truth value of x_i < y_i for each element x_i of the `array` with the respective element y_i of the `other` array.
        """
        ...

    def less_equal(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Computes the truth value of x_i <= y_i for each element x_i of the `array` with the respective element y_i of the `other` array.
        """
        ...

    def log(self) -> CompatArray[ArrayT]:
        """
        Calculates an implementation-dependent approximation to the natural (base e) logarithm for each element x_i of the `array`.
        """
        ...

    def log1p(self) -> CompatArray[ArrayT]:
        """
        Calculates an implementation-dependent approximation to log(1+x), where log refers to the natural (base e) logarithm, for each element x_i of the `array`.
        """
        ...

    def log2(self) -> CompatArray[ArrayT]:
        """
        Calculates an implementation-dependent approximation to the base 2 logarithm for each element x_i of the `array`.
        """
        ...

    def log10(self) -> CompatArray[ArrayT]:
        """
        Calculates an implementation-dependent approximation to the base 10 logarithm for each element x_i of the `array`.
        """
        ...

    def logaddexp(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Calculates the logarithm of the sum of exponentiations log(exp(x1) + exp(x2)) for each element x_i of the `array` with the respective element y_i of the `other` array.
        """
        ...

    def logical_and(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Computes the logical AND for each element x_i of the `array` with the respective element y_i of the `other` array.
        """
        ...

    def logical_not(self) -> CompatArray[ArrayT]:
        """
        Computes the logical NOT for each element x_i of the `array`.
        """
        ...

    def logical_or(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Computes the logical OR for each element x_i of the `array` with the respective element y_i of the `other` array.
        """
        ...

    def logical_xor(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Computes the logical XOR for each element x_i of the `array` with the respective element y_i of the `other` array.
        """
        ...

    def maximum(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Computes the maximum value for each element x_i of the `array` relative to the respective element y_i of the `other` array.
        """
        ...

    def minimum(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Computes the minimum value for each element x_i of the `array` relative to the respective element y_i of the `other` array.
        """
        ...

    def multiply(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Calculates the product for each element x_i of the `array` with the respective element y_i of the `other` array.
        """
        ...

    def negative(self) -> CompatArray[ArrayT]:
        """
        Computes the numerical negative of each element x_i (i.e., y_i = -x_i) of the `array`.
        """
        ...

    def nextafter(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Returns the next representable floating-point value for each element x_i of the `array` in the direction of the respective element y_i of the `other` array.
        """
        ...

    def not_equal(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Computes the truth value of x_i != y_i for each element x_i of the `array` with the respective element y_i of the `other` array.
        """
        ...

    def positive(self) -> CompatArray[ArrayT]:
        """
        Computes the numerical positive of each element x_i (i.e., y_i = +x_i) of the `array`.
        """
        ...

    def pow(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Calculates an implementation-dependent approximation of exponentiation by raising each element x_i (the base) of the `array` to the power of y_i (the exponent), where y_i is the corresponding element of the `other` array.
        """
        ...

    def real(self) -> CompatArray[ArrayT]:
        """
        Returns the real component of a complex number for each element x_i of the `array`.
        """
        ...

    def reciprocal(self) -> CompatArray[ArrayT]:
        """
        Returns the reciprocal for each element x_i of the `array`.
        """
        ...

    def remainder(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Returns the remainder of division for each element x_i of the `array` and the respective element y_i of the `other` array.
        """
        ...

    def round(self) -> CompatArray[ArrayT]:
        """
        Rounds each element x_i of the `array` to the nearest integer-valued number.
        """
        ...

    def sign(self) -> CompatArray[ArrayT]:
        """
        Returns an indication of the sign of a number for each element x_i of the `array`.
        """
        ...

    def signbit(self) -> CompatArray[ArrayT]:
        """
        Determines whether the sign bit is set for each element x_i of the `array`.
        """
        ...

    def sin(self) -> CompatArray[ArrayT]:
        """
        Calculates an implementation-dependent approximation to the sine for each element x_i of the `array`.
        """
        ...

    def sinh(self) -> CompatArray[ArrayT]:
        """
        Calculates an implementation-dependent approximation to the hyperbolic sine for each element x_i of the `array`.
        """
        ...

    def square(self) -> CompatArray[ArrayT]:
        """
        Squares each element x_i of the `array`.
        """
        ...

    def sqrt(self) -> CompatArray[ArrayT]:
        """
        Calculates the principal square root for each element x_i of the `array`.
        """
        ...

    def subtract(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Calculates the difference for each element x_i of the `array` with the respective element y_i of the `other` array.
        """
        ...

    def tan(self) -> CompatArray[ArrayT]:
        """
        Calculates an implementation-dependent approximation to the tangent for each element x_i of the `array`.
        """
        ...

    def tanh(self) -> CompatArray[ArrayT]:
        """
        Calculates an implementation-dependent approximation to the hyperbolic tangent for each element x_i of the `array`.
        """
        ...

    def trunc(self) -> CompatArray[ArrayT]:
        """
        Rounds each element x_i of the `array` to the nearest integer-valued number that is closer to zero than x_i.
        """

    # === Indexing functions ===
    def take(self, indices: ArrayLike, /, *, axis: Optional[int] = None) -> CompatArray[ArrayT]:
        """
        Returns elements of the `array` along an :param:`axis`.

        Parameters
        ----------
            indices : ArrayLike
                The :param:`indices` array must be one-dimensional and have an integer data type. For index in :param:`indices`:
                - _negative_: The function must determine the element to select along a specified axis (dimension) by counting from the last element (where -1 refers to the last element).

            axis : Optional[int], default to `None`
                Axis over which to select values.
                - _negative_int_: The function must determine the axis along which to select values by counting from the last dimension (where -1 refers to the last dimension).
                If the `array` is one-dimensional, providing an :param:`axis` is optional; however, if the `array` has more than one dimension, providing an :param:`axis` is required.

        Returns
        -------
            CompatArray[ArrayT]
                An array having the same data type as the `array`. The output array must have the same rank (i.e., number of dimensions) as the `array` and must have the same shape as the `array`, except for the axis specified by :param:`axis` whose size must equal the number of elements in :param:`indices`.
        """
        ...

    def take_along_axis(self, indices: ArrayLike, /, *, axis: int = -1) -> CompatArray[ArrayT]:
        """
        Returns elements from the `array` at the one-dimensional indices specified by :param:`indices` along a provided :param:`axis`.

        Parameters
        ----------
            indices : ArrayLike
                Must have the same rank (i.e., number of dimensions) as the `array`. For index in :param:`indices`:
                - _negative_: The function must determine the element to select along a specified axis (dimension) by counting from the last element (where -1 refers to the last element).

            axis : int, default to -1
                Axis along which to select values.
                - _negative_int_: The function must determine the axis along which to select values by counting from the last dimension (where -1 refers to the last dimension).

        Returns
        -------
            CompatArray[ArrayT]
                An array having the same data type as the `array`. Must have the same rank (i.e., number of dimensions) as the `array` and must have a shape determined according to Broadcasting, except for the axis (dimension) specified by :param:`axis` whose size must equal the size of the corresponding axis (dimension) in :param:`indices`.
        """
        ...

    # === Linear algebra functions ===
    def matmul(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Computes the matrix product.
        The `array` should have a numeric data type. Must have at least one dimension.
        - If the `array` is one-dimensional having shape (M,) and the `other` array has more than one dimension, the `array` must be promoted to a two-dimensional array by prepending 1 to its dimensions (i.e., must have shape (1, M)). After matrix multiplication, the prepended dimensions in the returned array must be removed.
        - If the `array` has more than one dimension (including after vector-to-matrix promotion), shape(`array`)[:-2] must be compatible with shape(`other`)[:-2] (after vector-to-matrix promotion) (see Broadcasting).
        - If the `array` has shape (..., M, K), the innermost two dimensions form matrices on which to perform matrix multiplication.

        Parameters
        ----------
            other : ArrayLike
                The `other` array. Should have a numeric data type. Must have at least one dimension.
                - If the `other` array is one-dimensional having shape (N,) and the `array` has more than one dimension, the `other` array must be promoted to a two-dimensional array by appending 1 to its dimensions (i.e., must have shape (N, 1)). After matrix multiplication, the appended dimensions in the returned array must be removed.
                - If the `other` array has more than one dimension (including after vector-to-matrix promotion), shape(`other`)[:-2] must be compatible with shape(`array`)[:-2] (after vector-to-matrix promotion) (see Broadcasting).
                - If the `other` array has shape (..., K, N), the innermost two dimensions form matrices on which to perform matrix multiplication.

        Returns
        -------
            CompatArray[ArrayT]
                The returned array must have a data type determined by Type Promotion Rules.
                - If both the `array` and the `other` array are one-dimensional arrays having shape (N,), a zero-dimensional array containing the inner product as its only element.
                - If the `array` is a two-dimensional array having shape (M, K) and the `other` array is a two-dimensional array having shape (K, N), a two-dimensional array containing the conventional matrix product and having shape (M, N).
                - If the `array` is a one-dimensional array having shape (K,) and the `other` array is an array having shape (..., K, N), an array having shape (..., N) (i.e., prepended dimensions during vector-to-matrix promotion must be removed) and containing the conventional matrix product.
                - If the `array` is an array having shape (..., M, K) and the `other` array is a one-dimensional array having shape (K,), an array having shape (..., M) (i.e., appended dimensions during vector-to-matrix promotion must be removed) and containing the conventional matrix product.
                - If the `array` is a two-dimensional array having shape (M, K) and the `other` array is an array having shape (..., K, N), an array having shape (..., M, N) and containing the conventional matrix product for each stacked matrix.
                - If the `array` is an array having shape (..., M, K) and the `other` array is a two-dimensional array having shape (K, N), an array having shape (..., M, N) and containing the conventional matrix product for each stacked matrix.
                - If either the `array` or the `other` array has more than two dimensions, an array having a shape determined by Broadcasting shape(`array`)[:-2] against shape(`other`)[:-2] and containing the conventional matrix product for each stacked matrix.
        """
        ...

    def tensordot(self, other: ArrayLike, /, *, axes: Union[int, Tuple[Sequence[int], Sequence[int]]] = 2) -> CompatArray[ArrayT]:
        """
        Returns a tensor contraction of x1 and x2 over specific axes. The tensordot function corresponds to the generalized matrix product.
        first input array. Should have a numeric data type.
        
        Parameters
        ----------
            other : ArrayLike
                second input array. Should have a numeric data type. Corresponding contracted axes of x1 and x2 must be equal.
                
                NOTE: Contracted axes (dimensions) must not be broadcasted.
            
            axes : Union[int, Tuple[Sequence[int], Sequence[int]]], default to 2
                number of axes (dimensions) to contract or explicit sequences of axis (dimension) indices for x1 and x2, respectively.

                If axes is an int equal to N, then contraction must be performed over the last N axes of x1 and the first N axes of x2 in order. The size of each corresponding axis (dimension) must match. Must be nonnegative.

                If N equals 0, the result is the tensor (outer) product.

                If N equals 1, the result is the tensor dot product.

                If N equals 2, the result is the tensor double contraction (default).

                If axes is a tuple of two sequences (x1_axes, x2_axes), the first sequence must apply to x1 and the second sequence to x2. Both sequences must have the same length. Each axis (dimension) x1_axes[i] for x1 must have the same size as the respective axis (dimension) x2_axes[i] for x2. Each index referred to in a sequence must be unique. If x1 has rank (i.e, number of dimensions) N, a valid x1 axis must reside on the half-open interval [-N, N). If x2 has rank M, a valid x2 axis must reside on the half-open interval [-M, M).
        
        Returns
        -------
            CompatArray[ArrayT]
                an array containing the tensor contraction whose shape consists of the non-contracted axes (dimensions) of the first array x1, followed by the non-contracted axes (dimensions) of the second array x2. The returned array must have a data type determined by Type Promotion Rules.
        
        Notes
        -----
            If either x1 or x2 has a complex floating-point data type, neither argument must be complex-conjugated or transposed. If conjugation and/or transposition is desired, these operations should be explicitly performed prior to computing the generalized matrix product.
        """
        ...

    def matrix_transpose(self) -> CompatArray[ArrayT]:
        """
        Transposes the matrix (or a stack of matrices) `array`.
        input array having shape (..., M, N) and whose innermost two dimensions form MxN matrices.

        Returns
        -------
            CompatArray[ArrayT]
                an array containing the transpose for each matrix and having shape (..., N, M). The returned array must have the same data type as x.
        """
        ...
    
    def vecdot(self, other: ArrayLike, /, *, axes: int = -1) -> CompatArray[ArrayT]:
        # FIXME
        """
        Compute $x^2$.
        """

    # === Manipulation functions ===
    def expand_dims(self, *, axis: int) -> CompatArray[ArrayT]:
        """
        
        """
        ...
    
    def flip(self, *, axis: int | Tuple[int, ...] | None = None) -> CompatArray[ArrayT]:
        """
        """
        ...
    
    def moveaxis(
        self,
        source: int | Tuple[int, ...],
        destination: int | Tuple[int, ...],
        /
    ) -> CompatArray[ArrayT]:
        """
        """
        ...

    def permute_dims(self, axes: Tuple[int, ...]) -> CompatArray[ArrayT]:
        """
        """
        ...

    def repeat(
        self,
        repeats: int | Array,
        /, *,
        axis: int | None = None
    ) -> CompatArray[ArrayT]:
        """
        """
        ...

    def reshape(
        self,
        shape: Tuple[int, ...],
        *,
        copy: bool | None = None
    ) -> CompatArray[ArrayT]:
        """
        """
        ...

    def roll(
        self,
        shift: int | Tuple[int, ...],
        *,
        axis: int | Tuple[int, ...] | None = None
    ) -> CompatArray[ArrayT]:
        """
        """
        ...

    def squeeze(self, axis: int | Tuple[int, ...]) -> CompatArray[ArrayT]:
        """
        """
        ...

    # def stack(
    #     arrays: Tuple[Array, ...] | List[Array],
    #     /,
    #     *,
    #     axis: int = 0
    # ) -> CompatArray[ArrayT]:
    #     """
    #     """
    #     ...

    def tile(self, repetitions: Tuple[int, ...], /) -> CompatArray[ArrayT]:
        """
        """
        ...

    def unstack(
        self, *,
        axis: int = 0
    ) -> Tuple[Array, ...]:
        """
        """
        ...

    # === searching_functions ===
    def argmax(
        self, *,
        axis: int | None = None,
        keepdims: bool = False
    ) -> CompatArray[ArrayT]:
        """
        """
        ...

    def argmin(
        self, *,
        axis: int | None = None,
        keepdims: bool = False
    ) -> CompatArray[ArrayT]:
        """
        """
        ...

    def nonzero(self) -> Tuple[Array, ...]:
        """
        """
        ...

    def count_nonzero(
        self, *,
        axis: int | Tuple[int, ...] | None = None,
        keepdims: bool = False
    ) -> CompatArray[ArrayT]:
        """
        """
        ...

    def searchsorted(
        x1: Array,
        x2: Array,
        /, *,
        side: Literal['left', 'right'] = "left",
        sorter: Array | None = None
    ) -> CompatArray[ArrayT]:
        """
        """
        ...

    def where(
        self,
        x1: bool | int | float | complex | Array,
        x2: bool | int | float | complex | Array,
        /
    ) -> CompatArray[ArrayT]:
        """
        """
        ...
    
    # === set_functions ===
    def unique_all(self) -> UniqueAllResult:
        """
        """
        ...

    def unique_counts(self) -> UniqueCountsResult:
        ...

    def unique_inverse(self) -> UniqueInverseResult:
        """
        """
        ...

    def unique_values(self) -> CompatArray[ArrayT]:
        """
        """
        ...
    
    # === _sorting_functions ===
    def argsort(
        self, *,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True
    ) -> CompatArray[ArrayT]:
        """
        """
        ...
        
    def sort(
        self, *,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True
    ) -> CompatArray[ArrayT]:
        """
        """
        ...
    
    # === _statistical_functions ===
    def cumulative_sum(
        self, *,
        axis: int | None = None,
        dtype: Dtype | None = None,
        include_initial: bool = False
    ) -> CompatArray[ArrayT]:
        """
        """
        ...

    def cumulative_prod(
        self, *,
        axis: int | None = None,
        dtype: Dtype | None = None,
        include_initial: bool = False
    ) -> CompatArray[ArrayT]:
        """
        """
        ...

    def max(
        self, *,
        axis: int | Tuple[int, ...] | None = None,
        keepdims: bool = False
    ) -> CompatArray[ArrayT]:
        """
        """
        ...

    def mean(
        self, *,
        axis: int | Tuple[int, ...] | None = None,
        keepdims: bool = False
    ) -> CompatArray[ArrayT]:
        """
        """
        ...

    def min(
        self, *,
        axis: int | Tuple[int, ...] | None = None,
        keepdims: bool = False
    ) -> CompatArray[ArrayT]:
        """
        """
        ...

    def prod(
        self, *,
        axis: int | Tuple[int, ...] | None = None,
        dtype: Dtype | None = None,
        keepdims: bool = False
    ) -> CompatArray[ArrayT]:
        """
        """
        ...

    def std(
        self, *,
        axis: int | Tuple[int, ...] | None = None,
        correction: int | float = 0,
        keepdims: bool = False
    ) -> CompatArray[ArrayT]:
        """
        """
        ...

    def sum(
        self, *,
        axis: int | Tuple[int, ...] | None = None,
        dtype: Dtype | None = None,
        keepdims: bool = False
    ) -> CompatArray[ArrayT]:
        """
        """
        ...

    def var(
        self, *,
        axis: int | Tuple[int, ...] | None = None,
        correction: int | float = 0,
        keepdims: bool = False
    ) -> CompatArray[ArrayT]:
        """
        """
        ...
    
    # === _utility_functions ===
    def all(
        self, *,
        axis: int | Tuple[int, ...] | None = None,
        keepdims: bool = False
    ) -> CompatArray[ArrayT]:
        """
        """
        ...

    def any(
        self, *,
        axis: int | Tuple[int, ...] | None = None,
        keepdims: bool = False
    ) -> CompatArray[ArrayT]:
        """
        """
        ...

    def diff(
        self, *,
        axis: int = -1,
        n: int = 1,
        prepend: Array | None = None,
        append: Array | None = None
    ) -> CompatArray[ArrayT]:
        """
        """
        ...

    # === Constants ===
    @property
    def e(self) -> float: ...
    @property
    def pi(self) -> float: ...
    @property
    def inf(self) -> float: ...
    @property
    def nan(self) -> float: ...
    @property
    def newaxis(self) -> None:
        """An alias for None which is useful for indexing arrays."""
        ...

    # === Data type ===
    @property
    def int8(self) -> DType: ...
    @property
    def int16(self) -> DType: ...
    @property
    def int32(self) -> DType: ...
    @property
    def int64(self) -> DType: ...
    @property
    def uint8(self) -> DType: ...
    @property
    def uint16(self) -> DType: ...
    @property
    def uint32(self) -> DType: ...
    @property
    def uint64(self) -> DType: ...
    @property
    def float32(self) -> DType: ...
    @property
    def float64(self) -> DType: ...
    @property
    def complex64(self) -> DType: ...
    @property
    def complex128(self) -> DType: ...
    @property
    def bool(self) -> DType: ...
    
    
array.dtype

Data type of the array elements.

array.device

Hardware device the array data resides on.

array.mT

Transpose of a matrix (or a stack of matrices).

array.ndim

Number of array dimensions (axes).

array.shape

Array dimensions.

array.size

Number of elements in an array.

array.T

Transpose of the array.
    
    
    
    
    
    
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

