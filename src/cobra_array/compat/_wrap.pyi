# -*- coding: utf-8 -*-
# Python version: 3.9
# @TianZhen

from torch import Tensor
from numpy.typing import NDArray
from array_api_compat.common._typing import Namespace
from typing import (Union, Tuple, List, Tuple, Optional, Any, Sequence, Generic, Literal)

from ..types import (ArrayT, DType, Device, ArrayLike, ArrayLibraryName)


class CompatArray(Generic[ArrayT]):
    """
    A backend-agnostic array abstraction compliant with the [`Python Array API standard`](https://data-apis.org/array-api/2024.12/API_specification/index.html).

    `CompatArray` provides a unified interface for numerical computation across multiple array backends (e.g., `NumPy`, `PyTorch`), strictly adhering to the `Python Array API standard`.
    Detailed documentation is provided for all supported operations to ensure consistent and predictable behavior.

    Notes
    -----
    - All operations follow the semantics defined by the `Python Array API standard`.
    - Methods correspond directly to standard functions, but are exposed in an object-oriented form.
    """
    @classmethod
    def from_other(cls, obj: object, xp: Union[Namespace, ArrayLibraryName], /) -> CompatArray[Any]: ...
    def __init__(self, arr: ArrayT, /, **kwargs): ...

    # === Conversion functions ===
    def to_numpy(self, *, copy: bool = False) -> NDArray[Any]:
        """
        Convert `self` to a `NumPy array`.
        """
        ...

    def to_tensor(
        self, *,
        device: Optional[Device] = None,
        copy: bool = False
    ) -> Tensor:
        """
        Convert `self` to a `PyTorch tensor`.
        """
        ...

    def to_list(self, *, copy: bool = False) -> List[Any]:
        """
        Convert `self` to a built-in `list`.
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
        Copies `self` to a specified data type irrespective of Type Promotion Rules rules.

        Parameters
        ----------
            dtype : DType
                Desired data type.

            copy : bool, default to `True`
                Specifies whether to copy an array when the specified dtype matches the data type of `self`.
                - `True`: A newly allocated array must always be returned;
                - `False`: Whether :param:`dtype` matches the data type of `self`:

                    - `True`: `self` must be returned;
                    - `False`: A newly allocated array must be returned.

            device : Optional[Device], default to `None`
                Device on which to place the returned array.
                - `None`: The output array device must be inferred from `self`.

        Returns
        -------
            CompatArray[Any]
        """
        ...

    def broadcast_to(self, shape: Tuple[int, ...]) -> CompatArray[ArrayT]: ...

    # === Elementwise functions ===
    def abs(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `absolute value` of `self`.
        """
        ...

    def acos(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `principal value of the inverse cosine` of `self`.
        """
        ...

    def acosh(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `inverse hyperbolic cosine` of `self`.
        """
        ...

    def add(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `sum` of `self` and `other`.
        """
        ...

    def asin(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `principal value of the inverse sine` of `self`.
        """
        ...

    def asinh(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `inverse hyperbolic sine` of `self`.
        """
        ...

    def atan(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `principal value of the inverse tangent` of `self`.
        """
        ...

    def atan2(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `inverse tangent` of `self / other`, taking into account the signs of both inputs.
        This allows determining the correct quadrant. Results are in radians in [-π, π].
        """
        ...

    def atanh(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `inverse hyperbolic tangent` of `self`.
        """
        ...

    def bitwise_and(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `bitwise AND` of `self` and `other`.
        """
        ...

    def bitwise_left_shift(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `bitwise left shift` of `self` by `other`.
        Each element is shifted left by the corresponding number of bits.
        """
        ...

    def bitwise_invert(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `bitwise NOT` of `self`.
        This operation flips every bit of each element.
        """
        ...

    def bitwise_or(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `bitwise OR` of `self` and `other`.
        """
        ...

    def bitwise_right_shift(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `bitwise right shift` of `self` by `other`.
        Each element is shifted right by the corresponding number of bits.
        """
        ...

    def bitwise_xor(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `bitwise XOR` of `self` and `other`.
        """
        ...

    def ceil(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `ceiling` of `self`.
        Each element is rounded to the smallest (i.e., closest to -infinity) integer-valued number that is not less than itself.
        """
        ...

    def clip(self, *, min: Optional[Any] = None, max: Optional[Any] = None) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `clipping` of `self` to the range [:param:`min`, :param:`max`].
        """
        ...

    def conj(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `complex conjugate` of `self`.
        """
        ...

    def copysign(self, other: Any, /) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `copysign` of `self` with `other`.
        Each element has the magnitude of `self` and the sign of `other`.
        """
        ...

    def cos(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `cosine` of `self`.
        """
        ...

    def cosh(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `hyperbolic cosine` of `self`.
        """
        ...

    def divide(self, other: Any, /) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `division` of `self` by `other`.
        """
        ...

    def equal(self, other: Any, /) -> CompatArray[ArrayT]:
        """
        Computes the element-wise truth value of `self == other`.

        The returned array must have a data type of `bool`.
        """
        ...

    def exp(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `exponential` (`exp(x)`) of `self`.
        """
        ...

    def expm1(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `exp(x) - 1` of `self`.
        """
        ...

    def floor(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `floor` of `self`.
        Each element is rounded to the largest (i.e., closest to +infinity) integer-valued number that is not greater than itself.
        """
        ...

    def floor_divide(self, other: Any, /) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `floor division` of `self` by `other`.
        Each element of the division result is rounded to the largest (i.e., closest to +infinity) integer-valued number that is not greater than itself.
        """
        ...

    def greater(self, other: Any, /) -> CompatArray[ArrayT]:
        """
        Computes the element-wise truth value of `self > other`.

        The returned array must have a data type of `bool`.
        """
        ...

    def greater_equal(self, other: Any, /) -> CompatArray[ArrayT]:
        """
        Computes the element-wise truth value of `self >= other`.

        The returned array must have a data type of `bool`.
        """
        ...

    def hypot(self, other: Any, /) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `hypotenuse` of `self` and `other`.
        Equivalent to `sqrt(self **2 + other **2)`, computed in a numerically stable way.
        """
        ...

    def imag(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `imaginary component` of `self`.
        """
        ...

    def isfinite(self) -> CompatArray[ArrayT]:
        """
        Tests the element-wise `finiteness` of `self`.

        The returned array must have a data type of `bool`.
        """
        ...

    def isinf(self) -> CompatArray[ArrayT]:
        """
        Tests the element-wise `infinity` of `self`.

        The returned array must have a data type of `bool`.
        """
        ...

    def isnan(self) -> CompatArray[ArrayT]:
        """
        Tests the element-wise `NaN` of `self`.

        The returned array must have a data type of `bool`.
        """
        ...

    def less(self, other: Any, /) -> CompatArray[ArrayT]:
        """
        Computes the element-wise truth value of `self < other`.

        The returned array must have a data type of `bool`.
        """
        ...

    def less_equal(self, other: Any, /) -> CompatArray[ArrayT]:
        """
        Computes the element-wise truth value of `self <= other`.

        The returned array must have a data type of `bool`.
        """
        ...

    def log(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `natural logarithm` (base `e`)  of `self`.
        """
        ...

    def log1p(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `log(1 + x)` (base `e`) of `self`.
        """
        ...

    def log2(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `base-2 logarithm` of `self`.
        """
        ...

    def log10(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `base-10 logarithm` of `self`.
        """
        ...

    def logaddexp(self, other: Any, /) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `logaddexp` of `self` and `other`.
        Equivalent to `log(exp(self) + exp(other))`.
        """
        ...

    def logical_and(self, other: Any, /) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `logical AND` of `self` and `other`.
        """
        ...

    def logical_not(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `logical NOT` of `self`.
        """
        ...

    def logical_or(self, other: Any, /) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `logical OR` of `self` and `other`.
        """
        ...

    def logical_xor(self, other: Any, /) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `logical XOR` of `self` and `other`.
        """
        ...

    def maximum(self, other: Any, /) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `maximum` of `self` and `other`.
        """
        ...

    def minimum(self, other: Any, /) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `minimum` of `self` and `other`.
        """
        ...

    def multiply(self, other: Any, /) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `product` of `self` and `other`.
        """
        ...

    def negative(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `numerical negative` of `self`.
        """
        ...

    def nextafter(self, other: Any, /) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `next representable floating-point value` of `self` toward `other`.
        """
        ...

    def not_equal(self, other: Any, /) -> CompatArray[ArrayT]:
        """
        Computes the element-wise truth value of `self != other`.

        The returned array must have a data type of `bool`.
        """
        ...

    def positive(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `numerical positive` of `self`.
        """
        ...

    def pow(self, other: Any, /) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `power` of `self` and `other`.
        Each element of `self` is raised to the corresponding power in `other`.
        """
        ...

    def real(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `real component` of `self`.
        """
        ...

    def reciprocal(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `reciprocal` of `self`.
        """
        ...

    def remainder(self, other: Any, /) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `remainder` of `self` divided by `other`.
        """
        ...

    def round(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `rounding` of `self` to the nearest integer-valued number.
        """
        ...

    def sign(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `indication of the sign` of `self`.
        """
        ...

    def signbit(self) -> CompatArray[ArrayT]:
        """
        Tests the element-wise `sign bit` of `self`.
        Tests each element for whenever is either `-0`, `less than zero`, or a signed `NaN` (i.e., a NaN value whose sign bit is 1).

        The returned array must have a data type of `bool`.
        """
        ...

    def sin(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `sine` of `self`.
        """
        ...

    def sinh(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `hyperbolic sine` of `self`.
        """
        ...

    def square(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `square` of `self`.
        """
        ...

    def sqrt(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `principal square root` of `self`.
        """
        ...

    def subtract(self, other: Any, /) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `difference` of `self` and `other`.
        """
        ...

    def tan(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `tangent` of `self`.
        """
        ...

    def tanh(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `hyperbolic tangent` of `self`.
        """
        ...

    def trunc(self) -> CompatArray[ArrayT]:
        """
        Computes the element-wise `truncation` of `self` toward `zero`.
        Each element is rounded to the nearest integer-valued number that is closer to `zero` than itself.
        """

    # === Indexing functions ===
    def take(self, indices: ArrayLike, /, *, axis: Optional[int] = None) -> CompatArray[ArrayT]:
        """
        Returns elements of `self` along an :param:`axis`.
        - `self` should have one or more dimensions (axes).

        Parameters
        ----------
            indices : ArrayLike
                The :param:`indices` array must be one-dimensional and have an integer data type. For index in :param:`indices`:
                - _negative_: The function must determine the element to select along a specified axis (dimension) by counting from the last element (where -1 refers to the last element).

            axis : Optional[int], default to `None`
                Axis over which to select values.
                - _negative_int_: The function must determine the axis along which to select values by counting from the last dimension (where -1 refers to the last dimension).

                For dimension of `self`:
                - `1`: Providing an :param:`axis` is optional;
                - `> 1`: Providing an :param:`axis` is required.

        Returns
        -------
            CompatArray[ArrayT]
                An array having the same data type as `self`. The output array must have the same rank (i.e., number of dimensions) as `self` and must have the same shape as `self`, except for the axis specified by :param:`axis` whose size must equal the number of elements in :param:`indices`.

        Notes
        -----
        - Conceptually, `take(x, indices, axis=3)` is equivalent to `x[:,:,:,indices,...]`; however, explicit indexing via arrays of indices is not currently supported in this specification due to concerns regarding __setitem__ and array mutation semantics;
        - This specification does not require bounds checking. The behavior for out-of-bounds indices is left unspecified;
        - When `self` is a zero-dimensional array, behavior is unspecified and thus implementation-defined.
        """
        ...

    def take_along_axis(self, indices: ArrayLike, /, *, axis: int = -1) -> CompatArray[ArrayT]:
        """
        Returns elements from `self` at the one-dimensional indices specified by :param:`indices` along a provided :param:`axis`.
        - `self` must be compatible with :param:`indices`, except for the :param:`axis` (dimension) specified by axis.

        Parameters
        ----------
            indices : ArrayLike
                Must have the same rank (i.e., number of dimensions) as `self`.
                For index in :param:`indices`:
                - _negative_: The function must determine the element to select along a specified axis (dimension) by counting from the last element (where -1 refers to the last element).

            axis : int, default to `-1`
                Axis along which to select values.
                - _negative_int_: The function must determine the axis along which to select values by counting from the last dimension (where -1 refers to the last dimension).

        Returns
        -------
            CompatArray[ArrayT]
                An array having the same data type as `self`. Must have the same rank (i.e., number of dimensions) as `self` and must have a shape determined according to Broadcasting, except for the axis (dimension) specified by :param:`axis` whose size must equal the size of the corresponding axis (dimension) in :param:`indices`.

        Notes
        -----
        - This specification does not require bounds checking. The behavior for out-of-bounds indices is left unspecified.
        """
        ...

    # === Linear algebra functions ===
    def matmul(self, other: ArrayLike, /) -> CompatArray[ArrayT]:
        """
        Computes the matrix product.
        - `self` should have a numeric data type. Must have at least one dimension.
            - If `self` is one-dimensional having shape `(M,)` and `other` has more than one dimension, `self` must be promoted to a two-dimensional array by prepending 1 to its dimensions (i.e., must have shape `(1, M)`). After matrix multiplication, the prepended dimensions in the returned array must be removed;
            - If `self` has more than one dimension (including after vector-to-matrix promotion), shape(`self`)[:-2] must be compatible with shape(`other`)[:-2] (after vector-to-matrix promotion);
            - If `self` has shape `(..., M, K)`, the innermost two dimensions form matrices on which to perform matrix multiplication.

        Parameters
        ----------
            other : ArrayLike
                Should have a numeric data type. Must have at least one dimension.
                - If `other` is one-dimensional having shape `(N,)` and `self` has more than one dimension, `other` must be promoted to a two-dimensional array by appending 1 to its dimensions (i.e., must have shape `(N, 1)`). After matrix multiplication, the appended dimensions in the returned array must be removed;
                - If `other` has more than one dimension (including after vector-to-matrix promotion), shape(`other`)[:-2] must be compatible with shape(`self`)[:-2] (after vector-to-matrix promotion);
                - If `other` has shape `(..., K, N)`, the innermost two dimensions form matrices on which to perform matrix multiplication.

        Returns
        -------
            CompatArray[ArrayT]
                The returned array must have a data type determined by Type Promotion Rules.
                - If both `self` and `other` are one-dimensional arrays having shape `(N,)`, a zero-dimensional array containing the inner product as its only element.
                - If `self` is a two-dimensional array having shape `(M, K)` and `other` is a two-dimensional array having shape `(K, N)`, a two-dimensional array containing the conventional matrix product and having shape `(M, N)`.
                - If `self` is a one-dimensional array having shape `(K,)` and `other` is an array having shape `(..., K, N)`, an array having shape `(..., N)` (i.e., prepended dimensions during vector-to-matrix promotion must be removed) and containing the conventional matrix product.
                - If `self` is an array having shape `(..., M, K)` and `other` is a one-dimensional array having shape `(K,)`, an array having shape `(..., M)` (i.e., appended dimensions during vector-to-matrix promotion must be removed) and containing the conventional matrix product.
                - If `self` is a two-dimensional array having shape `(M, K)` and `other` is an array having shape `(..., K, N)`, an array having shape `(..., M, N)` and containing the conventional matrix product for each stacked matrix.
                - If `self` is an array having shape `(..., M, K)` and `other` is a two-dimensional array having shape `(K, N)`, an array having shape `(..., M, N)` and containing the conventional matrix product for each stacked matrix.
                - If either `self` or `other` has more than two dimensions, an array having a shape determined by Broadcasting shape(`self`)[:-2] against shape(`other`)[:-2] and containing the conventional matrix product for each stacked matrix.
        """
        ...

    def tensordot(self, other: ArrayLike, /, *, axes: Union[int, Tuple[Sequence[int], Sequence[int]]] = 2) -> CompatArray[ArrayT]:
        """
        Returns a tensor contraction of `self` and `other` over specific axes.
        - The function corresponds to the generalized matrix product.
        - `self` should have a numeric data type.

        Parameters
        ----------
            other : ArrayLike
                Should have a numeric data type. Corresponding contracted axes of `self` and `other` must be equal.

                NOTE: Contracted axes (dimensions) must not be broadcasted.

            axes : Union[int, Tuple[Sequence[int], Sequence[int]]], default to `2`
                Number of axes (dimensions) to contract or explicit sequences of axis (dimension) indices for `self` and `other`, respectively.
                - _int_(`N`): Contraction must be performed over the last `N` axes of `self` and the first `N` axes of `other` in order. The size of each corresponding axis (dimension) must match. Must be nonnegative. For `N`:

                    - `0`: The result is the tensor (outer) product;
                    - `1`: The result is the tensor dot product;
                    - `2`: The result is the tensor double contraction (default).
                - _Tuple[`self_axes`, `other_axes`]_: The first sequence must apply to `self` and the second sequence to `other`. Both sequences must have the same length. Each axis (dimension) `self_axes[i]` for `self` must have the same size as the respective axis (dimension) `other_axes[i]` for `other`. Each index referred to in a sequence must be unique. If `self` has rank (i.e, number of dimensions) `N`, a valid `self` axis must reside on the half-open interval `[-N, N)`. If `other` has rank M, a valid `other` axis must reside on the half-open interval `[-M, M)`.

        Returns
        -------
            CompatArray[ArrayT]
                An array containing the tensor contraction whose shape consists of the non-contracted axes (dimensions) of the first array `self`, followed by the non-contracted axes (dimensions) of the second array `other`. The returned array must have a data type determined by Type Promotion Rules.
        """
        ...

    def matrix_transpose(self) -> CompatArray[ArrayT]:
        """
        Transposes the matrix (or a stack of matrices) `self`.
        - `self` having shape `(..., M, N)` and whose innermost two dimensions form `MxN` matrices.

        Returns
        -------
            CompatArray[ArrayT]
                An array containing the transpose for each matrix and having shape `(..., N, M)`.
                The returned array must have the same data type as `self`.
        """
        ...

    def vecdot(self, other: ArrayLike, /, *, axes: int = -1) -> CompatArray[ArrayT]:
        """
        Computes the (vector) dot product of two arrays.
        - `self` should have a floating-point data type.

        Parameters
        ----------
            other : ArrayLike
                Must be compatible with `self` for all non-contracted axes.
                The size of the axis over which to compute the dot product must be the same size as the respective axis in `self`.
                Should have a floating-point data type.

                NOTE: The contracted axis (dimension) must not be broadcasted.

            axes : int, default to `-1`
                The axis (dimension) of `self` and `other` containing the vectors for which to compute the dot product.
                Should be an integer on the interval `[-N, -1]`, where `N` is min(`self.ndim`, `other.ndim`).
                The function must determine the axis along which to compute the dot product by counting backward from the last dimension (where -1 refers to the last dimension).
                By default, the function must compute the dot product over the last axis.

        Returns
        -------
            CompatArray[ArrayT]
                The returned array must have a data type determined by Type Promotion Rules.
                - `self` & `other` is `1-D`: A zero-dimensional array containing the dot product as its only element.
                - _others_: A non-zero-dimensional array containing the dot products and having a shape determined according to Broadcasting along the non-contracted axes.
        """

    # === Manipulation functions ===
    def expand_dims(self, *, axis: int) -> CompatArray[ArrayT]:
        """
        Expands the shape of `self` by inserting a new axis (dimension) of size one at the position specified by :param:`axis`.

        Parameters
        ----------
            axis : int
                Axis position (zero-based).
                - `self` has rank (i.e, number of dimensions) `N`: A valid :param:`axis` must reside on the closed-interval `[-N-1, N]`;
                - _negative_: The axis position at which to insert a singleton dimension must be computed as `N + axis + 1`.

                    - `-1`: The resolved axis position must be `N` (i.e., a singleton dimension must be appended to `self`);
                    - `-N-1`: The resolved axis position must be `0` (i.e., a singleton dimension must be prepended to `self`).

        Returns
        -------
            CompatArray[ArrayT]
                An expanded output array having the same data type as `self`.

        Raises
        ------
            IndexError
                If provided an invalid :param:`axis` position.
        """
        ...

    def flip(self, *, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> CompatArray[ArrayT]:
        """
        Reverses the order of elements in `self` along the given axis. The shape of `self` must be preserved.

        Parameters
        ----------
            axis : Optional[Union[int, Tuple[int, ...]]], default to `None`
                Axis (or axes) along which to flip.
                - `None`: The function must flip all input array axes;
                - _negative_: The function must count from the last dimension;
                - more than one axis: The function must flip only the specified axes.


        Returns
        -------
            CompatArray[ArrayT]
                An output array having the same data type and shape as `self` and whose elements, relative to `self`, are reordered.
        """
        ...

    def moveaxis(
        self,
        source: Union[int, Tuple[int, ...]],
        destination: Union[int, Tuple[int, ...]],
        /
    ) -> CompatArray[ArrayT]:
        """
        Moves `self` axes (dimensions) to new positions, while leaving other axes in their original positions.

        Parameters
        ----------
            source : Union[int, Tuple[int, ...]]
                Axes to move. Provided axes must be unique.
                - `self` has rank (i.e, number of dimensions) `N`: A valid :param:`axis` must reside on the half-open interval `[-N, N)`.

            destination : Union[int, Tuple[int, ...]]
                Indices defining the desired positions for each respective :param:`source` axis index.
                Provided indices must be unique.
                - `self` has rank (i.e, number of dimensions) `N`: A valid :param:`axis` must reside on the half-open interval `[-N, N)`.

        Returns
        -------
            CompatArray[ArrayT]
                An array containing reordered axes. The returned array must have the same data type as `self`.
        """
        ...

    def permute_dims(self, axes: Tuple[int, ...]) -> CompatArray[ArrayT]:
        """
        Permutes the axes (dimensions) of `self`.

        Parameters
        ----------
            axes : Tuple[int, ...]
                Tuple containing a permutation of `(0, 1, ..., N-1)` where `N` is the number of axes (dimensions) of `self`.

        Returns
        -------
            CompatArray[ArrayT]
                An array containing the axes permutation. The returned array must have the same data type as `self`.
        """
        ...

    def repeat(
        self,
        repeats: Union[int, ArrayLike],
        /, *,
        axis: Optional[int] = None
    ) -> CompatArray[ArrayT]:
        """
        Repeats each element of `self` a specified number of times on a per-element basis.

        Parameters
        ----------
            repeats : Union[int, ArrayLike]
                The number of repetitions for each element.

                If :param:`axis` is `None`, let `N = prod(self.shape)` and
                - _ArrayLike_: :param:`repeats` must be broadcast compatible with the shape `(N,)` (i.e., be a one-dimensional array having shape `(1,)` or `(N,)`);
                - _int_: :param:`repeats` must be broadcasted to the shape `(N,)`.

                If :param:`axis` is not `None`, let `M = self.shape[axis]` and
                - _ArrayLike_: :param:`repeats` must be broadcast compatible with the shape `(M,)` (i.e., be a one-dimensional array having shape `(1,)` or `(M,)`);
                - _int_: :param:`repeats` must be broadcasted to the shape `(M,)`.

                If :param:`repeats` is an array, the array must have an integer data type.

            axis : Optional[int], default to `None`
                The axis along which to repeat elements.
                - `None`: The function must flatten `self` and then repeat elements of the flattened `self` and return the result as a one-dimensional output array. A flattened `self` must be flattened in row-major, C-style order.

        Returns
        -------
            CompatArray[ArrayT]
                An output array containing repeated elements.
                The returned array must have the same data type as `self`.
                For :param:`axis`:
                - `None`: the returned array must be a one-dimensional array;
                - _others_: the returned array must have the same shape as `self`, except for the axis (dimension) along which elements were repeated.
        """
        ...

    def reshape(
        self,
        shape: Tuple[int, ...],
        *,
        copy: Optional[bool] = None
    ) -> CompatArray[ArrayT]:
        """
        Reshapes `self` without changing its data.

        Parameters
        ----------
            shape : Tuple[int, ...]
                A new shape compatible with the original shape. One shape dimension is allowed to be -1.
                When a shape dimension is -1, the corresponding output array shape dimension must be inferred from the length of `self` and the remaining dimensions.

            copy : Optional[bool], default to `None`
                Whether or not to copy `self`.
                - `True`: The function must always copy;
                - `False`: The function must never copy;
                - `None`: The function must avoid copying, if possible, and may copy otherwise.

        Returns
        -------
            CompatArray[ArrayT]
                An output array having the same data type and elements as `self`.

        Raises
        ------
            ValueError
                If :param:`copy` is `False` and a copy would be necessary.
        """
        ...

    def roll(
        self,
        shift: Union[int, Tuple[int, ...]],
        *,
        axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> CompatArray[ArrayT]:
        """
        Rolls `self` elements along a specified axis.
        - Array elements that roll beyond the last position are re-introduced at the first position.
        - Array elements that roll beyond the first position are re-introduced at the last position.

        Parameters
        ----------
            shift : Union[int, Tuple[int, ...]]
                Number of places by which the elements are shifted.
                - _tuple_: :param:`axis` must be a tuple of the same size, and each of the given axes must be shifted by the corresponding element in :param:`shift`;
                - _int_ and :param:`axis` is _tuple_: the same :param:`shift` must be used for all specified axes;
                - _positive_: `self` elements must be shifted positively (toward larger indices) along the dimension of :param:`axis`;
                - _negative_: `self` elements must be shifted negatively (toward smaller indices) along the dimension of :param:`axis`.

            axis : Optional[Union[int, Tuple[int, ...]]], default to `None`
                Axis (or axes) along which elements to shift.
                - `None`: `self` must be flattened, shifted, and then restored to its original shape.

        Returns
        -------
            CompatArray[ArrayT]
                An output array having the same data type and shape as `self` and whose elements, relative to `self`, are shifted.
        """
        ...

    def squeeze(self, axis: Union[int, Tuple[int, ...]]) -> CompatArray[ArrayT]:
        """
        Removes singleton dimensions (axes) from `self`.

        Parameters
        ----------
            axis : Union[int, Tuple[int, ...]]
                Axis (or axes) to squeeze.

        Returns
        -------
            CompatArray[ArrayT]
                An output array having the same data type and elements as `self`.

        Raises
        ------
            ValueError
                If a specified axis has a size greater than one (i.e., it is not a singleton dimension).
        """
        ...

    def tile(self, repetitions: Tuple[int, ...], /) -> CompatArray[ArrayT]:
        """
        Constructs an array by tiling `self`.

        Parameters
        ----------
            repetitions : Tuple[int, ...]
                Number of repetitions along each axis (dimension).
                Let `N = len(self.shape)` and `M = len(repetitions)`.
                - `N > M`: The function must prepend ones until all axes (dimensions) are specified (e.g., if `self` has shape `(8,6,4,2)` and :param:`repetitions` is the tuple `(3,3)`, then :param:`repetitions` must be treated as `(1,1,3,3)`);
                - `N < M`: The function must prepend singleton axes (dimensions) to `self` until `self` has as many axes (dimensions) as :param:`repetitions` specifies (e.g., if `self` has shape `(4,2)` and :param:`repetitions` is the tuple `(3,3,3,3)`, then `self` must be treated as if it has shape `(1,1,4,2)`).

        Returns
        -------
            CompatArray[ArrayT]
                A tiled output array.
                The returned array must have the same data type as `self` and must have a rank (i.e., number of dimensions) equal to `max(N, M)`.
                If `S` is the shape of the tiled array after prepending singleton dimensions (if necessary) and `r` is the tuple of repetitions after prepending ones (if necessary), then the number of elements along each axis (dimension) must satisfy `S[i]*r[i]`, where `i` refers to the `i` th axis (dimension).
        """
        ...

    def unstack(self, *, axis: int = 0) -> Tuple[CompatArray[ArrayT], ...]:
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
            Tuple[CompatArray[ArrayT], ...]
                Tuple of slices along the given dimension.
                All the arrays have the same shape.
        """
        ...

    # === Searching functions ===
    def argmax(
        self, *,
        axis: Optional[int] = None,
        keepdims: bool = False
    ) -> CompatArray[ArrayT]:
        """
        Returns the indices of the maximum values along a specified axis.

        When the maximum value occurs multiple times, only the indices corresponding to the first occurrence are returned.

        - For backward compatibility, conforming implementations may support complex numbers; however, inequality comparison of complex numbers is unspecified and thus implementation-dependent.
        - `self` should have a real-valued data type.

        Parameters
        ----------
            axis : Optional[int], default to `None`
                Axis along which to search.
                - `None`: The function must return the index of the maximum value of the flattened `self`.

            keepdims : bool, default to `False`
                - `True`: The reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with `self`.
                - `False`: The reduced axes (dimensions) must not be included in the result.

        Returns
        -------
            CompatArray[ArrayT]
                The returned array must have be the default array index data type.
                For :param:`axis`:
                - `None`: A zero-dimensional array containing the index of the first occurrence of the maximum value;
                - _others_: A non-zero-dimensional array containing the indices of the maximum values.
        """
        ...

    def argmin(
        self, *,
        axis: Optional[int] = None,
        keepdims: bool = False
    ) -> CompatArray[ArrayT]:
        """
        Returns the indices of the minimum values along a specified axis.

        When the minimum value occurs multiple times, only the indices corresponding to the first occurrence are returned.

        - For backward compatibility, conforming implementations may support complex numbers; however, inequality comparison of complex numbers is unspecified and thus implementation-dependent.
        - `self` should have a real-valued data type.

        Parameters
        ----------
            axis : Optional[int], default to `None`
                Axis along which to search.
                - `None`: The function must return the index of the minimum value of the flattened `self`.

            keepdims : bool, default to `False`
                - `True`: The reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with `self`.
                - `False`: The reduced axes (dimensions) must not be included in the result.

        Returns
        -------
            CompatArray[ArrayT]
                The returned array must have be the default array index data type.
                For :param:`axis`:
                - `None`: A zero-dimensional array containing the index of the first occurrence of the minimum value;
                - _others_: A non-zero-dimensional array containing the indices of the minimum values.
        """
        ...

    def nonzero(self) -> Tuple[CompatArray[ArrayT], ...]:
        """
        Returns the indices of `self` elements which are non-zero.
        - `self` must have a positive rank. If `self` is zero-dimensional, the function must raise an exception.

        Returns
        -------
            Tuple[CompatArray[ArrayT], ...]
                A tuple of `k` arrays, one for each dimension of `self` and each of size `n` (where `n` is the total number of non-zero elements), containing the indices of the non-zero elements in that dimension.
                The indices must be returned in row-major, C-style order.
                The returned array must have the default array index data type.

        Notes
        -----
        - If `self` has a complex floating-point data type, non-zero elements are those elements having at least one component (real or imaginary) which is non-zero;
        - If `self` has a boolean data type, non-zero elements are those elements which are equal to `True`.
        """
        ...

    def count_nonzero(
        self, *,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False
    ) -> CompatArray[ArrayT]:
        """
        Counts the number of `self` elements which are non-zero.

        Parameters
        ----------
            axis : Optional[Union[int, Tuple[int, ...]]], default to `None`
                Axis or axes along which to count non-zero values.
                - `None`: The number of non-zero values must be computed over the entire `self`;
                - _Tuple[int, ...]_: The number of non-zero values must be computed over multiple axes.

            keepdims : bool, default to `False`
                - `True`: The reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with `self`;
                - `False`: The reduced axes (dimensions) must not be included in the result.

        Returns
        -------
            CompatArray[ArrayT]
                The returned array must have the default array index data type.
                - computed over the entire `self`: A zero-dimensional array containing the total number of non-zero values;
                - _others_: A non-zero-dimensional array containing the counts along the specified axes.

        Notes
        -----
        - If `self` has a complex floating-point data type, non-zero elements are those elements having at least one component (real or imaginary) which is non-zero;
        - If `self` has a boolean data type, non-zero elements are those elements which are equal to `True`.
        """
        ...

    def searchsorted(
        self,
        other: ArrayLike,
        /, *,
        side: Literal['left', 'right'] = "left",
        sorter: Optional[ArrayLike] = None
    ) -> CompatArray[ArrayT]:
        """
        Finds the indices into `self` such that, if the corresponding elements in `other` were inserted before the indices, the order of `self`, when sorted in ascending order, would be preserved.
        - `self` must be a one-dimensional array. Should have a real-valued data type.
            - :param:`sorter` is `None`: `self` must be sorted in ascending order;
            - :param:`sorter` is not `None`: :param:`sorter` must be an array of indices that sort `self` in ascending order.

        Parameters
        ----------
            other : ArrayLike
                Array containing search values. Should have a real-valued data type.

            side : Literal["left", "right"], default: "left"
                Argument controlling which index is returned if a value lands exactly on an edge.
                Let `v` be an element of `other` given by `v = other[j]`, where `j` refers to a valid index.
                - `v` < `self[any]`: `out[j]` must be `0`;
                - `v` > `self[any]`: `out[j]` must be `M`, where `M` is the number of elements in `self`.
                - _others_: Each returned index `i = out[j]` must satisfy an index condition:

                    - :param:`side` == `"left"`: `self[i-1] < v <= self[i]`;
                    - :param:`side` == `"right"`: `self[i-1] <= v < self[i]`.

            sorter : Optional[array], default: `None`
                Array of indices that sort `self` in ascending order.
                The array must have the same shape as `self` and have an integer data type.

        Returns
        -------
            CompatArray[ArrayT]
                An array of indices with the same shape as `other`.
                The returned array must have the default array index data type.

        Notes
        -----
        - For real-valued floating-point arrays, the sort order of NaNs and signed zeros is unspecified and thus implementation-dependent. Accordingly, when a real-valued floating-point array contains NaNs and signed zeros, what constitutes ascending order may vary among specification-conforming array libraries.
        - While behavior for arrays containing NaNs and signed zeros is implementation-dependent, specification-conforming libraries should, however, ensure consistency with :meth:`CompatArray.sort` and :meth:`CompatArray.argsort` (i.e., if a value in `other` is inserted into `self` according to the corresponding index in the output array and sort is invoked on the resultant array, the sorted result should be an array in the same order).
        """
        ...

    def where(
        self,
        X: Any,
        Y: Any,
        /
    ) -> CompatArray[ArrayT]:
        """
        Returns elements chosen from `X` or `Y` depending on condition `self`.
        - `self` is a condition array. Should have a boolean data type. Must be compatible with `X` and `Y`. For each element in `self`:
            - `True`: yield `X[i]`;
            - `False`: yield `Y[i]`.

        Parameters
        ----------
            X : Any
                First input array. Must be compatible with `self` and `Y`.

            Y : Any
                Second input array. Must be compatible with `self` and `X`.

        Returns
        -------
            CompatArray[ArrayT]
                An array with elements from `X` where `self` is `True`, and elements from `Y` elsewhere.
                The returned array must have a data type determined by Type Promotion Rules rules with the arrays `X` and `Y`.

        Notes
        - At least one of `X` and `Y` must be an array;
        - If either `X` or `Y` is a scalar value, the returned array must have a data type determined according to Mixing arrays with Python scalars.
        """
        ...

    # === Set functions ===
    def unique_all(self) -> UniqueAllResult:
        """
        Returns the unique elements of `self`, the first occurring indices for each unique element in `self`, the indices from the set of unique elements that reconstruct `self`, and the corresponding counts for each unique element in `self`.
        - `self`:
            - more than one dimension: the function must flatten `self` and return the unique elements of the flattened `self`.

        Returns
        -------
            
        
        
        out (Tuple[array, array, array, array]) – a namedtuple (values, indices, inverse_indices, counts) whose

        first element must have the field name values and must be a one-dimensional array containing the unique elements of x. The array must have the same data type as x.

        second element must have the field name indices and must be an array containing the indices (first occurrences) of a flattened x that result in values. The array must have the same shape as values and must have the default array index data type.

        third element must have the field name inverse_indices and must be an array containing the indices of values that reconstruct x. The array must have the same shape as x and must have the default array index data type.

        fourth element must have the field name counts and must be an array containing the number of times each unique element occurs in x. The order of the returned counts must match the order of values, such that a specific element in counts corresponds to the respective unique element in values. The returned array must have same shape as values and must have the default array index data type.

        Note

        The order of unique elements is not specified and may vary between implementations.
        
        Note

        Uniqueness should be determined based on value equality (see equal()). For input arrays having floating-point data types, value-based equality implies the following behavior.

        As nan values compare as False, nan values should be considered distinct.

        As complex floating-point values having at least one nan component compare as False, complex floating-point values having nan components should be considered distinct.

        As -0 and +0 compare as True, signed zeros should not be considered distinct, and the corresponding unique element will be implementation-dependent (e.g., an implementation could choose to return -0 if -0 occurs before +0).

        As signed zeros are not distinct, using inverse_indices to reconstruct the input array is not guaranteed to return an array having the exact same values.

        Each nan value and each complex floating-point value having a nan component should have a count of one, while the counts for signed zeros should be aggregated as a single count.
        """
        ...

    def unique_counts(self) -> UniqueCountsResult:
        """
        Returns the unique elements of an input array x and the corresponding counts for each unique element in x.
        
         input array. If x has more than one dimension, the function must flatten x and return the unique elements of the flattened array.

        Returns
        -------
        out (Tuple[array, array]) – a namedtuple (values, counts) whose

        first element must have the field name values and must be a one-dimensional array containing the unique elements of x. The array must have the same data type as x.

        second element must have the field name counts and must be an array containing the number of times each unique element occurs in x. The order of the returned counts must match the order of values, such that a specific element in counts corresponds to the respective unique element in values. The returned array must have same shape as values and must have the default array index data type.

        Note

        The order of unique elements is not specified and may vary between implementations.
        """
        ...

    def unique_inverse(self) -> UniqueInverseResult:
        """
        Returns the unique elements of an input array x and the indices from the set of unique elements that reconstruct x.
        
        Parameters
        ----------
        x (array) – input array. If x has more than one dimension, the function must flatten x and return the unique elements of the flattened array.

        Returns
        -------
        out (Tuple[array, array]) – a namedtuple (values, inverse_indices) whose

        first element must have the field name values and must be a one-dimensional array containing the unique elements of x. The array must have the same data type as x.

        second element must have the field name inverse_indices and must be an array containing the indices of values that reconstruct x. The array must have the same shape as x and have the default array index data type.

        Note

        The order of unique elements is not specified and may vary between implementations.
        
        """
        ...

    def unique_values(self) -> CompatArray[ArrayT]:
        """
        Returns the unique elements of an input array x.
        
        input array. If x has more than one dimension, the function must flatten x and return the unique elements of the flattened array.

        Returns
        -------
        out (array) – a one-dimensional array containing the set of unique elements in x. The returned array must have the same data type as x.

        Note

        The order of unique elements is not specified and may vary between implementations.
        
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
        Returns the indices that sort `self` along a specified axis.

        Note

        For backward compatibility, conforming implementations may support complex numbers; however, inequality comparison of complex numbers is unspecified and thus implementation-dependent (see Complex Number Ordering).

        Parameters
        ----------
        x (array) – input array. Should have a real-valued data type.

        axis (int) – axis along which to sort. If set to -1, the function must sort along the last axis. Default: -1.

        descending (bool) – sort order. If True, the returned indices sort x in descending order (by value). If False, the returned indices sort x in ascending order (by value). Default: False.

        stable (bool) – sort stability. If True, the returned indices must maintain the relative order of x values which compare as equal. If False, the returned indices may or may not maintain the relative order of x values which compare as equal (i.e., the relative order of x values which compare as equal is implementation-dependent). Default: True.

        Returns
        -------
        out (array) – an array of indices. The returned array must have the same shape as x. The returned array must have the default array index data type.
        """
        ...
        
    def sort(
        self, *,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True
    ) -> CompatArray[ArrayT]:
        """
        Returns a sorted copy of an input array x.

        Note

        For backward compatibility, conforming implementations may support complex numbers; however, inequality comparison of complex numbers is unspecified and thus implementation-dependent (see Complex Number Ordering).

        Parameters
        ----------
        x (array) – input array. Should have a real-valued data type.

        axis (int) – axis along which to sort. If set to -1, the function must sort along the last axis. Default: -1.

        descending (bool) – sort order. If True, the array must be sorted in descending order (by value). If False, the array must be sorted in ascending order (by value). Default: False.

        stable (bool) – sort stability. If True, the returned array must maintain the relative order of x values which compare as equal. If False, the returned array may or may not maintain the relative order of x values which compare as equal (i.e., the relative order of x values which compare as equal is implementation-dependent). Default: True.

        Returns
        -------
        out (array) – a sorted array. The returned array must have the same data type and shape as x.
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
        Calculates the cumulative sum of elements in the input array x.

        Parameters
        ----------
        x (array) – input array. Should have one or more dimensions (axes). Should have a numeric data type.

        axis (Optional[int]) –

        axis along which a cumulative sum must be computed. If axis is negative, the function must determine the axis along which to compute a cumulative sum by counting from the last dimension.

        If x is a one-dimensional array, providing an axis is optional; however, if x has more than one dimension, providing an axis is required.

        dtype (Optional[dtype]) –

        data type of the returned array. If None, the returned array must have the same data type as x, unless x has an integer data type supporting a smaller range of values than the default integer data type (e.g., x has an int16 or uint32 data type and the default integer data type is int64). In those latter cases:

        if x has a signed integer data type (e.g., int16), the returned array must have the default integer data type.

        if x has an unsigned integer data type (e.g., uint16), the returned array must have an unsigned integer data type having the same number of bits as the default integer data type (e.g., if the default integer data type is int32, the returned array must have a uint32 data type).

        If the data type (either specified or resolved) differs from the data type of x, the input array should be cast to the specified data type before computing the sum (rationale: the dtype keyword argument is intended to help prevent overflows). Default: None.

        include_initial (bool) – boolean indicating whether to include the initial value as the first value in the output. By convention, the initial value must be the additive identity (i.e., zero). Default: False.

        Returns
        -------
        out (array) – an array containing the cumulative sums. The returned array must have a data type as described by the dtype parameter above.

        Let N be the size of the axis along which to compute the cumulative sum. The returned array must have a shape determined according to the following rules:

        if include_initial is True, the returned array must have the same shape as x, except the size of the axis along which to compute the cumulative sum must be N+1.

        if include_initial is False, the returned array must have the same shape as x.

        Notes

        When x is a zero-dimensional array, behavior is unspecified and thus implementation-defined.
        """
        ...

    def cumulative_prod(
        self, *,
        axis: int | None = None,
        dtype: Dtype | None = None,
        include_initial: bool = False
    ) -> CompatArray[ArrayT]:
        """
        Calculates the cumulative product of elements in the input array x.

        Parameters
        ----------
        x (array) – input array. Should have one or more dimensions (axes). Should have a numeric data type.

        axis (Optional[int]) –

        axis along which a cumulative product must be computed. If axis is negative, the function must determine the axis along which to compute a cumulative product by counting from the last dimension.

        If x is a one-dimensional array, providing an axis is optional; however, if x has more than one dimension, providing an axis is required.

        dtype (Optional[dtype]) –

        data type of the returned array. If None, the returned array must have the same data type as x, unless x has an integer data type supporting a smaller range of values than the default integer data type (e.g., x has an int16 or uint32 data type and the default integer data type is int64). In those latter cases:

        if x has a signed integer data type (e.g., int16), the returned array must have the default integer data type.

        if x has an unsigned integer data type (e.g., uint16), the returned array must have an unsigned integer data type having the same number of bits as the default integer data type (e.g., if the default integer data type is int32, the returned array must have a uint32 data type).

        If the data type (either specified or resolved) differs from the data type of x, the input array should be cast to the specified data type before computing the product (rationale: the dtype keyword argument is intended to help prevent overflows). Default: None.

        include_initial (bool) – boolean indicating whether to include the initial value as the first value in the output. By convention, the initial value must be the multiplicative identity (i.e., one). Default: False.

        Returns
        -------
        out (array) – an array containing the cumulative products. The returned array must have a data type as described by the dtype parameter above.

        Let N be the size of the axis along which to compute the cumulative product. The returned array must have a shape determined according to the following rules:

        if include_initial is True, the returned array must have the same shape as x, except the size of the axis along which to compute the cumulative product must be N+1.

        if include_initial is False, the returned array must have the same shape as x.

        Notes

        When x is a zero-dimensional array, behavior is unspecified and thus implementation-defined.
        """
        ...

    def max(
        self, *,
        axis: int | Tuple[int, ...] | None = None,
        keepdims: bool = False
    ) -> CompatArray[ArrayT]:
        """
        Calculates the maximum value of the input array x.

        Parameters
        ----------
        x (array) – input array. Should have a real-valued data type.

        axis (Optional[Union[int, Tuple[int, ...]]]) – axis or axes along which maximum values must be computed. By default, the maximum value must be computed over the entire array. If a tuple of integers, maximum values must be computed over multiple axes. Default: None.

        keepdims (bool) – if True, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array. Otherwise, if False, the reduced axes (dimensions) must not be included in the result. Default: False.

        Returns
        -------
        out (array) – if the maximum value was computed over the entire array, a zero-dimensional array containing the maximum value; otherwise, a non-zero-dimensional array containing the maximum values. The returned array must have the same data type as x.

        Notes

        When the number of elements over which to compute the maximum value is zero, the maximum value is implementation-defined. Specification-compliant libraries may choose to raise an error, return a sentinel value (e.g., if x is a floating-point input array, return NaN), or return the minimum possible value for the input array x data type (e.g., if x is a floating-point array, return -infinity).

        The order of signed zeros is unspecified and thus implementation-defined. When choosing between -0 or +0 as a maximum value, specification-compliant libraries may choose to return either value.

        For backward compatibility, conforming implementations may support complex numbers; however, inequality comparison of complex numbers is unspecified and thus implementation-defined (see Complex Number Ordering).

        Special Cases

        For floating-point operands,

        If x_i is NaN, the maximum value is NaN (i.e., NaN values propagate).
        """
        ...

    def mean(
        self, *,
        axis: int | Tuple[int, ...] | None = None,
        keepdims: bool = False
    ) -> CompatArray[ArrayT]:
        """
        Calculates the arithmetic mean of the input array x.

        Parameters
        ----------
        x (array) – input array. Should have a floating-point data type.

        axis (Optional[Union[int, Tuple[int, ...]]]) – axis or axes along which arithmetic means must be computed. By default, the mean must be computed over the entire array. If a tuple of integers, arithmetic means must be computed over multiple axes. Default: None.

        keepdims (bool) – if True, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array. Otherwise, if False, the reduced axes (dimensions) must not be included in the result. Default: False.

        Returns
        -------
        out (array) – if the arithmetic mean was computed over the entire array, a zero-dimensional array containing the arithmetic mean; otherwise, a non-zero-dimensional array containing the arithmetic means. The returned array must have the same data type as x.

        Note

        While this specification recommends that this function only accept input arrays having a floating-point data type, specification-compliant array libraries may choose to accept input arrays having an integer data type. While mixed data type promotion is implementation-defined, if the input array x has an integer data type, the returned array must have the default real-valued floating-point data type.

        Notes

        Special Cases

        Let N equal the number of elements over which to compute the arithmetic mean. For real-valued operands,

        If N is 0, the arithmetic mean is NaN.

        If x_i is NaN, the arithmetic mean is NaN (i.e., NaN values propagate).

        For complex floating-point operands, real-valued floating-point special cases should independently apply to the real and imaginary component operations involving real numbers. For example, let a = real(x_i) and b = imag(x_i), and

        If N is 0, the arithmetic mean is NaN + NaN j.

        If a is NaN, the real component of the result is NaN.

        Similarly, if b is NaN, the imaginary component of the result is NaN.

        Note

        Array libraries, such as NumPy, PyTorch, and JAX, currently deviate from this specification in their handling of components which are NaN when computing the arithmetic mean. In general, consumers of array libraries implementing this specification should use isnan() to test whether the result of computing the arithmetic mean over an array have a complex floating-point data type is NaN, rather than relying on NaN propagation of individual components.
        """
        ...

    def min(
        self, *,
        axis: int | Tuple[int, ...] | None = None,
        keepdims: bool = False
    ) -> CompatArray[ArrayT]:
        """
        Calculates the minimum value of the input array x.

        Parameters
        ----------
        x (array) – input array. Should have a real-valued data type.

        axis (Optional[Union[int, Tuple[int, ...]]]) – axis or axes along which minimum values must be computed. By default, the minimum value must be computed over the entire array. If a tuple of integers, minimum values must be computed over multiple axes. Default: None.

        keepdims (bool) – if True, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array. Otherwise, if False, the reduced axes (dimensions) must not be included in the result. Default: False.

        Returns
        -------
        out (array) – if the minimum value was computed over the entire array, a zero-dimensional array containing the minimum value; otherwise, a non-zero-dimensional array containing the minimum values. The returned array must have the same data type as x.

        Notes

        When the number of elements over which to compute the minimum value is zero, the minimum value is implementation-defined. Specification-compliant libraries may choose to raise an error, return a sentinel value (e.g., if x is a floating-point input array, return NaN), or return the maximum possible value for the input array x data type (e.g., if x is a floating-point array, return +infinity).

        The order of signed zeros is unspecified and thus implementation-defined. When choosing between -0 or +0 as a minimum value, specification-compliant libraries may choose to return either value.

        For backward compatibility, conforming implementations may support complex numbers; however, inequality comparison of complex numbers is unspecified and thus implementation-defined (see Complex Number Ordering).

        Special Cases

        For floating-point operands,

        If x_i is NaN, the minimum value is NaN (i.e., NaN values propagate).
        """
        ...

    def prod(
        self, *,
        axis: int | Tuple[int, ...] | None = None,
        dtype: Dtype | None = None,
        keepdims: bool = False
    ) -> CompatArray[ArrayT]:
        """
        Calculates the product of input array x elements.

        Parameters
        ----------
        x (array) – input array. Should have a numeric data type.

        axis (Optional[Union[int, Tuple[int, ...]]]) – axis or axes along which products must be computed. By default, the product must be computed over the entire array. If a tuple of integers, products must be computed over multiple axes. Default: None.

        dtype (Optional[dtype]) –

        data type of the returned array. If None, the returned array must have the same data type as x, unless x has an integer data type supporting a smaller range of values than the default integer data type (e.g., x has an int16 or uint32 data type and the default integer data type is int64). In those latter cases:

        if x has a signed integer data type (e.g., int16), the returned array must have the default integer data type.

        if x has an unsigned integer data type (e.g., uint16), the returned array must have an unsigned integer data type having the same number of bits as the default integer data type (e.g., if the default integer data type is int32, the returned array must have a uint32 data type).

        If the data type (either specified or resolved) differs from the data type of x, the input array should be cast to the specified data type before computing the sum (rationale: the dtype keyword argument is intended to help prevent overflows). Default: None.

        keepdims (bool) – if True, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array. Otherwise, if False, the reduced axes (dimensions) must not be included in the result. Default: False.

        Returns
        -------
        out (array) – if the product was computed over the entire array, a zero-dimensional array containing the product; otherwise, a non-zero-dimensional array containing the products. The returned array must have a data type as described by the dtype parameter above.

        Notes

        Special Cases

        Let N equal the number of elements over which to compute the product.

        If N is 0, the product is 1 (i.e., the empty product).
        """
        ...

    def std(
        self, *,
        axis: int | Tuple[int, ...] | None = None,
        correction: int | float = 0,
        keepdims: bool = False
    ) -> CompatArray[ArrayT]:
        """
        Calculates the standard deviation of the input array x.

        Parameters
        ----------
        x (array) – input array. Should have a real-valued floating-point data type.

        axis (Optional[Union[int, Tuple[int, ...]]]) – axis or axes along which standard deviations must be computed. By default, the standard deviation must be computed over the entire array. If a tuple of integers, standard deviations must be computed over multiple axes. Default: None.

        correction (Union[int, float]) – degrees of freedom adjustment. Setting this parameter to a value other than 0 has the effect of adjusting the divisor during the calculation of the standard deviation according to N-c where N corresponds to the total number of elements over which the standard deviation is computed and c corresponds to the provided degrees of freedom adjustment. When computing the standard deviation of a population, setting this parameter to 0 is the standard choice (i.e., the provided array contains data constituting an entire population). When computing the corrected sample standard deviation, setting this parameter to 1 is the standard choice (i.e., the provided array contains data sampled from a larger population; this is commonly referred to as Bessel’s correction). Default: 0.

        keepdims (bool) – if True, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array. Otherwise, if False, the reduced axes (dimensions) must not be included in the result. Default: False.

        Returns
        -------
        out (array) – if the standard deviation was computed over the entire array, a zero-dimensional array containing the standard deviation; otherwise, a non-zero-dimensional array containing the standard deviations. The returned array must have the same data type as x.

        Note

        While this specification recommends that this function only accept input arrays having a real-valued floating-point data type, specification-compliant array libraries may choose to accept input arrays having an integer data type. While mixed data type promotion is implementation-defined, if the input array x has an integer data type, the returned array must have the default real-valued floating-point data type.

        Notes

        Special Cases

        Let N equal the number of elements over which to compute the standard deviation.

        If N - correction is less than or equal to 0, the standard deviation is NaN.

        If x_i is NaN, the standard deviation is NaN (i.e., NaN values propagate).
        """
        ...

    def sum(
        self, *,
        axis: int | Tuple[int, ...] | None = None,
        dtype: Dtype | None = None,
        keepdims: bool = False
    ) -> CompatArray[ArrayT]:
        """
        Calculates the sum of the input array x.

        Parameters
        ----------
        x (array) – input array. Should have a numeric data type.

        axis (Optional[Union[int, Tuple[int, ...]]]) – axis or axes along which sums must be computed. By default, the sum must be computed over the entire array. If a tuple of integers, sums must be computed over multiple axes. Default: None.

        dtype (Optional[dtype]) –

        data type of the returned array. If None, the returned array must have the same data type as x, unless x has an integer data type supporting a smaller range of values than the default integer data type (e.g., x has an int16 or uint32 data type and the default integer data type is int64). In those latter cases:

        if x has a signed integer data type (e.g., int16), the returned array must have the default integer data type.

        if x has an unsigned integer data type (e.g., uint16), the returned array must have an unsigned integer data type having the same number of bits as the default integer data type (e.g., if the default integer data type is int32, the returned array must have a uint32 data type).

        If the data type (either specified or resolved) differs from the data type of x, the input array should be cast to the specified data type before computing the sum (rationale: the dtype keyword argument is intended to help prevent overflows). Default: None.

        keepdims (bool) – if True, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array. Otherwise, if False, the reduced axes (dimensions) must not be included in the result. Default: False.

        Returns
        -------
        out (array) – if the sum was computed over the entire array, a zero-dimensional array containing the sum; otherwise, an array containing the sums. The returned array must have a data type as described by the dtype parameter above.

        Notes

        Special Cases

        Let N equal the number of elements over which to compute the sum.

        If N is 0, the sum is 0 (i.e., the empty sum).
        """
        ...

    def var(
        self, *,
        axis: int | Tuple[int, ...] | None = None,
        correction: int | float = 0,
        keepdims: bool = False
    ) -> CompatArray[ArrayT]:
        """
        Calculates the variance of the input array x.

        Parameters
        ----------
        x (array) – input array. Should have a real-valued floating-point data type.

        axis (Optional[Union[int, Tuple[int, ...]]]) – axis or axes along which variances must be computed. By default, the variance must be computed over the entire array. If a tuple of integers, variances must be computed over multiple axes. Default: None.

        correction (Union[int, float]) – degrees of freedom adjustment. Setting this parameter to a value other than 0 has the effect of adjusting the divisor during the calculation of the variance according to N-c where N corresponds to the total number of elements over which the variance is computed and c corresponds to the provided degrees of freedom adjustment. When computing the variance of a population, setting this parameter to 0 is the standard choice (i.e., the provided array contains data constituting an entire population). When computing the unbiased sample variance, setting this parameter to 1 is the standard choice (i.e., the provided array contains data sampled from a larger population; this is commonly referred to as Bessel’s correction). Default: 0.

        keepdims (bool) – if True, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array. Otherwise, if False, the reduced axes (dimensions) must not be included in the result. Default: False.

        Returns
        -------
        out (array) – if the variance was computed over the entire array, a zero-dimensional array containing the variance; otherwise, a non-zero-dimensional array containing the variances. The returned array must have the same data type as x.

        Note

        While this specification recommends that this function only accept input arrays having a real-valued floating-point data type, specification-compliant array libraries may choose to accept input arrays having an integer data type. While mixed data type promotion is implementation-defined, if the input array x has an integer data type, the returned array must have the default real-valued floating-point data type.

        Notes

        Special Cases

        Let N equal the number of elements over which to compute the variance.

        If N - correction is less than or equal to 0, the variance is NaN.

        If x_i is NaN, the variance is NaN (i.e., NaN values propagate).
        """
        ...
    
    # === _utility_functions ===
    def all(
        self, *,
        axis: int | Tuple[int, ...] | None = None,
        keepdims: bool = False
    ) -> CompatArray[ArrayT]:
        """
        Tests whether all input array elements evaluate to True along a specified axis.

        Note

        Positive infinity, negative infinity, and NaN must evaluate to True.

        Note

        If x has a complex floating-point data type, elements having a non-zero component (real or imaginary) must evaluate to True.

        Note

        If x is an empty array or the size of the axis (dimension) along which to evaluate elements is zero, the test result must be True.

        Parameters
        ----------
        x (array) – input array.

        axis (Optional[Union[int, Tuple[int, ...]]]) – axis or axes along which to perform a logical AND reduction. By default, a logical AND reduction must be performed over the entire array. If a tuple of integers, logical AND reductions must be performed over multiple axes. A valid axis must be an integer on the interval [-N, N), where N is the rank (number of dimensions) of x. If an axis is specified as a negative integer, the function must determine the axis along which to perform a reduction by counting backward from the last dimension (where -1 refers to the last dimension). If provided an invalid axis, the function must raise an exception. Default: None.

        keepdims (bool) – If True, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array. Otherwise, if False, the reduced axes (dimensions) must not be included in the result. Default: False.

        Returns
        -------
        out (array) – if a logical AND reduction was performed over the entire array, the returned array must be a zero-dimensional array containing the test result; otherwise, the returned array must be a non-zero-dimensional array containing the test results. The returned array must have a data type of bool.
        """
        ...

    def any(
        self, *,
        axis: int | Tuple[int, ...] | None = None,
        keepdims: bool = False
    ) -> CompatArray[ArrayT]:
        """
        Tests whether any input array element evaluates to True along a specified axis.

        Note

        Positive infinity, negative infinity, and NaN must evaluate to True.

        Note

        If x has a complex floating-point data type, elements having a non-zero component (real or imaginary) must evaluate to True.

        Note

        If x is an empty array or the size of the axis (dimension) along which to evaluate elements is zero, the test result must be False.

        Parameters
        ----------
        x (array) – input array.

        axis (Optional[Union[int, Tuple[int, ...]]]) – axis or axes along which to perform a logical OR reduction. By default, a logical OR reduction must be performed over the entire array. If a tuple of integers, logical OR reductions must be performed over multiple axes. A valid axis must be an integer on the interval [-N, N), where N is the rank (number of dimensions) of x. If an axis is specified as a negative integer, the function must determine the axis along which to perform a reduction by counting backward from the last dimension (where -1 refers to the last dimension). If provided an invalid axis, the function must raise an exception. Default: None.

        keepdims (bool) – If True, the reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with the input array. Otherwise, if False, the reduced axes (dimensions) must not be included in the result. Default: False.

        Returns
        -------
        out (array) – if a logical OR reduction was performed over the entire array, the returned array must be a zero-dimensional array containing the test result; otherwise, the returned array must be a non-zero-dimensional array containing the test results. The returned array must have a data type of bool.
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
        Calculates the n-th discrete forward difference along a specified axis.

        Parameters
        ----------
        x (array) – input array. Should have a numeric data type.

        axis (int) – axis along which to compute differences. A valid axis must be an integer on the interval [-N, N), where N is the rank (number of dimensions) of x. If an axis is specified as a negative integer, the function must determine the axis along which to compute differences by counting backward from the last dimension (where -1 refers to the last dimension). If provided an invalid axis, the function must raise an exception. Default: -1.

        n (int) – number of times to recursively compute differences. Default: 1.

        prepend (Optional[array]) – values to prepend to a specified axis prior to computing differences. Must have the same shape as x, except for the axis specified by axis which may have any size. Should have the same data type as x. Default: None.

        append (Optional[array]) – values to append to a specified axis prior to computing differences. Must have the same shape as x, except for the axis specified by axis which may have any size. Should have the same data type as x. Default: None.

        Returns
        -------
        out (array) – an array containing the n-th differences. Should have the same data type as x. Must have the same shape as x, except for the axis specified by axis which must have a size determined as follows:

        Let M be the number of elements along an axis specified by axis.

        Let N1 be the number of prepended values along an axis specified by axis.

        Let N2 be the number of appended values along an axis specified by axis.

        The final size of the axis specified by axis must be M + N1 + N2 - n.

        Notes

        The first-order differences are given by out[i] = x[i+1] - x[i] along a specified axis. Higher-order differences must be calculated recursively (e.g., by calling diff(out, axis=axis, n=n-1)).

        If a conforming implementation chooses to support prepend and append arrays which have a different data type than x, behavior is unspecified and thus implementation-defined. Implementations may choose to type promote (Type Promotion Rules), cast prepend and/or append to the same data type as x, or raise an exception.
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

    # === Array attributes ===
    @property
    def arr(self) -> ArrayT:
        """
        The backend-specific array instance managed by `CompatArray`.
        """
        ...

    @property
    def xp(self) -> Namespace:
        """
        The `array namespace` associated with `CompatArray`.
        """
        ...

    @property
    def dtype(self) -> DType:
        """
        Data type of the elements of `self`.
        """
        ...

    @property
    def device(self) -> Device:
        """
        Device on which `self` is stored.
        """
        ...

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Dimensions of `self`.
        """
        ...

    @property
    def ndim(self) -> int:
        """
        Number of `self` dimensions (axes).
        """
        ...

    @property
    def size(self) -> int:
        """
        Number of elements in `self`.
        """
        ...

    @property
    def T(self) -> CompatArray[ArrayT]:
        """
        Transpose of `self`.
        """
        ...




    
    
    
    
    
    
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

