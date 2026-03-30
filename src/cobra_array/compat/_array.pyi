# -*- coding: utf-8 -*-
# Python version: 3.9
# @TianZhen

from __future__ import annotations
from torch import Tensor
from numpy.typing import NDArray
from typing import (Union, List, Tuple, Optional, Any, Sequence, Generic, TypeVar, Literal, overload)

from ._base import Compat
from ..types import (
    T, DTypeT, DeviceT, dtypeT, deviceT, DType, Device,
    ArrayLike, ArrayLibraryName,
    ArrayOrAny, ArrayOrScalar, ArrayOrReal, ArrayOrIntLike, ArrayOrInt, ArrayOrbool,
    UniqueAllResult, UniqueCountsResult, UniqueInverseResult
)

TT = TypeVar("TT", bound=DType)
DT = TypeVar("DT", bound=Device)


class CompatArray(Compat, Generic[TT, DT]):
    @overload
    @classmethod
    def from_other(cls, obj: NDArray[dtypeT], /, xp: Literal["numpy"]) -> CompatArray[dtypeT, Literal["cpu"]]: ...
    @overload
    @classmethod
    def from_other(cls, obj: ArrayLike[dtypeT], /, xp: Literal["numpy"]) -> CompatArray[dtypeT, Literal["cpu"]]: ...
    @overload
    @classmethod
    def from_other(cls, obj: object, /, xp: Literal["numpy"]) -> CompatArray[Any, Literal["cpu"]]: ...
    @overload
    @classmethod
    def from_other(cls, obj: NDArray[dtypeT], /, xp: Union[object, ArrayLibraryName]) -> CompatArray[dtypeT, Any]: ...
    @overload
    @classmethod
    def from_other(cls, obj: ArrayLike[dtypeT], /, xp: Union[object, ArrayLibraryName]) -> CompatArray[dtypeT, Any]: ...
    @overload
    @classmethod
    def from_other(cls, obj: object, /, xp: Union[object, ArrayLibraryName]) -> CompatArray[Any, Any]: ...
    @classmethod
    def from_other(cls, obj: object, /, xp: Union[object, ArrayLibraryName]) -> CompatArray[Any, Any]: ...

    @overload
    def __new__(cls, arr: NDArray[dtypeT], /, **kwargs) -> CompatArray[dtypeT, Literal["cpu"]]: ...
    @overload
    def __new__(cls, arr: Tensor, /, **kwargs) -> CompatArray[Any, Any]: ...
    @overload
    def __new__(cls, arr: CompatArray[dtypeT, deviceT], /, **kwargs) -> CompatArray[dtypeT, deviceT]: ...
    @overload
    def __new__(cls, arr: ArrayLike[dtypeT], /, **kwargs) -> CompatArray[dtypeT, Any]: ...
    def __new__(cls, arr: ArrayLike[Any], /, **kwargs) -> CompatArray[Any, Any]: ...

    # === Conversion functions ===
    def to_numpy(self, *, copy: bool = False) -> NDArray[TT]: ...
    def to_tensor(self, *, device: Optional[DeviceT] = None, copy: bool = False) -> Tensor: ...
    def to_list(self, *, copy: bool = False) -> List[TT]: ...

    # === Device functions ===
    def to_device(self, device: DeviceT, /, *, stream: Optional[Any] = None) -> CompatArray[TT, DeviceT]: ...

    # === Data type functions ===
    @overload
    def astype(self, dtype: DTypeT, /, *, copy: bool = ...) -> CompatArray[DTypeT, DT]: ...
    @overload
    def astype(self, dtype: DTypeT, /, *, copy: bool = ..., device: DeviceT) -> CompatArray[DTypeT, DeviceT]: ...

    def astype(
        self,
        dtype: DType,
        /, *,
        copy: bool = True,
        device: Optional[Device] = None
    ) -> CompatArray[Any, Any]:
        """
        Copies `self` to a specified data type irrespective of Type Promotion Rules rules.

        Parameters
        ----------
            dtype : DTypeT
                Desired data type.

            copy : bool, default to `True`
                Specifies whether to copy an array when the specified dtype matches the data type of `self`.
                - `True`: A newly allocated array must always be returned;
                - `False`: Whether :param:`dtype` matches the data type of `self`:

                    - `True`: `self` must be returned;
                    - `False`: A newly allocated array must be returned.

            device : Optional[DeviceT], default to `None`
                DeviceT on which to place the returned array.
                - `None`: The output array device must be inferred from `self`.

        Returns
        -------
            CompatArray
                An array having the specified data type.
                The returned array must have the same shape as `self`.
        """
        ...

    # === Elementwise functions ===
    def abs(self) -> CompatArray[TT, DT]:
        """
        Computes the element-wise `absolute value` of `self`.
        """
        ...

    def acos(self) -> CompatArray[float, DT]:
        """
        Computes the element-wise `principal value of the inverse cosine` of `self`.
        - `self` should have a floating-point data type.
        """
        ...

    def acosh(self) -> CompatArray[float, DT]:
        """
        Computes the element-wise `inverse hyperbolic cosine` of `self`.
        - `self` should have a floating-point data type.
        """
        ...

    def add(self, other: ArrayOrScalar, /) -> CompatArray[Any, DT]:
        """
        Computes the element-wise `sum` of `self` and `other`.
        - `self` should have a numeric data type.
        """
        ...

    def asin(self) -> CompatArray[float, DT]:
        """
        Computes the element-wise `principal value of the inverse sine` of `self`.
        - `self` should have a floating-point data type.
        """
        ...

    def asinh(self) -> CompatArray[float, DT]:
        """
        Computes the element-wise `inverse hyperbolic sine` of `self`.
        - `self` should have a floating-point data type.
        """
        ...

    def atan(self) -> CompatArray[float, DT]:
        """
        Computes the element-wise `principal value of the inverse tangent` of `self`.
        - `self` should have a floating-point data type.
        """
        ...

    def atan2(self, other: ArrayOrReal, /) -> CompatArray[float, DT]:
        """
        Computes the element-wise `inverse tangent` of `self / other`, taking into account the signs of both inputs.
        - `self` should have a real-valued floating-point data type.
        - This allows determining the correct quadrant. Results are in radians in [-π, π].
        """
        ...

    def atanh(self) -> CompatArray[float, DT]:
        """
        Computes the element-wise `inverse hyperbolic tangent` of `self`.
        - `self` should have a floating-point data type.
        """
        ...

    def bitwise_and(self, other: ArrayOrIntLike, /) -> CompatArray[TT, DT]:
        """
        Computes the element-wise `bitwise AND` of `self` and `other`.
        - `self` should have an integer or boolean data type.
        """
        ...

    def bitwise_left_shift(self, other: ArrayOrInt, /) -> CompatArray[TT, DT]:
        """
        Computes the element-wise `bitwise left shift` of `self` by `other`.
        - `self` should have an integer data type.
        - Each element is shifted left by the corresponding number of bits.
        """
        ...

    def bitwise_invert(self) -> CompatArray[TT, DT]:
        """
        Computes the element-wise `bitwise NOT` of `self`.
        - `self` should have an integer or boolean data type.
        - This operation flips every bit of each element.
        """
        ...

    def bitwise_or(self, other: ArrayOrIntLike, /) -> CompatArray[TT, DT]:
        """
        Computes the element-wise `bitwise OR` of `self` and `other`.
        - `self` should have an integer or boolean data type.
        """
        ...

    def bitwise_right_shift(self, other: ArrayOrInt, /) -> CompatArray[TT, DT]:
        """
        Computes the element-wise `bitwise right shift` of `self` by `other`.
        - `self` should have an integer data type.
        - Each element is shifted right by the corresponding number of bits.
        """
        ...

    def bitwise_xor(self, other: ArrayOrIntLike, /) -> CompatArray[TT, DT]:
        """
        Computes the element-wise `bitwise XOR` of `self` and `other`.
        - `self` should have an integer or boolean data type.
        """
        ...

    def ceil(self) -> CompatArray[TT, DT]:
        """
        Computes the element-wise `ceiling` of `self`.
        - `self` should have a real-valued data type.
        - Each element is rounded to the smallest (i.e., closest to -infinity) integer-valued number that is not less than itself.
        """
        ...

    def clip(self, *, min: Optional[ArrayOrReal] = None, max: Optional[ArrayOrReal] = None) -> CompatArray[TT, DT]:
        """
        Computes the element-wise `clipping` of `self` to the range [:param:`min`, :param:`max`].
        - `self` should have a real-valued data type.
        """
        ...

    def conj(self) -> CompatArray[TT, DT]:
        """
        Computes the element-wise `complex conjugate` of `self`.
        - `self` must have a numeric data type.
        """
        ...

    def copysign(self, other: ArrayOrReal, /) -> CompatArray[float, DT]:
        """
        Computes the element-wise `copysign` of `self` with `other`.
        - `self` should have a real-valued floating-point data type.
        - Each element has the magnitude of `self` and the sign of `other`.
        """
        ...

    def cos(self) -> CompatArray[float, DT]:
        """
        Computes the element-wise `cosine` of `self`.
        - `self` should have a floating-point data type.
        """
        ...

    def cosh(self) -> CompatArray[float, DT]:
        """
        Computes the element-wise `hyperbolic cosine` of `self`.
        - `self` should have a floating-point data type.
        """
        ...

    def divide(self, other: ArrayOrScalar, /) -> CompatArray[float, DT]:
        """
        Computes the element-wise `division` of `self` by `other`.
        - `self` should have a numeric data type.
        """
        ...

    def equal(self, other: ArrayOrAny, /) -> CompatArray[bool, DT]:
        """
        Computes the element-wise truth value of `self == other`.
        """
        ...

    def exp(self) -> CompatArray[float, DT]:
        """
        Computes the element-wise `exponential` (`exp(x)`) of `self`.
        - `self` should have a floating-point data type.
        """
        ...

    def expm1(self) -> CompatArray[float, DT]:
        """
        Computes the element-wise `exp(x) - 1` of `self`.
        - `self` should have a floating-point data type.
        """
        ...

    def floor(self) -> CompatArray[TT, DT]:
        """
        Computes the element-wise `floor` of `self`.
        - `self` should have a real-valued data type.
        - Each element is rounded to the largest (i.e., closest to +infinity) integer-valued number that is not greater than itself.
        """
        ...

    def floor_divide(self, other: ArrayOrReal, /) -> CompatArray[TT, DT]:
        """
        Computes the element-wise `floor division` of `self` by `other`.
        - `self` should have a real-valued data type.
        - Each element of the division result is rounded to the largest (i.e., closest to +infinity) integer-valued number that is not greater than itself.
        """
        ...

    def greater(self, other: ArrayOrReal, /) -> CompatArray[bool, DT]:
        """
        Computes the element-wise truth value of `self > other`.
        - `self` should have a real-valued data type.
        """
        ...

    def greater_equal(self, other: ArrayOrReal, /) -> CompatArray[bool, DT]:
        """
        Computes the element-wise truth value of `self >= other`.
        - `self` should have a real-valued data type.
        """
        ...

    def hypot(self, other: ArrayOrReal, /) -> CompatArray[float, DT]:
        """
        Computes the element-wise `hypotenuse` of `self` and `other`.
        - `self` should have a real-valued floating-point data type.
        - Equivalent to `sqrt(self **2 + other **2)`, computed in a numerically stable way.
        """
        ...

    def imag(self) -> CompatArray[float, DT]:
        """
        Computes the element-wise `imaginary component` of `self`.
        - `self` should have a complex floating-point data type.
        """
        ...

    def isfinite(self) -> CompatArray[bool, DT]:
        """
        Tests the element-wise `finiteness` of `self`.
        - `self` should have a numeric data type.
        """
        ...

    def isinf(self) -> CompatArray[bool, DT]:
        """
        Tests the element-wise `infinity` of `self`.
        - - `self` should have a numeric data type.
        """
        ...

    def isnan(self) -> CompatArray[bool, DT]:
        """
        Tests the element-wise `NaN` of `self`.
        - `self` should have a numeric data type.
        """
        ...

    def less(self, other: ArrayOrReal, /) -> CompatArray[bool, DT]:
        """
        Computes the element-wise truth value of `self < other`.
        - `self` should have a real-valued data type.
        """
        ...

    def less_equal(self, other: ArrayOrReal, /) -> CompatArray[bool, DT]:
        """
        Computes the element-wise truth value of `self <= other`.
        - `self` should have a real-valued data type.
        """
        ...

    def log(self) -> CompatArray[float, DT]:
        """
        Computes the element-wise `natural logarithm` (base `e`)  of `self`.
        - `self` should have a floating-point data type.
        """
        ...

    def log1p(self) -> CompatArray[float, DT]:
        """
        Computes the element-wise `log(1 + x)` (base `e`) of `self`.
        - `self` should have a floating-point data type.
        """
        ...

    def log2(self) -> CompatArray[float, DT]:
        """
        Computes the element-wise `base-2 logarithm` of `self`.
        - `self` should have a floating-point data type.
        """
        ...

    def log10(self) -> CompatArray[float, DT]:
        """
        Computes the element-wise `base-10 logarithm` of `self`.
        - `self` should have a floating-point data type.
        """
        ...

    def logaddexp(self, other: ArrayOrReal, /) -> CompatArray[float, DT]:
        """
        Computes the element-wise `logaddexp` of `self` and `other`.
        - Equivalent to `log(exp(self) + exp(other))`.
        - `self` should have a floating-point data type.
        """
        ...

    def logical_and(self, other: ArrayOrbool, /) -> CompatArray[bool, DT]:
        """
        Computes the element-wise `logical AND` of `self` and `other`.
        - `self` should have a boolean data type.
        """
        ...

    def logical_not(self) -> CompatArray[bool, DT]:
        """
        Computes the element-wise `logical NOT` of `self`.
        - `self` should have a boolean data type.
        """
        ...

    def logical_or(self, other: ArrayOrbool, /) -> CompatArray[bool, DT]:
        """
        Computes the element-wise `logical OR` of `self` and `other`.
        - `self` should have a boolean data type.
        """
        ...

    def logical_xor(self, other: ArrayOrbool, /) -> CompatArray[bool, DT]:
        """
        Computes the element-wise `logical XOR` of `self` and `other`.
        - `self` should have a boolean data type.
        """
        ...

    def maximum(self, other: ArrayOrReal, /) -> CompatArray[Any, DT]:
        """
        Computes the element-wise `maximum` of `self` and `other`.
        - `self` should have a real-valued data type.
        """
        ...

    def minimum(self, other: ArrayOrReal, /) -> CompatArray[Any, DT]:
        """
        Computes the element-wise `minimum` of `self` and `other`.
        - `self` should have a real-valued data type.
        """
        ...

    def multiply(self, other: ArrayOrScalar, /) -> CompatArray[Any, DT]:
        """
        Computes the element-wise `product` of `self` and `other`.
        - `self` should have a numeric data type.
        """
        ...

    def negative(self) -> CompatArray[TT, DT]:
        """
        Computes the element-wise `numerical negative` of `self`.
        - `self` should have a numeric data type.
        """
        ...

    def nextafter(self, other: ArrayOrReal, /) -> CompatArray[TT, DT]:
        """
        Computes the element-wise `next representable floating-point value` of `self` toward `other`.
        - `self` should have a real-valued floating-point data type.
        """
        ...

    def not_equal(self, other: ArrayOrAny, /) -> CompatArray[bool, DT]:
        """
        Computes the element-wise truth value of `self != other`.
        """
        ...

    def positive(self) -> CompatArray[TT, DT]:
        """
        Computes the element-wise `numerical positive` of `self`.
        - `self` should have a numeric data type.
        """
        ...

    def pow(self, other: ArrayOrScalar, /) -> CompatArray[Any, DT]:
        """
        Computes the element-wise `power` of `self` and `other`.
        - `self` should have a numeric data type.
        - Each element of `self` is raised to the corresponding power in `other`.
        """
        ...

    def real(self) -> CompatArray[float, DT]:
        """
        Computes the element-wise `real component` of `self`.
        - `self` should have a numeric data type.
        """
        ...

    def reciprocal(self) -> CompatArray[float, DT]:
        """
        Computes the element-wise `reciprocal` of `self`.
        - `self` should have a floating-point data type.
        """
        ...

    def remainder(self, other: ArrayOrReal, /) -> CompatArray[Any, DT]:
        """
        Computes the element-wise `remainder` of `self` divided by `other`.
        - `self` should have a real-valued data type.
        """
        ...

    def round(self) -> CompatArray[TT, DT]:
        """
        Computes the element-wise `rounding` of `self` to the nearest integer-valued number.
        - `self` should have a numeric data type.
        """
        ...

    def sign(self) -> CompatArray[TT, DT]:
        """
        Computes the element-wise `indication of the sign` of `self`.
        - `self` should have a numeric data type.
        """
        ...

    def signbit(self) -> CompatArray[bool, DT]:
        """
        Tests the element-wise `sign bit` of `self`.
        - Tests each element for whenever is either `-0`, `less than zero`, or a signed `NaN` (i.e., a NaN value whose sign bit is 1).
        - `self` should have a real-valued floating-point data type.
        """
        ...

    def sin(self) -> CompatArray[float, DT]:
        """
        Computes the element-wise `sine` of `self`.
        - `self` should have a floating-point data type.
        """
        ...

    def sinh(self) -> CompatArray[float, DT]:
        """
        Computes the element-wise `hyperbolic sine` of `self`.
        - `self` should have a floating-point data type.
        """
        ...

    def square(self) -> CompatArray[TT, DT]:
        """
        Computes the element-wise `square` of `self`.
        - `self` should have a numeric data type.
        """
        ...

    def sqrt(self) -> CompatArray[float, DT]:
        """
        Computes the element-wise `principal square root` of `self`.
        - `self` should have a floating-point data type.
        """
        ...

    def subtract(self, other: ArrayOrScalar, /) -> CompatArray[Any, DT]:
        """
        Computes the element-wise `difference` of `self` and `other`.
        - `self` should have a numeric data type.
        """
        ...

    def tan(self) -> CompatArray[float, DT]:
        """
        Computes the element-wise `tangent` of `self`.
        - `self` should have a floating-point data type.
        """
        ...

    def tanh(self) -> CompatArray[float, DT]:
        """
        Computes the element-wise `hyperbolic tangent` of `self`.
        - `self` should have a floating-point data type.
        """
        ...

    def trunc(self) -> CompatArray[TT, DT]:
        """
        Computes the element-wise `truncation` of `self` toward `zero`.
        - `self` should have a real-valued data type.
        - Each element is rounded to the nearest integer-valued number that is closer to `zero` than itself.
        """

    # === Indexing functions ===
    def take(self, indices: ArrayLike[Any], /, *, axis: Optional[int] = None) -> CompatArray[TT, DT]:
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
            CompatArray
                A :class:`CompatArray` array having the same data type as `self`. The output array must have the same rank (i.e., number of dimensions) as `self` and must have the same shape as `self`, except for the axis specified by :param:`axis` whose size must equal the number of elements in :param:`indices`.

        Notes
        -----
        - Conceptually, `take(x, indices, axis=3)` is equivalent to `x[:,:,:,indices,...]`; however, explicit indexing via arrays of indices is not currently supported in this specification due to concerns regarding __setitem__ and array mutation semantics;
        - This specification does not require bounds checking. The behavior for out-of-bounds indices is left unspecified;
        - When `self` is a zero-dimensional array, behavior is unspecified and thus implementation-defined.
        """
        ...

    def take_along_axis(self, indices: ArrayLike[Any], /, *, axis: int = -1) -> CompatArray[TT, DT]:
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
            CompatArray
                A :class:`CompatArray` array having the same data type as `self`. Must have the same rank (i.e., number of dimensions) as `self` and must have a shape determined according to Broadcasting, except for the axis (dimension) specified by :param:`axis` whose size must equal the size of the corresponding axis (dimension) in :param:`indices`.

        Notes
        -----
        - This specification does not require bounds checking. The behavior for out-of-bounds indices is left unspecified.
        """
        ...

    # === Linear algebra functions ===
    def matmul(self, other: ArrayLike[Any], /) -> CompatArray[Any, DT]:
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
            CompatArray
                The returned :class:`CompatArray` array must have a data type determined by Type Promotion Rules.
                - If both `self` and `other` are one-dimensional arrays having shape `(N,)`, a zero-dimensional array containing the inner product as its only element.
                - If `self` is a two-dimensional array having shape `(M, K)` and `other` is a two-dimensional array having shape `(K, N)`, a two-dimensional array containing the conventional matrix product and having shape `(M, N)`.
                - If `self` is a one-dimensional array having shape `(K,)` and `other` is an array having shape `(..., K, N)`, an array having shape `(..., N)` (i.e., prepended dimensions during vector-to-matrix promotion must be removed) and containing the conventional matrix product.
                - If `self` is an array having shape `(..., M, K)` and `other` is a one-dimensional array having shape `(K,)`, an array having shape `(..., M)` (i.e., appended dimensions during vector-to-matrix promotion must be removed) and containing the conventional matrix product.
                - If `self` is a two-dimensional array having shape `(M, K)` and `other` is an array having shape `(..., K, N)`, an array having shape `(..., M, N)` and containing the conventional matrix product for each stacked matrix.
                - If `self` is an array having shape `(..., M, K)` and `other` is a two-dimensional array having shape `(K, N)`, an array having shape `(..., M, N)` and containing the conventional matrix product for each stacked matrix.
                - If either `self` or `other` has more than two dimensions, an array having a shape determined by Broadcasting shape(`self`)[:-2] against shape(`other`)[:-2] and containing the conventional matrix product for each stacked matrix.
        """
        ...

    def tensordot(self, other: ArrayLike[Any], /, *, axes: Union[int, Tuple[Sequence[int], Sequence[int]]] = 2) -> CompatArray[Any, DT]:
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
            CompatArray
                A :class:`CompatArray` array containing the tensor contraction whose shape consists of the non-contracted axes (dimensions) of the first array `self`, followed by the non-contracted axes (dimensions) of the second array `other`. The returned array must have a data type determined by Type Promotion Rules.
        """
        ...

    def matrix_transpose(self) -> CompatArray[TT, DT]:
        """
        Transposes the matrix (or a stack of matrices) `self`.
        - `self` having shape `(..., M, N)` and whose innermost two dimensions form `MxN` matrices.

        Returns
        -------
            CompatArray
                A :class:`CompatArray` array containing the transpose for each matrix and having shape `(..., N, M)`.
                The returned array must have the same data type as `self`.
        """
        ...

    def vecdot(self, other: ArrayLike[Any], /, *, axes: int = -1) -> CompatArray[Any, DT]:
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
            CompatArray
                The returned :class:`CompatArray` array must have a data type determined by Type Promotion Rules.
                - `self` & `other` is `1-D`: A zero-dimensional array containing the dot product as its only element.
                - _others_: A non-zero-dimensional array containing the dot products and having a shape determined according to Broadcasting along the non-contracted axes.
        """

    # === Manipulation functions ===
    def expand_dims(self, *, axis: int) -> CompatArray[TT, DT]:
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
            CompatArray
                An expanded output :class:`CompatArray` array having the same data type as `self`.

        Raises
        ------
            IndexError
                If provided an invalid :param:`axis` position.
        """
        ...

    def flip(self, *, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> CompatArray[TT, DT]:
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
            CompatArray
                An output :class:`CompatArray` array having the same data type and shape as `self` and whose elements, relative to `self`, are reordered.
        """
        ...

    def moveaxis(
        self,
        source: Union[int, Tuple[int, ...]],
        destination: Union[int, Tuple[int, ...]],
        /
    ) -> CompatArray[TT, DT]:
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
            CompatArray
                A :class:`CompatArray` array containing reordered axes.
                The returned array must have the same data type as `self`.
        """
        ...

    def permute_dims(self, axes: Tuple[int, ...]) -> CompatArray[TT, DT]:
        """
        Permutes the axes (dimensions) of `self`.

        Parameters
        ----------
            axes : Tuple[int, ...]
                Tuple containing a permutation of `(0, 1, ..., N-1)` where `N` is the number of axes (dimensions) of `self`.

        Returns
        -------
            CompatArray
                A :class:`CompatArray` array containing the axes permutation.
                The returned array must have the same data type as `self`.
        """
        ...

    def repeat(
        self,
        repeats: Union[int, ArrayLike[Any]],
        /, *,
        axis: Optional[int] = None
    ) -> CompatArray[TT, DT]:
        """
        Repeats each element of `self` a specified number of times on a per-element basis.

        Parameters
        ----------
            repeats : Union[int, ArrayLike[Any]]
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
            CompatArray
                An output :class:`CompatArray` array containing repeated elements.
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
    ) -> CompatArray[TT, DT]:
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
            CompatArray
                A :class:`CompatArray` output array having the same data type and elements as `self`.

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
    ) -> CompatArray[TT, DT]:
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
            CompatArray
                An output :class:`CompatArray` array having the same data type and shape as `self` and whose elements, relative to `self`, are shifted.
        """
        ...

    def squeeze(self, axis: Union[int, Tuple[int, ...]]) -> CompatArray[TT, DT]:
        """
        Removes singleton dimensions (axes) from `self`.

        Parameters
        ----------
            axis : Union[int, Tuple[int, ...]]
                Axis (or axes) to squeeze.

        Returns
        -------
            CompatArray
                An output :class:`CompatArray` array having the same data type and elements as `self`.

        Raises
        ------
            ValueError
                If a specified axis has a size greater than one (i.e., it is not a singleton dimension).
        """
        ...

    def tile(self, repetitions: Tuple[int, ...], /) -> CompatArray[TT, DT]:
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
            CompatArray
                A tiled output :class:`CompatArray` array.
                The returned array must have the same data type as `self` and must have a rank (i.e., number of dimensions) equal to `max(N, M)`.
                If `S` is the shape of the tiled array after prepending singleton dimensions (if necessary) and `r` is the tuple of repetitions after prepending ones (if necessary), then the number of elements along each axis (dimension) must satisfy `S[i]*r[i]`, where `i` refers to the `i` th axis (dimension).
        """
        ...

    def unstack(self, *, axis: int = 0) -> Tuple[CompatArray[TT, DT], ...]: ...

    # === Searching functions ===
    def argmax(
        self, *,
        axis: Optional[int] = None,
        keepdims: bool = False
    ) -> CompatArray[int, DT]:
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
            CompatArray
                The returned :class:`CompatArray` array must have be the default array index data type.
                For :param:`axis`:
                - `None`: A zero-dimensional array containing the index of the first occurrence of the maximum value;
                - _others_: A non-zero-dimensional array containing the indices of the maximum values.
        """
        ...

    def argmin(
        self, *,
        axis: Optional[int] = None,
        keepdims: bool = False
    ) -> CompatArray[int, DT]:
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
            CompatArray
                The returned :class:`CompatArray` array must have be the default array index data type.
                For :param:`axis`:
                - `None`: A zero-dimensional array containing the index of the first occurrence of the minimum value;
                - _others_: A non-zero-dimensional array containing the indices of the minimum values.
        """
        ...

    def nonzero(self) -> Tuple[CompatArray[int, DT], ...]: ...

    def count_nonzero(
        self, *,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False
    ) -> CompatArray[int, DT]:
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
            CompatArray
                The returned :class:`CompatArray` array must have the default array index data type.
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
        other: ArrayLike[Any],
        /, *,
        side: Literal['left', 'right'] = "left",
        sorter: Optional[ArrayLike[Any]] = None
    ) -> CompatArray[int, DT]:
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
            CompatArray
                A :class:`CompatArray` array of indices with the same shape as `other`.
                The returned array must have the default array index data type.

        Notes
        -----
        - For real-valued floating-point arrays, the sort order of NaNs and signed zeros is unspecified and thus implementation-dependent. Accordingly, when a real-valued floating-point array contains NaNs and signed zeros, what constitutes ascending order may vary among specification-conforming array libraries.
        - While behavior for arrays containing NaNs and signed zeros is implementation-dependent, specification-conforming libraries should, however, ensure consistency with :meth:`CompatArray.sort` and :meth:`CompatArray.argsort` (i.e., if a value in `other` is inserted into `self` according to the corresponding index in the output array and sort is invoked on the resultant array, the sorted result should be an array in the same order).
        """
        ...

    def where(self, X: ArrayOrAny, Y: ArrayOrAny, /) -> CompatArray[Any, DT]:
        """
        Returns elements chosen from `X` or `Y` depending on condition `self`.
        - `self` is a condition array. Should have a boolean data type. Must be compatible with `X` and `Y`. For each element in `self`:
            - `True`: yield `X[i]`;
            - `False`: yield `Y[i]`.

        Parameters
        ----------
            X : ArrayOrAny
                First input array. Must be compatible with `self` and `Y`.

            Y : ArrayOrAny
                Second input array. Must be compatible with `self` and `X`.

        Returns
        -------
            CompatArray
                A :class:`CompatArray` array with elements from `X` where `self` is `True`, and elements from `Y` elsewhere.
                The returned array must have a data type determined by Type Promotion Rules rules with the arrays `X` and `Y`.

        Notes
        - At least one of `X` and `Y` must be an array;
        - If either `X` or `Y` is a scalar value, the returned array must have a data type determined according to Mixing arrays with Python scalars.
        """
        ...

    # === Set functions ===
    def unique_all(self) -> UniqueAllResult[TT, DT]: ...
    def unique_counts(self) -> UniqueCountsResult[TT, DT]: ...
    def unique_inverse(self) -> UniqueInverseResult[TT, DT]: ...

    def unique_values(self) -> CompatArray[TT, DT]:
        """
        Returns the unique elements of an input array `self`.
        - `self`:
            - more than one dimension: the function must flatten `self` and return the unique elements of the flattened `self`.

        Returns
        -------
            CompatArray
                A one-dimensional :class:`CompatArray` array containing the set of unique elements in `self`.
                The returned array must have the same data type as `self`.
        """
        ...

    # === Sorting functions ===
    def argsort(
        self, *,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True
    ) -> CompatArray[int, DT]:
        """
        Returns the indices that sort `self` along a specified axis.

        Parameters
        ----------
            axis : int, default to `-1`
                Axis along which to sort.
                - `-1`: The function must sort along the last axis.

            descending : bool, default to `False`
                Sort order.
                - `True`: The returned indices sort `self` in descending order (by value);
                - `False`: The returned indices sort `self` in ascending order (by value).

            stable : bool, default to `True`
                Sort stability.
                - `True`: The returned indices must maintain the relative order of `self` values which compare as equal;
                - `False`: The returned indices may or may not maintain the relative order of `self` values which compare as equal (i.e., the relative order of `self` values which compare as equal is implementation-dependent).

        Returns
        -------
            CompatArray
                A :class:`CompatArray` array of indices. The returned array must have the same shape as `self`.
                The returned array must have the default array index data type.
        """
        ...

    def sort(
        self, *,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True
    ) -> CompatArray[TT, DT]:
        """
        Returns a sorted copy of an input array `self`.

        Parameters
        ----------
            axis : int, default to `-1`
                Axis along which to sort.
                - `-1`: The function must sort along the last axis.

            descending : bool, default to `False`
                Sort order.
                - `True`: The returned array must be sorted in descending order (by value);
                - `False`: The returned array must be sorted in ascending order (by value).

            stable : bool, default to `True`
                Sort stability.
                - `True`: The returned array must maintain the relative order of `self` values which compare as equal;
                - `False`: The returned array may or may not maintain the relative order of `self` values which compare as equal (i.e., the relative order of `self` values which compare as equal is implementation-dependent).

        Returns
        -------
            CompatArray
                A sorted :class:`CompatArray` array. The returned array must have the same data type and shape as `self`.
        """
        ...

    # === Statistical functions ===
    @overload
    def cumulative_sum(self, *, axis: Optional[int] = ..., dtype: None = ..., include_initial: bool = ...) -> CompatArray[TT, DT]: ...
    @overload
    def cumulative_sum(self, *, axis: Optional[int] = ..., dtype: DTypeT, include_initial: bool = ...) -> CompatArray[DTypeT, DT]: ...

    def cumulative_sum(
        self, *,
        axis: Optional[int] = None,
        dtype: Optional[DTypeT] = None,
        include_initial: bool = False
    ) -> CompatArray[TT, DT]:
        """
        Calculates the cumulative sum of elements in `self`.
        - `self` should have one or more dimensions (axes). Should have a numeric data type.

        Parameters
        ----------
            axis : Optional[int], default to `None`
                Axis along which a cumulative sum must be computed.
                - _negative_: The function must determine the axis along which to compute a cumulative sum by counting from the last dimension.

                For dimension of `self`:
                - `1`: Providing an :param:`axis` is optional;
                - `> 1`: Providing an :param:`axis` is required.

            dtype : Optional[dtype], default to `None`
                Data type of the returned array.
                - `None`: The returned array must have the same data type as `self`, unless `self` has an integer data type supporting a smaller range of values than the default integer data type (e.g., `self` has an `int16` or `uint32` data type and the default integer data type is `int64`). In those latter cases:

                    - `self` has a signed integer data type (e.g., `int16`): The returned array must have the default integer data type;
                    - `self` has an unsigned integer data type (e.g., `uint16`): The returned array must have an unsigned integer data type having the same number of bits as the default integer data type (e.g., if the default integer data type is `int32`, the returned array must have a `uint32` data type).
                - `data type` differs from `self`: The input array should be cast to the specified data type before computing the sum (rationale: the dtype keyword argument is intended to help prevent overflows).

            include_initial : bool, default to `False`
                Boolean indicating whether to include the initial value as the first value in the output.
                By convention, the initial value must be the additive identity (i.e., `zero`).

        Returns
        -------
            CompatArray
                A :class:`CompatArray` array containing the cumulative sums.
                The returned array must have a data type as described by the :param:`dtype` above.

                Let `N` be the size of the axis along which to compute the cumulative sum.
                The returned array must have a shape determined according to the following rules:
                - :param:`include_initial` is `True`: The returned array must have the same shape as `self`, except the size of the axis along which to compute the cumulative sum must be `N+1`;
                - :param:`include_initial` is `False`: The returned array must have the same shape as `self`.
        """
        ...

    @overload
    def cumulative_prod(self, *, axis: Optional[int] = ..., dtype: None = ..., include_initial: bool = ...) -> CompatArray[TT, DT]: ...
    @overload
    def cumulative_prod(self, *, axis: Optional[int] = ..., dtype: DTypeT, include_initial: bool = ...) -> CompatArray[DTypeT, DT]: ...

    def cumulative_prod(
        self, *,
        axis: Optional[int] = None,
        dtype: Optional[DTypeT] = None,
        include_initial: bool = False
    ) -> CompatArray[TT, DT]:
        """
        Calculates the cumulative product of elements in `self`.
        - `self` should have one or more dimensions (axes). Should have a numeric data type.

        Parameters
        ----------
            axis : Optional[int], default to `None`
                Axis along which a cumulative product must be computed.
                - _negative_: The function must determine the axis along which to compute a cumulative product by counting from the last dimension.

                For dimension of `self`:
                - `1`: Providing an :param:`axis` is optional;
                - `> 1`: Providing an :param:`axis` is required.

            dtype : Optional[dtype], default to `None`
                Data type of the returned array.
                - `None`: The returned array must have the same data type as `self`, unless `self` has an integer data type supporting a smaller range of values than the default integer data type (e.g., `self` has an `int16` or `uint32` data type and the default integer data type is `int64`). In those latter cases:

                    - `self` has a signed integer data type (e.g., `int16`): The returned array must have the default integer data type;
                    - `self` has an unsigned integer data type (e.g., `uint16`): The returned array must have an unsigned integer data type having the same number of bits as the default integer data type (e.g., if the default integer data type is `int32`, the returned array must have a `uint32` data type).
                - `data type` differs from `self`: The input array should be cast to the specified data type before computing the product (rationale: the dtype keyword argument is intended to help prevent overflows).

            include_initial : bool, default to `False`
                Boolean indicating whether to include the initial value as the first value in the output.
                By convention, the initial value must be the multiplicative identity (i.e., `one`).

        Returns
        -------
            CompatArray
                A :class:`CompatArray` array containing the cumulative products.
                The returned array must have a data type as described by the :param:`dtype` above.

                Let `N` be the size of the axis along which to compute the cumulative product.
                The returned array must have a shape determined according to the following rules:
                - :param:`include_initial` is `True`: The returned array must have the same shape as `self`, except the size of the axis along which to compute the cumulative product must be `N+1`;
                - :param:`include_initial` is `False`: The returned array must have the same shape as `self`.
        """
        ...

    def max(
        self, *,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False
    ) -> CompatArray[TT, DT]:
        """
        Calculates the maximum value of `self`.

        Parameters
        ----------
            axis : Optional[Union[int, Tuple[int, ...]]], default to `None`
                Axis or axes along which maximum values must be computed.
                - `None`: The maximum value must be computed over the entire `self`;
                - _Tuple[int, ...]_: The maximum value must be computed over multiple axes.

            keepdims : bool, default to `False`
                - `True`: The reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with `self`;
                - `False`: The reduced axes (dimensions) must not be included in the result.

        Returns
        -------
            CompatArray
                The returned :class:`CompatArray` array must have the same data type as `self`.
                - computed over the entire `self`: a zero-dimensional array containing the maximum value;
                - _others_: a non-zero-dimensional array containing the maximum values.

        Notes
        -----
        - For floating-point operands.
            - `self[any]` is `NaN`: The maximum value is `NaN` (i.e., `NaN` values propagate).
        """
        ...

    def mean(
        self, *,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False
    ) -> CompatArray[TT, DT]:
        """
        Calculates the arithmetic mean of `self`.

        Parameters
        ----------
            axis : Optional[Union[int, Tuple[int, ...]]], default to `None`
                Axis or axes along which arithmetic means must be computed.
                - `None`: The mean must be computed over the entire `self`;
                - _Tuple[int, ...]_: The arithmetic mean must be computed over multiple axes.

            keepdims : bool, default to `False`
                - `True`: The reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with `self`;
                - `False`: The reduced axes (dimensions) must not be included in the result.

        Returns
        -------
            CompatArray
                The returned :class:`CompatArray` array must have the same data type as `self`.
                - computed over the entire `self`: a zero-dimensional array containing the arithmetic mean;
                - _others_: a non-zero-dimensional array containing the arithmetic means.

        Notes
        -----
        - Let `N` equal the number of elements over which to compute the arithmetic mean. For real-valued operands,
            - `N` is `0`: The arithmetic mean is `NaN` (i.e., division by zero results in `NaN`);
            - `self[any]` is `NaN`: The arithmetic mean is `NaN` (i.e., `NaN` values propagate).
        - For complex floating-point operands, real-valued floating-point special cases should independently apply to the real and imaginary component operations involving real numbers. For example, let `a = real(x_i) and b = imag(x_i)`, and
            - `N` is `0`: The arithmetic mean is `NaN + NaN j` (i.e., division by zero results in `NaN` for both the real and imaginary components);
            - `a` is `NaN`: The real component of the arithmetic mean is `NaN` (i.e., division by zero results in `NaN` for the real component);
            - `b` is `NaN`: The imaginary component of the arithmetic mean is `NaN` (i.e., division by zero results in `NaN` for the imaginary component).
        """
        ...

    def min(
        self, *,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False
    ) -> CompatArray[TT, DT]:
        """
        Calculates the minimum value of `self`.

        Parameters
        ----------
            axis : Optional[Union[int, Tuple[int, ...]]], default to `None`
                Axis or axes along which minimum values must be computed.
                - `None`: The minimum value must be computed over the entire `self`;
                - _Tuple[int, ...]_: The minimum value must be computed over multiple axes.

            keepdims : bool, default to `False`
                - `True`: The reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with `self`;
                - `False`: The reduced axes (dimensions) must not be included in the result.

        Returns
        -------
            CompatArray
                The returned :class:`CompatArray` array must have the same data type as `self`.
                - computed over the entire `self`: a zero-dimensional array containing the minimum value;
                - _others_: a non-zero-dimensional array containing the minimum values.

        Notes
        -----
        - For floating-point operands.
            - `self[any]` is `NaN`: The minimum value is `NaN` (i.e., `NaN` values propagate).
        """
        ...

    @overload
    def prod(self, *, axis: Optional[Union[int, Tuple[int, ...]]] = ..., dtype: None = ..., keepdims: bool = ...) -> CompatArray[TT, DT]: ...
    @overload
    def prod(self, *, axis: Optional[Union[int, Tuple[int, ...]]] = ..., dtype: DTypeT, keepdims: bool = ...) -> CompatArray[DTypeT, DT]: ...

    def prod(
        self, *,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        dtype: Optional[DTypeT] = None,
        keepdims: bool = False
    ) -> CompatArray[TT, DT]:
        """
        Calculates the product of `self` elements.

        Parameters
        ----------
            axis : Optional[Union[int, Tuple[int, ...]]], default to `None`
                Axis or axes along which products must be computed.
                - `None`: The product must be computed over the entire `self`;
                - _Tuple[int, ...]_: The product must be computed over multiple axes.

            dtype : Optional[dtype], default to `None`
                Data type of the returned array.
                - `None`: The returned array must have the same data type as `self`, unless `self` has an integer data type supporting a smaller range of values than the default integer data type (e.g., `self` has an `int16` or `uint32` data type and the default integer data type is `int64`). In those latter cases:

                    - `self` has a signed integer data type (e.g., `int16`): The returned array must have the default integer data type;
                    - `self` has an unsigned integer data type (e.g., `uint16`): The returned array must have an unsigned integer data type having the same number of bits as the default integer data type (e.g., if the default integer data type is `int32`, the returned array must have a `uint32` data type).
                - `data type` differs from `self`: The input array should be cast to the specified data type before computing the product (rationale: the dtype keyword argument is intended to help prevent overflows).

            keepdims : bool, default to `False`
                - `True`: The reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with `self`;
                - `False`: The reduced axes (dimensions) must not be included in the result.

        Returns
        -------
            CompatArray
                The returned :class:`CompatArray` array must have a data type as described by the :param:`dtype` above.
                - computed over the entire `self`: a zero-dimensional array containing the product;
                - _others_: a non-zero-dimensional array containing the products.

        Notes
        -----
        - Let `N` equal the number of elements over which to compute the product.
            - `N` is `0`: The product is `1` (i.e., the empty product).
        """
        ...

    def std(
        self, *,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        correction: Union[int, float] = 0,
        keepdims: bool = False
    ) -> CompatArray[TT, DT]:
        """
        Calculates the standard deviation of `self`.

        Parameters
        ----------
            axis : Optional[Union[int, Tuple[int, ...]]], default to `None`
                Axis or axes along which standard deviations must be computed.
                - `None`: The standard deviation must be computed over the entire `self`;
                - _Tuple[int, ...]_: The standard deviation must be computed over multiple axes.

            correction : Union[int, float], default to `0`
                Degrees of freedom adjustment.
                Setting this parameter to a value other than `0` has the effect of adjusting the divisor during the calculation of the standard deviation according to `N-c` where `N` corresponds to the total number of elements over which the standard deviation is computed and `c` corresponds to the provided degrees of freedom adjustment.
                When computing the standard deviation of a population, setting this parameter to `0` is the standard choice (i.e., the provided array contains data constituting an entire population).
                When computing the corrected sample standard deviation, setting this parameter to `1` is the standard choice (i.e., the provided array contains data sampled from a larger population; this is commonly referred to as Bessel’s correction).

            keepdims : bool, default to `False`
                - `True`: The reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with `self`;
                - `False`: The reduced axes (dimensions) must not be included in the result.

        Returns
        -------
            CompatArray
                The returned :class:`CompatArray` array must have the same data type as `self`.
                - computed over the entire `self`: a zero-dimensional array containing the standard deviation;
                - _others_: a non-zero-dimensional array containing the standard deviations.

        Notes
        -----
        - Let `N` equal the number of elements over which to compute the standard deviation.
            - `N - correction <= 0`: The standard deviation is `NaN`;
            - `self[any]` is `NaN`: The standard deviation is `NaN` (i.e., `NaN` values propagate).
        """
        ...

    @overload
    def sum(self, *, axis: Optional[Union[int, Tuple[int, ...]]] = ..., dtype: None = ..., keepdims: bool = ...) -> CompatArray[TT, DT]: ...
    @overload
    def sum(self, *, axis: Optional[Union[int, Tuple[int, ...]]] = ..., dtype: DTypeT, keepdims: bool = ...) -> CompatArray[DTypeT, DT]: ...

    def sum(
        self, *,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        dtype: Optional[DTypeT] = None,
        keepdims: bool = False
    ) -> CompatArray[TT, DT]:
        """
        Calculates the sum of the `self`.

        Parameters
        ----------
            axis : Optional[Union[int, Tuple[int, ...]]], default to `None`
                Axis or axes along which sums must be computed.
                - `None`: The sum must be computed over the entire `self`;
                - _Tuple[int, ...]_: The sum must be computed over multiple axes.

            dtype : Optional[dtype], default to `None`
                Data type of the returned array.
                - `None`: The returned array must have the same data type as `self`, unless `self` has an integer data type supporting a smaller range of values than the default integer data type (e.g., `self` has an `int16` or `uint32` data type and the default integer data type is `int64`). In those latter cases:

                    - `self` has a signed integer data type (e.g., `int16`): The returned array must have the default integer data type;
                    - `self` has an unsigned integer data type (e.g., `uint16`): The returned array must have an unsigned integer data type having the same number of bits as the default integer data type (e.g., if the default integer data type is `int32`, the returned array must have a `uint32` data type).
                - `data type` differs from `self`: The input array should be cast to the specified data type before computing the sum (rationale: the dtype keyword argument is intended to help prevent overflows).

            keepdims : bool, default to `False`
                - `True`: The reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with `self`;
                - `False`: The reduced axes (dimensions) must not be included in the result.

        Returns
        -------
            CompatArray
                The returned :class:`CompatArray` array must have a data type as described by the :param:`dtype` above.
                - computed over the entire `self`: a zero-dimensional array containing the sum;
                - _others_: a non-zero-dimensional array containing the sums.

        Notes
        -----
        - Let `N` equal the number of elements over which to compute the sum.
            - `N` is `0`: The sum is `0` (i.e., the empty sum).
        """
        ...

    def var(
        self, *,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        correction: Union[int, float] = 0,
        keepdims: bool = False
    ) -> CompatArray[TT, DT]:
        """
        Calculates the variance of `self`.

        Parameters
        ----------
            axis : Optional[Union[int, Tuple[int, ...]]], default to `None`
                Axis or axes along which variances must be computed.
                - `None`: The variance must be computed over the entire `self`;
                - _Tuple[int, ...]_: The variance must be computed over multiple axes.

            correction : Union[int, float], default to `0`
                Degrees of freedom adjustment.
                Setting this parameter to a value other than `0` has the effect of adjusting the divisor during the calculation of the variance according to `N-c` where `N` corresponds to the total number of elements over which the variance is computed and `c` corresponds to the provided degrees of freedom adjustment.
                When computing the variance of a population, setting this parameter to `0` is the standard choice (i.e., the provided array contains data constituting an entire population).
                When computing the unbiased sample variance, setting this parameter to `1` is the standard choice (i.e., the provided array contains data sampled from a larger population; this is commonly referred to as Bessel’s correction).

            keepdims : bool, default to `False`
                - `True`: The reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with `self`;
                - `False`: The reduced axes (dimensions) must not be included in the result.

        Returns
        -------
            CompatArray
                The returned :class:`CompatArray` array must have the same data type as `self`.
                - computed over the entire `self`: a zero-dimensional array containing the variance;
                - _others_: a non-zero-dimensional array containing the variances.

        Notes
        -----
        - Let `N` equal the number of elements over which to compute the variance.
            - `N - correction <= 0`: The variance is `NaN` (i.e., division by zero results in `NaN`);
            - `self[any]` is `NaN`: The variance is `NaN` (i.e., `NaN` values propagate).
        """
        ...

    # === Utility functions ===
    def all(
        self, *,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False
    ) -> CompatArray[bool, DT]:
        """
        Tests whether all `self` elements evaluate to `True` along a specified axis.
        - `Positive infinity`, `negative infinity`, and `NaN` must evaluate to `True`;
        - If `self` is an empty array or the size of the axis (dimension) along which to evaluate elements is `zero`, the test result must be `True`.

        Parameters
        ----------
            axis : Optional[Union[int, Tuple[int, ...]]], default to `None`
                Axis or axes along which to perform a `logical AND` reduction.
                - `None`: A `logical AND` reduction must be performed over the entire array.
                - _Tuple[int, ...]_: `logical AND` reductions must be performed over multiple axes.
                A valid :param:`axis` must be an integer on the interval `[-N, N)`, where `N` is the rank (number of dimensions) of `self`. For each axis:
                - _negative_integer_: The function must determine the axis along which to perform a reduction by counting backward from the last dimension (where `-1` refers to the last dimension);
                - _invalid_: The function must raise an exception.

            keepdims : bool, default to `False`
                - `True`: The reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with `self`;
                - `False`: The reduced axes (dimensions) must not be included in the result.

        Returns
        -------
            CompatArray
                The returned :class:`CompatArray` array must have a data type of `bool`.
                - computed over the entire `self`: a zero-dimensional array containing the test result;
                - _others_: a non-zero-dimensional array containing the test results.
        """
        ...

    def any(
        self, *,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False
    ) -> CompatArray[bool, DT]:
        """
        Tests whether any `self` elements evaluate to `True` along a specified axis.
        - `Positive infinity`, `negative infinity`, and `NaN` must evaluate to `True`;
        - If `self` is an empty array or the size of the axis (dimension) along which to evaluate elements is `zero`, the test result must be `False`.

        Parameters
        ----------
            axis : Optional[Union[int, Tuple[int, ...]]], default to `None`
                Axis or axes along which to perform a `logical OR` reduction.
                - `None`: A `logical OR` reduction must be performed over the entire array.
                - _Tuple[int, ...]_: `logical OR` reductions must be performed over multiple axes.
                A valid :param:`axis` must be an integer on the interval `[-N, N)`, where `N` is the rank (number of dimensions) of `self`. For each axis:
                - _negative_integer_: The function must determine the axis along which to perform a reduction by counting backward from the last dimension (where `-1` refers to the last dimension);
                - _invalid_: The function must raise an exception.

            keepdims : bool, default to `False`
                - `True`: The reduced axes (dimensions) must be included in the result as singleton dimensions, and, accordingly, the result must be compatible with `self`;
                - `False`: The reduced axes (dimensions) must not be included in the result.

        Returns
        -------
            CompatArray
                The returned :class:`CompatArray` array must have a data type of `bool`.
                - computed over the entire `self`: a zero-dimensional array containing the test result;
                - _others_: a non-zero-dimensional array containing the test results.
        """
        ...

    def diff(
        self, *,
        axis: int = -1,
        n: int = 1,
        prepend: Optional[ArrayLike[Any]] = None,
        append: Optional[ArrayLike[Any]] = None
    ) -> CompatArray[TT, DT]:
        """
        Calculates the n-th discrete forward difference along a specified axis.

        Parameters
        ----------
            axis : Optional[Union[int, Tuple[int, ...]]], default to `None`
                Axis or axes along which to compute differences.
                A valid :param:`axis` must be an integer on the interval `[-N, N)`, where `N` is the rank (number of dimensions) of `self`. For each axis:
                - _negative_integer_: The function must determine the axis along which to perform a reduction by counting backward from the last dimension (where `-1` refers to the last dimension);
                - _invalid_: The function must raise an exception.

            n : int, default to `1`
                Number of times to recursively compute differences.

            prepend : Optional[array], default to `None`
                Values to prepend to a specified axis prior to computing differences.
                Must have the same shape as `self`, except for the axis specified by :param:`axis` which may have any size. Should have the same data type as `self`.

            append : Optional[array], default to `None`
                Values to append to a specified axis prior to computing differences.
                Must have the same shape as `self`, except for the axis specified by :param:`axis` which may have any size. Should have the same data type as `self`.

        Returns
        -------
            CompatArray
                A :class:`CompatArray` array containing the `n`-th differences.
                Should have the same data type as `self`.
                Must have the same shape as `self`, except for the axis specified by :param:`axis` which must have a size determined as follows:
                - Let M be the number of elements along an axis specified by :param:`axis`.
                - Let N1 be the number of prepended values along an axis specified by :param:`axis`.
                - Let N2 be the number of appended values along an axis specified by :param:`axis`.

                The final size of the axis specified by :param:`axis` must be `M + N1 + N2 - n`.

        Notes
        -----
        - The first-order differences are given by `out[i] = self[i+1] - self[i]` along a specified axis.
        Higher-order differences must be calculated recursively (e.g., by calling `diff(out, axis=axis, n=n-1)`).
        - If a conforming implementation chooses to support :param:`prepend` and :param:`append` arrays which have a different data type than `self`, behavior is unspecified and thus implementation-defined.
        Implementations may choose to type promote (Type Promotion Rules), cast :param:`prepend` and/or :param:`append` to the same data type as `self`, or raise an exception.
        """
        ...

    # === Others ===
    def copy(self) -> CompatArray[TT, DT]: ...

    # === Array attributes ===
    @property
    def arr(self) -> ArrayLike[TT]: ...
    @property
    def dtype(self) -> TT: ...
    @property
    def device(self) -> DT: ...
    @property
    def shape(self) -> Tuple[int, ...]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def size(self) -> Optional[int]: ...
    @property
    def T(self) -> CompatArray[TT, DT]: ...
    @property
    def mT(self) -> CompatArray[TT, DT]: ...

    def __array__(self) -> NDArray[TT]: ...

    def __len__(self) -> int: ...
    def __abs__(self) -> CompatArray[TT, DT]: ...
    def __add__(self, other: ArrayOrScalar, /) -> CompatArray[Any, DT]: ...
    def __and__(self, other: ArrayOrIntLike, /) -> CompatArray[TT, DT]: ...
    def __bool__(self) -> bool: ...
    def __complex__(self) -> complex: ...
    def __eq__(self, other: ArrayOrAny, /) -> CompatArray[bool, DT]: ...
    def __float__(self) -> float: ...
    def __floordiv__(self, other: ArrayOrReal, /) -> CompatArray[TT, DT]: ...
    def __ge__(self, other: ArrayOrReal, /) -> CompatArray[bool, DT]: ...
    def __getitem__(self, key: Any, /) -> CompatArray[TT, DT]: ...
    def __gt__(self, other: ArrayOrReal, /) -> CompatArray[bool, DT]: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __invert__(self) -> CompatArray[TT, DT]: ...
    def __le__(self, other: ArrayOrReal, /) -> CompatArray[bool, DT]: ...
    def __lshift__(self, other: ArrayOrInt, /) -> CompatArray[TT, DT]: ...
    def __lt__(self, other: ArrayOrReal, /) -> CompatArray[bool, DT]: ...
    def __matmul__(self, other: ArrayLike[Any], /) -> CompatArray[Any, DT]: ...
    def __mod__(self, other: ArrayOrReal, /) -> CompatArray[Any, DT]: ...
    def __mul__(self, other: ArrayOrScalar, /) -> CompatArray[Any, DT]: ...
    def __ne__(self, other: ArrayOrAny, /) -> CompatArray[bool, DT]: ...
    def __neg__(self) -> CompatArray[TT, DT]: ...
    def __or__(self, other: ArrayOrIntLike, /) -> CompatArray[TT, DT]: ...
    def __pos__(self) -> CompatArray[TT, DT]: ...
    def __pow__(self, other: ArrayOrScalar, /) -> CompatArray[Any, DT]: ...
    def __rshift__(self, other: ArrayOrInt, /) -> CompatArray[TT, DT]: ...
    def __setitem__(self, key: Any, value: Any, /): ...
    def __sub__(self, other: ArrayOrScalar, /) -> CompatArray[Any, DT]: ...
    def __truediv__(self, other: ArrayOrScalar, /) -> CompatArray[float, DT]: ...
    def __xor__(self, other: ArrayOrIntLike, /) -> CompatArray[TT, DT]: ...


@overload
def unwrap(obj: CompatArray[dtypeT, Any]) -> ArrayLike[dtypeT]: ...
@overload
def unwrap(obj: ArrayLike[dtypeT]) -> ArrayLike[dtypeT]: ...
@overload
def unwrap(obj: T) -> T: ...


def unwrap(obj: Any) -> Any: ...


@overload
def wrap_arraylike(arr: NDArray[dtypeT], xp: Optional[object] = ...) -> CompatArray[dtypeT, Literal["cpu"]]: ...
@overload
def wrap_arraylike(arr: CompatArray[dtypeT, deviceT], xp: Optional[object] = ...) -> CompatArray[dtypeT, deviceT]: ...
@overload
def wrap_arraylike(arr: ArrayLike[dtypeT], xp: Optional[object] = ...) -> CompatArray[dtypeT, Any]: ...
@overload
def wrap_arraylike(arr: T, xp: Optional[object] = ...) -> T: ...


def wrap_arraylike(arr: Any, xp: Optional[object] = None) -> Any: ...
