# -*- coding: utf-8 -*-
# Python version: 3.9
# @TianZhen

from __future__ import annotations
from numpy.typing import NDArray
from array_api_compat.common._typing import Namespace
from typing import (Union, List, Tuple, Optional, Any, Literal, overload)

from ._array import CompatArray
from ..types import (
    DTypeT, DeviceT, dtypeT, DType, Device,
    ValueT, Value, ArrayLike
)


class NameSpace(Namespace):
    def __init__(self, xp: Namespace, /): ...

    # === Creation functions ===
    @overload
    def asarray(self, obj: NDArray[dtypeT], /, *, dtype: None = ..., device: Optional[Device] = ..., copy: Optional[bool] = ...) -> CompatArray[dtypeT, Literal["cpu"]]: ...
    @overload
    def asarray(self, obj: ArrayLike[dtypeT], /, *, dtype: None = ..., device: None = ..., copy: Optional[bool] = ...) -> CompatArray[dtypeT, Any]: ...
    @overload
    def asarray(self, obj: object, /, *, dtype: None = ..., device: None = ..., copy: Optional[bool] = ...) -> CompatArray[Any, Any]: ...
    @overload
    def asarray(self, obj: object, /, *, dtype: None = ..., device: DeviceT, copy: Optional[bool] = ...) -> CompatArray[Any, DeviceT]: ...
    @overload
    def asarray(self, obj: NDArray[Any], /, *, dtype: DTypeT, device: Optional[Device] = ..., copy: Optional[bool] = ...) -> CompatArray[DTypeT, Literal["cpu"]]: ...
    @overload
    def asarray(self, obj: object, /, *, dtype: DTypeT, device: None = ..., copy: Optional[bool] = ...) -> CompatArray[DTypeT, Any]: ...
    @overload
    def asarray(self, obj: object, /, *, dtype: DTypeT, device: DeviceT, copy: Optional[bool] = ...) -> CompatArray[DTypeT, DeviceT]: ...

    def asarray(
        self,
        obj: object,
        /, *,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        copy: Optional[bool] = None
    ) -> CompatArray[Any, Any]:
        """
        Convert the input to a `CompatArray` array.

        Parameters
        ----------
            obj : object
                Object to be converted to a `CompatArray` array.
                May be a Python scalar, a (possibly nested) sequence of Python scalars, or an object supporting the Python buffer protocol.

            dtype : Optional[DType], default to `None`
                Output array data type.
                - `None`: The output array data type must be inferred from the data type(s) in obj.
                If all input values are Python scalars, then, in order of precedence:
                1. If all values are of type `bool`, the output data type must be `bool`;
                2. If all values are of type `int` or are a mixture of `bool` and `int`, the output data type must be the default `integer` data type;
                3. If one or more values are `complex` numbers, the output data type must be the default `complex` floating-point data type;
                4. If one or more values are `floats`, the output data type must be the default real-valued floating-point data type.

            device : Optional[Device], default to `None`
                Device on which to place the created array.
                - `None`: If :param:`obj` is an array, the output array device must be inferred from :param:`obj`.

            copy : Optional[bool], default to `None`
                Boolean indicating whether or not to copy the input.
                - `True`: The function must always copy (see Copy keyword argument behavior);
                - `False`: The function must never copy for input which supports the buffer protocol and must raise a ValueError in case a copy would be necessary;
                - `None`: The function must reuse existing memory buffer if possible and copy otherwise.

        Returns
        -------
            CompatArray
                A `CompatArray` array containing the data from :param:`obj`.
        """
        ...

    @overload
    def arange(self, start: int, /, stop: Optional[int] = ..., step: int = ..., *, dtype: None = ..., device: None = ...) -> CompatArray[int, Literal["cpu"]]: ...
    @overload
    def arange(self, start: Union[int, float], /, stop: Optional[Union[int, float]] = ..., step: Union[int, float] = ..., *, dtype: None = ..., device: None = ...) -> CompatArray[float, Literal["cpu"]]: ...
    @overload
    def arange(self, start: int, /, stop: Optional[int] = ..., step: int = ..., *, dtype: None = ..., device: DeviceT) -> CompatArray[int, DeviceT]: ...
    @overload
    def arange(self, start: Union[int, float], /, stop: Optional[Union[int, float]] = ..., step: Union[int, float] = ..., *, dtype: None = ..., device: DeviceT) -> CompatArray[float, DeviceT]: ...
    @overload
    def arange(self, start: Union[int, float], /, stop: Optional[Union[int, float]] = ..., step: Union[int, float] = ..., *, dtype: DTypeT, device: None = ...) -> CompatArray[DTypeT, Literal["cpu"]]: ...
    @overload
    def arange(self, start: Union[int, float], /, stop: Optional[Union[int, float]] = ..., step: Union[int, float] = ..., *, dtype: DTypeT, device: Device) -> CompatArray[DTypeT, Device]: ...

    def arange(
        self,
        start: Union[int, float],
        /,
        stop: Optional[Union[int, float]] = None,
        step: Union[int, float] = 1,
        *,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None
    ) -> CompatArray[Any, Any]:
        """
        Returns evenly spaced values within the half-open interval `[start, stop)` as a one-dimensional `CompatArray` array.

        Parameters
        ----------
            start : Union[int, float]
                - :param:`stop` is specified: the start of interval (inclusive);
                - :param:`stop` is not specified: the end of the interval (exclusive), and the default is `0`.

            stop : Optional[Union[int, float]], default to `None`
                The end of the interval.

            step : Union[int, float], default to `1`
                The distance between two adjacent elements (`out[i+1] - out[i]`).
                Must not be `0`; may be `negative`, this results in an empty array if :param:`stop` >= :param:`start`.

            dtype : Optional[DType], default to `None`
                 Output array data type.
                 - `None`: The output array data type must be inferred from :param:`start`, :param:`stop` and :param:`step`. For :param:`start`, :param:`stop` and :param:`step`:

                    - all integers: the output data type must be the default `integer` data type;
                    - one or more floats: the output data type must be the default real-valued floating-point data type.

            device : Optional[Device], default to `None`
                 Device on which to place the created array.

        Returns
        -------
            CompatArray
                A one-dimensional `CompatArray` array containing evenly spaced values.
                The length of the output array must be `ceil((stop-start)/step)` if `stop - start` and :param:`step` have the same sign, and length `0` otherwise.
        """
        ...

    @overload
    def empty(self, shape: Union[int, Tuple[int, ...]], *, dtype: None = ..., device: None = ...) -> CompatArray[float, Literal["cpu"]]: ...
    @overload
    def empty(self, shape: Union[int, Tuple[int, ...]], *, dtype: None = ..., device: DeviceT) -> CompatArray[float, DeviceT]: ...
    @overload
    def empty(self, shape: Union[int, Tuple[int, ...]], *, dtype: DTypeT, device: None = ...) -> CompatArray[DTypeT, Literal["cpu"]]: ...
    @overload
    def empty(self, shape: Union[int, Tuple[int, ...]], *, dtype: DTypeT, device: DeviceT) -> CompatArray[DTypeT, DeviceT]: ...

    def empty(
        self,
        shape: Union[int, Tuple[int, ...]],
        *,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None
    ) -> CompatArray[Any, Any]:
        """
        Returns an uninitialized `CompatArray` array having a specified shape.

        Parameters
        ----------
            shape : Union[int, Tuple[int, ...]]
                Output array shape.

            dtype : Optional[DType], default to `None`
                 Output array data type.
                 - `None`: The output array data type must be the default real-valued floating-point data type.

            device : Optional[Device], default to `None`
                 Device on which to place the created array.

        Returns
        -------
            CompatArray
                a `CompatArray` array containing uninitialized data.
        """
        ...

    @overload
    def empty_like(self, x: NDArray[dtypeT], /, *, dtype: None = ..., device: Optional[Device] = ...) -> CompatArray[dtypeT, Literal["cpu"]]: ...
    @overload
    def empty_like(self, x: ArrayLike[dtypeT], /, *, dtype: None = ..., device: None = ...) -> CompatArray[dtypeT, Any]: ...
    @overload
    def empty_like(self, x: ArrayLike[dtypeT], /, *, dtype: None = ..., device: DeviceT) -> CompatArray[dtypeT, DeviceT]: ...
    @overload
    def empty_like(self, x: ArrayLike[Any], /, *, dtype: DTypeT, device: None = ...) -> CompatArray[DTypeT, Any]: ...
    @overload
    def empty_like(self, x: ArrayLike[Any], /, *, dtype: DTypeT, device: DeviceT) -> CompatArray[DTypeT, DeviceT]: ...

    def empty_like(
        self,
        x: ArrayLike[Any],
        /, *,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None
    ) -> CompatArray[Any, Any]:
        """
        Returns an uninitialized `CompatArray` array with the same shape as an input array :param:`x`.

        Parameters
        ----------
            x : ArrayLike
                Input array from which to derive the output array shape.

            dtype : Optional[DType], default to `None`
                 Output array data type.
                 - `None`: The output array data type must be inferred from :param:`x`.

            device : Optional[Device], default to `None`
                 Device on which to place the created array.
                 - `None`: The output array device must be inferred from :param:`x`.

        Returns
        -------
            CompatArray
                a `CompatArray` array having the same shape as :param:`x` and containing uninitialized data.
        """
        ...

    @overload
    def eye(self, n_rows: int, n_cols: Optional[int] = ..., /, *, k: int = ..., dtype: None = ..., device: None = ...) -> CompatArray[float, Literal["cpu"]]: ...
    @overload
    def eye(self, n_rows: int, n_cols: Optional[int] = ..., /, *, k: int = ..., dtype: None = ..., device: DeviceT) -> CompatArray[float, DeviceT]: ...
    @overload
    def eye(self, n_rows: int, n_cols: Optional[int] = ..., /, *, k: int = ..., dtype: DTypeT, device: None = ...) -> CompatArray[DTypeT, Literal["cpu"]]: ...
    @overload
    def eye(self, n_rows: int, n_cols: Optional[int] = ..., /, *, k: int = ..., dtype: DTypeT, device: DeviceT) -> CompatArray[DTypeT, DeviceT]: ...

    def eye(
        self,
        n_rows: int,
        n_cols: Optional[int] = None,
        /, *,
        k: int = 0,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None
    ) -> CompatArray[Any, Any]:
        """
        Returns a two-dimensional `CompatArray` array with ones on the :param:`k`th diagonal and zeros elsewhere.

        Parameters
        ----------
            n_rows : int
                Number of rows in the output array.

            n_cols : Optional[int], default to `None`
                Number of columns in the output array.
                - `None`: The default number of columns in the output array is equal to :param:`n_rows`.

            k : int, default to `0`
                Index of the diagonal.
                - _positive_: Upper diagonal;
                - _negative_: Lower diagonal;
                - `0`: Main diagonal.

            dtype : Optional[DType], default to `None`
                 Output array data type.
                 - `None`: The output array data type must be the default real-valued floating-point data type.

            device : Optional[Device], default to `None`
                 Device on which to place the created array.

        Returns
        -------
            CompatArray
                A `CompatArray` array where all elements are equal to zero, except for the kth diagonal, whose values are equal to one.
        """
        ...

    @overload
    def from_dlpack(self, x: NDArray[dtypeT], /, *, device: None = ..., copy: Optional[bool] = ...) -> CompatArray[dtypeT, Literal["cpu"]]: ...
    @overload
    def from_dlpack(self, x: ArrayLike[dtypeT], /, *, device: None = ..., copy: Optional[bool] = ...) -> CompatArray[dtypeT, Any]: ...
    @overload
    def from_dlpack(self, x: object, /, *, device: None = ..., copy: Optional[bool] = ...) -> CompatArray[Any, Any]: ...
    @overload
    def from_dlpack(self, x: ArrayLike[dtypeT], /, *, device: DeviceT, copy: Optional[bool] = ...) -> CompatArray[dtypeT, DeviceT]: ...
    @overload
    def from_dlpack(self, x: object, /, *, device: DeviceT, copy: Optional[bool] = ...) -> CompatArray[Any, DeviceT]: ...

    def from_dlpack(
        self,
        x: object,
        /, *,
        device: Optional[Device] = None,
        copy: Optional[bool] = None
    ) -> CompatArray[Any, Any]:
        """
        Returns a new `CompatArray` array containing the data from another (array) object with a `__dlpack__` method.

        Parameters
        ----------
            x : object
                Input (array) object.

            device : Optional[Device], default to `None`
                Device on which to place the created array.
                - `None`:  If :param:`x` supports `DLPack`, the output array must be on the same device as :param:`x`.

            copy : Optional[bool], default to `None`
                Boolean indicating whether or not to copy the input.
                - `True`: The function must always copy;
                - `False`: The function must never copy, and raise `BufferError` in case a copy is deemed necessary (e.g. if a cross-device data movement is requested, and it is not possible without a copy);
                - `None`: The function must reuse existing memory buffer if possible and copy otherwise.

        Returns
        -------
            CompatArray
                A `CompatArray` array containing the data in :param:`x`.
        """
        ...

    @overload
    def full(self, shape: Union[int, Tuple[int, ...]], fill_value: ValueT, *, dtype: None = ..., device: None = ...) -> CompatArray[ValueT, Literal["cpu"]]: ...
    @overload
    def full(self, shape: Union[int, Tuple[int, ...]], fill_value: ValueT, *, dtype: None = ..., device: DeviceT) -> CompatArray[ValueT, DeviceT]: ...
    @overload
    def full(self, shape: Union[int, Tuple[int, ...]], fill_value: Value, *, dtype: DTypeT, device: None = ...) -> CompatArray[DTypeT, Literal["cpu"]]: ...
    @overload
    def full(self, shape: Union[int, Tuple[int, ...]], fill_value: Value, *, dtype: DTypeT, device: DeviceT) -> CompatArray[DTypeT, DeviceT]: ...

    def full(
        self,
        shape: Union[int, Tuple[int, ...]],
        fill_value: Value,
        *,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None
    ) -> CompatArray[Any, Any]:
        """
        Returns a new `CompatArray` array having a specified :param:`shape` and filled with :param:`fill_value`.

        Parameters
        ----------
            shape : Union[int, Tuple[int, ...]]
                Output array shape.

            fill_value : Value
                Fill value.

            dtype : Optional[DType], default to `None`
                Output array data type.
                - `None`: The output array data type must be inferred from :param:`fill_value` according to the following rules, for :param:`fill_value`:

                    - _int_: The output array data type must be the default `integer` data type;
                    - _float_: The output array data type must be the default real-valued floating-point data type;
                    - _complex_: The output array data type must be the default `complex` floating-point data type;
                    - _bool_: The output array data type must be `bool`.

            device : Optional[Device], default to `None`
                Device on which to place the created array.

        Returns
        -------
            CompatArray
                A `CompatArray` array where every element is equal to :param:`fill_value`.
        """
        ...

    @overload
    def full_like(self, x: NDArray[dtypeT], /, fill_value: Value, *, dtype: None = ..., device: Optional[Device] = ...) -> CompatArray[dtypeT, Literal["cpu"]]: ...
    @overload
    def full_like(self, x: ArrayLike[dtypeT], /, fill_value: Value, *, dtype: None = ..., device: None = ...) -> CompatArray[dtypeT, Any]: ...
    @overload
    def full_like(self, x: ArrayLike[dtypeT], /, fill_value: Value, *, dtype: None = ..., device: DeviceT) -> CompatArray[dtypeT, DeviceT]: ...
    @overload
    def full_like(self, x: ArrayLike[Any], /, fill_value: Value, *, dtype: DTypeT, device: None = ...) -> CompatArray[DTypeT, Any]: ...
    @overload
    def full_like(self, x: ArrayLike[Any], /, fill_value: Value, *, dtype: DTypeT, device: DeviceT) -> CompatArray[DTypeT, DeviceT]: ...

    def full_like(
        self,
        x: ArrayLike[Any],
        /,
        fill_value: Value,
        *,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None
    ) -> CompatArray[Any, Any]:
        """
        Returns a `CompatArray` array with the same shape as an input array :param:`x` and filled with :param:`fill_value`.

        Parameters
        ----------
            x : ArrayLike
                Input array from which to derive the output array shape.

            fill_value : Value
                Fill value.

            dtype : Optional[DType], default to `None`
                 Output array data type.
                 - `None`: The output array data type must be inferred from :param:`x`.

            device : Optional[Device], default to `None`
                 Device on which to place the created array.
                 - `None`: The output array device must be inferred from :param:`x`.

        Returns
        -------
            CompatArray
                a `CompatArray` array having the same shape as :param:`x` and where every element is equal to :param:`fill_value`.
        """
        ...

    @overload
    def linspace(self, start: Union[int, float, complex], stop: Union[int, float, complex], /, num: int, *, dtype: None = ..., device: None = ..., endpoint: bool = ...) -> CompatArray[float, Literal["cpu"]]: ...
    @overload
    def linspace(self, start: Union[int, float, complex], stop: Union[int, float, complex], /, num: int, *, dtype: None = ..., device: DeviceT, endpoint: bool = ...) -> CompatArray[float, DeviceT]: ...
    @overload
    def linspace(self, start: Union[int, float, complex], stop: Union[int, float, complex], /, num: int, *, dtype: DTypeT, device: None = ..., endpoint: bool = ...) -> CompatArray[DTypeT, Literal["cpu"]]: ...
    @overload
    def linspace(self, start: Union[int, float, complex], stop: Union[int, float, complex], /, num: int, *, dtype: DTypeT, device: DeviceT, endpoint: bool = ...) -> CompatArray[DTypeT, DeviceT]: ...

    def linspace(
        self,
        start: Union[int, float, complex],
        stop: Union[int, float, complex],
        /,
        num: int,
        *,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        endpoint: bool = True
    ) -> CompatArray[Any, Any]:
        """
        Returns evenly spaced numbers over a specified interval.

        Parameters
        ----------
            start : Union[int, float, complex]
                The start of the interval.

            stop : Union[int, float, complex]
                The end of the interval.
                - :param:`endpoint` is `False`: The function must generate a sequence of `num+1` evenly spaced numbers starting with :param:`start` and ending with :param:`stop` and exclude the :param:`stop` from the returned array such that the returned array consists of evenly spaced numbers over the half-open interval `[start, stop)`;
                - :param:`endpoint` is `True`: The output array must consist of evenly spaced numbers over the closed interval `[start, stop]`.

                NOTE: The step size changes when endpoint is False.

            num : int
                Number of samples.
                Must be a nonnegative `integer` value.

            dtype : Optional[DType], default to `None`
                Output array data type. Should be a floating-point data type.
                - `None`: For :param:`start` and :param:`stop`:

                    - either one or both are `complex` numbers: The output data type must be the default `complex` floating-point data type;
                    - both are real-valued: The output data type must be the default real-valued floating-point data type.

            device : Optional[Device], default to `None`
                The device on which to place the output array.

            endpoint : bool, default to `True`
                Boolean indicating whether to include :param:`stop` in the interval.

        Returns
        -------
            CompatArray
                A one-dimensional `CompatArray` array containing evenly spaced values.
        """
        ...

    def meshgrid(self, *arrays: ArrayLike[Any], indexing: Literal["xy", "ij"] = "xy") -> List[CompatArray[Any, Any]]:
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
        ...

    @overload
    def ones(self, shape: Union[int, Tuple[int, ...]], *, dtype: None = ..., device: None = ...) -> CompatArray[float, Literal["cpu"]]: ...
    @overload
    def ones(self, shape: Union[int, Tuple[int, ...]], *, dtype: None = ..., device: DeviceT) -> CompatArray[float, DeviceT]: ...
    @overload
    def ones(self, shape: Union[int, Tuple[int, ...]], *, dtype: DTypeT, device: None = ...) -> CompatArray[DTypeT, Literal["cpu"]]: ...
    @overload
    def ones(self, shape: Union[int, Tuple[int, ...]], *, dtype: DTypeT, device: DeviceT) -> CompatArray[DTypeT, DeviceT]: ...

    def ones(
        self,
        shape: Union[int, Tuple[int, ...]],
        *,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None
    ) -> CompatArray[Any, Any]:
        """
        Returns a `CompatArray` array having a specified shape and filled with ones.

        Parameters
        ----------
            shape : Union[int, Tuple[int, ...]]
                Output array shape.

            dtype : Optional[DType], default to `None`
                 Output array data type.
                 - `None`: The output array data type must be the default real-valued floating-point data type.

            device : Optional[Device], default to `None`
                 Device on which to place the created array.

        Returns
        -------
            CompatArray
                a `CompatArray` array containing ones.
        """
        ...

    @overload
    def ones_like(self, x: NDArray[dtypeT], /, *, dtype: None = ..., device: Optional[Device] = ...) -> CompatArray[dtypeT, Literal["cpu"]]: ...
    @overload
    def ones_like(self, x: ArrayLike[dtypeT], /, *, dtype: None = ..., device: None = ...) -> CompatArray[dtypeT, Any]: ...
    @overload
    def ones_like(self, x: ArrayLike[dtypeT], /, *, dtype: None = ..., device: DeviceT) -> CompatArray[dtypeT, DeviceT]: ...
    @overload
    def ones_like(self, x: ArrayLike[Any], /, *, dtype: DTypeT, device: None = ...) -> CompatArray[DTypeT, Any]: ...
    @overload
    def ones_like(self, x: ArrayLike[Any], /, *, dtype: DTypeT, device: DeviceT) -> CompatArray[DTypeT, DeviceT]: ...

    def ones_like(
        self,
        x: ArrayLike[Any],
        /, *,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None
    ) -> CompatArray[Any, Any]:
        """
        Returns a `CompatArray` array filled with ones with the same shape as an input array :param:`x`.

        Parameters
        ----------
            x : ArrayLike
                Input array from which to derive the output array shape.

            dtype : Optional[DType], default to `None`
                 Output array data type.
                 - `None`: The output array data type must be inferred from :param:`x`.

            device : Optional[Device], default to `None`
                 Device on which to place the created array.
                 - `None`: The output array device must be inferred from :param:`x`.

        Returns
        -------
            CompatArray
                a `CompatArray` array having the same shape as :param:`x` and containing ones.
        """
        ...

    @overload
    def tril(self, x: NDArray[dtypeT], /, *, k: int = 0) -> CompatArray[dtypeT, Literal["cpu"]]: ...
    @overload
    def tril(self, x: ArrayLike[dtypeT], /, *, k: int = 0) -> CompatArray[dtypeT, Any]: ...

    def tril(self, x: ArrayLike[Any], /, *, k: int = 0) -> CompatArray[Any, Any]:
        """
        Returns the lower triangular part of a matrix (or a stack of matrices) :param:`x`.

        Parameters
        ----------
            x : ArrayLike
                Input array. Must have a shape of `(..., M, N)`, and whose innermost two dimensions form `MxN` matrices.

            k : int, default to `0`
                Diagonal above which to zero elements.
                - `k = 0`: Main diagonal;
                - `k < 0`: The diagonal is below the main diagonal;
                - `k > 0`: The diagonal is above the main diagonal.

        Returns
        -------
            CompatArray
                a `CompatArray` array containing the lower triangular part(s).
                The returned array must have the same shape and data type as :param:`x`.
                All elements above the specified diagonal :param:`k` must be zeroed.
                The returned array should be allocated on the same device as :param:`x`.
        """
        ...

    @overload
    def triu(self, x: NDArray[dtypeT], /, *, k: int = 0) -> CompatArray[dtypeT, Literal["cpu"]]: ...
    @overload
    def triu(self, x: ArrayLike[dtypeT], /, *, k: int = 0) -> CompatArray[dtypeT, Any]: ...

    def triu(
        self,
        x: ArrayLike[Any],
        /, *,
        k: int = 0
    ) -> CompatArray[Any, Any]:
        """
        Returns the upper triangular part of a matrix (or a stack of matrices) :param:`x`.

        Parameters
        ----------
            x : ArrayLike
                Input array. Must have a shape of `(..., M, N)`, and whose innermost two dimensions form `MxN` matrices.

            k : int, default to `0`
                Diagonal above which to zero elements.
                - `k = 0`: Main diagonal;
                - `k < 0`: The diagonal is below the main diagonal;
                - `k > 0`: The diagonal is above the main diagonal.

        Returns
        -------
            CompatArray
                a `CompatArray` array containing the upper triangular part(s).
                The returned array must have the same shape and data type as :param:`x`.
                All elements above the specified diagonal :param:`k` must be zeroed.
                The returned array should be allocated on the same device as :param:`x`.
        """
        ...

    @overload
    def zeros(self, shape: Union[int, Tuple[int, ...]], *, dtype: None = ..., device: None = ...) -> CompatArray[float, Literal["cpu"]]: ...
    @overload
    def zeros(self, shape: Union[int, Tuple[int, ...]], *, dtype: None = ..., device: DeviceT) -> CompatArray[float, DeviceT]: ...
    @overload
    def zeros(self, shape: Union[int, Tuple[int, ...]], *, dtype: DTypeT, device: None = ...) -> CompatArray[DTypeT, Literal["cpu"]]: ...
    @overload
    def zeros(self, shape: Union[int, Tuple[int, ...]], *, dtype: DTypeT, device: DeviceT) -> CompatArray[DTypeT, DeviceT]: ...

    def zeros(
        self,
        shape: Union[int, Tuple[int, ...]],
        *,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None
    ) -> CompatArray[Any, Any]:
        """
        Returns a `CompatArray` array having a specified shape and filled with zeros.

        Parameters
        ----------
            shape : Union[int, Tuple[int, ...]]
                Output array shape.

            dtype : Optional[DType], default to `None`
                 Output array data type.
                 - `None`: The output array data type must be the default real-valued floating-point data type.

            device : Optional[Device], default to `None`
                 Device on which to place the created array.

        Returns
        -------
            CompatArray
                a `CompatArray` array containing zeros.
        """
        ...

    @overload
    def zeros_like(self, x: NDArray[dtypeT], /, *, dtype: None = ..., device: Optional[Device] = ...) -> CompatArray[dtypeT, Literal["cpu"]]: ...
    @overload
    def zeros_like(self, x: ArrayLike[dtypeT], /, *, dtype: None = ..., device: None = ...) -> CompatArray[dtypeT, Any]: ...
    @overload
    def zeros_like(self, x: ArrayLike[dtypeT], /, *, dtype: None = ..., device: DeviceT) -> CompatArray[dtypeT, DeviceT]: ...
    @overload
    def zeros_like(self, x: ArrayLike[Any], /, *, dtype: DTypeT, device: None = ...) -> CompatArray[DTypeT, Any]: ...
    @overload
    def zeros_like(self, x: ArrayLike[Any], /, *, dtype: DTypeT, device: DeviceT) -> CompatArray[DTypeT, DeviceT]: ...

    def zeros_like(
        self,
        x: ArrayLike[Any],
        /, *,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None
    ) -> CompatArray[Any, Any]:
        """
        Returns a `CompatArray` array filled with zeros with the same shape as an input array :param:`x`.

        Parameters
        ----------
            x : ArrayLike
                Input array from which to derive the output array shape.

            dtype : Optional[DType], default to `None`
                 Output array data type.
                 - `None`: The output array data type must be inferred from :param:`x`.

            device : Optional[Device], default to `None`
                 Device on which to place the created array.
                 - `None`: The output array device must be inferred from :param:`x`.

        Returns
        -------
            CompatArray
                a `CompatArray` array having the same shape as :param:`x` and containing zeros.
        """
        ...

    # === Data Type functions ===
    def can_cast(self, from_: Union[DType, ArrayLike[Any]], to: DType, /) -> bool:
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
        ...

    def finfo(self, type: Union[DType, ArrayLike[Any]], /) -> Any:
        """
        Machine limits for floating-point data types.

        Parameters
        ----------
            type : Union[DType, ArrayLike[Any]]
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
        ...

    def iinfo(self, type: Union[DType, ArrayLike[Any]], /) -> Any:
        """
        Machine limits for integer data types.

        Parameters
        ----------
            type : Union[DType, ArrayLike[Any]]
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
        ...

    def isdtype(
        self,
        dtype: DType,
        kind: Union[str, DType, Tuple[Union[str, DType], ...]]
    ) -> bool:
        """
        # TODO
        Returns a boolean indicating whether a provided dtype is of a specified data type “kind”.

        Parameters:
        dtype (dtype) – the input dtype.

        kind (Union[str, dtype, Tuple[Union[str, dtype], ...]]) –

        data type kind.

        If kind is a dtype, the function must return a boolean indicating whether the input dtype is equal to the dtype specified by kind.

        If kind is a string, the function must return a boolean indicating whether the input dtype is of a specified data type kind. The following dtype kinds must be supported:

        'bool': boolean data types (e.g., bool).

        'signed integer': signed integer data types (e.g., int8, int16, int32, int64).

        'unsigned integer': unsigned integer data types (e.g., uint8, uint16, uint32, uint64).

        'integral': integer data types. Shorthand for ('signed integer', 'unsigned integer').

        'real floating': real-valued floating-point data types (e.g., float32, float64).

        'complex floating': complex floating-point data types (e.g., complex64, complex128).

        'numeric': numeric data types. Shorthand for ('integral', 'real floating', 'complex floating').

        If kind is a tuple, the tuple specifies a union of dtypes and/or kinds, and the function must return a boolean indicating whether the input dtype is either equal to a specified dtype or belongs to at least one specified data type kind.
        
        Returns:
        out (bool) – boolean indicating whether a provided dtype is of a specified data type kind.
        """
        ...

    def result_type(*arrays_and_dtypes: Union[ArrayLike[Any], int, float, complex, bool, DType]) -> DType:
        """
        # TODO
        Returns the dtype that results from applying type promotion rules (see Type Promotion Rules) to the arguments.

        Parameters:
        arrays_and_dtypes (Union[array, int, float, complex, bool, dtype]) – an arbitrary number of input arrays, scalars, and/or dtypes.

        Returns:
        out (dtype) – the dtype resulting from an operation involving the input arrays, scalars, and/or dtypes.
        
        """
        ...

    # === Manipulation functions ===
    def broadcast_arrays(self, *arrays: ArrayLike[Any]) -> List[CompatArray[Any, Any]]:
        """
        Broadcasts one or more arrays against one another.

        Parameters
        ----------
            arrays : ArrayLike[Any]
                An arbitrary number of to-be broadcasted arrays.

        Returns
        -------
            List[CompatArray]
                A list of broadcasted `CompatArray` arrays.
                Each array must have the same shape.
                Each array must have the same dtype as its corresponding input array.
        """
        ...

    @overload
    def broadcast_to(self, x: NDArray[dtypeT], shape: Tuple[int, ...]) -> CompatArray[dtypeT, Literal["cpu"]]: ...
    @overload
    def broadcast_to(self, x: ArrayLike[dtypeT], shape: Tuple[int, ...]) -> CompatArray[dtypeT, Any]: ...

    def broadcast_to(self, x: ArrayLike[Any], shape: Tuple[int, ...]) -> CompatArray[Any, Any]:
        """
        Broadcasts an array to a specified shape.

        Parameters
        ----------
            x : ArrayLike
                Array to broadcast.
                Must be capable of being broadcast to the specified shape (see Broadcasting).
                If the array is incompatible with the specified shape, the function must raise an exception.

            shape : Tuple[int, ...]
                The shape to broadcast to.

        Returns
        -------
            CompatArray
                A `CompatArray` array having the specified shape.
                Must have the same data type as :param:`x`.
        """
        ...

    def concat(
        self,
        arrays: Union[Tuple[ArrayLike[Any], ...], List[ArrayLike[Any]]],
        /, *,
        axis: Optional[int] = 0
    ) -> CompatArray[Any, Any]:
        """
        Joins a sequence of arrays along an existing axis.

        Parameters
        ----------
            arrays : Union[Tuple[ArrayLike[Any], ...], List[ArrayLike[Any]]]
                Input arrays to join.
                The arrays must have the same shape, except in the dimension specified by axis.

            axis : int, default to `0`
                Axis along which the arrays will be joined.
                - `None`: Arrays must be flattened before concatenation;
                - _negative_: The function must determine the axis along which to join by counting from the last dimension.

        Returns
        -------
            CompatArray
                A `CompatArray` output array containing the concatenated values. 
        """
        ...

    def stack(
        self,
        arrays: Union[Tuple[ArrayLike[Any], ...], List[ArrayLike[Any]]],
        /, *,
        axis: int = 0
    ) -> CompatArray[Any, Any]:
        """
        Joins a sequence of arrays along a new axis.

        Parameters
        ----------
            arrays : Union[Tuple[ArrayLike[Any], ...], List[ArrayLike[Any]]]
                 Input arrays to join.
                 Each array must have the same shape.

            axis : int, default to `0`
                Axis along which the arrays will be joined.
                Providing an :param:`axis` specifies the index of the new axis in the dimensions of the result.
                For example,
                - `0`: The new axis will be the first dimension and the output array will have shape `(N, A, B, C)`;
                - `1`: The new new axis will be the second dimension and the output array will have shape `(A, N, B, C)`;
                - `-1`: new axis will be the last dimension and the output array will have shape `(A, B, C, N)`.

                A valid axis must be on the interval `[-N, N)`, where `N` is the rank (number of dimensions) of `x`.
                If provided an axis outside of the required interval, the function must raise an exception.

        Returns
        -------
            CompatArray
                An output array having rank `N+1`, where `N` is the rank (number of dimensions) of `x`.
                If the input arrays have different data types, normal Type Promotion Rules must apply.
                If the input arrays have the same data type, the output array must have the same data type as the input arrays.
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
