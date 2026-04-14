# -*- coding: utf-8 -*-
# Python version: 3.9
# @TianZhen

from __future__ import annotations
from numpy.typing import NDArray
from types import ModuleType
from typing import (Union, List, Tuple, Optional, Any, Literal, overload)

from ._base import Compat
from ._array import CompatArray
from ..types import (
    dtypeT, DTypeT, deviceT, DeviceT, DType, AnyDevice,
    ValueT, Value, ArrayLike, ArrayOrAny
)


class CompatNamespace(Compat):
    def __new__(cls, xp: object, /): ...

    # === Creation functions ===
    @overload
    def asarray(self, obj: NDArray[dtypeT], /, *, dtype: None = ..., device: Optional[AnyDevice] = ..., copy: Optional[bool] = ...) -> CompatArray[dtypeT, Literal["cpu"]]: ...
    @overload
    def asarray(self, obj: ArrayLike[dtypeT], /, *, dtype: None = ..., device: None = ..., copy: Optional[bool] = ...) -> CompatArray[dtypeT, AnyDevice]: ...
    @overload
    def asarray(self, obj: object, /, *, dtype: None = ..., device: None = ..., copy: Optional[bool] = ...) -> CompatArray[Any, AnyDevice]: ...
    @overload
    def asarray(self, obj: object, /, *, dtype: None = ..., device: DeviceT, copy: Optional[bool] = ...) -> CompatArray[Any, DeviceT]: ...
    @overload
    def asarray(self, obj: object, /, *, dtype: None = ..., device: AnyDevice, copy: Optional[bool] = ...) -> CompatArray[Any, AnyDevice]: ...
    @overload
    def asarray(self, obj: NDArray[Any], /, *, dtype: DTypeT, device: Optional[AnyDevice] = ..., copy: Optional[bool] = ...) -> CompatArray[DTypeT, Literal["cpu"]]: ...
    @overload
    def asarray(self, obj: object, /, *, dtype: DTypeT, device: None = ..., copy: Optional[bool] = ...) -> CompatArray[DTypeT, AnyDevice]: ...
    @overload
    def asarray(self, obj: object, /, *, dtype: DTypeT, device: DeviceT, copy: Optional[bool] = ...) -> CompatArray[DTypeT, DeviceT]: ...
    @overload
    def asarray(self, obj: object, /, *, dtype: DTypeT, device: AnyDevice, copy: Optional[bool] = ...) -> CompatArray[DTypeT, AnyDevice]: ...

    def asarray(
        self,
        obj: object,
        /, *,
        dtype: Optional[DType] = None,
        device: Optional[AnyDevice] = None,
        copy: Optional[bool] = None
    ) -> CompatArray[Any, AnyDevice]:
        """
        Convert the input to a :class:`CompatArray` array.

        Parameters
        ----------
            obj : object
                Object to be converted to a :class:`CompatArray` array.
                May be a Python scalar, a (possibly nested) sequence of Python scalars, or an object supporting the Python buffer protocol.

            dtype : Optional[DType], default to `None`
                Output array data type.
                - `None`: The output array data type must be inferred from the data type(s) in obj.
                If all input values are Python scalars, then, in order of precedence:
                1. If all values are of type `bool`, the output data type must be `bool`;
                2. If all values are of type `int` or are a mixture of `bool` and `int`, the output data type must be the default `integer` data type;
                3. If one or more values are `complex` numbers, the output data type must be the default `complex` floating-point data type;
                4. If one or more values are `floats`, the output data type must be the default real-valued floating-point data type.

            device : Optional[AnyDevice], default to `None`
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
                A :class:`CompatArray` array containing the data from :param:`obj`.
        """
        ...

    @overload
    def arange(self, start: int, /, stop: Optional[int] = ..., step: int = ..., *, dtype: None = ..., device: None = ...) -> CompatArray[int, Literal["cpu"]]: ...
    @overload
    def arange(self, start: Union[int, float], /, stop: Optional[Union[int, float]] = ..., step: Union[int, float] = ..., *, dtype: None = ..., device: None = ...) -> CompatArray[float, Literal["cpu"]]: ...
    @overload
    def arange(self, start: int, /, stop: Optional[int] = ..., step: int = ..., *, dtype: None = ..., device: DeviceT) -> CompatArray[int, DeviceT]: ...
    @overload
    def arange(self, start: int, /, stop: Optional[int] = ..., step: int = ..., *, dtype: None = ..., device: AnyDevice) -> CompatArray[int, AnyDevice]: ...
    @overload
    def arange(self, start: Union[int, float], /, stop: Optional[Union[int, float]] = ..., step: Union[int, float] = ..., *, dtype: None = ..., device: DeviceT) -> CompatArray[float, DeviceT]: ...
    @overload
    def arange(self, start: Union[int, float], /, stop: Optional[Union[int, float]] = ..., step: Union[int, float] = ..., *, dtype: None = ..., device: AnyDevice) -> CompatArray[float, AnyDevice]: ...
    @overload
    def arange(self, start: Union[int, float], /, stop: Optional[Union[int, float]] = ..., step: Union[int, float] = ..., *, dtype: DTypeT, device: None = ...) -> CompatArray[DTypeT, Literal["cpu"]]: ...
    @overload
    def arange(self, start: Union[int, float], /, stop: Optional[Union[int, float]] = ..., step: Union[int, float] = ..., *, dtype: DTypeT, device: AnyDevice) -> CompatArray[DTypeT, AnyDevice]: ...

    def arange(
        self,
        start: Union[int, float],
        /,
        stop: Optional[Union[int, float]] = None,
        step: Union[int, float] = 1,
        *,
        dtype: Optional[DType] = None,
        device: Optional[AnyDevice] = None
    ) -> CompatArray[Any, AnyDevice]:
        """
        Returns evenly spaced values within the half-open interval `[start, stop)` as a one-dimensional :class:`CompatArray` array.

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

            device : Optional[AnyDevice], default to `None`
                 Device on which to place the created array.

        Returns
        -------
            CompatArray
                A one-dimensional :class:`CompatArray` array containing evenly spaced values.
                The length of the output array must be `ceil((stop-start)/step)` if `stop - start` and :param:`step` have the same sign, and length `0` otherwise.
        """
        ...

    @overload
    def empty(self, shape: Union[int, Tuple[int, ...]], *, dtype: None = ..., device: None = ...) -> CompatArray[float, Literal["cpu"]]: ...
    @overload
    def empty(self, shape: Union[int, Tuple[int, ...]], *, dtype: None = ..., device: DeviceT) -> CompatArray[float, DeviceT]: ...
    @overload
    def empty(self, shape: Union[int, Tuple[int, ...]], *, dtype: None = ..., device: AnyDevice) -> CompatArray[float, AnyDevice]: ...
    @overload
    def empty(self, shape: Union[int, Tuple[int, ...]], *, dtype: DTypeT, device: None = ...) -> CompatArray[DTypeT, Literal["cpu"]]: ...
    @overload
    def empty(self, shape: Union[int, Tuple[int, ...]], *, dtype: DTypeT, device: DeviceT) -> CompatArray[DTypeT, DeviceT]: ...
    @overload
    def empty(self, shape: Union[int, Tuple[int, ...]], *, dtype: DTypeT, device: AnyDevice) -> CompatArray[DTypeT, AnyDevice]: ...

    def empty(
        self,
        shape: Union[int, Tuple[int, ...]],
        *,
        dtype: Optional[DType] = None,
        device: Optional[AnyDevice] = None
    ) -> CompatArray[Any, AnyDevice]:
        """
        Returns an uninitialized :class:`CompatArray` array having a specified shape.

        Parameters
        ----------
            shape : Union[int, Tuple[int, ...]]
                Output array shape.

            dtype : Optional[DType], default to `None`
                 Output array data type.
                 - `None`: The output array data type must be the default real-valued floating-point data type.

            device : Optional[AnyDevice], default to `None`
                 Device on which to place the created array.

        Returns
        -------
            CompatArray
                a :class:`CompatArray` array containing uninitialized data.
        """
        ...

    @overload
    def empty_like(self, x: NDArray[dtypeT], /, *, dtype: None = ..., device: Optional[AnyDevice] = ...) -> CompatArray[dtypeT, Literal["cpu"]]: ...
    @overload
    def empty_like(self, x: ArrayLike[dtypeT], /, *, dtype: None = ..., device: None = ...) -> CompatArray[dtypeT, AnyDevice]: ...
    @overload
    def empty_like(self, x: ArrayLike[dtypeT], /, *, dtype: None = ..., device: DeviceT) -> CompatArray[dtypeT, DeviceT]: ...
    @overload
    def empty_like(self, x: ArrayLike[dtypeT], /, *, dtype: None = ..., device: AnyDevice) -> CompatArray[dtypeT, AnyDevice]: ...
    @overload
    def empty_like(self, x: ArrayLike[Any], /, *, dtype: DTypeT, device: None = ...) -> CompatArray[DTypeT, AnyDevice]: ...
    @overload
    def empty_like(self, x: ArrayLike[Any], /, *, dtype: DTypeT, device: DeviceT) -> CompatArray[DTypeT, DeviceT]: ...
    @overload
    def empty_like(self, x: ArrayLike[Any], /, *, dtype: DTypeT, device: AnyDevice) -> CompatArray[DTypeT, AnyDevice]: ...

    def empty_like(
        self,
        x: ArrayLike[Any],
        /, *,
        dtype: Optional[DType] = None,
        device: Optional[AnyDevice] = None
    ) -> CompatArray[Any, AnyDevice]:
        """
        Returns an uninitialized :class:`CompatArray` array with the same shape as an input array :param:`x`.

        Parameters
        ----------
            x : ArrayLike
                Input array from which to derive the output array shape.

            dtype : Optional[DType], default to `None`
                 Output array data type.
                 - `None`: The output array data type must be inferred from :param:`x`.

            device : Optional[AnyDevice], default to `None`
                 Device on which to place the created array.
                 - `None`: The output array device must be inferred from :param:`x`.

        Returns
        -------
            CompatArray
                a :class:`CompatArray` array having the same shape as :param:`x` and containing uninitialized data.
        """
        ...

    @overload
    def eye(self, n_rows: int, n_cols: Optional[int] = ..., /, *, k: int = ..., dtype: None = ..., device: None = ...) -> CompatArray[float, Literal["cpu"]]: ...
    @overload
    def eye(self, n_rows: int, n_cols: Optional[int] = ..., /, *, k: int = ..., dtype: None = ..., device: DeviceT) -> CompatArray[float, DeviceT]: ...
    @overload
    def eye(self, n_rows: int, n_cols: Optional[int] = ..., /, *, k: int = ..., dtype: None = ..., device: AnyDevice) -> CompatArray[float, AnyDevice]: ...
    @overload
    def eye(self, n_rows: int, n_cols: Optional[int] = ..., /, *, k: int = ..., dtype: DTypeT, device: None = ...) -> CompatArray[DTypeT, Literal["cpu"]]: ...
    @overload
    def eye(self, n_rows: int, n_cols: Optional[int] = ..., /, *, k: int = ..., dtype: DTypeT, device: DeviceT) -> CompatArray[DTypeT, DeviceT]: ...
    @overload
    def eye(self, n_rows: int, n_cols: Optional[int] = ..., /, *, k: int = ..., dtype: DTypeT, device: AnyDevice) -> CompatArray[DTypeT, AnyDevice]: ...

    def eye(
        self,
        n_rows: int,
        n_cols: Optional[int] = None,
        /, *,
        k: int = 0,
        dtype: Optional[DType] = None,
        device: Optional[AnyDevice] = None
    ) -> CompatArray[Any, AnyDevice]:
        """
        Returns a two-dimensional :class:`CompatArray` array with ones on the :param:`k`th diagonal and zeros elsewhere.

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

            device : Optional[AnyDevice], default to `None`
                 Device on which to place the created array.

        Returns
        -------
            CompatArray
                A :class:`CompatArray` array where all elements are equal to zero, except for the kth diagonal, whose values are equal to one.
        """
        ...

    @overload
    def from_dlpack(self, x: NDArray[dtypeT], /, *, device: None = ..., copy: Optional[bool] = ...) -> CompatArray[dtypeT, Literal["cpu"]]: ...
    @overload
    def from_dlpack(self, x: ArrayLike[dtypeT], /, *, device: None = ..., copy: Optional[bool] = ...) -> CompatArray[dtypeT, AnyDevice]: ...
    @overload
    def from_dlpack(self, x: object, /, *, device: None = ..., copy: Optional[bool] = ...) -> CompatArray[Any, AnyDevice]: ...
    @overload
    def from_dlpack(self, x: ArrayLike[dtypeT], /, *, device: DeviceT, copy: Optional[bool] = ...) -> CompatArray[dtypeT, DeviceT]: ...
    @overload
    def from_dlpack(self, x: ArrayLike[dtypeT], /, *, device: AnyDevice, copy: Optional[bool] = ...) -> CompatArray[dtypeT, AnyDevice]: ...
    @overload
    def from_dlpack(self, x: object, /, *, device: DeviceT, copy: Optional[bool] = ...) -> CompatArray[Any, DeviceT]: ...
    @overload
    def from_dlpack(self, x: object, /, *, device: AnyDevice, copy: Optional[bool] = ...) -> CompatArray[Any, AnyDevice]: ...

    def from_dlpack(
        self,
        x: object,
        /, *,
        device: Optional[AnyDevice] = None,
        copy: Optional[bool] = None
    ) -> CompatArray[Any, AnyDevice]:
        """
        Returns a new :class:`CompatArray` array containing the data from another (array) object with a `__dlpack__` method.

        Parameters
        ----------
            x : object
                Input (array) object.

            device : Optional[AnyDevice], default to `None`
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
                A :class:`CompatArray` array containing the data in :param:`x`.
        """
        ...

    @overload
    def full(self, shape: Union[int, Tuple[int, ...]], fill_value: ValueT, *, dtype: None = ..., device: None = ...) -> CompatArray[ValueT, Literal["cpu"]]: ...
    @overload
    def full(self, shape: Union[int, Tuple[int, ...]], fill_value: ValueT, *, dtype: None = ..., device: DeviceT) -> CompatArray[ValueT, DeviceT]: ...
    @overload
    def full(self, shape: Union[int, Tuple[int, ...]], fill_value: ValueT, *, dtype: None = ..., device: AnyDevice) -> CompatArray[ValueT, AnyDevice]: ...
    @overload
    def full(self, shape: Union[int, Tuple[int, ...]], fill_value: Value, *, dtype: DTypeT, device: None = ...) -> CompatArray[DTypeT, Literal["cpu"]]: ...
    @overload
    def full(self, shape: Union[int, Tuple[int, ...]], fill_value: Value, *, dtype: DTypeT, device: DeviceT) -> CompatArray[DTypeT, DeviceT]: ...
    @overload
    def full(self, shape: Union[int, Tuple[int, ...]], fill_value: Value, *, dtype: DTypeT, device: AnyDevice) -> CompatArray[DTypeT, AnyDevice]: ...

    def full(
        self,
        shape: Union[int, Tuple[int, ...]],
        fill_value: Value,
        *,
        dtype: Optional[DType] = None,
        device: Optional[AnyDevice] = None
    ) -> CompatArray[Any, AnyDevice]:
        """
        Returns a new :class:`CompatArray` array having a specified :param:`shape` and filled with :param:`fill_value`.

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

            device : Optional[AnyDevice], default to `None`
                Device on which to place the created array.

        Returns
        -------
            CompatArray
                A :class:`CompatArray` array where every element is equal to :param:`fill_value`.
        """
        ...

    @overload
    def full_like(self, x: NDArray[dtypeT], /, fill_value: Value, *, dtype: None = ..., device: Optional[AnyDevice] = ...) -> CompatArray[dtypeT, Literal["cpu"]]: ...
    @overload
    def full_like(self, x: ArrayLike[dtypeT], /, fill_value: Value, *, dtype: None = ..., device: None = ...) -> CompatArray[dtypeT, AnyDevice]: ...
    @overload
    def full_like(self, x: ArrayLike[dtypeT], /, fill_value: Value, *, dtype: None = ..., device: DeviceT) -> CompatArray[dtypeT, DeviceT]: ...
    @overload
    def full_like(self, x: ArrayLike[dtypeT], /, fill_value: Value, *, dtype: None = ..., device: AnyDevice) -> CompatArray[dtypeT, AnyDevice]: ...
    @overload
    def full_like(self, x: ArrayLike[Any], /, fill_value: Value, *, dtype: DTypeT, device: None = ...) -> CompatArray[DTypeT, AnyDevice]: ...
    @overload
    def full_like(self, x: ArrayLike[Any], /, fill_value: Value, *, dtype: DTypeT, device: DeviceT) -> CompatArray[DTypeT, DeviceT]: ...
    @overload
    def full_like(self, x: ArrayLike[Any], /, fill_value: Value, *, dtype: DTypeT, device: AnyDevice) -> CompatArray[DTypeT, AnyDevice]: ...

    def full_like(
        self,
        x: ArrayLike[Any],
        /,
        fill_value: Value,
        *,
        dtype: Optional[DType] = None,
        device: Optional[AnyDevice] = None
    ) -> CompatArray[Any, AnyDevice]:
        """
        Returns a :class:`CompatArray` array with the same shape as an input array :param:`x` and filled with :param:`fill_value`.

        Parameters
        ----------
            x : ArrayLike
                Input array from which to derive the output array shape.

            fill_value : Value
                Fill value.

            dtype : Optional[DType], default to `None`
                 Output array data type.
                 - `None`: The output array data type must be inferred from :param:`x`.

            device : Optional[AnyDevice], default to `None`
                 Device on which to place the created array.
                 - `None`: The output array device must be inferred from :param:`x`.

        Returns
        -------
            CompatArray
                a :class:`CompatArray` array having the same shape as :param:`x` and where every element is equal to :param:`fill_value`.
        """
        ...

    @overload
    def linspace(self, start: Union[int, float, complex], stop: Union[int, float, complex], /, num: int, *, dtype: None = ..., device: None = ..., endpoint: bool = ...) -> CompatArray[float, Literal["cpu"]]: ...
    @overload
    def linspace(self, start: Union[int, float, complex], stop: Union[int, float, complex], /, num: int, *, dtype: None = ..., device: DeviceT, endpoint: bool = ...) -> CompatArray[float, DeviceT]: ...
    @overload
    def linspace(self, start: Union[int, float, complex], stop: Union[int, float, complex], /, num: int, *, dtype: None = ..., device: AnyDevice, endpoint: bool = ...) -> CompatArray[float, AnyDevice]: ...
    @overload
    def linspace(self, start: Union[int, float, complex], stop: Union[int, float, complex], /, num: int, *, dtype: DTypeT, device: None = ..., endpoint: bool = ...) -> CompatArray[DTypeT, Literal["cpu"]]: ...
    @overload
    def linspace(self, start: Union[int, float, complex], stop: Union[int, float, complex], /, num: int, *, dtype: DTypeT, device: DeviceT, endpoint: bool = ...) -> CompatArray[DTypeT, DeviceT]: ...
    @overload
    def linspace(self, start: Union[int, float, complex], stop: Union[int, float, complex], /, num: int, *, dtype: DTypeT, device: AnyDevice, endpoint: bool = ...) -> CompatArray[DTypeT, AnyDevice]: ...

    def linspace(
        self,
        start: Union[int, float, complex],
        stop: Union[int, float, complex],
        /,
        num: int,
        *,
        dtype: Optional[DType] = None,
        device: Optional[AnyDevice] = None,
        endpoint: bool = True
    ) -> CompatArray[Any, AnyDevice]:
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

            device : Optional[AnyDevice], default to `None`
                The device on which to place the output array.

            endpoint : bool, default to `True`
                Boolean indicating whether to include :param:`stop` in the interval.

        Returns
        -------
            CompatArray
                A one-dimensional :class:`CompatArray` array containing evenly spaced values.
        """
        ...

    def meshgrid(self, *arrays: ArrayLike[Any], indexing: Literal["xy", "ij"] = "xy") -> List[CompatArray[Any, AnyDevice]]: ...

    @overload
    def ones(self, shape: Union[int, Tuple[int, ...]], *, dtype: None = ..., device: None = ...) -> CompatArray[float, Literal["cpu"]]: ...
    @overload
    def ones(self, shape: Union[int, Tuple[int, ...]], *, dtype: None = ..., device: DeviceT) -> CompatArray[float, DeviceT]: ...
    @overload
    def ones(self, shape: Union[int, Tuple[int, ...]], *, dtype: None = ..., device: AnyDevice) -> CompatArray[float, AnyDevice]: ...
    @overload
    def ones(self, shape: Union[int, Tuple[int, ...]], *, dtype: DTypeT, device: None = ...) -> CompatArray[DTypeT, Literal["cpu"]]: ...
    @overload
    def ones(self, shape: Union[int, Tuple[int, ...]], *, dtype: DTypeT, device: DeviceT) -> CompatArray[DTypeT, DeviceT]: ...
    @overload
    def ones(self, shape: Union[int, Tuple[int, ...]], *, dtype: DTypeT, device: AnyDevice) -> CompatArray[DTypeT, AnyDevice]: ...

    def ones(
        self,
        shape: Union[int, Tuple[int, ...]],
        *,
        dtype: Optional[DType] = None,
        device: Optional[AnyDevice] = None
    ) -> CompatArray[Any, AnyDevice]:
        """
        Returns a :class:`CompatArray` array having a specified shape and filled with ones.

        Parameters
        ----------
            shape : Union[int, Tuple[int, ...]]
                Output array shape.

            dtype : Optional[DType], default to `None`
                 Output array data type.
                 - `None`: The output array data type must be the default real-valued floating-point data type.

            device : Optional[AnyDevice], default to `None`
                 Device on which to place the created array.

        Returns
        -------
            CompatArray
                a :class:`CompatArray` array containing ones.
        """
        ...

    @overload
    def ones_like(self, x: NDArray[dtypeT], /, *, dtype: None = ..., device: Optional[AnyDevice] = ...) -> CompatArray[dtypeT, Literal["cpu"]]: ...
    @overload
    def ones_like(self, x: ArrayLike[dtypeT], /, *, dtype: None = ..., device: None = ...) -> CompatArray[dtypeT, AnyDevice]: ...
    @overload
    def ones_like(self, x: ArrayLike[dtypeT], /, *, dtype: None = ..., device: DeviceT) -> CompatArray[dtypeT, DeviceT]: ...
    @overload
    def ones_like(self, x: ArrayLike[dtypeT], /, *, dtype: None = ..., device: AnyDevice) -> CompatArray[dtypeT, AnyDevice]: ...
    @overload
    def ones_like(self, x: ArrayLike[Any], /, *, dtype: DTypeT, device: None = ...) -> CompatArray[DTypeT, AnyDevice]: ...
    @overload
    def ones_like(self, x: ArrayLike[Any], /, *, dtype: DTypeT, device: DeviceT) -> CompatArray[DTypeT, DeviceT]: ...
    @overload
    def ones_like(self, x: ArrayLike[Any], /, *, dtype: DTypeT, device: AnyDevice) -> CompatArray[DTypeT, AnyDevice]: ...

    def ones_like(
        self,
        x: ArrayLike[Any],
        /, *,
        dtype: Optional[DType] = None,
        device: Optional[AnyDevice] = None
    ) -> CompatArray[Any, AnyDevice]:
        """
        Returns a :class:`CompatArray` array filled with ones with the same shape as an input array :param:`x`.

        Parameters
        ----------
            x : ArrayLike
                Input array from which to derive the output array shape.

            dtype : Optional[DType], default to `None`
                 Output array data type.
                 - `None`: The output array data type must be inferred from :param:`x`.

            device : Optional[AnyDevice], default to `None`
                 Device on which to place the created array.
                 - `None`: The output array device must be inferred from :param:`x`.

        Returns
        -------
            CompatArray
                a :class:`CompatArray` array having the same shape as :param:`x` and containing ones.
        """
        ...

    @overload
    def tril(self, x: NDArray[dtypeT], /, *, k: int = 0) -> CompatArray[dtypeT, Literal["cpu"]]: ...
    @overload
    def tril(self, x: ArrayLike[dtypeT], /, *, k: int = 0) -> CompatArray[dtypeT, AnyDevice]: ...

    def tril(self, x: ArrayLike[Any], /, *, k: int = 0) -> CompatArray[Any, AnyDevice]:
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
                a :class:`CompatArray` array containing the lower triangular part(s).
                The returned array must have the same shape and data type as :param:`x`.
                All elements above the specified diagonal :param:`k` must be zeroed.
                The returned array should be allocated on the same device as :param:`x`.
        """
        ...

    @overload
    def triu(self, x: NDArray[dtypeT], /, *, k: int = 0) -> CompatArray[dtypeT, Literal["cpu"]]: ...
    @overload
    def triu(self, x: ArrayLike[dtypeT], /, *, k: int = 0) -> CompatArray[dtypeT, AnyDevice]: ...

    def triu(
        self,
        x: ArrayLike[Any],
        /, *,
        k: int = 0
    ) -> CompatArray[Any, AnyDevice]:
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
                a :class:`CompatArray` array containing the upper triangular part(s).
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
    def zeros(self, shape: Union[int, Tuple[int, ...]], *, dtype: None = ..., device: AnyDevice) -> CompatArray[float, AnyDevice]: ...
    @overload
    def zeros(self, shape: Union[int, Tuple[int, ...]], *, dtype: DTypeT, device: None = ...) -> CompatArray[DTypeT, Literal["cpu"]]: ...
    @overload
    def zeros(self, shape: Union[int, Tuple[int, ...]], *, dtype: DTypeT, device: DeviceT) -> CompatArray[DTypeT, DeviceT]: ...
    @overload
    def zeros(self, shape: Union[int, Tuple[int, ...]], *, dtype: DTypeT, device: AnyDevice) -> CompatArray[DTypeT, AnyDevice]: ...

    def zeros(
        self,
        shape: Union[int, Tuple[int, ...]],
        *,
        dtype: Optional[DType] = None,
        device: Optional[AnyDevice] = None
    ) -> CompatArray[Any, AnyDevice]:
        """
        Returns a :class:`CompatArray` array having a specified shape and filled with zeros.

        Parameters
        ----------
            shape : Union[int, Tuple[int, ...]]
                Output array shape.

            dtype : Optional[DType], default to `None`
                 Output array data type.
                 - `None`: The output array data type must be the default real-valued floating-point data type.

            device : Optional[AnyDevice], default to `None`
                 Device on which to place the created array.

        Returns
        -------
            CompatArray
                a :class:`CompatArray` array containing zeros.
        """
        ...

    @overload
    def zeros_like(self, x: NDArray[dtypeT], /, *, dtype: None = ..., device: Optional[AnyDevice] = ...) -> CompatArray[dtypeT, Literal["cpu"]]: ...
    @overload
    def zeros_like(self, x: ArrayLike[dtypeT], /, *, dtype: None = ..., device: None = ...) -> CompatArray[dtypeT, AnyDevice]: ...
    @overload
    def zeros_like(self, x: ArrayLike[dtypeT], /, *, dtype: None = ..., device: DeviceT) -> CompatArray[dtypeT, DeviceT]: ...
    @overload
    def zeros_like(self, x: ArrayLike[dtypeT], /, *, dtype: None = ..., device: AnyDevice) -> CompatArray[dtypeT, AnyDevice]: ...
    @overload
    def zeros_like(self, x: ArrayLike[Any], /, *, dtype: DTypeT, device: None = ...) -> CompatArray[DTypeT, AnyDevice]: ...
    @overload
    def zeros_like(self, x: ArrayLike[Any], /, *, dtype: DTypeT, device: DeviceT) -> CompatArray[DTypeT, DeviceT]: ...
    @overload
    def zeros_like(self, x: ArrayLike[Any], /, *, dtype: DTypeT, device: AnyDevice) -> CompatArray[DTypeT, AnyDevice]: ...

    def zeros_like(
        self,
        x: ArrayLike[Any],
        /, *,
        dtype: Optional[DType] = None,
        device: Optional[AnyDevice] = None
    ) -> CompatArray[Any, AnyDevice]:
        """
        Returns a :class:`CompatArray` array filled with zeros with the same shape as an input array :param:`x`.

        Parameters
        ----------
            x : ArrayLike
                Input array from which to derive the output array shape.

            dtype : Optional[DType], default to `None`
                 Output array data type.
                 - `None`: The output array data type must be inferred from :param:`x`.

            device : Optional[AnyDevice], default to `None`
                 Device on which to place the created array.
                 - `None`: The output array device must be inferred from :param:`x`.

        Returns
        -------
            CompatArray
                a :class:`CompatArray` array having the same shape as :param:`x` and containing zeros.
        """
        ...

    # === Data Type functions ===
    def can_cast(self, from_: Union[DType, ArrayLike[Any]], to: DType, /) -> bool: ...

    def finfo(self, type_: Union[DType, ArrayLike[Any]], /) -> Any: ...

    def iinfo(self, type_: Union[DType, ArrayLike[Any]], /) -> Any: ...

    def isdtype(
        self,
        dtype: DType,
        kind: Union[str, DType, Tuple[Union[str, DType], ...]]
    ) -> bool: ...

    def result_type(self, *arrays_and_dtypes: Union[ArrayOrAny, DType]) -> DType: ...

    # === Manipulation functions ===
    def broadcast_arrays(self, *arrays: ArrayLike[Any]) -> List[CompatArray[Any, AnyDevice]]: ...

    @overload
    def broadcast_to(self, x: NDArray[dtypeT], shape: Tuple[int, ...]) -> CompatArray[dtypeT, Literal["cpu"]]: ...
    @overload
    def broadcast_to(self, x: ArrayLike[dtypeT], shape: Tuple[int, ...]) -> CompatArray[dtypeT, AnyDevice]: ...

    def broadcast_to(self, x: ArrayLike[Any], shape: Tuple[int, ...]) -> CompatArray[Any, AnyDevice]:
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
                A :class:`CompatArray` array having the specified shape.
                Must have the same data type as :param:`x`.
        """
        ...

    def concat(
        self,
        arrays: Union[Tuple[ArrayLike[Any], ...], List[ArrayLike[Any]]],
        /, *,
        axis: Optional[int] = 0
    ) -> CompatArray[Any, AnyDevice]:
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
                A :class:`CompatArray` output array containing the concatenated values.
        """
        ...

    def stack(
        self,
        arrays: Union[Tuple[ArrayLike[Any], ...], List[ArrayLike[Any]]],
        /, *,
        axis: int = 0
    ) -> CompatArray[Any, AnyDevice]:
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

    # === Linear Algebra Extension ===
    @overload
    def vector_norm(self, x: NDArray[Any], /, *, axis: Optional[Union[int, Tuple[int, ...]]] = ..., keepdims: bool = ..., ord: Union[int, float, Literal["inf", "-inf"]] = ...) -> CompatArray[float, Literal["cpu"]]: ...
    @overload
    def vector_norm(self, x: CompatArray[Any, deviceT], /, *, axis: Optional[Union[int, Tuple[int, ...]]] = ..., keepdims: bool = ..., ord: Union[int, float, Literal["inf", "-inf"]] = ...) -> CompatArray[float, deviceT]: ...
    @overload
    def vector_norm(self, x: ArrayLike[Any], /, *, axis: Optional[Union[int, Tuple[int, ...]]] = ..., keepdims: bool = ..., ord: Union[int, float, Literal["inf", "-inf"]] = ...) -> CompatArray[float, AnyDevice]: ...
    def vector_norm(self, x: ArrayLike[Any], /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False, ord: Union[int, float, Literal["inf", "-inf"]] = 2) -> CompatArray[float, AnyDevice]: ...

    @overload
    def matrix_norm(self, x: NDArray[Any], /, *, keepdims: bool = ..., ord: Optional[Union[int, float, Literal["inf", "-inf", "fro", "nuc"]]] = ...) -> CompatArray[float, Literal["cpu"]]: ...
    @overload
    def matrix_norm(self, x: CompatArray[Any, deviceT], /, *, keepdims: bool = ..., ord: Optional[Union[int, float, Literal["inf", "-inf", "fro", "nuc"]]] = ...) -> CompatArray[float, deviceT]: ...
    @overload
    def matrix_norm(self, x: ArrayLike[Any], /, *, keepdims: bool = ..., ord: Optional[Union[int, float, Literal["inf", "-inf", "fro", "nuc"]]] = ...) -> CompatArray[float, AnyDevice]: ...
    def matrix_norm(self, x: ArrayLike[Any], /, *, keepdims: bool = False, ord: Optional[Union[int, float, Literal["inf", "-inf", "fro", "nuc"]]] = "fro") -> CompatArray[float, AnyDevice]: ...

    @property
    def linalg(self) -> ModuleType: ...

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
    def newaxis(self) -> None: ...

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

    @property
    def __name__(self) -> str: ...

    def __getattr__(self, name: str) -> Any: ...
