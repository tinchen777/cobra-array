# -*- coding: utf-8 -*-
# Python version: 3.9
# @TianZhen

from __future__ import annotations
from functools import wraps
from itertools import (islice, chain)
import array_api_compat as api
from contextvars import ContextVar
from typing import (Any, Union, Optional, Dict, overload, TYPE_CHECKING)

from .convert import (to_xp, as_array)
from ._utils import ArraySpec
from .default import default_array_spec
from .exceptions import (
    NoArrayInputsError,
    GetArrayNamespaceError,
    MissingDependencyError,
    NotArrayAPIObjectError
)

if TYPE_CHECKING:
    from array_api_compat.common._typing import Namespace
    from .types import (T, DTypeT, DeviceT, ArrayLike, ArrayLibraryName)


def array_spec(
    *arrays: object,
    kw_arrays: Optional[Dict[str, object]] = None,
    ref: Optional[Union[str, int]] = None,
    filter_array_like: bool = False,
    api_version: Optional[str] = None,
    use_compat: Optional[bool] = None
) -> ArraySpec:
    """
    Determine the array API compatible `namespace`, `dtype` and `device` from the provided array arguments and the reference array.

    Parameters
    ----------
        arrays : object
            Positional array-like objects to determine the `array namespace`.

        kw_arrays : Optional[Dict[str, object]], default to `None`
            Keyword array-like objects to determine the `array namespace`.

        ref : Optional[Union[str, int]], default to `None`
            Reference array to determine the `array namespace`, `dtype` and `device`.
            - `None`: Use all provided arrays in `arrays` and `kw_arrays` to determine the `array namespace`, and return `None` for `dtype` and `device`;
            - _str_: Use the specified keyword array in `kw_arrays` as reference array to determine the `array namespace`, `dtype` and `device`;
            - _int_: Use the specified positional array in `arrays` and `kw_arrays` (in order) as reference array to determine the `array namespace`, `dtype` and `device`. The index and valid range are affected by :param:`filter_array_like`.

        filter_array_like : bool, default to `False`
            Whether to filter the provided inputs to all array-likes via :func:`array_api_compat.is_array_api_obj` when determining the `array namespace`.

        api_version : Optional[str], default to `None`
            The target array API version for the returned `array namespace`. See also :param:`api_version` in :func:`array_api_compat.array_namespace`.

        use_compat : Optional[bool], default to `None`
            See also :param:`use_compat` in :func:`array_api_compat.array_namespace`.
            - `None`: Return the native namespace if it is already Array API–compatible, otherwise return a compat wrapper;
            - `True`: Always return the compat-wrapped namespace;
            - `False`: Return the native namespace.

    Returns
    -------
        ArraySpec
            An :class:`ArraySpec` named tuple containing the determined `xp`(`array namespace`), `dtype` and `device`. If :param:`ref` is `None`, the returned `dtype` and `device` will be `None`.

    Raises
    ------
        NoArrayInputsError
            If no array inputs are provided in `arrays` and `kw_arrays`.
        GetArrayNamespaceError
            If an error occurs while determining the `array namespace` from the provided inputs or the reference array.
        KeyError
            If `ref` is a string but not a key in `kw_arrays`.
        IndexError
            If `ref` is an integer but out of range for the array inputs.
        TypeError
            If `ref` is not `None`, a string, or an integer.
        NotArrayAPIObjectError
            If the reference array determined by `ref` is not an array API compatible array object.
    """
    kw_arrays = kw_arrays or {}

    if len(arrays) == 0 and len(kw_arrays) == 0:
        # no arrays provided
        raise NoArrayInputsError("Expected at least one array input.")

    # iterator of all arrays
    all_arrays = chain(arrays, kw_arrays.values())
    if filter_array_like:
        all_arrays = (a for a in all_arrays if api.is_array_api_obj(a))

    if ref is None:
        # use arrays and kw_arrays in order to determine the namespace
        try:
            return ArraySpec(
                api.array_namespace(
                    *all_arrays,
                    api_version=api_version,
                    use_compat=use_compat
                ), None, None
            )
        except Exception as e:
            raise GetArrayNamespaceError(
                "Failed to determine the `array namespace` from the provided inputs."
            ) from e

    if isinstance(ref, str):
        # use the specified kw_array as reference array to determine the namespace
        try:
            ref_arr = kw_arrays[ref]
        except KeyError:
            raise KeyError(
                f"Parameter `ref` of `array_spec()` must be a key in `kw_arrays`, got {ref!r}."
            ) from None
    elif isinstance(ref, int):
        # use the specified array as reference array to determine the namespace
        try:
            ref_arr = next(islice(all_arrays, ref, ref + 1))
        except ValueError:
            raise IndexError(
                    "Parameter `ref` of `array_spec()` must be a "
                    f"non-negative index for the array inputs, got {ref!r}."
                ) from None
        except StopIteration:
            if filter_array_like:
                raise IndexError(
                    "Parameter `ref` of `array_spec()` is out of range "
                    f"for the array-like inputs, got {ref!r}."
                ) from None
            raise IndexError(
                "Parameter `ref` of `array_spec()` must be in the range "
                f"[0, {len(arrays) + len(kw_arrays)}) for the array inputs, got {ref!r}."
            ) from None
    else:
        raise TypeError(
            "Parameter `ref` of `array_spec()` must be `str`, "
            f"`int` or `NoneType`, got {type(ref)}."
        )

    if not filter_array_like and not api.is_array_api_obj(ref_arr):
        raise NotArrayAPIObjectError(
            f"Reference array must be an array API compatible array object, got {ref_arr!r}."
        )

    dtype = getattr(ref_arr, "dtype", None)
    device = api.device(ref_arr)
    try:
        return ArraySpec(
            api.array_namespace(
                ref_arr,
                api_version=api_version,
                use_compat=use_compat
            ), dtype, device
        )
    except Exception as e:
        raise GetArrayNamespaceError(
            "Failed to determine the `array namespace` from the reference array."
        ) from e


# initialize the context variable for array specification
_arr_spec_var = ContextVar("arr_spec")


def context_array_spec() -> ArraySpec:
    """
    Get the `array namespace`, `dtype` and `device` associated with the most recent :func:`unify_array_args`-decorated function call in the current context.
    If there is no such function call in the current context, return the default `array namespace`, `dtype` and `device` from :func:`default.default_array_spec`.

    Returns
    -------
        ArraySpec
            An :class:`ArraySpec` named tuple containing the determined `xp`(`array namespace`), `dtype` and `device`.

    Raises
    ------
        Refer to :func:`default.default_array_spec` for possible exceptions.
    """
    try:
        return _arr_spec_var.get()
    except LookupError:
        return default_array_spec()


def context_namespace() -> Namespace:
    """
    Get the `array namespace` associated with the most recent :func:`unify_array_args`-decorated function call in the current context.
    If there is no such function call in the current context, return the default `array namespace` from :func:`default.default_array_spec`.

    Returns
    -------
        Namespace
            The `array namespace`.

    Raises
    ------
        Refer to :func:`default.default_array_spec` for possible exceptions.
    """
    return context_array_spec().xp


def as_context_array(
    obj: object,
    /,
    unify_dtype: bool = True,
    unify_device: bool = True,
    copy: bool = False
) -> Any:
    """
    Convert the given object to an array in the current context `array namespace`, with the `dtype` and `device` unified to the current context if specified.

    Parameters
    ----------
        obj : object
            The object to be converted to an array.

        unify_dtype : bool, default to `True`
            Whether to unify the `dtype` of the converted array to that of the current context.

        unify_device : bool, default to `True`
            Whether to unify the `device` of the converted array to that of the current context.

        copy : bool, default to `False`
            Whether to return a copy of the array if it is already in the current context namespace.

    Returns
    -------
        Any
        The converted array representation of the object in the current context `array namespace`, with the `dtype` and `device` unified to the current context if specified.

    Raises
    ------
        Refer to :func:`convert.as_array`, :func:`context_array_spec` for possible exceptions.
    """
    arr_spec = context_array_spec()
    return as_array(
        obj, arr_spec.xp,
        arr_spec.dtype if unify_dtype else None,
        arr_spec.device if unify_device else None,
        copy
    )


@overload
def as_context_array_if_like(
    obj: ArrayLike,
    /,
    unify_dtype: bool = ...,
    unify_device: bool = ...,
    copy: bool = ...
) -> Any: ...
@overload
def as_context_array_if_like(obj: T, /) -> T: ...


def as_context_array_if_like(
    obj: object,
    /,
    unify_dtype: bool = True,
    unify_device: bool = True,
    copy: bool = False
) -> Any:
    """
    Convert an array-like object to an array in the current context `array namespace`, with the `dtype` and `device` unified to the current context if specified. Otherwise return the object itself.

    Returns
    -------
        Any
        The converted array representation of the object in the current context `array namespace` if it is array-like, with the `dtype` and `device` unified to the current context if specified; otherwise, return the object itself.

    Raises
    ------
        Refer to :func:`as_context_array` for possible exceptions.

    Notes
    -----
    - All parameters follow the usage conventions of :func:`as_context_array`.
    """
    if api.is_array_api_obj(obj):
        return as_context_array(obj, unify_dtype=unify_dtype, unify_device=unify_device, copy=copy)
    return obj


class array_context:
    """
    **Context Manager** to set the context `array namespace`, `dtype` and `device` for the enclosed block of code.
    """
    @classmethod
    def from_array_spec(cls, arr_spec: ArraySpec, /):
        """
        Create an :class:`array_context` from an :class:`ArraySpec` named tuple.

        Parameters
        ----------
            arr_spec : ArraySpec
                An :class:`ArraySpec` named tuple containing the `xp`(`array namespace`), `dtype` and `device`.
        """
        return cls(xp=arr_spec.xp, dtype=arr_spec.dtype, device=arr_spec.device)

    def __init__(
        self,
        xp: Optional[Union[Namespace, ArrayLibraryName]] = None,
        dtype: Optional[DTypeT] = None,
        device: Optional[DeviceT] = None
    ):
        """
        Initialize the context manager with the specified `array namespace`, `dtype` and `device`.

        Parameters
        ----------
            xp : Optional[Union[Namespace, ArrayLibraryName]], default to `None`
                The target `array namespace` or array library name for the context.
                - `None`: Use the `array namespace` from the context;
                - _others_: See also :param:`xp` in :func:`convert.as_array`.

            dtype : Optional[DTypeT], default to `None`
                See also :param:`dtype` in :func:`convert.as_array`.

            device : Optional[DeviceT], default to `None`
                See also :param:`device` in :func:`convert.as_array`.
        """
        self.xp = to_xp(xp) if xp is not None else context_array_spec().xp
        self.dtype = dtype
        self.device = device

    def __enter__(self):
        """Set the context variable to the specified namespace, dtype and device when entering the context."""
        self._token = _arr_spec_var.set(ArraySpec(
            xp=self.xp,
            dtype=self.dtype,
            device=self.device
        ))
        return self

    def __exit__(self, *args):
        """Reset the context variable to its previous value when exiting the context."""
        _arr_spec_var.reset(self._token)


def unify_array_args(
    ref: Optional[Union[str, int]] = 0,
    filter_array_like: bool = True,
    api_version: Optional[str] = None,
    use_compat: Optional[bool] = None,
    unify_dtype: bool = False,
    unify_device: bool = True
):
    """
    **Decorator** to unify array arguments of a function to the same `array namespace`, `dtype` and `device` determined by the provided array arguments and the reference array.

    Parameters
    ----------
        ref : Optional[Union[str, int]], default to `None`
            Reference array to determine the `array namespace`, `dtype` and `device`.
            See also :param:`ref` in :func:`array_spec`.

        filter_array_like : bool, default to `False`
            Whether to filter the provided inputs to all array-likes via :func:`array_api_compat.is_array_api_obj` when determining the `array namespace`.

        api_version : Optional[str], default to `None`
            The target array API version for the returned `array namespace`. See also :param:`api_version` in :func:`array_api_compat.array_namespace`.

        use_compat : Optional[bool], default to `None`
            See also :param:`use_compat` in :func:`array_api_compat.array_namespace`.
            - `None`: Return the native namespace if it is already Array API–compatible, otherwise return a compat wrapper;
            - `True`: Always return the compat-wrapped namespace;
            - `False`: Return the native namespace.

        unify_dtype : bool, default to `False`
            Whether to unify the `dtype` of all array arguments to that of the reference array.

        unify_device : bool, default to `True`
            Whether to unify the `device` of all array arguments to that of the reference array.

    Raises
    ------
        Refer to :func:`default.default_array_spec`, :func:`as_context_array_if_like` for possible exceptions.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # determine the namespace, dtype and device for the array inputs
            try:
                arr_spec = array_spec(
                    *args, kw_arrays=kwargs, ref=ref,
                    filter_array_like=filter_array_like,
                    api_version=api_version,
                    use_compat=use_compat
                )
            except Exception:
                try:
                    # fall back to the default namespace
                    arr_spec = default_array_spec()
                except MissingDependencyError:
                    # just run the function without conversion
                    return func(*args, **kwargs)

            with array_context.from_array_spec(arr_spec):
                out_args = tuple(
                    as_context_array_if_like(a, unify_dtype, unify_device, False) for a in args
                )
                out_kwargs = {
                    k: as_context_array_if_like(v, unify_dtype, unify_device, False)
                    for k, v in kwargs.items()
                }
                return func(*out_args, **out_kwargs)
        return wrapper
    return decorator
