# -*- coding: utf-8 -*-
# Python version: 3.9
# @TianZhen

from __future__ import annotations
from functools import wraps
from itertools import (islice, chain)
import array_api_compat as api
from contextvars import ContextVar
from typing import (Any, Literal, Union, Optional, Dict, overload, TYPE_CHECKING)

from .compat import wrap_arraylike
from .convert import (to_xp, as_array)
from .default import (ArraySpec, default_spec)
from .exceptions import (
    NoArrayInputsError,
    GetArrayNamespaceError,
    MissingDependencyError,
    NotArrayAPIObjectError
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from .compat import CompatArray
    from .types import (T, dtypeT, DType, AnyDevice, ArrayLike, ArrayLibraryName)


def array_spec(
    *arrays: object,
    kw_arrays: Optional[Dict[str, object]] = None,
    ref: Optional[Union[str, int]] = None,
    filter_arraylike: bool = False,
    api_version: Optional[str] = None,
    use_compat: Optional[bool] = None
) -> ArraySpec:
    """
    Determine the array API compatible `compatibility namespace`, `dtype` and `device` from the provided array arguments and the reference array.

    Parameters
    ----------
        arrays : object
            Positional array-like objects to determine the `compatibility namespace`.

        kw_arrays : Optional[Dict[str, object]], default to `None`
            Keyword array-like objects to determine the `compatibility namespace`.

        ref : Optional[Union[str, int]], default to `None`
            Reference array to determine the `compatibility namespace`, `dtype` and `device`.
            - `None`: Use all provided arrays in `arrays` and `kw_arrays` to determine the `compatibility namespace`, and return `None` for `dtype` and `device`;
            - _str_: Use the specified keyword array in `kw_arrays` as reference array to determine the `compatibility namespace`, `dtype` and `device`;
            - _int_: Use the specified positional array in `arrays` and `kw_arrays` (in order) as reference array to determine the `compatibility namespace`, `dtype` and `device`. The index and valid range are affected by :param:`filter_arraylike`.

        filter_arraylike : bool, default to `False`
            Whether to filter the provided inputs to all array-likes via :func:`array_api_compat.is_array_api_obj` when determining the `compatibility namespace`.

        api_version : Optional[str], default to `None`
            The target array API version for the returned `compatibility namespace`. See also :param:`api_version` in :func:`array_api_compat.array_namespace`.

        use_compat : Optional[bool], default to `None`
            See also :param:`use_compat` in :func:`array_api_compat.array_namespace`.
            - `None`: Return the native namespace if it is already Array API–compatible, otherwise return a compat wrapper;
            - `True`: Always return the compat-wrapped namespace;
            - `False`: Return the native namespace.

            NOTE: The compat-wrapped namespace is NOT `compatibility namespace`. The former is a wrapper in :pkg:`array_api_compat`.

    Returns
    -------
        ArraySpec
            An :class:`ArraySpec` named tuple containing the determined `cxp`(`compatibility namespace`), `dtype` and `device`.
            - If :param:`ref` is `None`, the returned `dtype` and `device` will be `None`.
            - `compatibility namespace` is a wrapper of the native `array namespace` provides a compatibility layer for backend-agnostic array operations. See also :class:`compat.CompatNamespace`.

    Raises
    ------
        NoArrayInputsError
            If no array inputs are provided in `arrays` and `kw_arrays`.
        GetArrayNamespaceError
            If an error occurs while determining the `compatibility namespace` from the provided inputs or the reference array.
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
    if filter_arraylike:
        all_arrays = (a for a in all_arrays if api.is_array_api_obj(a))

    if ref is None:
        # use arrays and kw_arrays in order to determine the namespace
        try:
            return ArraySpec.create(
                api.array_namespace(
                    *all_arrays,
                    api_version=api_version,
                    use_compat=use_compat
                ), None, None
            )
        except Exception as e:
            raise GetArrayNamespaceError(
                "Failed to determine the `compatibility namespace` from the provided inputs."
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
            if filter_arraylike:
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

    if not filter_arraylike and not api.is_array_api_obj(ref_arr):
        raise NotArrayAPIObjectError(
            f"Reference array must be an array API compatible array object, got {ref_arr!r}."
        )

    dtype = getattr(ref_arr, "dtype", None)
    device = api.device(ref_arr)
    try:
        return ArraySpec.create(
            api.array_namespace(
                ref_arr,
                api_version=api_version,
                use_compat=use_compat
            ), dtype, device
        )
    except Exception as e:
        raise GetArrayNamespaceError(
            "Failed to determine the `compatibility namespace` from the reference array."
        ) from e


# initialize the context variable for array specification
_arr_spec_var = ContextVar("arr_spec")


def context_spec() -> ArraySpec:
    """
    Get the `compatibility namespace`, `dtype` and `device` associated with the most recent :func:`unify_array_args`-decorated function call in the current context.
    If there is no such function call in the current context, return the default `compatibility namespace`, `dtype` and `device` from :func:`default.default_spec`.

    Returns
    -------
        ArraySpec
            An :class:`ArraySpec` named tuple containing the determined `cxp`(`compatibility namespace`), `dtype` and `device`.

    Raises
    ------
        Refer to :func:`default.default_spec` for possible exceptions.
    """
    try:
        return _arr_spec_var.get()
    except LookupError:
        return default_spec()


@overload
def as_context(obj: NDArray[dtypeT], /, *, unify_dtype: Literal[False], unify_device: bool = ..., copy: bool = ..., arraylike_only: bool = ...) -> CompatArray[dtypeT, Literal["cpu"]]: ...
@overload
def as_context(obj: ArrayLike[dtypeT], /, *, unify_dtype: Literal[False], unify_device: bool = ..., copy: bool = ..., arraylike_only: bool = ...) -> CompatArray[dtypeT, AnyDevice]: ...
@overload
def as_context(obj: ArrayLike[Any], /, *, unify_dtype: Literal[True] = ..., unify_device: bool = ..., copy: bool = ..., arraylike_only: bool = ...) -> CompatArray[Any, AnyDevice]: ...
@overload
def as_context(obj: object, /, *, unify_dtype: bool = ..., unify_device: bool = ..., copy: bool = ..., arraylike_only: Literal[False] = ...) -> CompatArray[Any, AnyDevice]: ...
@overload
def as_context(obj: T, /, *, unify_dtype: bool = ..., unify_device: bool = ..., copy: bool = ..., arraylike_only: Literal[True]) -> T: ...


def as_context(
    obj: object,
    /,
    unify_dtype: bool = True,
    unify_device: bool = True,
    copy: bool = False,
    arraylike_only: bool = False
) -> Any:
    """
    Convert the given object to a :class:`CompatArray` array in the current context `compatibility namespace`, with the `dtype` and `device` unified to the current context if specified.

    Parameters
    ----------
        obj : object
            The object to be converted to a :class:`CompatArray` array.

        unify_dtype : bool, default to `True`
            Whether to unify the `dtype` of the converted array to that of the current context.

        unify_device : bool, default to `True`
            Whether to unify the `device` of the converted array to that of the current context.

        copy : bool, default to `False`
            Whether to return a copy of the array if it is already in the current context namespace.

        arraylike_only : bool, default to `False`
            Whether to only convert array-like objects to arrays in the current context namespace, and return the object itself if it is not array-like.

    Returns
    -------
        CompatArray[Any, AnyDevice]
            The converted array representation of the object in the current context `compatibility namespace`, with the current context `dtype` and `device` if specified.
        object
            If :param:`arraylike_only` is `True` and the object is not array-like.

    Raises
    ------
        Refer to :func:`convert.as_array`, :func:`context_spec` for possible exceptions.
    """
    spec = context_spec()
    return wrap_arraylike(as_array(
        obj, spec.cxp,
        dtype=spec.dtype if unify_dtype else None,
        device=spec.device if unify_device else None,
        copy=copy,
        arraylike_only=arraylike_only
    ), xp=spec.cxp)


class array_context:
    """
    **Context Manager** to set the context `compatibility namespace`, `dtype` and `device` for the enclosed block of code.
    """
    @classmethod
    def from_array_spec(cls, arr_spec: ArraySpec, /):
        """
        Create an :class:`array_context` from an :class:`ArraySpec` named tuple.

        Parameters
        ----------
            arr_spec : ArraySpec
                An :class:`ArraySpec` named tuple containing the `cxp`(`compatibility namespace`), `dtype` and `device`.
        """
        return cls(xp=arr_spec.cxp, dtype=arr_spec.dtype, device=arr_spec.device)

    def __init__(
        self,
        xp: Optional[Union[object, ArrayLibraryName]] = None,
        dtype: Optional[DType] = None,
        device: Optional[AnyDevice] = None
    ):
        """
        Initialize the context manager with the specified `array namespace`, `dtype` and `device`.

        Parameters
        ----------
            xp : Optional[Union[object, ArrayLibraryName]], default to `None`
                The target `array namespace` or array library name for the context.
                - `None`: Use the `compatibility namespace` from the context;

            dtype : Optional[DType], default to `None`
                The target `dtype` for the context.
                - `None`: Use the `dtype` from the context.

            device : Optional[AnyDevice], default to `None`
                The target `device` for the context.
                - `None`: Use the `device` from the context.
        """
        spec = context_spec()
        self.cur_spec = ArraySpec.create(
            to_xp(xp) if xp is not None else spec.cxp,
            dtype if dtype is not None else spec.dtype,
            device if device is not None else spec.device
        )

    def __enter__(self):
        """Set the context variable to the specified namespace, dtype and device when entering the context."""
        self._token = _arr_spec_var.set(self.cur_spec)
        return self.cur_spec

    def __exit__(self, *args):
        """Reset the context variable to its previous value when exiting the context."""
        _arr_spec_var.reset(self._token)


def unify_args(
    ref: Optional[Union[str, int]] = 0,
    /,
    filter_arraylike: bool = True,
    api_version: Optional[str] = None,
    use_compat: Optional[bool] = None,
    unify_dtype: bool = False,
    unify_device: bool = True,
    arraylike_only: bool = True,
    strict: bool = True
):
    """
    **Decorator** to unify arguments of a function to the same `compatibility namespace`, `dtype` and `device` determined by the provided array arguments and the reference array.

    Parameters
    ----------
        ref : Optional[Union[str, int]], default to `None`
            Reference array to determine the `compatibility namespace`, `dtype` and `device`.
            See also :param:`ref` in :func:`array_spec`.

        filter_arraylike : bool, default to `False`
            Whether to filter the provided inputs to all array-likes via :func:`array_api_compat.is_array_api_obj` when determining the `compatibility namespace`.

        api_version : Optional[str], default to `None`
            See also :param:`api_version` in :func:`array_spec`.

        use_compat : Optional[bool], default to `None`
            See also :param:`use_compat` in :func:`array_spec`.

        unify_dtype : bool, default to `False`
            Whether to unify the `dtype` of arguments to that of the reference array.

        unify_device : bool, default to `True`
            Whether to unify the `device` of arguments to that of the reference array.

        arraylike_only : bool, default to `True`
            Whether to only convert array-like objects to arrays in the determined namespace, and return the object itself if it is not array-like.

        strict : bool, default to `True`
            Whether to raise exceptions when failing to determine the `compatibility namespace`.
            - `True`: Raise exceptions when failing to determine the `compatibility namespace` from the provided inputs or the reference array;
            - `False`: Fall back to the default `compatibility namespace` if an error occurs. If all default array libraries are missing, just run the function without conversion.

    Raises
    ------
        Refer to :func:`default.default_spec`, :func:`as_context` for possible exceptions.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # determine the namespace, dtype and device for the array inputs
            try:
                spec = array_spec(
                    *args, kw_arrays=kwargs, ref=ref,
                    filter_arraylike=filter_arraylike,
                    api_version=api_version,
                    use_compat=use_compat
                )
            except Exception as e:
                if strict:
                    raise e
                try:
                    # fall back to the default namespace
                    spec = default_spec()
                except MissingDependencyError:
                    # just run the function without conversion
                    return func(*args, **kwargs)

            with array_context.from_array_spec(spec):
                out_args = tuple(
                    as_context(a, unify_dtype=unify_dtype, unify_device=unify_device, arraylike_only=arraylike_only) for a in args
                )
                out_kwargs = {
                    k: as_context(v, unify_dtype=unify_dtype, unify_device=unify_device, arraylike_only=arraylike_only)
                    for k, v in kwargs.items()
                }
                return func(*out_args, **out_kwargs)
        return wrapper
    return decorator
