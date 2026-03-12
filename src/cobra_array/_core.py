# -*- coding: utf-8 -*-
# Python version: 3.9
# @TianZhen

from __future__ import annotations
from functools import wraps
from itertools import (islice, chain)
import array_api_compat as api
from array_api_compat.common._typing import Namespace
from contextvars import ContextVar
from typing import (Union, Optional, Dict, Tuple, Any)

from .convert import (as_array, as_array_if_like, np, torch)
from .exceptions import (
    NoArrayInputsError,
    GetArrayNamespaceError,
    MissingDependencyError,
    NotArrayAPIObjectError
)


def array_spec(
    *arrays: object,
    kw_arrays: Optional[Dict[str, object]] = None,
    ref: Optional[Union[str, int]] = None,
    filter_array_like: bool = False,
    api_version: Optional[str] = None,
    use_compat: Optional[bool] = None
) -> Tuple[Namespace, Tuple[Any, Any]]:
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
        Tuple[Namespace, Tuple[Optional[DTypeT], Optional[DeviceT]]
            A tuple containing the determined `array namespace` and a tuple of the `dtype` and `device` of the reference array (or `None` if `ref` is `None`).

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
            return api.array_namespace(
                *all_arrays,
                api_version=api_version,
                use_compat=use_compat
            ), (None, None)
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
        return api.array_namespace(
            ref_arr,
            api_version=api_version,
            use_compat=use_compat
        ), (dtype, device)
    except Exception as e:
        raise GetArrayNamespaceError(
            "Failed to determine the `array namespace` from the reference array."
        ) from e


# get defaults
DEFAULT_DTYPE = float
DEFAULT_DEVICE = "cpu"
DEFAULT_TORCH_NAMESPACE = api.array_namespace(torch.empty(0)) if torch is not None else None
DEFAULT_NUMPY_NAMESPACE = api.array_namespace(np.empty(0)) if np is not None else None


def default_array_spec() -> Tuple[Namespace, Tuple[Any, Any]]:
    """
    Try to get a suitable `array namespace` from the available array libraries in order of `PyTorch` > `NumPy`, and return it along with the default `dtype` and `device`.

    Returns
    -------
        Tuple[Namespace, Tuple[DTypeT, DeviceT]
            A tuple containing the default `array namespace` and a tuple of the default `dtype` and default `device`.

    Raises
    ------
        MissingDependencyError
            If all default array libraries (`NumPy` and `PyTorch`) are missing.
    """
    default_xp = DEFAULT_TORCH_NAMESPACE or DEFAULT_NUMPY_NAMESPACE
    if default_xp is None:
        raise MissingDependencyError(
            "Missing all default array libraries (`PyTorch` > `NumPy`)."
        )
    return default_xp, (DEFAULT_DTYPE, DEFAULT_DEVICE)


# initial namespace context variable
_xp_var = ContextVar("xp")


def context_array_spec() -> Tuple[Namespace, Tuple[Any, Any]]:
    """
    Get the `array namespace`, `dtype` and `device` associated with the most recent :func:`unify_array_args`-decorated function call in the current context.
    If there is no such function call in the current context, return the default `array namespace`, `dtype` and `device` from :func:`default_array_spec`.

    Returns
    -------
        Tuple[Namespace, Tuple[DTypeT, DeviceT]
            A tuple containing the `array namespace` and a tuple of the `dtype` and `device`.

    Raises
    ------
        Refer to :func:`default_array_spec` for possible exceptions.
    """
    try:
        return _xp_var.get()
    except LookupError:
        return default_array_spec()


def context_namespace() -> Namespace:
    """
    Get the `array namespace` associated with the most recent :func:`unify_array_args`-decorated function call in the current context.
    If there is no such function call in the current context, return the default `array namespace` from :func:`default_array_spec`.

    Returns
    -------
        Namespace
            The `array namespace`.

    Raises
    ------
        Refer to :func:`default_array_spec` for possible exceptions.
    """
    return context_array_spec()[0]


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
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # determine the namespace, dtype and device for the array inputs
            try:
                xp, (dtype, device) = array_spec(
                    *args, kw_arrays=kwargs, ref=ref,
                    filter_array_like=filter_array_like,
                    api_version=api_version,
                    use_compat=use_compat
                )
            except Exception:
                try:
                    # fall back to the default namespace
                    xp, (dtype, device) = default_array_spec()
                except MissingDependencyError:
                    # just run the function without conversion
                    return func(*args, **kwargs)
            # set the determined namespace, dtype and device in the context variable
            token = _xp_var.set((xp, (dtype, device)))

            dtype = dtype if unify_dtype else None
            device = device if unify_device else None

            out_args = tuple(
                as_array_if_like(a, xp, dtype, device, False) for a in args
            )
            out_kwargs = {
                k: as_array_if_like(v, xp, dtype, device, False)
                for k, v in kwargs.items()
            }
            # call the original function with the converted array arguments
            try:
                return func(*out_args, **out_kwargs)
            finally:
                # reset the context variable to its previous value
                _xp_var.reset(token)
        return wrapper
    return decorator


def as_context_array(
    obj: object,
    /,
    unify_dtype: bool = True,
    unify_device: bool = True,
    copy: bool = False
):
    """
    Convert an array-like object to an array in the current context `array namespace`, with the `dtype` and `device` unified to the current context if specified.

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
        ArrayLike
        The converted array representation of the object in the current context `array namespace`, with the `dtype` and `device` unified to the current context if specified.

    Raises
    ------
        Refer to :func:`as_array` for possible exceptions.
    """
    xp, (dtype, device) = context_array_spec()
    return as_array(
        obj, xp,
        dtype if unify_dtype else None,
        device if unify_device else None,
        copy
    )
