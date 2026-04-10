# src/cobra_array/__init__.py
"""
cobra-array
===========

A unified array interface for multiple array libraries, providing seamless interoperability and convenient utilities for array manipulation and conversion.
A backend-agnostic array utility library that unifies array conversion, context control, and cross-library operations across NumPy/PyTorch-style ecosystems.

Modules
-------
- :mod:`cobra_array.compat`: Compatibility utilities for `compatibility namespaces` and `compatibility arrays`.
- :mod:`cobra_array.convert`: Utilities for converting between different array types and namespaces.
- :mod:`cobra_array.default`: Default utilities for array specifications and namespaces.
- :mod:`cobra_array.array_api`: Utilities for working with array namespaces and devices.

Functions
---------
- :func:`array_spec`: Get the array specification of an object.
- :func:`context_spec`: Get the context specification of an object.
- :func:`as_context`: Convert an array-like object to an array in the determined context namespace.
- :func:`array_context`: Get the context namespace for a given array or set of arrays.
- :func:`unify_args`: A decorator to unify the array arguments of a function to the same namespace, dtype, and device.
- :func:`array_namespace_alias`: Get the alias of an array namespace if it exists.
- :func:`is_compat_namespace`: Check if an object is a `compatibility namespace`.
- :func:`is_array_namespace`: Check if an object is an `array namespace`.

Examples
--------
- Basic conversions::

    import numpy as np
    from cobra_array.convert import to_numpy, to_tensor, to_list

    data = [[1, 2], [3, 4]]

    arr_np = to_numpy(data, dtype=np.float32)  # numpy.ndarray float32

    arr_torch = to_tensor(data, device="cpu")

    back_to_list = to_list(arr_np)  # [[1.0, 2.0], [3.0, 4.0]]

- Context-based conversion::

    import numpy as np
    from cobra_array import array_context, as_context, context_spec

    with array_context(xp="numpy", dtype=np.float32, device="cpu"):
        x = as_context([1, 2, 3])
        y = as_context(np.array([4, 5]))
        spec = context_spec()

- Auto-unify function arguments::

    import numpy as np
    from cobra_array import unify_args

    @unify_args(ref=0, unify_dtype=True, unify_device=True, arraylike_only=True)
    def add_and_mean(a, b):
        c = a + b
        return c.mean()

    out = add_and_mean(np.array([1, 2, 3]), [4, 5, 6])

- Default backend strategy::

    from cobra_array.default import as_default, default_spec

    spec = default_spec()

    x = as_default([1, 2, 3], unify_dtype=True, unify_device=True)
"""

from ._core import (
    array_spec,
    context_spec,
    as_context,
    array_context,
    unify_args
)
from ._utils import (
    array_namespace_alias,
    is_compat_namespace,
    is_array_namespace
)

__author__ = "Zhen Tian"
__version__ = "0.1.4"

__all__ = [
    "array_spec",
    "context_spec",
    "as_context",
    "array_context",
    "unify_args",
    "array_namespace_alias",
    "is_compat_namespace",
    "is_array_namespace"
]
