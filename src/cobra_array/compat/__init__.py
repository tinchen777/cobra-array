# src/cobra_array/compat/__init__.py
"""
Compatibility utilities for :pkg:`cobra_color`.

Functions
---------
- :func:`wrap_arraylike`: Wraps an array-like object in a :class:`CompatArray` array if it is an array API object.
- :func:`unwrap`: Unwraps a :class:`CompatArray` array to get the backend-specific array instance, or returns the object itself if it is not a :class:`CompatArray` array.
Classes
-------
- :class:`CompatArray`: A backend-agnostic array abstraction compliant with the `Python Array API standard`.
- :class::class:`CompatNamespace`: A wrapper around an `array namespace` providing a unified, backend-agnostic functional interface.
"""

from ._array import (CompatArray, wrap_arraylike, unwrap)
from ._namespace import CompatNamespace

__all__ = [
    "CompatArray",
    "wrap_arraylike",
    "unwrap",
    "CompatNamespace",
]
