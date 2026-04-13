# -*- coding: utf-8 -*-
# Python version: 3.9
# @TianZhen

from __future__ import annotations
from typing import (Any, TYPE_CHECKING, Callable)

from .._utils import array_namespace_alias

if TYPE_CHECKING:
    from array_api_compat.common._typing import Namespace


class Compat:
    """
    A base class for creating compatibility wrappers for :class:`CompatNamespace` and :class:`CompatArray`.
    """
    _xp: Namespace
    _xp_name: str

    _UNWRAP_COMPAT: bool = True

    def __new__(cls, xp: Any, /):
        obj = super().__new__(cls)
        obj._xp_name = array_namespace_alias(xp)
        obj._xp = xp

        return obj

    def _get_xp_attr(self, name: str):
        """Try to get the attribute `name` from the `array namespace`."""
        try:
            return getattr(self._xp, name)
        except AttributeError:
            raise AttributeError(f"Namespace `{self._xp_name}` of `{self.__class__.__name__}` has no attribute `{name}`.") from None

    @property
    def xp(self) -> Namespace:
        """
        The `array namespace`.
        """
        return self._xp

    @property
    def xp_name(self) -> str:
        """
        The alias of the `array namespace`.
        """
        return self._xp_name

    def __array_namespace__(self, *, api_version=None) -> Any:
        """Returns an object that has all the array API functions on it."""
        raise NotImplementedError(f"`__array_namespace__()` is not implemented for `{self.__class__.__name__}`.")
