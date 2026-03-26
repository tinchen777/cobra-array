# -*- coding: utf-8 -*-
# Python version: 3.9
# @TianZhen

from __future__ import annotations
from array_api_compat.common._typing import Namespace

from .._utils import array_namespace_alias


class NameSpace(Namespace):
    def __init__(self, xp, /):
        self._xp_name = array_namespace_alias(xp)
        self._xp = xp

    def __getattr__(self, name):
        try:
            return getattr(self._xp, name)
        except AttributeError:
            raise AttributeError(
                f"Namespace `{self._xp_name}` have no attribute `{name}`."
            ) from None
