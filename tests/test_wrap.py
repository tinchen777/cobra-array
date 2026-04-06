import sys
from typing import Any, cast

import numpy as np
import pytest

sys.path.insert(0, "/data/tianzhen/my_packages/cobra-array/src")

import cobra_array._utils as utils
from cobra_array._utils import array_namespace_alias, is_array_namespace, is_compat_namespace
from cobra_array.compat import CompatArray, CompatNamespace, unwrap, wrap_arraylike
from cobra_array.exceptions import NotArrayAPIObjectError, UnsupportedNamespaceError


def test_array_namespace_alias_numpy():
    assert array_namespace_alias(np) == "NumPy"


def test_array_namespace_alias_invalid_raises():
    with pytest.raises(UnsupportedNamespaceError):
        array_namespace_alias(object())


def test_is_array_namespace_true_and_false():
    assert is_array_namespace(np) is True
    assert is_array_namespace(object()) is False


def test_is_compat_namespace_true_and_false():
    cxp = CompatNamespace(np)
    assert is_compat_namespace(cxp) is True
    assert is_compat_namespace(np) is False


def test_warn_falls_back_to_python_warning(monkeypatch):
    monkeypatch.setattr(utils, "_WARN_AVAILABLE", False)
    with pytest.warns(UserWarning):
        utils.warn("hello", category=UserWarning)


def test_unwrap_and_wrap_arraylike():
    arr = np.array([1, 2, 3])
    wrapped = wrap_arraylike(arr)
    assert isinstance(wrapped, CompatArray)
    assert unwrap(wrapped) is arr
    assert wrap_arraylike("x") == "x"
    assert unwrap("x") == "x"


def test_compatnamespace_wraps_namespace():
    cxp = CompatNamespace(np)
    assert isinstance(cxp, CompatNamespace)
    assert cxp.xp is np


def test_compatarray_invalid_init_raises():
    with pytest.raises(NotArrayAPIObjectError):
        CompatArray(cast(Any, "not-an-array"))
