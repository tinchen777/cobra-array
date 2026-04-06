import sys

import numpy as np
import pytest
import array_api_compat as api

sys.path.insert(0, "/data/tianzhen/my_packages/cobra-array/src")

import cobra_array.convert as convert
import cobra_array._utils as utils
from cobra_array.array_api import torch_xp
from cobra_array.convert import as_array, to_list, to_numpy, to_tensor, to_xp
from cobra_array.exceptions import (
    ArrayConversionError,
    ConvertNoneTypeError,
    NumPyConversionError,
    ParameterIgnoredWarning,
    UnsupportedArrayLibraryNameError,
)


def test_to_numpy_from_list_and_dtype():
    arr = to_numpy([[1, 2], [3, 4]], dtype=np.float32)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32
    assert arr.shape == (2, 2)


def test_to_numpy_from_set():
    arr = to_numpy({1, 2, 3}, copy=True)
    assert isinstance(arr, np.ndarray)
    assert set(arr.tolist()) == {1, 2, 3}


def test_to_numpy_invalid_dtype_raises():
    with pytest.raises(NumPyConversionError):
        to_numpy([1, 2, 3], dtype="not-a-valid-dtype")


def test_to_tensor_none_raises():
    with pytest.raises(ConvertNoneTypeError):
        to_tensor(None)


@pytest.mark.skipif(torch_xp is None, reason="PyTorch not available")
def test_to_tensor_from_list_cpu():
    t = to_tensor([[1, 2], [3, 4]], device="cpu", copy=False)
    assert isinstance(t, torch_xp.Tensor)
    assert tuple(t.shape) == (2, 2)


def test_to_list_for_iterable_and_scalar_and_copy_behavior():
    src = [1, 2, 3]
    out_copy = to_list(src, copy=True)
    out_ref = to_list(src, copy=False)
    assert out_copy == [1, 2, 3]
    assert out_ref == [1, 2, 3]
    assert out_copy is not src
    assert out_ref is src
    assert to_list(7) == [7]


def test_to_list_none_raises():
    with pytest.raises(ConvertNoneTypeError):
        to_list(None)


def test_to_xp_and_alias_function():
    xp = to_xp("numpy")
    assert api.is_numpy_namespace(xp)
    alias_func = getattr(convert, "to_array_namespace")
    assert api.is_numpy_namespace(alias_func("numpy"))


def test_to_xp_unsupported_name_raises():
    with pytest.raises(UnsupportedArrayLibraryNameError):
        to_xp("invalid-lib")


def test_as_array_arraylike_only_passthrough_non_array():
    obj = 123
    assert as_array(obj, "numpy", arraylike_only=True) == obj


def test_as_array_numpy_ignores_non_cpu_device_warning(monkeypatch):
    monkeypatch.setattr(utils, "_WARN_AVAILABLE", False)
    with pytest.warns(ParameterIgnoredWarning):
        arr = as_array([1, 2, 3], "numpy", device="cuda:0")
    assert isinstance(arr, np.ndarray)


def test_as_array_wraps_namespace_exception(monkeypatch):
    class DummyNamespace:
        __name__ = "dummy"

        def asarray(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(convert, "to_xp", lambda _: DummyNamespace())
    with pytest.raises(ArrayConversionError):
        convert.as_array([1, 2, 3], "numpy")
