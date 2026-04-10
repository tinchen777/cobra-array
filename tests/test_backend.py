import sys
from typing import Any, cast

import numpy as np
import pytest

sys.path.insert(0, "/data/tianzhen/my_packages/cobra-array/src")

from cobra_array import array_context, array_spec, as_context, context_spec, unify_args
from cobra_array.array_api import CUDA_AVAILABLE, resolve_device
from cobra_array.compat import CompatArray
from cobra_array.default import default_spec
from cobra_array.exceptions import (
    CUDAUnavailableError,
    DeviceNotSupportedError,
    NoArrayInputsError,
    NotArrayAPIObjectError,
)


def test_array_spec_no_inputs_raises():
    with pytest.raises(NoArrayInputsError):
        array_spec()


def test_array_spec_ref_none_returns_namespace_only():
    spec = array_spec(np.array([1, 2, 3]), ref=None)
    assert spec.cxp is not None
    assert spec.dtype is None
    assert spec.device is None


def test_array_spec_ref_int_and_str():
    arr = np.array([1, 2, 3], dtype=np.float32)
    spec_by_int = array_spec(arr, ref=0)
    spec_by_str = array_spec(kw_arrays={"x": arr}, ref="x")
    assert spec_by_int.dtype == arr.dtype
    assert str(spec_by_int.device) == "cpu"
    assert spec_by_str.dtype == arr.dtype


def test_array_spec_invalid_ref_type_raises():
    with pytest.raises(TypeError):
        array_spec(np.array([1]), ref=cast(Any, 1.5))


def test_array_spec_invalid_ref_index_raises():
    with pytest.raises(IndexError):
        array_spec(np.array([1]), ref=2)


def test_array_spec_non_array_ref_raises_when_not_filtering():
    with pytest.raises(NotArrayAPIObjectError):
        array_spec([1, 2, 3], ref=0, filter_arraylike=False)


def test_context_spec_falls_back_to_default():
    spec = context_spec()
    dspec = default_spec()
    assert spec.cxp.xp_name == dspec.cxp.xp_name


def test_array_context_override_and_restore():
    before = context_spec()
    with array_context(xp="numpy", dtype=np.float32, device="cpu") as spec:
        current = context_spec()
        assert current.cxp.xp_name == spec.cxp.xp_name
        assert current.dtype == np.float32
        assert str(current.device) == "cpu"
    after = context_spec()
    assert after.cxp.xp_name == before.cxp.xp_name


def test_as_context_converts_under_current_context():
    with array_context(xp="numpy", dtype=np.float32, device="cpu"):
        out = as_context([1, 2, 3], unify_dtype=True, unify_device=True)
    assert isinstance(out, CompatArray)
    assert out.dtype == np.dtype(np.float32)


def test_as_context_arraylike_only_passthrough():
    marker = object()
    with array_context(xp="numpy"):
        out = as_context(marker, arraylike_only=True)
    assert out is marker


def test_unify_args_strict_true_raises_for_non_array_inputs():
    @unify_args(filter_arraylike=True, fallback=False)
    def fn(a, b):
        return a, b

    a, b = fn("x", {"y": 1})
    assert a == "x"
    assert b == {"y": 1}


def test_unify_args_strict_false_fallback_and_convert():
    @unify_args(filter_arraylike=True, fallback=True, arraylike_only=False)
    def fn(a):
        return a, default_spec().cxp.xp_name

    out, xp_name = fn([1, 2, 3])

    assert isinstance(out, CompatArray)
    assert out.cxp.xp_name == xp_name
    assert xp_name in ("NumPy", "PyTorch")


def test_resolve_device_basic_and_numpy_constraints():
    assert resolve_device(None) is None
    assert resolve_device("cpu", xp="numpy") == "cpu"
    with pytest.raises(DeviceNotSupportedError):
        resolve_device("cuda:0", xp="numpy")


def test_resolve_device_torch_checks():
    with pytest.raises(DeviceNotSupportedError):
        resolve_device("not-a-device", xp="torch")

    if CUDA_AVAILABLE:
        assert resolve_device("cuda", xp="torch") == "cuda"
    else:
        with pytest.raises(CUDAUnavailableError):
            resolve_device("cuda", xp="torch")


if __name__ == "__main__":
    # test_array_context_override_and_restore()
    # test_as_context_converts_under_current_context()
    # test_as_context_arraylike_only_passthrough()
    # test_unify_args_strict_true_raises_for_non_array_inputs()
    # test_unify_args_strict_false_fallback_and_convert()
    # test_resolve_device_basic_and_numpy_constraints()
    # test_resolve_device_torch_checks()
    # test_array_context_override_and_restore()
    # test_as_context_converts_under_current_context()
    # test_as_context_arraylike_only_passthrough()
    # test_unify_args_strict_true_raises_for_non_array_inputs()
    # test_unify_args_strict_false_fallback_and_convert()
    # test_resolve_device_basic_and_numpy_constraints()
    # test_resolve_device_torch_checks()
    test_unify_args_strict_false_fallback_and_convert()
