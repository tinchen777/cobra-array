import sys

import numpy as np

sys.path.insert(0, "/data/tianzhen/my_packages/cobra-array/src")

from cobra_array.compat import CompatArray, CompatNamespace
from cobra_array.default import ArraySpec, as_default, default_spec


def test_arrayspec_create_wraps_namespace():
    spec = ArraySpec.create(np, np.float32, "cpu")
    assert isinstance(spec.cxp, CompatNamespace)
    assert spec.dtype == np.float32
    assert spec.device == "cpu"


def test_default_spec_returns_valid_defaults():
    spec = default_spec()
    assert isinstance(spec, ArraySpec)
    assert isinstance(spec.cxp, CompatNamespace)
    assert spec.dtype is float
    assert spec.device == "cpu"


def test_as_default_returns_compat_array():
    out = as_default([1, 2, 3], unify_dtype=True, unify_device=True)
    assert isinstance(out, CompatArray)


def test_as_default_arraylike_only_passthrough_non_array():
    marker = object()
    out = as_default(marker, arraylike_only=True)
    assert out is marker


def test_as_default_unify_dtype_false_keeps_original_dtype():
    src = np.array([1, 2, 3], dtype=np.int32)
    out = as_default(src, unify_dtype=False)
    assert isinstance(out, CompatArray)
    assert out.to_list() == [1, 2, 3]
    assert "int32" in str(out.dtype)
