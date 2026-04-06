import sys

import numpy as np
import pytest

sys.path.insert(0, "/data/tianzhen/my_packages/cobra-array/src")

from cobra_array.compat import CompatArray, CompatNamespace


def _cxp() -> CompatNamespace:
    return CompatNamespace(np)


def test_new_with_namespace_and_compatnamespace():
    a = CompatNamespace(np)
    b = CompatNamespace(a)

    assert isinstance(a, CompatNamespace)
    assert b is a
    assert a.xp_name == "NumPy"


def test_meshgrid_and_broadcast_arrays():
    cxp = _cxp()
    x = CompatArray(np.array([1, 2], dtype=np.int32))
    y = CompatArray(np.array([10, 20, 30], dtype=np.int32))

    grids = cxp.meshgrid(x, y, indexing="xy")
    b1, b2 = cxp.broadcast_arrays(cxp.asarray([1, 2, 3]), cxp.asarray([[1], [2]]))

    assert isinstance(grids, list)
    assert len(grids) == 2
    assert all(isinstance(g, CompatArray) for g in grids)
    assert grids[0].shape == (3, 2)
    assert grids[1].shape == (3, 2)

    assert isinstance(b1, CompatArray)
    assert isinstance(b2, CompatArray)
    assert b1.shape == (2, 3)
    assert b2.shape == (2, 3)


def test_can_cast_finfo_iinfo_isdtype_result_type():
    cxp = _cxp()

    assert cxp.can_cast(cxp.int32, cxp.float64) is True

    fi = cxp.finfo(cxp.float32)
    ii = cxp.iinfo(cxp.int32)

    assert fi.bits > 0
    assert ii.bits > 0
    assert ii.max > ii.min

    with pytest.raises(AttributeError):
        cxp.isdtype(cxp.float32, "real floating")

    rt = cxp.result_type(cxp.asarray([1, 2], dtype=cxp.int32), cxp.float64)
    assert rt == cxp.float64


def test_constants_and_dtype_properties_and_name():
    cxp = _cxp()

    assert isinstance(cxp.e, float)
    assert isinstance(cxp.pi, float)
    assert cxp.inf > 1e100
    assert np.isnan(cxp.nan)
    assert cxp.newaxis is None

    assert cxp.int8 is np.int8
    assert cxp.int16 is np.int16
    assert cxp.int32 is np.int32
    assert cxp.int64 is np.int64
    assert cxp.uint8 is np.uint8
    assert cxp.uint16 is np.uint16
    assert cxp.uint32 is np.uint32
    assert cxp.uint64 is np.uint64
    assert cxp.float32 is np.float32
    assert cxp.float64 is np.float64
    assert cxp.complex64 is np.complex64
    assert cxp.complex128 is np.complex128
    # assert cxp.bool in (bool, np.bool_)

    assert cxp.__name__.startswith("(compat)")


def test_getattr_dynamic_function_wrapping_from_pyi_methods():
    cxp = _cxp()

    a = cxp.asarray([1, 2, 3], dtype=cxp.float32)
    b = cxp.arange(0, 6, 2, dtype=cxp.int32)
    c = cxp.eye(2, dtype=cxp.float32)
    d = cxp.empty_like(c, dtype=cxp.float64)

    assert isinstance(a, CompatArray)
    assert isinstance(b, CompatArray)
    assert isinstance(c, CompatArray)
    assert isinstance(d, CompatArray)

    assert a.shape == (3,)
    assert b.to_list() == [0, 2, 4]
    assert c.shape == (2, 2)
    assert str(d.dtype) == "float64"


def test_getattr_dynamic_unwrap_inputs_and_wrap_outputs():
    cxp = _cxp()

    a = CompatArray(np.array([1, 2, 3], dtype=np.int32))
    b = CompatArray(np.array([10, 20, 30], dtype=np.int32))
    out = getattr(cxp, "add")(a, b)

    assert isinstance(out, CompatArray)
    assert out.to_list() == [11, 22, 33]


def test_getattr_missing_attribute_raises_attribute_error():
    cxp = _cxp()
    with pytest.raises(AttributeError):
        _ = getattr(cxp, "this_attr_does_not_exist")
