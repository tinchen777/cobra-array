import sys
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, "/data/tianzhen/my_packages/cobra-array/src")

from cobra_array.array_api import torch_xp
from cobra_array.compat import CompatArray, unwrap, wrap_arraylike
from cobra_array.exceptions import CompatArrayAttributeError, NotArrayAPIObjectError


def _arr_1d():
    return CompatArray(np.array([1.0, 2.0, 3.0], dtype=np.float32))


def _arr_2d():
    return CompatArray(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))


def test_from_other_with_numpy_namespace():
    out = CompatArray.from_other([1, 2, 3], xp="numpy")
    assert isinstance(out, CompatArray)
    assert out.xp_name == "NumPy"
    assert out.to_list() == [1, 2, 3]


def test_new_from_compatarray_copy_and_no_copy():
    a = _arr_1d()
    b = CompatArray(a)
    c = CompatArray(a, copy=True)

    assert b is a
    assert c is not a
    assert c.to_list() == a.to_list()


def test_new_invalid_input_raises():
    with pytest.raises(NotArrayAPIObjectError):
        CompatArray("not-array")  # type: ignore[arg-type]


def test_to_numpy_to_list_and_array_protocol():
    a = _arr_1d()

    n1 = a.to_numpy(copy=False)
    n2 = np.asarray(a)

    assert isinstance(n1, np.ndarray)
    assert isinstance(n2, np.ndarray)
    assert n1.tolist() == [1.0, 2.0, 3.0]
    assert n2.tolist() == [1.0, 2.0, 3.0]


def test_to_device_on_numpy_backend():
    a = _arr_1d()
    moved = a.to_device("cpu:0")
    
    f = a.astype(int, device="cpu: 0")  # type: ignore[call-arg]

    assert isinstance(moved, np.ndarray)
    assert moved.tolist() == a.to_list()


@pytest.mark.skipif(torch_xp is None, reason="PyTorch not available")
def test_to_tensor_available_backend():
    a = _arr_1d()
    t = a.to_tensor(device="cpu")

    assert isinstance(t, torch_xp.Tensor)
    assert tuple(t.shape) == (3,)


def test_unstack_and_nonzero():
    a = _arr_2d()

    pieces = a.unstack(axis=0)
    nz = a.nonzero()

    assert isinstance(pieces, tuple)
    assert len(pieces) == 2
    assert all(isinstance(x, CompatArray) for x in pieces)
    assert pieces[0].to_list() == [1.0, 2.0]

    assert isinstance(nz, tuple)
    assert len(nz) == 2
    assert all(isinstance(x, CompatArray) for x in nz)


def test_unique_all_unique_counts_unique_inverse():
    a = CompatArray(np.array([1, 2, 2, 1, 3], dtype=np.int32))

    all_result = a.unique_all()
    counts_result = a.unique_counts()
    inverse_result = a.unique_inverse()

    assert isinstance(all_result.values, CompatArray)
    assert isinstance(all_result.indices, CompatArray)
    assert isinstance(all_result.inverse_indices, CompatArray)
    assert isinstance(all_result.counts, CompatArray)

    assert isinstance(counts_result.values, CompatArray)
    assert isinstance(counts_result.counts, CompatArray)

    assert isinstance(inverse_result.values, CompatArray)
    assert isinstance(inverse_result.inverse_indices, CompatArray)


def test_copy_and_basic_properties():
    a = _arr_2d()
    b = a.copy()

    assert isinstance(a.arr, np.ndarray)
    assert isinstance(b, CompatArray)
    assert b is not a

    assert str(a.device) == "cpu"
    assert a.shape == (2, 2)
    assert a.ndim == 2
    assert a.size == 4


def test_transpose_properties():
    a = _arr_2d()

    t = a.T

    assert isinstance(t, CompatArray)
    assert t.shape == (2, 2)

    with pytest.raises(AttributeError):
        _ = a.mT


def test_len_and_repr():
    a = _arr_1d()
    z = CompatArray(np.array(5))

    assert len(a) == 3
    assert "NumPy_Array" in repr(a)

    with pytest.raises(TypeError):
        len(z)


def test_getitem_and_setitem():
    a = CompatArray(np.array([10, 20, 30], dtype=np.int32))

    assert int(a[1]) == 20

    a[1] = 99
    assert a.to_list() == [10, 99, 30]


def test_scalar_conversions():
    b = CompatArray(np.array(True))
    i = CompatArray(np.array(7))
    f = CompatArray(np.array(1.5, dtype=np.float32))
    c = CompatArray(np.array(2 + 3j, dtype=np.complex64))

    assert bool(b) is True
    assert int(i) == 7
    assert i.__index__() == 7
    assert float(f) == pytest.approx(1.5)
    assert complex(c) == complex(2, 3)


def test_operator_overloads():
    a = CompatArray(np.array([1, 2, 3], dtype=np.int32))
    b = CompatArray(np.array([3, 2, 1], dtype=np.int32))

    assert a.__abs__().to_list() == [1, 2, 3]
    assert (a + b).to_list() == [4, 4, 4]
    assert (a - b).to_list() == [-2, 0, 2]
    assert (a * b).to_list() == [3, 4, 3]
    assert (a / 2).to_list() == [0.5, 1.0, 1.5]
    assert (a // 2).to_list() == [0, 1, 1]
    assert (a % 2).to_list() == [1, 0, 1]
    assert (a & b).to_list() == [1, 2, 1]
    assert (a | b).to_list() == [3, 2, 3]
    assert (a ^ b).to_list() == [2, 0, 2]
    assert (a << 1).to_list() == [2, 4, 6]
    assert (a >> 1).to_list() == [0, 1, 1]

    assert (-a).to_list() == [-1, -2, -3]
    assert (+a).to_list() == [1, 2, 3]
    assert (~a).to_list() == [-2, -3, -4]

    assert (a == b).to_list() == [False, True, False]
    assert (a != b).to_list() == [True, False, True]
    assert (a > b).to_list() == [False, False, True]
    assert (a >= b).to_list() == [False, True, True]
    assert (a < b).to_list() == [True, False, False]
    assert (a <= b).to_list() == [True, True, False]


def test_pow_operator_matches_current_implementation_behavior():
    a = _arr_1d()
    out = a ** 2
    assert isinstance(out, CompatArray)
    assert out.to_list() == [1.0, 4.0, 9.0]


def test_matmul_operator():
    x = CompatArray(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    y = CompatArray(np.array([[2.0, 0.0], [1.0, 2.0]], dtype=np.float32))

    out = x @ y

    assert isinstance(out, CompatArray)
    assert out.to_list() == [[4.0, 4.0], [10.0, 8.0]]


def test_getattr_rejects_non_callable_namespace_attrs():
    a = _arr_1d()
    with pytest.raises(CompatArrayAttributeError):
        _ = getattr(a, "pi")


def test_module_helpers_unwrap_wrap_to_cxp():
    raw = np.array([1, 2, 3])
    wrapped = wrap_arraylike(raw)

    assert isinstance(wrapped, CompatArray)
    assert unwrap(wrapped) is raw
    assert unwrap(raw) is raw

    obj: Any = {"k": 1}
    assert wrap_arraylike(obj) is obj

    out = CompatArray(raw, xp=np)
    assert out.xp_name == "NumPy"


def test_some_pyi_annotated_methods_via_dynamic_dispatch():
    a = _arr_2d()

    b = a.astype(np.float64)
    c = a.abs()
    d = a.add(1)
    s = a.sum()
    r = a.reshape((4,))
    t = a.take(np.array([0], dtype=np.int32), axis=1)

    assert isinstance(b, CompatArray)
    assert str(b.dtype) == "float64"
    assert isinstance(c, CompatArray)
    assert isinstance(d, CompatArray)
    assert isinstance(s, CompatArray)
    assert isinstance(r, CompatArray)
    assert isinstance(t, CompatArray)

    assert r.shape == (4,)
    assert t.shape == (2, 1)
