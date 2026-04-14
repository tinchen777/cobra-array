import sys
from typing import Any

import numpy as np
import pytest
import time
from types import ModuleType

sys.path.insert(0, "/data/tianzhen/my_packages/cobra-array/src")

from cobra_array.array_api import torch_xp
from cobra_array.compat import CompatArray, unwrap, wrap_arraylike
from cobra_array.exceptions import CompatArrayAttributeError, NotArrayAPIObjectError
from cobra_array.compat._array import UniqueResult


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
    moved = a.to_device("cpu")

    assert isinstance(moved, np.ndarray)
    assert moved.tolist() == a.to_list()


@pytest.mark.skipif(torch_xp is None, reason="PyTorch not available")
def test_to_tensor_available_backend():
    a = _arr_1d()
    
    aa = a.to_device("cpu")
    
    aa = CompatArray(a)
    
    t = a.to_tensor(device="cpu")

    assert isinstance(t, torch_xp.Tensor)
    assert tuple(t.shape) == (3,)
    
    aa = wrap_arraylike(a)


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
    
    print(type(a) is CompatArray)
    
    start = time.process_time()
    all_result = a.cxp.unique_all(a)
    # counts_result = a.unique_counts()
    # inverse_result = a.unique_inverse()
    
    # print(all_result.values)
    
    # assert isinstance(all_result.values, np.ndarray)
    # assert isinstance(all_result.indices, np.ndarray)
    # assert isinstance(all_result.inverse_indices, np.ndarray)
    # assert isinstance(all_result.counts, np.ndarray)

    print(f"(cxp1)Unique operations test took {time.process_time() - start:.5f} seconds")
    

    start = time.process_time()
    all_result = a.unique_all()
    # counts_result = a.unique_counts()
    # inverse_result = a.unique_inverse()
    
    # assert isinstance(all_result.values, CompatArray)
    # assert isinstance(all_result.indices, CompatArray)
    # assert isinstance(all_result.inverse_indices, CompatArray)
    # assert isinstance(all_result.counts, CompatArray)

    print(f"Unique operations test took {time.process_time() - start:.5f} seconds")

    start = time.process_time()
    all_result = a.cxp.unique_all(a)
    # counts_result = a.unique_counts()
    # inverse_result = a.unique_inverse()
    
    # print(all_result.values)
    
    # assert isinstance(all_result.values, np.ndarray)
    # assert isinstance(all_result.indices, np.ndarray)
    # assert isinstance(all_result.inverse_indices, np.ndarray)
    # assert isinstance(all_result.counts, np.ndarray)

    print(f"(cxp)Unique operations test took {time.process_time() - start:.5f} seconds")

    # assert isinstance(counts_result.values, CompatArray)
    # assert isinstance(counts_result.counts, CompatArray)

    # assert isinstance(inverse_result.values, CompatArray)
    # assert isinstance(inverse_result.inverse_indices, CompatArray)
    
    a = np.array([1, 2, 2, 1, 3], dtype=np.int32)

    start = time.process_time()
    all_result = np.unique(a, return_index=True, return_inverse=True, return_counts=True)
    
    UniqueResult(
        CompatArray(all_result[0], xp=np),
        CompatArray(all_result[1], xp=np),
        CompatArray(all_result[2], xp=np),
        CompatArray(all_result[3], xp=np)
    )
    
    print(f"(RAW)Unique operations test took {time.process_time() - start:.5f} seconds")
    
    # start = time.process_time()
    # all_result = torch_xp.unique(torch.tensor(a), return_inverse=True, return_counts=True)
  
    # print(f"API operations test took {time.process_time() - start:.4f} seconds")
    
    from array_api_compat import get_namespace
    xp = get_namespace(a)

    start = time.process_time()
    all_result = xp.unique_all(a)
    
    UniqueResult(
        CompatArray(all_result.values, xp=xp),
        CompatArray(all_result.indices, xp=xp),
        CompatArray(all_result.inverse_indices, xp=xp),
        CompatArray(all_result.counts, xp=xp)
    )
    assert isinstance(all_result.values, xp.ndarray)
    assert isinstance(all_result.indices, xp.ndarray)
    assert isinstance(all_result.inverse_indices, xp.ndarray)
    assert isinstance(all_result.counts, xp.ndarray)
    
    print(f"API operations test took {time.process_time() - start:.5f} seconds")
    
    print(type(xp) is ModuleType)


def test_add():

    a = CompatArray(np.array([1, 2, 2, 1, 3], dtype=np.int32))
    print(type(a) is CompatArray)

    start = time.time()
    all_result = a.add(1)
    all_result = a.add(1)
    # counts_result = a.unique_counts()
    # inverse_result = a.unique_inverse()

    print(f"add operations test took {time.time() - start:.5f} seconds")

    start = time.time()
    all_result = a.cxp.add(a, 1)
    all_result = a.cxp.add(a, 1)

    print(f"(cxp)add operations test took {time.time() - start:.5f} seconds")
    
    a = np.array([1, 2, 2, 1, 3], dtype=np.int32)

    start = time.time()
    all_result = np.add(a, 1)

    print(f"(RAW)add operations test took {time.time() - start:.5f} seconds")

    from array_api_compat import get_namespace
    xp = get_namespace(a)

    start = time.time()
    all_result = xp.add(a, 1)

    print(f"API operations test took {time.time() - start:.5f} seconds")


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


def test_numpy_operator_overloads():
    

    a = np.array([1, 2, 3], dtype=np.int32)
    b = np.array([3, 2, 1], dtype=np.int32)

    start = time.process_time()
    assert a.__abs__().tolist() == [1, 2, 3]
    assert (a + b).tolist() == [4, 4, 4]
    assert (a - b).tolist() == [-2, 0, 2]
    assert (a * b).tolist() == [3, 4, 3]
    assert (a / 2).tolist() == [0.5, 1.0, 1.5]
    assert (a // 2).tolist() == [0, 1, 1]
    assert (a % 2).tolist() == [1, 0, 1]
    assert (a & b).tolist() == [1, 2, 1]
    assert (a | b).tolist() == [3, 2, 3]
    assert (a ^ b).tolist() == [2, 0, 2]
    assert (a << 1).tolist() == [2, 4, 6]
    assert (a >> 1).tolist() == [0, 1, 1]

    assert (-a).tolist() == [-1, -2, -3]
    assert (+a).tolist() == [1, 2, 3]
    assert (~a).tolist() == [-2, -3, -4]

    assert (a == b).tolist() == [False, True, False]
    assert (a != b).tolist() == [True, False, True]
    assert (a > b).tolist() == [False, False, True]
    assert (a >= b).tolist() == [False, True, True]
    assert (a < b).tolist() == [True, False, False]
    assert (a <= b).tolist() == [True, True, False]

    print(f"NumPy operator overloads test took {time.process_time() - start:.4f} seconds")
    
    a = CompatArray(np.array([1, 2, 3], dtype=np.int32))
    b = CompatArray(np.array([3, 2, 1], dtype=np.int32))
    
    start = time.process_time()
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

    print(f"Operator overloads test took {time.process_time() - start:.4f} seconds")


# def test_torch_operator_overloads():
#     a = CompatArray(torch.tensor([1, 2, 3], dtype=torch.int32))
#     b = CompatArray(torch.tensor([3, 2, 1], dtype=torch.int32))

#     start = time.process_time()
#     assert a.__abs__().to_list() == [1, 2, 3]
#     assert (a + b).to_list() == [4, 4, 4]
#     assert (a - b).to_list() == [-2, 0, 2]
#     assert (a * b).to_list() == [3, 4, 3]
#     assert (a / 2).to_list() == [0.5, 1.0, 1.5]
#     assert (a // 2).to_list() == [0, 1, 1]
#     assert (a % 2).to_list() == [1, 0, 1]
#     assert (a & b).to_list() == [1, 2, 1]
#     assert (a | b).to_list() == [3, 2, 3]
#     assert (a ^ b).to_list() == [2, 0, 2]
#     assert (a << 1).to_list() == [2, 4, 6]
#     assert (a >> 1).to_list() == [0, 1, 1]

#     assert (-a).to_list() == [-1, -2, -3]
#     assert (+a).to_list() == [1, 2, 3]
#     assert (~a).to_list() == [-2, -3, -4]

#     assert (a == b).to_list() == [False, True, False]
#     assert (a != b).to_list() == [True, False, True]
#     assert (a > b).to_list() == [False, False, True]
#     assert (a >= b).to_list() == [False, True, True]
#     assert (a < b).to_list() == [True, False, False]
#     assert (a <= b).to_list() == [True, True, False]

#     print(f"Operator overloads test took {time.process_time() - start:.4f} seconds")

#     a = torch.tensor([1, 2, 3], dtype=torch.int32)
#     b = torch.tensor([3, 2, 1], dtype=torch.int32)

#     start = time.process_time()
#     assert a.__abs__().tolist() == [1, 2, 3]
#     assert (a + b).tolist() == [4, 4, 4]
#     assert (a - b).tolist() == [-2, 0, 2]
#     assert (a * b).tolist() == [3, 4, 3]
#     assert (a / 2).tolist() == [0.5, 1.0, 1.5]
#     assert (a // 2).tolist() == [0, 1, 1]
#     assert (a % 2).tolist() == [1, 0, 1]
#     assert (a & b).tolist() == [1, 2, 1]
#     assert (a | b).tolist() == [3, 2, 3]
#     assert (a ^ b).tolist() == [2, 0, 2]
#     assert (a << 1).tolist() == [2, 4, 6]
#     assert (a >> 1).tolist() == [0, 1, 1]

#     assert (-a).tolist() == [-1, -2, -3]
#     assert (+a).tolist() == [1, 2, 3]
#     assert (~a).tolist() == [-2, -3, -4]

#     assert (a == b).tolist() == [False, True, False]
#     assert (a != b).tolist() == [True, False, True]
#     assert (a > b).tolist() == [False, False, True]
#     assert (a >= b).tolist() == [False, True, True]
#     assert (a < b).tolist() == [True, False, False]
#     assert (a <= b).tolist() == [True, True, False]

#     print(f"Torch operator overloads test took {time.process_time() - start:.4f} seconds")


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


def test_cxp_of_compatarray_matches_array_namespace():
    a = _arr_1d()
    cxp = a.cxp

    assert cxp is not None
    assert cxp.xp_name == a.xp_name


def test_linalg():
    a = CompatArray(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    cxp = a.cxp
    print(a)

    f = cxp.vector_norm(a, axis=0, ord="-inf")

    print(f)

    f = cxp.matrix_norm(a, ord="nuc")

    print(f)

def test_linalg2():
    import torch
    a = CompatArray(torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda:0"))
    cxp = a.cxp
    print(a)

    f = cxp.vector_norm(a, axis=0, ord="-inf")

    print(f.device)

    print(f)

    f = cxp.matrix_norm(a, ord="nuc")

    print(f)
    print(f.device)


if __name__ == "__main__":
    # test_unique_all_unique_counts_unique_inverse()
    # print("=" * 40)
    
    # test_numpy_operator_overloads()
    # print("=" * 40)
    # test_torch_operator_overloads()
    # print("=" * 40)

    # test_add()
    
    test_linalg2()
