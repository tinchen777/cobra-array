import sys
sys.path.insert(0, "/data/tianzhen/my_packages/cobra-array/src")
# sys.path.insert(0, "/Users/apple/Develop/Python_WorkSpace/my_packages/cobra-array/src")

from cobra_array.compat._array import CompatArray
from cobra_array.compat._namespace import NameSpace
import numpy as np
from numpy.typing import NDArray
import torch
import time
import array_api_compat as api
import array_api_strict
import inspect



def test_class():
    a = torch.tensor([[1, 2], [3, 4], [5, 6]])
    b = np.array([[11, 12, 13], [14, 15, 16]], dtype=float)
    # b = np.array([1, 1])
    c = [[21, 22], [23, 24], [25, 26]]
    d = "hello"
    e = iter(a)

    np_m = api.array_namespace(b)
    torch_m = api.array_namespace(a)

    
    ba = CompatArray.from_other(b, xp="numpy")
    ba = CompatArray.from_other(ba, xp="torch")
    ba = CompatArray(ba)
    
    # ba = ba.astype(np.float128)

    bxp = ba.xp
    
    

    print(ba)
    print(ba.arr)
    print(bxp)
    print(ba.to_numpy())
    # print(ba.to_tensor())
    print(ba.to_list())
    print("=====" * 10)
    # f = ba.astype(np.float128, copy=False, device="cuda")
    # print("astype", f)
    f = ba.abs()
    print("abs", f)
    f = ba.acos()
    print("acos", f)
    f = ba.acosh()
    print("acosh", f)
    f = ba.add(ba)
    print("add", f)
    f = ba.asin()
    print("asin", f)
    f = ba.asinh()
    print("asinh", f)
    f = ba.atan()
    print("atan", f)
    f = ba.atan2(ba)
    print("atan2", f)
    f = ba.atanh()
    print("atanh", f)
    # f = ba.ceil()
    # print("ceil", f)
    # f = ba.clip(12, 15)
    # print("clip", f)
    
    f = abs(ba)
    f = ba + ba
    print(f)
    
    f = ba.cumulative_sum(axis=1)
    
    f = ba.unique_all()
    print(f)
   
    print(f.counts, type(f.counts), f.counts.dtype)
    print(f.indices, type(f.indices), f.indices.dtype)
    print(f.inverse_indices, type(f.inverse_indices), f.inverse_indices.dtype)
    print(f.values, type(f.values), f.values.dtype)



if __name__ == "__main__":
    test_class()
