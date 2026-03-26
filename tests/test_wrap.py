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
    b = np.array([[11, 12, 13], [14, 15, 16]], dtype=np.float32)
    c = [[21, 22], [23, 24], [25, 26]]
    d = "hello"
    e = iter(a)

    np_m = api.array_namespace(b)
    torch_m = api.array_namespace(a)
    np_xp = NameSpace(np_m)
    torch_xp: NameSpace[NDArray] = NameSpace(torch_m)
    
    print(torch_m.__name__)
    print(torch_xp.__name__)
    
    # print(np_xp.float645)
    print(np_m.float645)

    ba = CompatArray(b)
    aa = CompatArray(a)
    aa.arr
    
    bxp = ba.xp
    
    print(ba)
    print(ba.arr)
    print(bxp)
    print(ba.to_numpy())
    print(ba.to_tensor())
    print(ba.to_list())
    print("=====" * 10)
    s = ba.unique_all()
    print(s.counts.uint16)
    print(type(s.counts))
    
    
    exit()
    n = bxp.astype(ba.arr, ba.float64)
    print(n)
    
    
    print(ba.astype(ba.float64))
    
    
    
    
    


if __name__ == "__main__":
    test_class()
