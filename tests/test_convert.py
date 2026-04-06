import sys
sys.path.insert(0, "/data/tianzhen/my_packages/cobra-array/src")
# sys.path.insert(0, "/Users/apple/Develop/Python_WorkSpace/my_packages/cobra-array/src")

from cobra_array.convert import to_numpy, to_tensor, to_list, as_array
import numpy as np
import torch
import time


def test_to():
    a = torch.tensor([[1, 2], [3, 4], [5, 6]])
    b = np.array([[1, 2, 3], [4, 5, 6]])
    c = [[1, 2], [3, 4], [5, 6]]
    d = "hello"
    e = iter(a)

    a1 = to_numpy(a, copy=False, dtype=float)
    print(a1, type(a1), a1.dtype, id(a) == id(a1))
    a1 = to_numpy(a1, copy=False)

    b1 = to_numpy(a, copy=False, dtype=None)
    print(b1, type(b1), b1.dtype, id(b) == id(b1))
    
    a3 = to_list(a)
    a3 = to_list(a1)
    for i in a3:
        
        print(i, type(i))
    

    start = time.thread_time()
    a2 = to_tensor(a, copy=False, dtype=float, device="cuda: 1")
    print(a2, type(a2), a2.dtype, id(a) == id(a2))
    print("Time taken for to_tensor:", time.thread_time() - start)
  
    start = time.thread_time()
    b2 = to_tensor(b, copy=False, dtype=float, device="cpu")
    print(b2, type(b2), b2.dtype, id(b) == id(b2))
    print("Time taken for to_tensor:", time.thread_time() - start)


def test_to_2():
    a = [[1, 2], [3, 4], [5, 6]]
    b = [[1, 2, 3], [4, 5, 6]]
    c = iter(a)

    a1 = to_numpy(a, dtype=np.float32)
    print(a1, type(a1), a1.dtype)
    c1 = as_array(a1, np, device="cpu")

    b1 = to_tensor(b)
    print(b1, type(b1), b1.dtype)
    

    c1 = as_array(a, "torch", device="cpu")
    print(c1, type(c1), c1.dtype)
    c1.device

    a = as_array(a1, "numpy", arraylike_only=False)
    a = as_array(1, "numpy", dtype=np.float128, arraylike_only=True)
    print(a)
    print(a.dtype)


if __name__ == "__main__":
    test_to()
    test_to_2()
