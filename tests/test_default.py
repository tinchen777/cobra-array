import sys
sys.path.insert(0, "/data/tianzhen/my_packages/cobra-array/src")
# sys.path.insert(0, "/Users/apple/Develop/Python_WorkSpace/my_packages/cobra-array/src")

from cobra_array.default import default_spec, as_default
import numpy as np
import torch
import time


def test_to():
    a = torch.tensor([[1, 2], [3, 4], [5, 6]])
    b = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    c = [[1, 2], [3, 4], [5, 6]]
    d = "hello"
    e = iter(a)


    start = time.thread_time()
    a2 = as_default(a, copy=False, arraylike_only=True)
    print(a2, type(a2), a2.dtype, id(a) == id(a2))
    print("Time taken for to_tensor:", time.thread_time() - start)

    start = time.thread_time()
    b2 = as_default(b, copy=False)
    print(b2, type(b2), b2.dtype, id(b) == id(b2))
    print("Time taken for to_tensor:", time.thread_time() - start)


if __name__ == "__main__":
    test_to()
