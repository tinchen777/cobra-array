import sys
# sys.path.insert(0, "/data/tianzhen/my_packages/cobra-array/src")
sys.path.insert(0, "/Users/apple/Develop/Python_WorkSpace/my_packages/cobra-array/src")

from cobra_array._wrap import CompatArray
import numpy as np
import torch
import time


def test_class():
    a = torch.tensor([[1, 2], [3, 4], [5, 6]])
    b = np.array([[1, 2, 3], [4, 5, 6]])
    c = [[1, 2], [3, 4], [5, 6]]
    d = "hello"
    e = iter(a)


    start = time.thread_time()
    a2 = CompatArray(b)
    
    result = a2.mean(axis=1)
    print(result)
    
    a2.to_numpy()
    
    print("Time taken for to_tensor:", time.thread_time() - start)

    r2 = a2.mean(axis=1)
    print(r2)
    
    r3 = a2.mT
    print(r3)
    
    


if __name__ == "__main__":
    test_class()
