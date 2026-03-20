import sys
sys.path.insert(0, "/data/tianzhen/my_packages/cobra-array/src")
# sys.path.insert(0, "/Users/apple/Develop/Python_WorkSpace/my_packages/cobra-array/src")

from cobra_array.compat._wrap import CompatArray
import numpy as np
import torch
import time
import array_api_compat as api
import array_api_strict
import inspect



def test_class():
    a = torch.tensor([[1, 2], [3, 4], [5, 6]])
    b = np.array([[1, 2, 3], [4, 5, 6]])
    c = [[1, 2], [3, 4], [5, 6]]
    d = "hello"
    e = iter(a)
    
    array_api_strict.arange


    np_m = api.array_namespace(b)
    torch_m = api.array_namespace(a)

    with open("/data/tianzhen/my_packages/cobra-array/strict.txt", "r") as f:
        l = f.readlines()
        for i in l:
            i = i.strip()
            if len(i) == 0:
                continue

            print("=====" * 10)
            np_attr = getattr(np_m, i, None)
            if np_attr is not None:
                try:
                    print(f"func-np.{i}:", inspect.signature(np_attr))
                except Exception:
                    print(f"attr-np.{i}:", np_attr)
            else:
                print(f"None-np.{i}:")

            torch_attr = getattr(torch_m, i, None)
            if torch_attr is not None:
                try:
                    print(f"func-torch.{i}:", inspect.signature(torch_attr))
                except Exception:
                    print(f"attr-torch.{i}:", torch_attr)
            else:
                print(f"None-torch.{i}:")

if __name__ == "__main__":
    test_class()
