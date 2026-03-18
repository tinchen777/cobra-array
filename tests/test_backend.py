import sys
sys.path.insert(0, "/data/tianzhen/my_packages/cobra-array/src")
# sys.path.insert(0, "/Users/apple/Develop/Python_WorkSpace/my_packages/cobra-array/src")

from cobra_array._core import array_spec, unify_array_args, context_namespace, as_context_array
import numpy as np
import torch
import array_api_compat as api


def test_array_namespace():
    a = torch.tensor([[1, 2], [3, 4], [5, 6]])
    b = np.array([[1, 2, 3], [4, 5, 6]])
    c = [[1, 2], [3, 4], [5, 6]]
    d = "hello"
    e = iter(a)
    f = torch.tensor([[1, 2], [3, 4], [5, 6]])

    arr_spec = array_spec(c, f, kw_arrays={"d": d}, ref=0, filter_array_like=True)
    # xp, arr = array_namespace(f, a, kw_arrays={"a": a}, ref=2)
    print(arr_spec.xp)
    print(arr_spec.dtype)
    print(arr_spec.device)


def test_unify_array_args():
    a = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    b = np.array([[1, 2, 3], [4, 5, 6]])
    bb = np.array(1)
    aa = torch.tensor(1)
    c = [[1, 2], [3, 4], [5, 6]]
    d = "hello"
    e = iter(a)
    f = torch.tensor([[1, 2], [3, 4], [5, 6]])
    
    
    print(a.dtype)
    print(b.dtype)
    

    @unify_array_args(1, unify_device=False, filter_array_like=True)
    def func(*args, **kwargs):
        print(args)
        print(kwargs)
        xp = context_namespace()
        print(xp.__name__)
        
        # print(a.size, b.size)
        print(len(aa))
        
        # print(xp.shape(a))
        # print(api.device(b))

        
        

        new = [1, 2, 3]
        new = as_context_array(new)
        print(repr(new))

        a1 = xp.mean(args[0], axis=1)
        print(a1)
        # raise ValueError("Test unify_array_args")

    func(c, b, a)


if __name__ == "__main__":
    test_array_namespace()
    print("=====")
    test_unify_array_args()
