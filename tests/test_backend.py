import sys
sys.path.insert(0, "/data/tianzhen/my_packages/cobra-array/src")

from cobra_array._core import array_spec, unify_array_args, context_namespace, as_context_array
import numpy as np
import torch


def test_array_namespace():
    a = torch.tensor([[1, 2], [3, 4], [5, 6]], device="cuda:0")
    b = np.array([[1, 2, 3], [4, 5, 6]])
    c = [[1, 2], [3, 4], [5, 6]]
    d = "hello"
    e = iter(a)
    f = torch.tensor([[1, 2], [3, 4], [5, 6]])

    xp, arr = array_spec(c, f, kw_arrays={"d": d}, ref=0, filter_array_like=True)
    # xp, arr = array_namespace(f, a, kw_arrays={"a": a}, ref=2)
    
    print(xp)
    print(arr)


def test_unify_array_args():
    a = torch.tensor([[1, 2], [3, 4], [5, 6]], device="cuda:0")
    b = np.array([[1, 2, 3], [4, 5, 6]])
    c = [[1, 2], [3, 4], [5, 6]]
    d = "hello"
    e = iter(a)
    f = torch.tensor([[1, 2], [3, 4], [5, 6]])

    @unify_array_args(unify_device=False)
    def func(*args, **kwargs):
        print(args)
        print(kwargs)
        xp = context_namespace()
        print(xp.__name__)
        
        new = [1, 2, 3]
        new = as_context_array(new)
        print(repr(new))
        
    
    
    func(c, b, a)
    
    
    


if __name__ == "__main__":
    test_array_namespace()
    print("=====")
    test_unify_array_args()
