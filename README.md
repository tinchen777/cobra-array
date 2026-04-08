<div align="center">

<h2 id="title">
🍁 cobra-array 🍁<br>
<sub>Unified Array Utilities with Python Array API Compatibility</sub>
</h2>

[![PyPI version](https://img.shields.io/pypi/v/cobra-array.svg)](https://pypi.org/project/cobra-array/)
![Python](https://img.shields.io/pypi/pyversions/cobra-array?color=brightgreen)
[![codecov](https://codecov.io/gh/tinchen777/cobra-array/branch/main/graph/badge.svg)](https://codecov.io/gh/tinchen777/cobra-array)
![License](https://img.shields.io/github/license/tinchen777/cobra-array.svg)

[![Tests](https://github.com/tinchen777/cobra-array/actions/workflows/test.yml/badge.svg)](https://github.com/tinchen777/cobra-array/actions/workflows/test.yml)
![Github stars](https://img.shields.io/github/stars/tinchen777/cobra-array.svg)

</div>

## About

`cobra-array` is a backend-agnostic array utility library that unifies array conversion, context control, and cross-library operations across `NumPy`/`PyTorch`-style ecosystems.

- Python: 3.9+
- Runtime deps: `array-api-compat` (>= 1.11.2)

## Features

- 🚀 **Ackend-agnostic API**: Work with different array backends through one consistent interface.
- 🚀 **Context-driven unification**: Automatically align namespace, dtype, and device via context managers and decorators.
- 🚀 **Compatibility wrappers**: `CompatArray` and `CompatNamespace` provide a clean, consistent layer over native backend behavior.

## Installation

### Install from PyPI

```bash
pip install cobra-array
```

## Quick Start

- Basic conversions:

    ```python
    import numpy as np
    from cobra_array.convert import to_numpy, to_tensor, to_list

    data = [[1, 2], [3, 4]]

    arr_np = to_numpy(data, dtype=np.float32)  # numpy.ndarray float32

    arr_torch = to_tensor(data, device="cpu")

    back_to_list = to_list(arr_np)  # [[1.0, 2.0], [3.0, 4.0]]
    ```

- Context-based conversion:

    ```python
    import numpy as np
    from cobra_array import array_context, as_context, context_spec

    with array_context(xp="numpy", dtype=np.float32, device="cpu"):
        x = as_context([1, 2, 3])
        y = as_context(np.array([4, 5]))
        spec = context_spec()
    ```

- Auto-unify function arguments:

    ```python
    import numpy as np
    from cobra_array import unify_args

    @unify_args(ref=0, unify_dtype=True, unify_device=True, arraylike_only=True)
    def add_and_mean(a, b):
        c = a + b
        return c.mean()

    out = add_and_mean(np.array([1, 2, 3]), [4, 5, 6])
    ```

- Default backend strategy:

    ```python
    from cobra_array.default import as_default, default_spec

    spec = default_spec()

    x = as_default([1, 2, 3], unify_dtype=True, unify_device=True)
    ```

## Requirements

- Python >= 3.9
- `array-api-compat` >= 1.11.2

## License

See LICENSE in the repository.

## Links

- [Homepage/Repo](https://github.com/tinchen777/cobra-array.git)
- [Issues](https://github.com/tinchen777/cobra-array.git/issues)
