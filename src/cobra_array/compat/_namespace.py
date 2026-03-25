


# === Creation functions ===
def asarray(
    obj: Array | bool | int | float | NestedSequence[bool | int | float] | Any,
    /,
    *,
    dtype: Dtype | None = None,
    device: Unknown | None = None,
    copy: bool | None = None
) -> Array

def arange(
    start: int | float,
    /,
    stop: int | float | None = None,
    step: int | float = 1,
    *,
    dtype: Dtype | None = None,
    device: Unknown | None = None
) -> Array

def empty(
    shape: int | Tuple[int, ...],
    *,
    dtype: Dtype | None = None,
    device: Unknown | None = None
) -> Array

def empty_like(
    x: Array,
    /,
    *,
    dtype: Dtype | None = None,
    device: Unknown | None = None
) -> Array

def eye(
    n_rows: int,
    n_cols: int | None = None,
    /,
    *,
    k: int = 0,
    dtype: Dtype | None = None,
    device: Unknown | None = None
) -> Array

def from_dlpack(
    x: object,
    /,
    *,
    device: Unknown | None = _default,
    copy: bool | None = _default
) -> Array

def full(
    shape: int | Tuple[int, ...],
    fill_value: int | float,
    *,
    dtype: Dtype | None = None,
    device: Unknown | None = None
) -> Array

def full_like(
    x: Array,
    /,
    fill_value: int | float,
    *,
    dtype: Dtype | None = None,
    device: Unknown | None = None
) -> Array

def linspace(
    start: int | float,
    stop: int | float,
    /,
    num: int,
    *,
    dtype: Dtype | None = None,
    device: Unknown | None = None,
    endpoint: bool = True
) -> Array

def meshgrid(
    *arrays: Array,
    indexing: str = "xy"
) -> List[Array]

def ones(
    shape: int | Tuple[int, ...],
    *,
    dtype: Dtype | None = None,
    device: Unknown | None = None
) -> Array

def ones_like(
    x: Array,
    /,
    *,
    dtype: Dtype | None = None,
    device: Unknown | None = None
) -> Array

def tril(
    x: Array,
    /,
    *,
    k: int = 0
) -> Array

def triu(
    x: Array,
    /,
    *,
    k: int = 0
) -> Array

def zeros(
    shape: int | Tuple[int, ...],
    *,
    dtype: Dtype | None = None,
    device: Unknown | None = None
) -> Array

def zeros_like(
    x: Array,
    /,
    *,
    dtype: Dtype | None = None,
    device: Unknown | None = None
) -> Array



# === data_type_functions ===
def broadcast_arrays(*arrays: Array) -> List[Array]


def can_cast(
    from_: Dtype | Array,
    to: Dtype,
    /
) -> bool

def finfo(
    type: Dtype | Array,
    /
) -> finfo_object

def isdtype(
    dtype: Dtype,
    kind: Dtype | str | Tuple[Dtype | str, ...]
) -> bool

def iinfo(
    type: Dtype | Array,
    /
) -> iinfo_object

def result_type(*arrays_and_dtypes: Array | Dtype | int | float | complex | bool) -> Dtype


# === Manipulation functions ===
    def concat(
        arrays: Tuple[Array, ...] | List[Array],
        /,
        *,
        axis: int | None = 0
    ) -> Array:
        """
        Joins a sequence of arrays along an existing axis.

        Parameters
        ----------
            arrays : Tuple[Array, ...] | List[Array]
                input arrays to join. The arrays must have the same shape, except in the dimension specified by axis.

            axis : int, default to `0`
                axis along which the arrays will be joined. If axis is None, arrays must be flattened before concatenation. If axis is negative, the function must determine the axis along which to join by counting from the last dimension.
        
        Returns
        -------
            CompatArray[ArrayT]
                an output array containing the concatenated values. 
        """
        ...
    
    def stack(
        arrays: Tuple[Array, ...] | List[Array],
        /,
        *,
        axis: int = 0
    ) -> CompatArray[ArrayT]:
        """
        Joins a sequence of arrays along a new axis.

        Parameters:
        arrays (Union[Tuple[array, ...], List[array]]) – input arrays to join. Each array must have the same shape.

        axis (int) – axis along which the arrays will be joined. Providing an axis specifies the index of the new axis in the dimensions of the result. For example, if axis is 0, the new axis will be the first dimension and the output array will have shape (N, A, B, C); if axis is 1, the new axis will be the second dimension and the output array will have shape (A, N, B, C); and, if axis is -1, the new axis will be the last dimension and the output array will have shape (A, B, C, N). A valid axis must be on the interval [-N, N), where N is the rank (number of dimensions) of x. If provided an axis outside of the required interval, the function must raise an exception. Default: 0.

        Returns:
        out (array) – an output array having rank N+1, where N is the rank (number of dimensions) of x. If the input arrays have different data types, normal Type Promotion Rules must apply. If the input arrays have the same data type, the output array must have the same data type as the input arrays.

        Note

        This specification leaves type promotion between data type families (i.e., intxx and floatxx) unspecified.
        
        """
        ...




“Answer the question.”

你局部优化：

“Answer carefully”
“Answer step by step”
“Answer in detail”

👉 你永远不会得到：

“Use chain-of-thought reasoning with explicit intermediate steps and verification.”

