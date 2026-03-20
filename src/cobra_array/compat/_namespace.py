


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
    ) -> Array
    
    def stack(
        arrays: Tuple[Array, ...] | List[Array],
        /,
        *,
        axis: int = 0
    ) -> CompatArray[ArrayT]:
        """
        """
        ...







