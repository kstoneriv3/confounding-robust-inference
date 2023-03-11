import numpy as np
import torch

# As inconsistent use of dtypes in torch operation results in error, we set our default.
_DEFAULT_TORCH_FLOAT_DTYPE = torch.float64


def get_default_torch_dtype() -> torch.dtype:
    """Get the value of _DEFAULT_TORCH_FLOAT_DTYPE.

    The initial value of _DEFAULT_TORCH_FLOAT_DTYPE is torch.float64.
    """
    return _DEFAULT_TORCH_FLOAT_DTYPE


def set_default_torch_dtype(dtype: torch.dtype) -> None:
    """Set the value of _DEFAULT_TORCH_FLOAT_DTYPE."""
    _DEFAULT_TORCH_FLOAT_DTYPE = dtype


def as_tensor(arr: float | list | np.ndarray, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(arr)
    if tensor.is_floating_point():
        return tensor.to(_DEFAULT_TORCH_FLOAT_DTYPE)
    else:
        return tensor


def as_tensors(*ndarrays: np.ndarray) -> tuple[torch.Tensor, ...]:
    return tuple(as_tensor(a) for a in ndarrays)


def as_ndarrays(*tensors: torch.Tensor) -> tuple[np.ndarray, ...]:
    return tuple(t.data.numpy() for t in tensors)
