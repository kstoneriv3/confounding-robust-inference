import numpy as np
import torch


def as_tensors(*ndarrays: np.ndarray) -> tuple[torch.Tensor, ...]:
    return tuple(torch.as_tensor(a) for a in ndarrays)


def as_ndarrays(*tensors: torch.Tensor) -> tuple[np.ndarray, ...]:
    return tuple(t.data.cpu().numpy() for t in tensors)
