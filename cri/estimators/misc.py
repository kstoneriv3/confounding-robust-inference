import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Kernel, WhiteKernel


def select_kernel(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
) -> Kernel:
    """Learn the kernel parameters by fitting Gaussian process regression of Y on (T, X)."""
    TX = np.concatenate([T[:, None], X], axis=1)
    TX /= TX.std(axis=0)[None, :]
    kernel = WhiteKernel() + ConstantKernel() * RBF()
    # As GP gets slower for large sample size, we truncate data by n=1000.
    model = GaussianProcessRegressor(kernel=kernel).fit(TX[:1000], Y[:1000])
    # Omit the while kernel part (model.kernel_.k1), which is the observation noise.
    return model.kernel_.k2


def normalize_p_t(p_t: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Normalize p_t for individual treatment T when treatment space is discrete.

    The normlization enforces the constraint E[1 / p_t|T = t] = 1 for any t at the empirical level.
    """
    p_t_new = torch.zeros_like(p_t)
    for t in set(T):
        p_t_new[T == t] = p_t[T == t] * torch.mean((T == t) / p_t)
    return p_t_new


def assert_input(
    Y: torch.Tensor,
    T: torch.Tensor,
    X: torch.Tensor,
    p_t: torch.Tensor,
) -> None:
    assert len(Y.shape) == 1
    assert len(T.shape) == 1
    assert len(X.shape) == 2
    assert len(p_t.shape) == 1

    assert Y.shape[0] == T.shape[0] == X.shape[0]
    assert Y.dtype == X.dtype == X.dtype
