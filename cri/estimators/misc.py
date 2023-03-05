from typing import Callable

import cvxpy as cp
import numpy as np
import torch
from sklearn.decomposition import KernelPCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Kernel, WhiteKernel
from sklearn.preprocessing import StandardScaler


F_DIVERGENCES = [
    "KL",
    "inverse_KL",
    "Jensen_Shannon",
    "squared_Hellinger",
    "Pearson_chi_squared",
    "Neyman_chi_squared",
    "total_variation",
]

CVXPY_F_DIV_FUNCTIONS: dict[str, Callable[[cp.Expression], cp.Expression]] = {
    "KL": lambda u: -cp.entr(u),
    "inverse_KL": lambda u: -cp.log(u),
    #'Jensen_Shannon': lambda u: -(u + 1) * cp.log(u + 1) + (u + 1) * np.log(2.) + u * cp.log(u)
    "Jensen_Shannon": lambda u: cp.entr(u + 1) + (u + 1) * np.log(2.0) - cp.entr(u),
    "squared_Hellinger": lambda u: u - 2 * cp.sqrt(u) + 1,
    "Pearson_chi_squared": lambda u: cp.square(u) - 1,
    "Neyman_chi_squared": lambda u: cp.inv_pos(u) - 1,
    "total_variation": lambda u: 0.5 * cp.abs(u - 1),
}


def get_orthogonal_basis(
    T: np.ndarray,
    X: np.ndarray,
    D: int,
    kernel: Kernel,
) -> np.ndarray:
    """Calculate kernel PCA's (empirically) orthogonal features and intercept (constant feature).

    Returns:
        Feature matrix of shape (n, D + 1).
    """
    TX = np.concatenate([T[:, None], X], axis=1)
    TX = StandardScaler().fit_transform(TX)
    Psi = KernelPCA(D, kernel=kernel).fit_transform(TX)
    Psi = np.concatenate([Psi, np.ones_like(Psi[:, :1])], axis=1)
    return Psi


def select_kernel(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
) -> Kernel:
    """Learn the kernel parameters by fitting Gaussian process regression of Y on (T, X)."""
    TX = np.concatenate([T[:, None], X], axis=1)
    TX = StandardScaler().fit_transform(TX)
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
