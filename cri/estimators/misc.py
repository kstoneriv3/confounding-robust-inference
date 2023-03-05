from typing import List

import cvxpy as cp
import numpy as np
import torch
from sklearn.decomposition import KernelPCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Kernel, WhiteKernel

from cri.utils.quantile_regression import TorchQuantileRegressor
from cri.utils.types import as_ndarrays, as_tensors


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


def get_hajek_constraints(
    w: cp.Variable, T: np.ndarray, p_t: np.ndarray
) -> List[cp.constraints.Constraint]:
    n = T.shape[0]
    constraints = []
    for t in set(T):
        constraints.append(cp.sum(w[T == t]) == n)
    return constraints


def get_a_b(p_t: np.ndarray, Gamma: float) -> tuple[np.ndarray, np.ndarray]:
    a = 1 + 1 / Gamma * (1 / p_t - 1)
    b = 1 + Gamma * (1 / p_t - 1)
    return a, b


def get_zsb_box_constraints(
    w: cp.Variable,
    T: np.ndarray,
    p_t: np.ndarray,
    Gamma: float,
) -> List[cp.constraints.Constraint]:
    scale = cp.Variable(1)
    a, b = get_a_b(p_t, Gamma)
    return [0 <= scale, scale * a <= w, w <= scale * b]


def get_box_constraints(
    w: cp.Variable,
    T: np.ndarray,
    p_t: np.ndarray,
    Gamma: float,
) -> List[cp.constraints.Constraint]:
    a, b = get_a_b(p_t, Gamma)
    return [a <= w, w <= b]


def get_qb_constraint(
    w: cp.Variable,
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    p_t: np.ndarray,
    pi: np.ndarray,
    Gamma: float,
    D: int,
) -> List[cp.constraints.Constraint]:
    TX = np.concatenate([T[:, None], X], axis=1)
    TX /= TX.std(axis=0)[None, :]
    kernel = select_kernel(Y, T, X)
    Psi = KernelPCA(D, kernel=kernel).fit_transform(TX)
    tau = 1 / (Gamma + 1)
    tY, tPsi = as_tensors(Y, Psi)
    # sklearn's QuantileRegressor is slow for large data set.
    tQ = TorchQuantileRegressor(quantile=tau).fit(tPsi, tY).predict(tPsi)
    (Q,) = as_ndarrays(tQ)
    # Carveat: np.ones(n) * w is NOT the element-wise product in cvxpy!!!
    return [cp.scalar_product(pi * Q, w) == np.sum(pi * Q / p_t)]


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
    return model.kernel_
