from typing import List

import cvxpy as cp
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.gaussian_process.kernels import Kernel

from cri.utils.quantile_regression import TorchQuantileRegressor
from cri.utils.types import as_ndarrays, as_tensors


def get_hajek_constraints(w: cp.Variable, T: np.ndarray, p_t: np.ndarray) -> List[cp.Constraint]:
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
) -> List[cp.Constraint]:
    scale = cp.Variable(1)
    a, b = get_a_b(p_t, Gamma)
    return [0 <= scale, scale * a <= w, w <= scale * b]


def get_box_constraints(
    w: cp.Variable,
    T: np.ndarray,
    p_t: np.ndarray,
    Gamma: float,
) -> List[cp.Constraint]:
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
    kernel: Kernel,
) -> List[cp.Constraint]:
    TX = np.concatenate([T[:, None], X], axis=1)
    TX /= TX.std(axis=0)[None, :]
    Psi = KernelPCA(D, kernel=kernel).fit_transform(TX)
    tau = 1 / (Gamma + 1)
    tY, tPsi = as_tensors(Y, Psi)
    # sklearn's QuantileRegressor is slow for large data set.
    tQ = TorchQuantileRegressor(quantile=tau).fit(tPsi, tY).predict(tPsi)
    (Q,) = as_ndarrays(tQ)
    # Carveat: np.ones(n) * w is NOT the element-wise product in cvxpy!!!
    return [cp.scalar_product(pi * Q, w) == np.sum(pi * Q / p_t)]
