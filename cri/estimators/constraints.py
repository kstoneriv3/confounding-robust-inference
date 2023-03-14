from typing import List

import cvxpy as cp
import numpy as np
from scipy.stats import chi2
from sklearn.gaussian_process.kernels import Kernel

from cri.estimators.misc import CVXPY_F_DIV_FUNCTIONS, get_a_b
from cri.utils.quantile_regression import TorchQuantileRegressor
from cri.utils.types import as_ndarrays, as_tensors


def get_hajek_constraints(w: cp.Variable, T: np.ndarray, p_t: np.ndarray) -> List[cp.Constraint]:
    n = T.shape[0]
    constraints = []
    for t in set(T):
        constraints.append(cp.sum(w[T == t]) == n)
    return constraints


def get_zsb_box_constraints(
    w: cp.Variable,
    T: np.ndarray,
    p_t: np.ndarray,
    Gamma: float,
    const_type: str,
) -> List[cp.Constraint]:
    setT = set(T)
    scale = cp.Variable(len(setT))
    a, b = get_a_b(p_t, Gamma, const_type)
    constraints = [np.zeros(len(setT)) <= scale]
    for t in setT:
        constraints.append(scale[t] * a[T == t] <= w[T == t])
        constraints.append(w[T == t] <= scale[t] * b[T == t])
    return constraints


def get_box_constraints(
    w: cp.Variable,
    p_t: np.ndarray,
    Gamma: float,
    const_type: str,
) -> List[cp.Constraint]:
    a, b = get_a_b(p_t, Gamma, const_type)
    return [a <= w, w <= b]


def get_qb_constraint(
    w: cp.Variable,
    Y: np.ndarray,
    Psi: np.ndarray,
    p_t: np.ndarray,
    pi: np.ndarray,
    Gamma: float,
    D: int,
    kernel: Kernel,
) -> List[cp.Constraint]:
    tau = 1 / (Gamma + 1)
    tY, tPsi = as_tensors(Y, Psi)
    # sklearn's QuantileRegressor is slow for large data set.
    tQ = TorchQuantileRegressor(quantile=tau).fit(tPsi, tY).predict(tPsi)
    (Q,) = as_ndarrays(tQ)
    # Carveat: np.ones(n) * w is NOT the element-wise product in cvxpy!!!
    return [cp.scalar_product(pi * Q, w) == np.sum(pi * Q / p_t)]


def get_kernel_constraints(
    w: cp.Variable,
    p_t: np.ndarray,
    Psi: np.ndarray,
) -> List[cp.Constraint]:
    # Carveat: np.ones(n) * w is NOT the element-wise product in cvxpy!!!
    # Numpy's broadcasting MATCHES THE LOWER DIMENSIONS and assume shape 1 at higher dimensions.
    return [(Psi.T * p_t) @ w == Psi.T @ np.ones_like(p_t)]


def get_gp_constraints(
    w: cp.Variable,
    p_t: np.ndarray,
    pi: np.ndarray,
    Psi: np.ndarray,
    sigma2: float,
    alpha: float,
) -> List[cp.Constraint]:
    """Returns GP-KCMC using low-rank approximation.

    Args:
        w: Weight for the optimization problem
        p_t: Nominal propensity p_obs(T|X)
        pi: Probability of the target policy pi(T|X)
        Psi: orthogonal basis obtained by kernel PCA
        sigma2: conditional variance Var[e|t, x] of e:= w * p_t - 1
        alpha: (Baysian) significance level for credible set

    Returns:
        GP constraints
    """
    n, d = Psi.shape
    S = np.sum(Psi**2, axis=0)  # Here, diag(S) = Psi.T @ Psi, as Psi's columns are orthogonal.
    e = cp.Variable(n)
    # z := [\Psi (\Psi^T \Psi + \sigma2 \I_d)^{-1} Psi.T]^{1 / 2} e so that
    # e^T [...] e = \|z\|^2.
    z = cp.Variable(d)
    chi2_bound = chi2(df=d).ppf(1 - alpha)
    # Carveat: np.ones(n) * w is NOT the element-wise product in cvxpy!!!
    constraints = [
        e == 1 - cp.multiply(p_t, w),
        z == (np.diag(np.sqrt(1 / (S + sigma2))) @ Psi.T) @ e,
        cp.sum(z**2) <= chi2_bound,
    ]
    return constraints


def get_f_div_constraint(
    w: cp.Variable,
    p_t: np.ndarray,
    gamma: float,
    f_div_type: str,
) -> List[cp.Constraint]:
    n = p_t.shape[0]
    f = CVXPY_F_DIV_FUNCTIONS[f_div_type]
    # Carveat: np.ones(n) * w is NOT the element-wise product in cvxpy!!!
    constraints = [
        cp.sum(f(cp.multiply(w, p_t))) <= gamma * n,
    ]
    return constraints
