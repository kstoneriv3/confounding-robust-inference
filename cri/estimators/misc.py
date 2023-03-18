from typing import Callable

import cvxpy as cp
import numpy as np
import torch
from sklearn.decomposition import KernelPCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Kernel, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from cri.utils.types import _DEFAULT_TORCH_FLOAT_DTYPE, as_ndarrays, as_tensors

F_DIVERGENCES = [
    "KL",
    "inverse_KL",
    # "Jensen_Shannon",  # Unfortunately, CVXPY does not recognize it as convex
    "squared_Hellinger",
    "Pearson_chi_squared",
    "Neyman_chi_squared",
    "total_variation",
]

CONSTRAINT_TYPES = F_DIVERGENCES + ["Tan_box", "lr_box"]

# Some of the f_conjugate have limited domain, which makes it difficult to fit with dual problem.
DUAL_FEASIBLE_CONSTRAINT_TYPES = [
    "KL",
    "Pearson_chi_squared",
    "Tan_box",
    "lr_box",
]

CVXPY_F_DIV_FUNCTIONS: dict[str, Callable[[cp.Expression], cp.Expression]] = {
    "KL": lambda u: -cp.entr(u),
    "inverse_KL": lambda u: -cp.log(u),
    "Jensen_Shannon": lambda u: (
        -0.5 * (u + 1) * cp.log(u + 1) + 0.5 * (u + 1) * np.log(2.0) + 0.5 * u * cp.log(u)
    ),
    "squared_Hellinger": lambda u: u - 2 * cp.sqrt(u) + 1,
    "Pearson_chi_squared": lambda u: cp.square(u - 1),
    "Neyman_chi_squared": lambda u: cp.inv_pos(u) - 1,
    "total_variation": lambda u: 0.5 * cp.abs(u - 1),
}

TORCH_F_DIV_FUNCTIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "KL": lambda u: u * torch.log(u),
    "inverse_KL": lambda u: -torch.log(u),
    "Jensen_Shannon": lambda u: -(u + 1) * torch.log((1 + u) / 2) + u * torch.log(u),
    "squared_Hellinger": lambda u: u - 2 * torch.sqrt(u) + 1,
    "Pearson_chi_squared": lambda u: (u - 1) ** 2,
    "Neyman_chi_squared": lambda u: 1.0 / u - 1,
    "total_variation": lambda u: 0.5 * torch.abs(u - 1),
}

# See
# - https://arxiv.org/abs/1606.00709
# - http://www.stat.yale.edu/~yw562/teaching/598/lec06.pdf
# - https://en.wikipedia.org/wiki/F-divergence
TORCH_F_DIV_CONJUGATE_FUNCTIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "KL": lambda v: torch.exp(v - 1),
    "inverse_KL": lambda v: -1 - torch.log(-v),
    "Jensen_Shannon": lambda v: torch.where(
        v < torch.log(torch.as_tensor(2.0)), -torch.log(2 - torch.exp(v)), torch.inf
    ),
    "squared_Hellinger": lambda v: torch.where(v < 1.0, v / (1 - v), torch.inf),
    "Pearson_chi_squared": lambda v: v**2 / 4.0 + v,
    "Neyman_chi_squared": lambda v: torch.where(
        torch.abs(v) <= 0.0, -2 * torch.sqrt(-v) + 1, torch.inf
    ),
    "total_variation": lambda v: torch.where(torch.abs(v) < 0.5, v, torch.inf),
}


def get_dual_objective(
    Y: torch.Tensor,
    p_t: torch.Tensor,
    pi: torch.Tensor,
    eta_cmc: torch.Tensor,
    eta_f: torch.Tensor,
    gamma: float,
    Gamma: float,
    const_type: str,
) -> torch.Tensor:
    f_conj = get_f_conjugate(p_t, Gamma, const_type)
    dual = -gamma * eta_f + eta_cmc - eta_f * f_conj((eta_cmc - Y * pi / p_t) / eta_f)
    # TODO
    print(list(map(torch.mean, (-gamma * eta_f, eta_cmc, -eta_f * f_conj((eta_cmc - Y * pi / p_t) / eta_f)))))
    return dual


def get_f_conjugate(
    p_t: torch.Tensor, Gamma: float, const_type: str
) -> Callable[[torch.Tensor], torch.Tensor]:
    if "box" in const_type:
        (p_t_np,) = as_ndarrays(p_t)
        a_np, b_np = get_a_b(p_t_np, Gamma, const_type)
        a, b = as_tensors(a_np, b_np)
        a_w_tilde = a * p_t
        b_w_tilde = b * p_t
        f_conj: Callable[[torch.Tensor], torch.Tensor]

        def f_conj(v: torch.Tensor) -> torch.Tensor:
            return torch.where(v < 0.0, a_w_tilde * v, b_w_tilde * v)

    else:
        f_conj = TORCH_F_DIV_CONJUGATE_FUNCTIONS[const_type]
    return f_conj


class OrthogonalBasis:
    """Calculate kernel PCA's (empirically) orthogonal features and constant feature (intercept)."""

    def __init__(self, D: int, kernel: Kernel) -> None:
        add_intercept = FunctionTransformer(
            lambda X: np.concatenate([X, np.ones_like(X[:, :1])], axis=1)
        )
        self.pipeline = Pipeline(
            [
                ("StandardScalaer", StandardScaler()),
                ("KernelPCA", KernelPCA(D, kernel=kernel)),
                ("AddIntercept", add_intercept),
            ]
        )

    def fit(self, X: np.ndarray) -> "OrthogonalBasis":
        self.pipeline.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.fit_transform(X)


def select_kernel(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
) -> Kernel:
    """Learn the kernel parameters by fitting Gaussian process regression of Y on (T, X)."""
    Y = Y - Y.mean()  # It is OK to demean Y, as Psi contains intercept term (constant feature).
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


def get_a_b(p_t: np.ndarray, Gamma: float, const_type: str) -> tuple[np.ndarray, np.ndarray]:
    # Tan's Marginal Sensitivity Model (MSM), 1 / Gamma <= {p_t (1 - pi)} / {(1 - p_t) pi} <= Gamma.
    if const_type == "Tan_box":
        a = 1 + 1 / Gamma * (1 / p_t - 1)
        b = 1 + Gamma * (1 / p_t - 1)
    # Likelihood ratio constraints 1 / Gamma <= p_t / pi <= Gamma.
    elif const_type == "lr_box":
        a = (1 / Gamma) * (1 / p_t)
        b = Gamma * (1 / p_t)
    else:
        raise ValueError('A valid const_type is either "tan_box" or "lr_box"')
    return a, b


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

    n = Y.shape[0]
    assert T.shape[0] == n
    assert X.shape[0] == n
    assert p_t.shape[0] == n
    assert Y.dtype == _DEFAULT_TORCH_FLOAT_DTYPE
    assert X.dtype == _DEFAULT_TORCH_FLOAT_DTYPE
    assert p_t.dtype == _DEFAULT_TORCH_FLOAT_DTYPE
