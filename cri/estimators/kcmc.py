import warnings
from typing import Callable, Dict, List

import cvxpy as cp
import numpy as np
import torch
from sklearn.decomposition import KernelPCA
from sklearn.gaussian_process.kernels import RBF, Kernel

from cri.estimators.base import BaseEstimator
from cri.estimators.constraints import get_box_constraints
from cri.estimators.misc import assert_input
from cri.policies import BasePolicy
from cri.utils.types import as_ndarrays

F_DIVERGENCES = [
    "KL",
    "inverse_KL",
    "Jensen_Shannon",
    "squared_Hellinger",
    "Pearson_chi_squared",
    "Neyman_chi_squared",
    "total_variation",
]

CONST_TYPES = F_DIVERGENCES + ["box"]

CVXPY_F_DIV_FUNCTIONS: Dict[str, Callable[[cp.Expression], cp.Expression]] = {
    "KL": lambda u: -cp.entr(u),
    "inverse_KL": lambda u: -cp.log(u),
    #'Jensen_Shannon': lambda u: -(u + 1) * cp.log(u + 1) + (u + 1) * np.log(2.) + u * cp.log(u)
    "Jensen_Shannon": lambda u: cp.entr(u + 1) + (u + 1) * np.log(2.0) - cp.entr(u),
    "squared_Hellinger": lambda u: u - 2 * cp.sqrt(u) + 1,
    "Pearson_chi_squared": lambda u: cp.square(u) - 1,
    "Neyman_chi_squared": lambda u: cp.inv_pos(u) - 1,
    "total_variation": lambda u: 0.5 * cp.abs(u - 1),
}


class KCMCEstimator(BaseEstimator):
    """Kernel Conditional Moment Constraints (KCMC) Estimator.

    Args:
        const_type: Type of the constraint used. It must be one of "box", "KL", "inverse_KL",
            "Jensen_Shannon", "squared_Hellinger", "Pearson_chi_squared", "Neyman_chi_squared",
            and "total_variation".
        gamma: Sensitivity parameter for f-divergence constraint satisfying Gamma >= 1.0.
            Default is None. When gamma == 0.0, QB estimator is equivalent to the IPW estimator.
        Gamma: Sensitivity parameter for box constraints satisfying Gamma >= 1.0. Default is None.
            When Gamma == 1.0, QB estimator is equivalent to the IPW estimator.
        D: Dimension of the low-rank approximation used in the kernel quantile regression.
        kernel: Kernel used in the low-rank kernel quantile regression.
    """

    def __init__(
        self,
        const_type: str,
        gamma: float | None = None,
        Gamma: float | None = None,
        D: int = 30,
        kernel: Kernel = RBF(),
    ) -> None:
        assert const_type in CONST_TYPES
        assert gamma is None or gamma >= 0
        assert Gamma is None or Gamma >= 1
        self.gamma = gamma if gamma is not None else 0.0
        self.Gamma = Gamma if Gamma is not None else 1.0
        self.D = D
        self.kernel = kernel

    def fit(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
        X: torch.Tensor,
        p_t: torch.Tensor,
        policy: BasePolicy,
    ) -> "BaseEstimator":
        assert_input(Y, T, X, p_t)
        self.Y = Y
        self.T = T
        self.X = X
        self.p_t = p_t
        self.policy = policy
        n = T.shape[0]

        # Necessary for ensuring the feasibility for small Gamma under box and Hajek constraints.
        pi = policy.prob(X, T)
        r = Y * pi
        r_np, Y_np, T_np, X_np, p_t_np, pi_np = as_ndarrays(r, Y, T, X, p_t, pi)

        # For avoiding user warning about multiplication operator with `*` and `@`
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            w = cp.Variable(n)

            objective = cp.Minimize(cp.sum(r_np * w))

            constraints: List[cp.Constraint] = [np.zeros(n) <= w]
            constraints.extend(
                get_kernel_constraints(w, T_np, X_np, p_t_np, pi_np, self.D, self.kernel)
            )
            if self.const_type == "box":
                constraints.extend(get_box_constraints(w, T_np, p_t_np, self.Gamma))
            else:
                constraints.extend(get_f_div_constraint(w, p_t_np, self.gamma, self.const_type))

            problem = cp.Problem(objective, constraints)
            problem.solve()

        if problem.status != "optimal":
            raise ValueError(
                "The optimizer found the associated convex programming to be {}.".format(
                    problem.status
                )
            )

        self.w = torch.zeros_like(p_t)
        self.w[:] = torch.as_tensor(w.value)
        self.fitted_lower_bound = torch.mean(w * r)
        self.problem = problem
        return self

    def predict(self) -> torch.Tensor:
        return self.fitted_lower_bound

    def predict_dual(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
        X: torch.Tensor,
        p_t: torch.Tensor,
        policy: BasePolicy,
    ) -> torch.Tensor:
        raise NotImplementedError


def get_kernel_constraints(
    w: cp.Variable,
    T: np.ndarray,
    X: np.ndarray,
    p_t: np.ndarray,
    pi: np.ndarray,
    D: int,
    kernel: Kernel,
) -> List[cp.Constraint]:
    TX = np.concatenate([T[:, None], X], axis=1)
    TX /= TX.std(axis=0)[None, :]
    Psi = KernelPCA(D, kernel=kernel).fit_transform(TX)
    # Carveat: np.ones(n) * w is NOT the element-wise product in cvxpy!!!
    # Numpy's broadcasting MATCHES THE LOWER DIMENSIONS and assume shape 1 at higher dimensions.
    return [(Psi.T * pi) @ w == np.sum(Psi.T * pi / p_t)]


def get_f_div_constraint(
    w: cp.Variable,
    p_t: np.ndarray,
    gamma: float,
    f_div_type: str,
) -> List[cp.Constraint]:
    n = p_t.shape[0]
    # Carveat: np.ones(n) * w is NOT the element-wise product in cvxpy!!!
    f = CVXPY_F_DIV_FUNCTIONS[f_div_type]
    constraints = [
        cp.sum(f(cp.multiply(w, p_t))) <= gamma * n,
        cp.scalar_product(w, p_t) == n,
    ]
    return constraints


class DualKCMCEstimator(BaseEstimator):
    pass


class GPKCMCEstimator(BaseEstimator):
    pass
