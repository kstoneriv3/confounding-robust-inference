import warnings
from typing import List, Literal

import cvxpy as cp
import numpy as np
import torch
from sklearn.gaussian_process.kernels import Kernel

from cri.estimators.base import BaseEstimator
from cri.estimators.constraints import (
    get_box_constraints,
    get_hajek_constraints,
    get_qb_constraint,
    get_zsb_box_constraints,
)
from cri.estimators.misc import DEFAULT_KERNEL, OrthogonalBasis, assert_input, normalize_p_t
from cri.policies import BasePolicy
from cri.utils.types import as_ndarrays, as_tensor

# one of default, ignore, error, always, module, once
WARNINGS_MODE: Literal["default", "error", "ignore", "always", "module", "once"] = "ignore"


class IPWEstimator(BaseEstimator):
    """Inverse Probability Weightning (IPW) estimator."""

    def fit(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
        X: torch.Tensor,
        p_t: torch.Tensor,
        policy: BasePolicy,
    ) -> BaseEstimator:
        assert_input(Y, T, X, p_t)
        self.Y = Y
        self.T = T
        self.X = X
        self.p_t = p_t
        self.policy = policy
        return self

    def predict(self) -> torch.Tensor:
        return torch.mean(self.Y * self.policy.prob(self.T, self.X) / self.p_t)


class HajekEstimator(BaseEstimator):
    """Hajek estimator."""

    def fit(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
        X: torch.Tensor,
        p_t: torch.Tensor,
        policy: BasePolicy,
    ) -> BaseEstimator:
        assert_input(Y, T, X, p_t)
        self.Y = Y
        self.T = T
        self.X = X
        self.p_t = p_t
        self.policy = policy
        return self

    def predict(self) -> torch.Tensor:
        p_t = normalize_p_t(self.p_t, self.T)
        return torch.mean(self.Y * self.policy.prob(self.T, self.X) / p_t)


class ZSBEstimator(BaseEstimator):
    """ZSB estimator from Zhao, Small, and Bhattacharya (2019) and MSM from Tan (2006).

    Args:
        Gamma: Sensitivity parameter satisfying Gamma >= 1.0. When Gamma == 1.0, ZSB estimator is
            equivalent to the Hajek estimator.
        use_fractional_programming: If True, linear fractional programming is used as in
            Zhao, Small, and Bhattacharya (2019). If False, it does not use linear fractional
            programming for lower bound calculation and instead normalize nominal propensity p_t
            and uses takes more primitive approach used by the Marginal Sensitivity Model (MSM)
            by Tan (2006).
    """

    def __init__(
        self, const_type: str, Gamma: float, use_fractional_programming: bool = True
    ) -> None:
        assert Gamma >= 1
        assert const_type in ["Tan_box", "lr_box"]
        self.Gamma = Gamma
        self.const_type = const_type
        self.use_fractional_programming = use_fractional_programming

    def fit(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
        X: torch.Tensor,
        p_t: torch.Tensor,
        policy: BasePolicy,
    ) -> BaseEstimator:
        assert_input(Y, T, X, p_t)
        self.Y = Y
        self.T = T
        self.X = X
        self.p_t = p_t
        self.policy = policy
        n = T.shape[0]

        # Necessary for ensuring the feasibility for small Gamma under box and Hajek constraints.
        if not self.use_fractional_programming:
            p_t = normalize_p_t(p_t, T)
        pi = policy.prob(T, X)
        r = Y * pi
        r_np, Y_np, T_np, p_t_np = as_ndarrays(r, Y, T, p_t)

        # For avoiding user warning about multiplication operator with `*` and `@`
        with warnings.catch_warnings():
            warnings.simplefilter(WARNINGS_MODE)

            w = cp.Variable(n)

            objective = cp.Minimize(cp.sum(r_np * w))

            constraints: List[cp.constraints.Constraint] = [np.zeros(n) <= w]
            constraints.extend(get_hajek_constraints(w, T_np, p_t_np))
            if self.use_fractional_programming:
                constraints.extend(
                    get_zsb_box_constraints(w, T_np, p_t_np, self.Gamma, self.const_type)
                )
            else:
                constraints.extend(get_box_constraints(w, p_t_np, self.Gamma, self.const_type))

            problem = cp.Problem(objective, constraints)
            problem.solve()

        if problem.status != "optimal":
            raise ValueError(
                "The optimizer found the associated convex programming to be {}.".format(
                    problem.status
                )
            )

        self.w = torch.zeros_like(p_t)
        self.w[:] = as_tensor(w.value)
        self.fitted_lower_bound = torch.mean(self.w * r)
        self.problem = problem
        return self

    def predict(self) -> torch.Tensor:
        return self.fitted_lower_bound


class QBEstimator(BaseEstimator):
    """Quantile balancing (QB) estimator by Dorn and Guo (2022).

    Args:
        Gamma: Sensitivity parameter satisfying Gamma >= 1.0. When Gamma == 1.0, QB estimator is
            equivalent to the IPW estimator.
        D: Dimension of the low-rank approximation used in the kernel quantile regression.
        kernel: Kernel used in the low-rank kernel quantile regression.
        const_type: Type of box constraints. A valid argument is either "Tan_box" (Tan's MSN) or
            "lr_box" (likelihood ratio).
    """

    def __init__(
        self,
        const_type: str,
        Gamma: float,
        D: int = 30,
        kernel: Kernel | None = None,
    ) -> None:
        assert const_type in ["Tan_box", "lr_box"]
        assert Gamma >= 1
        self.const_type = const_type
        self.Gamma = Gamma
        self.D = D
        self.kernel = kernel

    def fit(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
        X: torch.Tensor,
        p_t: torch.Tensor,
        policy: BasePolicy,
    ) -> BaseEstimator:
        assert_input(Y, T, X, p_t)
        self.Y = Y
        self.T = T
        self.X = X
        self.p_t = p_t
        self.policy = policy
        n = T.shape[0]

        pi = policy.prob(T, X)
        r = Y * pi
        r_np, Y_np, T_np, X_np, p_t_np, pi_np = as_ndarrays(r, Y, T, X, p_t, pi)
        TX_np = np.concatenate([T_np[:, None], X_np], axis=1)

        self.kernel = self.kernel if self.kernel is not None else DEFAULT_KERNEL
        self.Psi_np = OrthogonalBasis(self.D, self.kernel).fit_transform(TX_np)

        # For avoiding user warning about multiplication operator with `*` and `@`
        with warnings.catch_warnings():
            warnings.simplefilter(WARNINGS_MODE)

            w = cp.Variable(n)

            objective = cp.Minimize(r_np.T @ w)

            constraints: List[cp.constraints.Constraint] = [np.zeros(n) <= w]
            constraints.extend(get_box_constraints(w, p_t_np, self.Gamma, self.const_type))
            constraints.extend(
                get_qb_constraint(
                    w, Y_np, self.Psi_np, p_t_np, pi_np, self.Gamma, self.D, self.kernel
                )
            )

            problem = cp.Problem(objective, constraints)
            problem.solve()

        if problem.status != "optimal":
            raise ValueError(
                "The optimizer found the associated convex programming to be {}.".format(
                    problem.status
                )
            )

        self.w = torch.zeros_like(p_t)
        self.w[:] = as_tensor(w.value)
        self.fitted_lower_bound = torch.mean(self.w * r)
        self.problem = problem
        return self

    def predict(self) -> torch.Tensor:
        return self.fitted_lower_bound
