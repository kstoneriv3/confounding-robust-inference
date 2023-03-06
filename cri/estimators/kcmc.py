import warnings

import cvxpy as cp
import numpy as np
import torch
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Kernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from torch.autograd.functional import hessian, jacobian

from cri.estimators.base import BaseEstimator
from cri.estimators.constraints import (
    get_a_b,
    get_box_constraints,
    get_f_div_constraint,
    get_gp_constraints,
    get_kernel_constraints,
)
from cri.estimators.misc import (
    F_DIVERGENCES,
    assert_input,
    get_f_conjugate,
    get_orthogonal_basis,
    select_kernel,
)
from cri.policies import BasePolicy
from cri.utils.types import as_ndarrays

CONSTRAINT_TYPES = F_DIVERGENCES + ["Tan_box", "lr_box"]


class KCMCEstimator(BaseEstimator):
    """Kernel Conditional Moment Constraints (KCMC) Estimator.

    Args:
        const_type: Type of the constraint used. It must be one of "Tan_box", "lr_box", "KL",
            "inverse_KL", "Jensen_Shannon", "squared_Hellinger", "Pearson_chi_squared",
            "Neyman_chi_squared", and "total_variation".
        gamma: Sensitivity parameter for f-divergence constraint satisfying Gamma >= 1.0.
            When gamma == 0.0, QB estimator is equivalent to the IPW estimator.
        Gamma: Sensitivity parameter for box constraints satisfying Gamma >= 1.0.
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
        kernel: Kernel | None = None,
    ) -> None:
        assert const_type in CONSTRAINT_TYPES
        if "box" in const_type:
            assert Gamma is not None and Gamma >= 1
        else:
            assert gamma is not None and gamma >= 0
        self.const_type = const_type
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
        self.pi = policy.prob(X, T)

        n = T.shape[0]
        r = Y * self.pi
        r_np, Y_np, T_np, X_np, p_t_np, pi_np = as_ndarrays(r, Y, T, X, p_t, self.pi)

        self.kernel = self.kernel if self.kernel is not None else select_kernel(Y_np, T_np, X_np)
        self.Psi_np = get_orthogonal_basis(T_np, X_np, self.D, self.kernel)

        # For avoiding user warning about multiplication operator with `*` and `@`
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            w = cp.Variable(n)

            objective = cp.Minimize(cp.sum(r_np * w))

            constraints: list[cp.Constraint] = [np.zeros(n) <= w]
            kernel_consts = get_kernel_constraints(w, p_t_np, pi_np, self.Psi_np)
            constraints.extend(kernel_consts)
            if "box" in self.const_type:
                constraints.extend(
                    get_box_constraints(w, T_np, p_t_np, self.Gamma, self.const_type)
                )
            else:
                f_div_const = get_f_div_constraint(w, p_t_np, self.gamma, self.const_type)
                constraints.extend(f_div_const)

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
        self.eta_kcmc = torch.as_tensor(kernel_consts[0].dual_value)  # need to match sign!
        # For box constraints, the dual objective does not depend on eta_f so it does not matter.
        self.eta_f = torch.as_tensor(
            f_div_const[0].dual_value if "box" not in self.const_type else 1.0
        )
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
        assert hasattr(self, "fitted_lower_bound")
        T_np, X_np = as_ndarrays(T, X)
        Psi = torch.as_tensor(get_orthogonal_basis(T_np, X_np, self.D, self.kernel))
        pi = policy.prob(T, X)
        dual = dual_objective(
            Y, Psi, p_t, pi, self.eta_kcmc, self.eta_f, self.gamma, self.Gamma, self.const_type
        )
        return dual.mean()

    def predict_gic(self) -> torch.Tensor:
        n = self.Y.shape[0]
        scores = self._get_dual_jacobian()
        V = scores.T @ scores / n
        J = self._get_dual_hessian()
        J_inv = torch.pinv(J)
        gic = self.fitted_lower_bound + torch.einsum("ij, ji->", J_inv, V) / n
        return gic

    def _get_fitted_dual_loss(self, eta: torch.Tensor) -> torch.Tensor:
        if "box" in self.const_type:
            eta_kcmc = eta
            eta_f = torch.zeros(1)
        else:
            eta_kcmc = eta[:-1]
            eta_f = eta[-1]
        loss = dual_objective(
            self.Y,
            self.Psi,
            self.p_t,
            self.pi,
            eta_kcmc,
            eta_f,
            self.gamma,
            self.Gamma,
            self.const_type,
        )
        return loss

    def _get_dual_hessian(self) -> torch.Tensor:
        if "box" in self.const_type:
            # Use numpy for computation in this block.
            Y_np, T_np, X_np, p_t_np, pi_np, eta_kcmc_np = as_ndarrays(
                self.Y, self.T, self.X, self.p_t, self.pi, self.eta_kcmc
            )
            n = Y_np.shape[0]
            a, b = get_a_b(p_t_np, self.Gamma, self.const_type)

            # estimate p_{y|tx}(Pshi @ eta_kcmc)
            TX_np = np.concatenate([T_np, X_np], axis=1)
            TX_np = StandardScaler().fit_transform(TX_np)
            kernel = WhiteKernel() + ConstantKernel() * RBF()
            model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
            model.fit(TX_np[:1000], Y_np[:1000])
            y_mean, y_std = model.predict(TX_np, return_std=True)
            r_mean, r_std = map(lambda x: x * pi_np / p_t_np, [y_mean, y_std])
            conditional_pdf = norm.pdf(self.Psi_np @ eta_kcmc_np, loc=r_mean, scale=r_std)

            diag = np.diag(p_t_np * (b - a) * conditional_pdf)
            # TODO: fix dtype for torch.as_tensor
            H = torch.as_tensor(self.Psi_np.T @ diag @ self.Psi_np / n)
        else:
            eta = torch.concat([self.eta_kcmc, self.eta_f])
            eta = torch.tensor(eta.data, requires_grad=True)
            H = hessian(lambda eta: self.get_fitted_dual_loss(eta).mean(), eta)  # type: ignore
        return H

    def _get_dual_jacobian(self) -> torch.Tensor:
        if "box" in self.const_type:
            eta = torch.tensor(self.eta_kcmc.data, requires_grad=True)
        else:
            eta = torch.concat([self.eta_kcmc, self.eta_f])
            eta = torch.tensor(eta.data, requires_grad=True)
        H = jacobian(lambda eta: self.get_fitted_dual_loss(eta), eta)  # type: ignore
        raise H


def dual_objective(
    Y: torch.Tensor,
    Psi: torch.Tensor,
    p_t: torch.Tensor,
    pi: torch.Tensor,
    eta_kcmc: torch.Tensor,
    eta_f: torch.Tensor,
    gamma: float,
    Gamma: float,
    const_type: str,
) -> torch.Tensor:
    f_conj = get_f_conjugate(p_t, Gamma, const_type)
    dual = -eta_f * gamma + Psi @ eta_kcmc - eta_f * f_conj((Psi @ eta_kcmc - Y * pi / p_t) / eta_f)
    return dual


class DualKCMCEstimator(BaseEstimator):
    """TODO: Maybe fix torch.as_tensor's dtype issue first."""

    pass


class GPKCMCEstimator(BaseEstimator):
    """Gaussian process Kernel Conditional Moment Constraints (GP-KCMC) Estimator.

    Args:
        const_type: Type of the constraint used. It must be one of "box", "KL", "inverse_KL",
            "Jensen_Shannon", "squared_Hellinger", "Pearson_chi_squared", "Neyman_chi_squared",
            and "total_variation".
        gamma: Sensitivity parameter for f-divergence constraint satisfying Gamma >= 1.0.
            When gamma == 0.0, GP-KCMC estimator is equivalent to the IPW estimator.
        Gamma: Sensitivity parameter for box constraints satisfying Gamma >= 1.0.
            When Gamma == 1.0, GP-KCMC estimator is equivalent to the IPW estimator.
        alpha: (Bayesian) Significance level of credible interval.
        sigma2: Noise level if the GP model.
        kernel: Kernel used by Gaussian process.
    """

    def __init__(
        self,
        const_type: str,
        gamma: float | None = None,
        Gamma: float | None = None,
        alpha: float = 0.05,
        sigma2: float = 1.0,
        kernel: Kernel | None = None,
    ) -> None:
        assert const_type in CONSTRAINT_TYPES
        if "box" in const_type:
            assert Gamma is not None and Gamma >= 1
        else:
            assert gamma is not None and gamma >= 0
        assert sigma2 > 0
        assert 0 < alpha < 1
        self.const_type = const_type
        self.gamma = gamma if gamma is not None else 0.0
        self.Gamma = Gamma if Gamma is not None else 1.0
        self.alpha = alpha
        self.sigma2 = sigma2
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
        pi = policy.prob(X, T)
        r = Y * pi
        r_np, Y_np, T_np, X_np, p_t_np, pi_np = as_ndarrays(r, Y, T, X, p_t, pi)

        self.kernel = self.kernel if self.kernel is not None else select_kernel(Y_np, T_np, X_np)
        self.Psi_np = get_orthogonal_basis(T_np, X_np, self.D, self.kernel)

        # For avoiding user warning about multiplication operator with `*` and `@`
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            w = cp.Variable(n)

            objective = cp.Minimize(cp.sum(r_np * w))

            constraints: list[cp.Constraint] = [np.zeros(n) <= w]
            constraints.extend(
                get_gp_constraints(w, p_t_np, pi_np, self.Psi_np, self.sigma2, self.alpha)
            )
            if "box" in self.const_type:
                constraints.extend(
                    get_box_constraints(w, T_np, p_t_np, self.Gamma, self.const_type)
                )
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
