import warnings

import cvxpy as cp
import numpy as np
import torch
from scipy.linalg import eigh
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Kernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from torch.autograd.functional import hessian, jacobian
from torch.optim import SGD

from cri.estimators.base import BaseEstimator
from cri.estimators.constraints import (
    get_a_b,
    get_box_constraints,
    get_f_div_constraint,
    get_gp_constraints,
    get_kernel_constraints,
)
from cri.estimators.misc import (
    CONSTRAINT_TYPES,
    DUAL_FEASIBLE_CONSTRAINT_TYPES,
    OrthogonalBasis,
    assert_input,
    get_dual_objective,
    select_kernel,
)
from cri.policies import BasePolicy
from cri.utils.types import _DEFAULT_TORCH_FLOAT_DTYPE, as_ndarrays, as_tensor


class KCMCEstimator(BaseEstimator):
    """Kernel Conditional Moment Constraints (KCMC) Estimator.

    Args:
        const_type: Type of the constraint used. It must be one of "Tan_box", "lr_box", "KL",
            "inverse_KL", "squared_Hellinger", "Pearson_chi_squared",
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
            # For box constraint, it is convenient to assume gamma = 0.0, as eta_f -> +0
            # in the dual problem.
            gamma = 0.0
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
        self.pi = policy.prob(T, X)

        n = T.shape[0]
        r = Y * self.pi
        r_np, Y_np, T_np, X_np, p_t_np, pi_np = as_ndarrays(r, Y, T, X, p_t, self.pi)
        TX_np = np.concatenate([T_np[:, None], X_np], axis=1)

        self.kernel = self.kernel if self.kernel is not None else select_kernel(Y_np, T_np, X_np)
        self.Psi_np_pipeline = OrthogonalBasis(self.D, self.kernel)
        self.Psi_np = self.Psi_np_pipeline.fit_transform(TX_np) / p_t_np[:, None] # this makes it harder somehow...

        # For avoiding user warning about multiplication operator with `*` and `@`
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            w = cp.Variable(n)

            objective = cp.Minimize(cp.sum(r_np * w))

            constraints: list[cp.Constraint] = [np.zeros(n) <= w] 
            # constraints: list[cp.Constraint] = [np.zeros(n) <= w, w == 1 / p_t_np]  # TODO
            kernel_consts = get_kernel_constraints(w, p_t_np, self.Psi_np)
            constraints.extend(kernel_consts)
            if "box" in self.const_type:
                constraints.extend(get_box_constraints(w, p_t_np, self.Gamma, self.const_type))
            else:
                f_div_const = get_f_div_constraint(w, p_t_np, self.gamma, self.const_type)
                constraints.extend(f_div_const)

            problem = cp.Problem(objective, constraints)
            solvers = [cp.ECOS, cp.SCS]  # Available solvers can be check cp.installed_solvers()
            self.try_solvers(problem, solvers)
            # problem.solve(solver=cp.ECOS, max_iters=1000, abstol=1e-12) #, verbose=True)  # TODO

        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise ValueError(
                "The optimizer found the associated convex programming to be {}.".format(
                    problem.status
                )
            )

        self.w = torch.zeros_like(p_t)
        self.w[:] = as_tensor(w.value)

        print(problem.status)  # TODO
        print(torch.sum((self.w * p_t) ** 2 - 1) / p_t.shape[0])  # TODO
        print(torch.mean(self.w * r))

       # The solution is very inaccurate for p_t >= eps!
        eps = 5e-2
        print(torch.mean(self.w * (p_t < eps) * r + 1 / p_t * (p_t >= eps) * r))  # close to the solution
        print(torch.mean(self.w * (p_t >= eps) * r + 1 / p_t * (p_t < eps) * r))  # far from the solution

        print(torch.mean((r * (self.w - 1 / p_t) * (p_t < eps))**))
        print(torch.mean((r * (self.w - 1 / p_t) * (p_t >= eps))**))

        self.fitted_lower_bound = torch.mean(self.w * r)
        self.problem = problem
        self.eta_kcmc = as_tensor(-kernel_consts[0].dual_value)  # need to match sign!
        # For box constraints, the dual objective does not depend on eta_f so it does not matter.
        self.eta_f = as_tensor(f_div_const[0].dual_value if "box" not in self.const_type else 1.0)
        return self

    @staticmethod  # maybe make it a standalone function
    def try_solvers(problem: cp.Problem, solvers: list[str]):
        for solver in solvers:
            try:
                problem.solve(solver=solver) #, verbose=True)
            except cp.error.SolverError:
                pass
            if problem.status == "optimal":
                break
        print(solver)

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
        assert_input(Y, T, X, p_t)
        pi = policy.prob(T, X)
        T_np, X_np, p_t_np = as_ndarrays(T, X, p_t)
        TX_np = np.concatenate([T_np[:, None], X_np], axis=1)
        Psi_np = self.Psi_np_pipeline.transform(TX_np) / p_t_np[:, None]
        Psi = as_tensor(Psi_np)
        eta_cmc = Psi @ self.eta_kcmc * pi / p_t
        dual = get_dual_objective(
            Y, p_t, pi, eta_cmc, self.eta_f, self.gamma, self.Gamma, self.const_type
        )
        return dual.mean()

    def predict_gic(self) -> torch.Tensor:
        """Calculate the Generalized Information Criterion (GIC) of the lower bound."""
        n = self.Y.shape[0]
        scores = self._get_dual_jacobian()
        V = scores.T @ scores / n
        J = self._get_dual_hessian()
        J_inv = torch.pinv(J)
        gic = self.fitted_lower_bound - torch.einsum("ij, ji->", J_inv, V) / n
        return gic

    def predict_ci(
        self, n_boot: int = 2**20, alpha: float = 0.05
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate confidence interval of the lower bound.

        Bootstrap with used for calculating the percentile of the asymptotic distribution.

        Args:
            n_boot: The number of Monte Carlo samples used to calculate the percentile of the
                asymptotic distribution by bootstraping.
            alpha: Significance level of used for the confidence interval.
        """
        n = self.Y.shape[0]
        losses = self._get_fitted_dual_loss(self.eta)
        scores = self._get_dual_jacobian()
        l_s = torch.concat([losses, scores], dim=1)
        V_joint = l_s.T @ l_s.T / n
        J = self._get_dual_hessian()
        J_inv = torch.pinv(J)

        S, U = eigh(V_joint / n)
        en = torch.quasirandom.SobolEngine(1 + self.eta.shape[0])  # type: ignore
        normal_samples = en.draw(n_boot)
        boot_l_s = U @ torch.sqrt(S) @ normal_samples
        boot_l_s[:, 0] += self.fitted_lower_bound
        boot_losses, boot_scores = torch.split(boot_l_s, [1, self.eta_shape[0]])
        boot_lb = boot_losses - torch.einsum("ni,ij,jn->n", boot_scores.T, J_inv, boot_scores) / n
        low = torch.quantile(boot_lb, alpha / 2)
        high = torch.quantile(boot_lb, 1 - alpha / 2)
        return low, high

    def _get_fitted_dual_loss(self, eta: torch.Tensor) -> torch.Tensor:
        # The dual objective does not depend on eta_f for box constraints
        if "box" in self.const_type:
            eta_kcmc = eta
            eta_f = torch.zeros(1)
        else:
            eta_kcmc = eta[:-1]
            eta_f = eta[-1]
        eta_cmc = as_tensor(self.Psi_np) @ eta_kcmc * self.pi / self.p_t
        loss = get_dual_objective(
            self.Y,
            self.p_t,
            self.pi,
            eta_cmc,
            eta_f,
            self.gamma,
            self.Gamma,
            self.const_type,
        )
        return loss

    @property
    def eta(self) -> torch.Tensor:
        # The dual objective does not depend on eta_f for box constraints so ignore eta_f then.
        if "box" in self.const_type:
            return self.eta_kcmc.data
        else:
            return torch.concat([self.eta_kcmc, self.eta_f])

    def _get_dual_hessian(self) -> torch.Tensor:
        if "box" in self.const_type:
            # Use numpy for computation in this block.
            Y_np, T_np, X_np, p_t_np, pi_np, eta_kcmc_np = as_ndarrays(
                self.Y, self.T, self.X, self.p_t, self.pi, self.eta_kcmc
            )
            n = Y_np.shape[0]
            a, b = get_a_b(p_t_np, self.Gamma, self.const_type)

            # estimate p_{y|tx}(Pshi @ eta_kcmc)
            TX_np = np.concatenate([T_np[:, None], X_np], axis=1)
            TX_np = StandardScaler().fit_transform(TX_np)
            kernel = WhiteKernel() + ConstantKernel() * RBF()
            model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
            model.fit(TX_np[:1000], Y_np[:1000])
            y_mean, y_std = model.predict(TX_np, return_std=True)
            r_mean, r_std = map(lambda x: x * pi_np / p_t_np, [y_mean, y_std])
            conditional_pdf = norm.pdf(self.Psi_np @ eta_kcmc_np, loc=r_mean, scale=r_std)

            diag = np.diag(p_t_np * (b - a) * conditional_pdf)
            H = as_tensor(self.Psi_np.T @ diag @ self.Psi_np / n)
        else:
            eta = torch.tensor(self.eta.data, requires_grad=True)
            H = hessian(lambda eta: self.get_fitted_dual_loss(eta).mean(), eta)  # type: ignore
        return H

    def _get_dual_jacobian(self) -> torch.Tensor:
        eta = torch.tensor(self.eta.data, requires_grad=True)
        H = jacobian(lambda eta: self.get_fitted_dual_loss(eta), eta)  # type: ignore
        raise H


class DualKCMCEstimator(BaseEstimator):
    """Dual Kernel Conditional Moment Constraints (KCMC) Estimator.

    Args:
        const_type: Type of the constraint used. It must be one of "Tan_box", "lr_box", "KL",
            "inverse_KL", "Pearson_chi_squared".
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
        assert const_type in DUAL_FEASIBLE_CONSTRAINT_TYPES, f"{const_type} is not supported."
        if "box" in const_type:
            assert Gamma is not None and Gamma >= 1
            # For box constraint, it is convenient to assume gamma = 0.0, as eta_f -> +0
            # in the dual problem.
            gamma = 0.0
        else:
            assert gamma is not None and gamma >= 0
        self.const_type = const_type
        self.gamma = gamma if gamma is not None else 0.0
        self.Gamma = Gamma if Gamma is not None else 1.0
        self.D = D
        self.kernel = kernel
        self.eta_kcmc = torch.zeros(D + 1, dtype=_DEFAULT_TORCH_FLOAT_DTYPE, requires_grad=True)
        self.log_eta_f = torch.zeros(1, dtype=_DEFAULT_TORCH_FLOAT_DTYPE, requires_grad=True)

    def fit(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
        X: torch.Tensor,
        p_t: torch.Tensor,
        policy: BasePolicy,
        lr: float = 3e-2,
        n_steps: int = 50,
        batch_size: int = 1024,
        seed: int = 0,
    ) -> "BaseEstimator":
        assert_input(Y, T, X, p_t)
        self.Y = Y
        self.T = T
        self.X = X
        self.p_t = p_t
        self.policy = policy
        self.pi = policy.prob(T, X)
        np.random.seed(seed)

        n = T.shape[0]
        batch_size = min(n, batch_size)
        r = Y * self.pi
        r_np, Y_np, T_np, X_np, p_t_np, pi_np = as_ndarrays(r, Y, T, X, p_t, self.pi)
        TX_np = np.concatenate([T_np[:, None], X_np], axis=1)

        self.kernel = self.kernel if self.kernel is not None else select_kernel(Y_np, T_np, X_np)
        self.Psi_np_pipeline = OrthogonalBasis(self.D, self.kernel)
        self.Psi_np = self.Psi_np_pipeline.fit_transform(TX_np) / p_t_np[:, None]

        optimizer = SGD(params=[self.eta_kcmc, self.log_eta_f], lr=lr)
        for i in range(n_steps):
            train_idx = torch.randint(n, (batch_size,))
            eta_cmc = (
                as_tensor(self.Psi_np)[train_idx]
                @ self.eta_kcmc
                * self.pi[train_idx]
                / self.p_t[train_idx]
            )
            objective = -get_dual_objective(
                self.Y[train_idx],
                self.p_t[train_idx],
                self.pi[train_idx],
                eta_cmc,
                self.eta_f,
                self.gamma,
                self.Gamma,
                self.const_type,
            ).mean()
            objective.backward()  # type: ignore
            optimizer.step()
            optimizer.zero_grad()

        m = 1024
        lower_bounds = torch.zeros(n)
        for i in range((n + m - 1) // m):
            val_idx = slice(m * i, min(n, m * (i + 1)))
            with torch.no_grad():
                eta_cmc = (
                    as_tensor(self.Psi_np)[val_idx]
                    @ self.eta_kcmc
                    * self.pi[val_idx]
                    / self.p_t[val_idx]
                )
                lower_bounds[val_idx] = get_dual_objective(
                    self.Y[val_idx],
                    self.p_t[val_idx],
                    self.pi[val_idx],
                    eta_cmc,
                    self.eta_f,
                    self.gamma,
                    self.Gamma,
                    self.const_type,
                )
        self.fitted_lower_bound = torch.mean(lower_bounds).to(_DEFAULT_TORCH_FLOAT_DTYPE)
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
        assert_input(Y, T, X, p_t)
        T_np, X_np, p_t_np = as_ndarrays(T, X, p_t)
        TX_np = np.concatenate([T_np[:, None], X_np], axis=1)
        Psi_np = self.Psi_np_pipeline.transform(TX_np) / p_t_np[:, None]
        pi = policy.prob(T, X)
        eta_cmc = as_tensor(Psi_np) @ self.eta_kcmc * pi / p_t
        dual = get_dual_objective(
            Y, p_t, pi, eta_cmc, self.eta_f, self.gamma, self.Gamma, self.const_type
        )
        return dual.mean()

    @property
    def eta_f(self) -> torch.Tensor:
        return self.log_eta_f.exp()


class GPKCMCEstimator(BaseEstimator):
    """Gaussian Process Kernel Conditional Moment Constraints (GP-KCMC) Estimator.

    Args:
        const_type: Type of the constraint used. It must be one of "box", "KL", "inverse_KL",
            "squared_Hellinger", "Pearson_chi_squared", "Neyman_chi_squared", and
            "total_variation".
        gamma: Sensitivity parameter for f-divergence constraint satisfying Gamma >= 1.0.
            When gamma == 0.0, GP-KCMC estimator is equivalent to the IPW estimator.
        Gamma: Sensitivity parameter for box constraints satisfying Gamma >= 1.0.
            When Gamma == 1.0, GP-KCMC estimator is equivalent to the IPW estimator.
        D: Dimension of the low-rank approximation used in the kernel quantile regression.
        alpha: (Bayesian) Significance level of credible interval.
        sigma2: Noise level if the GP model.
        kernel: Kernel used by Gaussian process.


    Here, as low-rank GP is equivalent to Bayesian ridge regression with design matrix
    :math:`\\Psi_{n,d}=\psi_d(t_n, x_n)`. Thus, we use the following model with
    :math:`\\sigma^2` estimated by empirical Bayes\:

    .. math::
       :nowrap:

       \\begin{eqnarray}
          \\beta &\\sim N(0, I_d), \\\\
          e &= \\Psi \\beta + \\varepsilon, \\\\
          \\varepsilon &\sim N(0, \sigma^2 I_n). \\\\
       \\end{eqnarray}

    For this model, the posterior and the credible set (highest posterior density set) of
    :math:`\\beta` are

    .. math::
       :nowrap:

       \\begin{eqnarray}
          \\beta | e &\\sim N(\\mu_{\\beta|e}, \\Sigma_{\\beta|e})  \\\\
       \\end{eqnarray}

    and

    .. math::
       :nowrap:

       \\begin{eqnarray}
          \\mathrm{CI}_{\\beta|e}(1 - \\alpha) =
          \\{\\beta:
              (\\beta - \\mu_{\\beta|e})^T \\Sigma_{\\beta|e}^{-1} (\\beta - \\mu_{\\beta|e})
              \\leq \chi^2_d(1 - \\alpha)
          \\},
       \\end{eqnarray}

    where :math:`\\mu_\\beta=(\\Psi^T\\Psi+\\sigma^2I_d)^{-1}\\Psi^Te`
    and :math:`\\Sigma_\\beta=(\\Psi^T\\Psi+\\sigma^2I_d)^{-1}`. Therefore, the condition 
    :math:`0_d\\in \mathrm{CI}_{\\beta|e}(1 - \\alpha)` can be written as

    .. math::
       :nowrap:

       \\begin{equation}
          \\mu_{\\beta|e}^T \\Sigma_{\\beta|e}^{-1} \\mu_{\\beta|e}
          = e^T \\Psi (\\Psi^T\\Psi + \\sigma^2I_d)^{-1} \\Psi^Te
          \\leq \chi^2_d(1 - \\alpha).
       \\end{equation}
    """

    def __init__(
        self,
        const_type: str,
        gamma: float | None = None,
        Gamma: float | None = None,
        D: int = 30,
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
        self.D = D
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
        pi = policy.prob(T, X)
        r = Y * pi
        r_np, Y_np, T_np, X_np, p_t_np, pi_np = as_ndarrays(r, Y, T, X, p_t, pi)
        TX_np = np.concatenate([T_np[:, None], X_np], axis=1)

        self.kernel = self.kernel if self.kernel is not None else select_kernel(Y_np, T_np, X_np)
        self.Psi_np_pipeline = OrthogonalBasis(self.D, self.kernel)
        self.Psi_np = self.Psi_np_pipeline.fit_transform(TX_np)

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
                constraints.extend(get_box_constraints(w, p_t_np, self.Gamma, self.const_type))
            else:
                constraints.extend(get_f_div_constraint(w, p_t_np, self.gamma, self.const_type))

            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.SCS)

        if problem.status not in ("optimal", "optimal_inaccurate"):
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
