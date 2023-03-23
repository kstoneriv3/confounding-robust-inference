import warnings
from typing import Any, Type

import cvxpy as cp
import numpy as np
import torch
from scipy.linalg import eigh
from scipy.stats import norm, gaussian_kde
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel
from sklearn.preprocessing import StandardScaler
from torch.autograd.functional import hessian, jacobian
from torch.optim import SGD, Optimizer

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
    DEFAULT_KERNEL,
    DUAL_FEASIBLE_CONSTRAINT_TYPES,
    OrthogonalBasis,
    assert_input,
    get_dual_objective,
)
from cri.policies import BasePolicy
from cri.utils.types import _DEFAULT_TORCH_FLOAT_DTYPE, as_ndarrays, as_tensor, as_tensors


def try_solvers(problem: cp.Problem, solvers: list[str]) -> None:
    installed_solvers = [sol for sol in solvers if sol in cp.installed_solvers()]
    for solver in installed_solvers:
        try:
            problem.solve(solver=solver)
        except cp.error.SolverError:
            pass
        if problem.status == "optimal":
            break

    if problem.status != "optimal":
        raise ValueError(
            "The optimizer found the associated convex programming to be {}.".format(problem.status)
        )


def apply_black_magic(Psi: np.ndarray, p_t: np.ndarray) -> np.ndarray:
    """Black magic for choosing orthogonal basis."""
    # Rescale the kernel per sample so that it's scaler better matches that of Y
    Psi = Psi / p_t[:, None]
    # constant basis is essential for numerical stability of f-divergence constraint
    Psi = np.concatenate([Psi, np.ones_like(p_t)[:, None]], axis=1)
    # Rescale per basis
    Psi = Psi / np.linalg.norm(Psi, axis=0, keepdims=True)
    return Psi


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
        should_augment_data: If True, use argumented data for estimation of hessian of the dual
            loss and asymptotic variance of its gradient. Empirically, this technique sometimes
            helps to stabilize the estimation of these quantities. For data augmentation, a kernel
            density estimator is used.
    """

    def __init__(
        self,
        const_type: str,
        gamma: float | None = None,
        Gamma: float | None = None,
        D: int = 30,
        kernel: Kernel | None = None,
        should_augment_data: bool = False,
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
        self.should_augment_data = should_augment_data

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

        self.kernel = self.kernel if self.kernel is not None else DEFAULT_KERNEL
        self.Psi_np_pipeline = OrthogonalBasis(self.D, self.kernel)
        self.Psi_np = self.Psi_np_pipeline.fit_transform(TX_np)
        self.Psi_np = apply_black_magic(self.Psi_np, p_t_np)

        # For avoiding user warning about multiplication operator with `*` and `@`
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            w = cp.Variable(n)

            objective = cp.Minimize(r_np.T @ w)

            constraints: list[cp.Constraint] = [np.zeros(n) <= w]
            kernel_consts = get_kernel_constraints(w, p_t_np, self.Psi_np)
            constraints.extend(kernel_consts)
            if "box" in self.const_type:
                constraints.extend(get_box_constraints(w, p_t_np, self.Gamma, self.const_type))
            else:
                f_div_const = get_f_div_constraint(w, p_t_np, self.gamma, self.const_type)
                constraints.extend(f_div_const)

            problem = cp.Problem(objective, constraints)
            solvers = [cp.MOSEK, cp.ECOS, cp.SCS]
            try_solvers(problem, solvers)

        self.w = torch.zeros_like(p_t)
        self.w[:] = as_tensor(w.value)
        self.fitted_lower_bound = torch.mean(self.w * r)
        self.problem = problem
        self.eta_kcmc = as_tensor(-kernel_consts[0].dual_value)  # need to match sign!
        # For box constraints, the dual objective does not depend on eta_f so it does not matter.
        self.eta_f = as_tensor(
            [f_div_const[0].dual_value] if "box" not in self.const_type else [1.0]
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
        assert_input(Y, T, X, p_t)
        pi = policy.prob(T, X)
        T_np, X_np, p_t_np = as_ndarrays(T, X, p_t)
        TX_np = np.concatenate([T_np[:, None], X_np], axis=1)
        Psi_np = self.Psi_np_pipeline.transform(TX_np)
        Psi_np = apply_black_magic(Psi_np, p_t_np)
        Psi = as_tensor(Psi_np)
        eta_cmc = Psi @ self.eta_kcmc
        dual = get_dual_objective(
            Y, p_t, pi, eta_cmc, self.eta_f, self.gamma, self.Gamma, self.const_type
        )
        return dual.mean()

    def predict_gic(self) -> torch.Tensor:
        """Calculate the Generalized Information Criterion (GIC) of the lower bound."""
        n = self.Y.shape[0]
        _, scores = self._get_dual_loss_and_jacobian()
        mask = torch.any(torch.isnan(scores), axis=1)
        V = scores.T @ scores / n
        J = self._get_dual_hessian()  # negative definite, as dual objective is concave
        J_inv = torch.pinverse(J)
        gic = self.fitted_lower_bound + torch.einsum("ij, ji->", J_inv, V) / n
        print(torch.einsum("ij, ji->", J_inv, V) / n)  # TODO, maybe scaling issue?
        if self.D == 10:
            S, _ = np.linalg.eigh(V)
            print((V[:4, :4], J[:4, :4], S))
            # assert False, 
        return gic

    def predict_ci(
        self, n_boot: int = 2**20, alpha: float = 0.05
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate confidence interval of the lower bound.

        Bootstrap is used for calculating the percentile of the asymptotic distribution.

        Args:
            n_boot: The number of Monte Carlo samples used to calculate the percentile of the
                asymptotic distribution by bootstraping.
            alpha: Significance level of used for the confidence interval.
        """
        n = self.Y.shape[0]
        losses, scores = self._get_dual_loss_and_jacobian()
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
        if self.const_type in ("Tan_box", "lr_box", "total_variation"):
            eta_kcmc = eta
            if "box" in self.const_type:
                eta_f = torch.ones((1,))
            elif self.const_type == "total_variation":
                # The dual objective of the total variation can only be used evaluation,
                # and it is not differentiable.
                assert hasattr(self, "fitted_lower_bound")
                eta_f = self.eta_f
        else:
            eta_kcmc = eta[:-1]
            eta_f = eta[-1]

        if self.should_augment_data:
            eta_cmc = as_tensor(self.Psi_np) @ eta_kcmc
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
        else:
            # Argument data to get non-singular estimates of the Hessian and the Jacobian
            Y, T, X, p_t, pi = self.augment_data(3000)
            T_np, X_np, p_t_np = as_ndarrays(T, X, p_t)
            TX_np = np.concatenate([T_np[:, None], X_np], axis=1)
            Psi_np = self.Psi_np_pipeline.transform(TX_np)
            Psi_np = apply_black_magic(Psi_np, p_t_np)
            Psi = as_tensor(Psi_np)
            eta_cmc = Psi @ eta_kcmc
            loss = get_dual_objective(
                Y,
                p_t,
                pi,
                eta_cmc,
                eta_f,
                self.gamma,
                self.Gamma,
                self.const_type,
            )
            if torch.any(torch.isfinite(loss)):
                # Even if we select samples whose loss is not nan or infinite, 
                # the backward pass tries to propagete zero grad for remaining samples,
                # resulting in nan gradients.
                mask = torch.isfinite(loss)
                loss = get_dual_objective(
                    Y[mask],
                    p_t[mask],
                    pi[mask],
                    eta_cmc[mask],
                    eta_f,
                    self.gamma,
                    self.Gamma,
                    self.const_type,
                )
        return loss

    def augment_data(self, n: int, seed: int = 0):
        Y_np, T_np, X_np, p_t_np = as_ndarrays(self.Y, self.T, self.X, self.p_t)
        log_p_t_np = np.log(p_t_np)
        T_unique, T_count = np.unique(T_np, return_counts=True)
        if len(T_unique) == len(T_np):  # T: continuous
            data = np.concatenate([Y_np[:, None], T_np[:, None], X_np, log_p_t_np[:, None]], axis=1)
            scaler = StandardScaler().fit(data)
            scaled_data = scaler.transform(data)
            scaled_augmented_data = gaussian_kde(scaled_data.T).resample(n, seed).T
            augmented_data = scaler.inverse_transform(scaled_augmented_data)
            Y_arg_np, T_arg_np, X_arg_np, log_p_t_arg_np = [
               (x[:, 0] if x.shape[1] == 1 else x) for x in np.split(augmented_data, [1, 2, -1], axis=1)
            ]
        else:
            data = np.concatenate([Y_np[:, None], X_np, log_p_t_np[:, None]], axis=1)
            T_prob = T_count / np.sum(T_count)
            T_arg_np = np.random.choice(T_unique, size=n, p=T_prob)
            augmented_data = np.empty((n, data.shape[1]))
            for i, t in enumerate(T_unique):
                n_t = sum(T_arg_np == t)
                data_t = data[T_np == t]
                scaler_t = StandardScaler().fit(data_t)
                scaled_data_t = scaler_t.transform(data_t)
                scaled_augmented_data_t = gaussian_kde(scaled_data_t.T).resample(n_t, seed).T
                augmented_data[T_arg_np == t] = scaler_t.inverse_transform(scaled_augmented_data_t)
            Y_arg_np, X_arg_np, log_p_t_arg_np = [
               (x[:, 0] if x.shape[1] == 1 else x) for x in np.split(augmented_data, [1, -1], axis=1)
            ]
            log_p_t_arg_np = np.minimum(0., log_p_t_arg_np)  # discrete probability mass must be <= 1.
        p_t_arg_np = np.exp(log_p_t_arg_np)
        p_t_arg_np = np.clip(p_t_arg_np, 0.5 * min(p_t_np), 2 * max(p_t_np))  # for numerical stability
        Y_arg, T_arg, X_arg, p_t_arg = as_tensors(Y_arg_np, T_arg_np, X_arg_np, p_t_arg_np)
        pi_arg = self.policy.prob(T_arg, X_arg)
        return Y_arg, T_arg, X_arg, p_t_arg, pi_arg

    @property
    def eta(self) -> torch.Tensor:
        # The dual objective does not depend on eta_f for box constraints so ignore eta_f then.
        if self.const_type in ("Tan_box", "lr_box", "total_variation"):
            return self.eta_kcmc.data
        else:
            return torch.concat([self.eta_kcmc, self.eta_f])

    def _get_dual_hessian(self) -> torch.Tensor:
        if self.const_type in ("Tan_box", "lr_box", "total_variation"):
            # Use numpy for computation in this block.
            Y_np, T_np, X_np, p_t_np, pi_np, eta_kcmc_np = as_ndarrays(
                self.Y, self.T, self.X, self.p_t, self.pi, self.eta_kcmc
            )
            n = Y_np.shape[0]

            # estimate p_{y|tx}(Pshi @ eta_kcmc)
            TX_np = np.concatenate([T_np[:, None], X_np], axis=1)
            TX_np = StandardScaler().fit_transform(TX_np)
            model = GaussianProcessRegressor(kernel=DEFAULT_KERNEL, alpha=0.25, normalize_y=True)
            model.fit(TX_np[:1000], (pi_np * Y_np)[:1000])
            r_mean, r_std = model.predict(TX_np, return_std=True)
            eta_cmc_mean, eta_cmc_std = map(lambda x: x / p_t_np, [r_mean, r_std])
            eta_cmc = self.Psi_np @ eta_kcmc_np

            if self.const_type == "total_variation":
                conditional_pdf = norm.pdf(eta_cmc + 0.5, loc=eta_cmc_mean, scale=eta_cmc_std)
                diag = np.diag(conditional_pdf)
            else:
                a, b = get_a_b(p_t_np, self.Gamma, self.const_type)
                conditional_pdf = norm.pdf(eta_cmc, loc=eta_cmc_mean, scale=eta_cmc_std)
                diag = np.diag(p_t_np * (b - a) * conditional_pdf)

            H = -as_tensor(self.Psi_np.T @ diag @ self.Psi_np / n)
        else:
            eta = torch.tensor(self.eta.data, requires_grad=True)
            H = hessian(lambda eta: self._get_fitted_dual_loss(eta).mean(), eta)  # type: ignore
        return H

    def _get_dual_loss_and_jacobian(self) -> torch.Tensor:
        eta = self.eta.clone().detach().requires_grad_(True)
        one = torch.ones((1,), requires_grad=True)
        one_and_eta = torch.concat([eta, one])
        loss_and_J = jacobian(
            lambda x: self._get_fitted_dual_loss(x[1:]) * x[0],
            one_and_eta,
            vectorize=True,
        )  # type: ignore
        loss = loss_and_J[:, 0]
        J = loss_and_J[:, 1:]
        return loss, J


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
        self.eta_kcmc = torch.zeros(D + 2, dtype=_DEFAULT_TORCH_FLOAT_DTYPE, requires_grad=True)
        self.log_eta_f = torch.zeros(1, dtype=_DEFAULT_TORCH_FLOAT_DTYPE, requires_grad=True)

    def fit(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
        X: torch.Tensor,
        p_t: torch.Tensor,
        policy: BasePolicy,
        optimizer_cls: Type[Optimizer] = SGD,
        optimizer_kwargs: dict[str, Any] | None = None,
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

        self.kernel = self.kernel if self.kernel is not None else DEFAULT_KERNEL
        self.Psi_np_pipeline = OrthogonalBasis(self.D, self.kernel)
        self.Psi_np = self.Psi_np_pipeline.fit_transform(TX_np)
        self.Psi_np = apply_black_magic(self.Psi_np, p_t_np)

        kwargs = {
            "lr": 3e-2,
            "params": [self.eta_kcmc, self.log_eta_f],
        }
        if optimizer_kwargs:
            kwargs.update(optimizer_kwargs)
        optimizer = optimizer_cls(**kwargs)  # type: ignore

        for i in range(n_steps):
            train_idx = torch.randint(n, (batch_size,))
            eta_cmc = as_tensor(self.Psi_np)[train_idx] @ self.eta_kcmc
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
                eta_cmc = as_tensor(self.Psi_np)[val_idx] @ self.eta_kcmc
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
        Psi_np = self.Psi_np_pipeline.transform(TX_np)
        Psi_np = apply_black_magic(Psi_np, p_t_np)
        pi = policy.prob(T, X)
        eta_cmc = as_tensor(Psi_np) @ self.eta_kcmc
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
    :math:`\\Psi_{n,d}=\\psi_d(t_n, x_n)`. Thus, we use the following model with
    :math:`\\sigma^2` estimated by empirical Bayes as

    .. math::
       :nowrap:

       \\begin{eqnarray}
          \\beta &\\sim N(0, I_d), \\\\
          e &= \\Psi \\beta + \\varepsilon, \\\\
          \\varepsilon &\\sim N(0, \\sigma^2 I_n). \\\\
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
              \\leq \\chi^2_d(1 - \\alpha)
          \\},
       \\end{eqnarray}

    where :math:`\\mu_\\beta=(\\Psi^T\\Psi+\\sigma^2I_d)^{-1}\\Psi^Te`
    and :math:`\\Sigma_\\beta=(\\Psi^T\\Psi+\\sigma^2I_d)^{-1}`. Therefore, the condition 
    :math:`0_d\\in \\mathrm{CI}_{\\beta|e}(1 - \\alpha)` can be written as

    .. math::
       :nowrap:

       \\begin{equation}
          \\mu_{\\beta|e}^T \\Sigma_{\\beta|e}^{-1} \\mu_{\\beta|e}
          = e^T \\Psi (\\Psi^T\\Psi + \\sigma^2I_d)^{-1} \\Psi^Te
          \\leq \\chi^2_d(1 - \\alpha).
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
        self.kernel = kernel if kernel is not None else DEFAULT_KERNEL

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

        self.Psi_np_pipeline = OrthogonalBasis(self.D, self.kernel)
        self.Psi_np = self.Psi_np_pipeline.fit_transform(TX_np)

        # For avoiding user warning about multiplication operator with `*` and `@`
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            w = cp.Variable(n)

            objective = cp.Minimize(r_np.T @ w)

            constraints: list[cp.Constraint] = [np.zeros(n) <= w]
            constraints.extend(
                get_gp_constraints(w, p_t_np, pi_np, self.Psi_np, self.sigma2, self.alpha)
            )
            if "box" in self.const_type:
                constraints.extend(get_box_constraints(w, p_t_np, self.Gamma, self.const_type))
            else:
                constraints.extend(get_f_div_constraint(w, p_t_np, self.gamma, self.const_type))

            problem = cp.Problem(objective, constraints)
            solvers = [cp.MOSEK, cp.ECOS, cp.SCS]
            try_solvers(problem, solvers)

        self.w = torch.zeros_like(p_t)
        self.w[:] = as_tensor(w.value)
        self.fitted_lower_bound = torch.mean(self.w * r)
        self.problem = problem
        return self

    def predict(self) -> torch.Tensor:
        return self.fitted_lower_bound
