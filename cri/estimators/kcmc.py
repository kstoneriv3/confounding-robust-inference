import warnings
from typing import Callable, Dict, List

import cvxpy as cp
import numpy as np
import torch
from sklearn.decomposition import KernelPCA
from sklearn.gaussian_process.kernels import Kernel

from cri.estimators.base import BaseEstimator
from cri.estimators.constraints import get_box_constraints
from cri.estimators.misc import assert_input, select_kernel
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
            When gamma == 0.0, QB estimator is equivalent to the IPW estimator.
        Gamma: Sensitivity parameter for box constraints satisfying Gamma >= 1.0.
            When Gamma == 1.0, QB estimator is equivalent to the IPW estimator.
        D: Dimension of the low-rank approximation used in the kernel quantile regression.
        alpha: (Bayesian) Significance level of credible interval.
        kernel: Kernel used in the low-rank kernel quantile regression.
    """

    def __init__(
        self,
        const_type: str,
        gamma: float | None = None,
        Gamma: float | None = None,
        D: int = 30,
        alpha: float = 0.05,
        kernel: Kernel | None = None,
    ) -> None:
        assert const_type in CONST_TYPES
        if const_type == "box":
            assert Gamma is not None and Gamma >= 1
        else:
            assert gamma is not None and gamma >= 0
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
        pi = policy.prob(X, T)
        r = Y * pi
        r_np, Y_np, T_np, X_np, p_t_np, pi_np = as_ndarrays(r, Y, T, X, p_t, pi)

        self.kernel = self.kernel if self.kernel is not None else select_kernel(Y_np, T_np, X_np)
        self.Psi = get_orthogonal_functions(T, X, D, kernel)

        # For avoiding user warning about multiplication operator with `*` and `@`
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            w = cp.Variable(n)

            objective = cp.Minimize(cp.sum(r_np * w))

            constraints: List[cp.Constraint] = [np.zeros(n) <= w]
            constraints.extend(get_kernel_constraints(w, p_t_np, pi_np, self.Psi))
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

    def dual_objective(self) -> torch.Tensor:
        raise NotImplementedError

    def get_dual_hessian(self) -> torch.Tensor:
        raise NotImplementedError

    def get_dual_jacobian(self) -> torch.Tensor:
        raise NotImplementedError

    def get_gp_const(self) -> None:
        """Returns GP-KCMC using low-rank approximation.

        Here, as low-rank GP is equivalent to Bayesian ridge regression with design matrix
        :math:`X_{n,d}=\psi_d(t_n, x_n)`. Thus, we use the following model with
        :math:`\\sigma^2` estimated by empirical Bayes\:

        .. math::
           :nowrap:

           \\begin{eqnarray}
              \\beta &\\sim N(0, I_d), \\\\
              y &= X \\beta + \\varepsilon, \\\\
              \\varepsilon &\sim N(0, \sigma^2 I_n). \\\\
           \\end{eqnarray}

        For this model, the posterior and the credible set (highest posterior density set) of
        :math:`\\beta` are

        .. math::
           :nowrap:

           \\begin{eqnarray}
              \\beta | y &\\sim N(\\mu_{\\beta|y}, \\Sigma_{\\beta|y})  \\\\
           \\end{eqnarray}

        and

        .. math::
           :nowrap:

           \\begin{eqnarray}
              \\mathrm{CI}_{\\beta|y}(1 - \\alpha) =
              \\{\\beta:
                  (\\beta - \\mu_{\\beta|y})^T \\Sigma_{\\beta|y} (\\beta - \\mu_{\\beta|y})
                  \\leq \chi^2_d(1 - \\alpha)
              \\},
           \\end{eqnarray}

        where :math:`\\mu_\\beta=(X^TX+\\sigma^2I_d)^{-1}X^Ty`
        and :math:`\\Sigma_\\beta=(X^TX+\\sigma^2I_d)^{-1}`. Therefore, the condition 
        :math:`0_d\\in \mathrm{CI}_{\\beta|y}(1 - \\alpha)` can be written as

        .. math::
           :nowrap:

           \\begin{equation}
              \\mu_{\\beta|y}^T \\Sigma_{\\beta|y} \\mu_{\\beta|y}
              = y^T X(X^TX + \\sigma^2I_d)X^Ty
              \\leq \chi^2_d(1 - \\alpha).
           \\end{equation}

        Lastly, for empirical Bayes estimation of :math:`\\sigma^2`, we maximize the log marginal
        likelihood of the model, which is concave w.r.t. :math:`\\sigma^2` and can be reformuted
        in low-rank manner by Woodbury's formula\:

        .. math::
           :nowrap:

           \\begin{align}
           \\log p(y;\\sigma^2)
           &= - \\frac{1}{2} \\log\\det(XX^T + \\sigma^2 I_n)
              - \\frac{1}{2} y^T (XX^T + \\sigma^2 I_n)^{-1} y
              + \\text{const.} \\\\
           &= - \\frac{1}{2} \\log\\det(I_d + \\frac{1}{\\sigma^2} X^TX)
              - \\frac{n}{2} \\log \\sigma^2 \\\\
           &\quad\quad - \\frac{1}{2\\sigma^2} \\|y\\|^2
              - \\frac{1}{2\\sigma^2} y^T X (\\sigma^2 I_d +  X^TX)^{-1}X^T y
              + \\text{const.} \\\\
           \\end{align}

        As :math:`X^TX` is a diagonal matrix, by the property of kernel PCA
        (or eigen decomposition), this further simplifies to

        .. math::
           :nowrap:

           \\begin{align}
           \\log p(y;\\sigma^2)
           &= - \\frac{1}{2} \\sum_{i=1}^d \\log (1 + s_i \\cdot \\frac{1}{\\sigma^2})
              + \\frac{n}{2} \\log \\sigma^2 \\\\
           &\quad\quad - \\frac{1}{2\\sigma^2} \\|y\\|^2
              - \\frac{1}{2\\sigma^4} \\sum_{i=1}^d
              \\frac{1}{1 + s_i \\cdot \\frac{1}{\\sigma^2}} |y^T X_{\\cdot, i}|^2 
              + \\text{const.}
           \\end{align}

        where :math:`X^TX = \\mathrm{diag}(s_i)`. As the product of positive convex functions is
        convex, we can see that :math:`p(y;\\sigma^2)` is concave w.r.t
        :math:`\\frac{1}{\\sigma^2}`.
        """
        raise NotImplementedError


def get_orthogonal_functions(
    T: np.ndarray,
    X: np.ndarray,
    D: int,
    kernel: Kernel,
) -> np.ndarray:
    TX = np.concatenate([T[:, None], X], axis=1)
    TX /= TX.std(axis=0)[None, :]
    Psi = KernelPCA(D, kernel=kernel).fit_transform(TX)
    return Psi


def get_kernel_constraints(
    w: cp.Variable,
    p_t: np.ndarray,
    pi: np.ndarray,
    Psi: np.ndarray,
) -> List[cp.Constraint]:
    # Carveat: np.ones(n) * w is NOT the element-wise product in cvxpy!!!
    # Numpy's broadcasting MATCHES THE LOWER DIMENSIONS and assume shape 1 at higher dimensions.
    n, d = Psi.shape
    X = Psi
    sigma2 = cp.Variable(1, pos=True)
    S = np.sum(X**2, axis=0)
    log_py = (
        -0.5 * cp.sum(cp.log(sigma2 + S))
        - 0.5 * (n - d) * cp.log(sigma2)
        - 0.5 * cp.inv_pos(sigma2) * np.sum(y**2)
        - 0.5 * cp.inv_pos(sigma2) * cp.sum(cp.multiply((X.T @ Y) ** 2, cp.inv_pos(sigma2 + S)))
    )
    problem = cp.Problem(cp.Maximize(log_py))
    problem.solve(gp=True)
    sigma2_hat = sigma2.value
    return [(Psi.T * pi) @ w == np.sum(Psi.T * pi / p_t)]


def get_gp_constraints(
    w: cp.Variable,
    p_t: np.ndarray,
    pi: np.ndarray,
    Psi: np.ndarray,
) -> List[cp.Constraint]:
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
    f = CVXPY_F_DIV_FUNCTIONS[f_div_type]
    # Carveat: np.ones(n) * w is NOT the element-wise product in cvxpy!!!
    constraints = [
        cp.sum(f(cp.multiply(w, p_t))) <= gamma * n,
        cp.scalar_product(w, p_t) == n,
    ]
    return constraints


class DualKCMCEstimator(BaseEstimator):
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
        kernel: Kernel used by Gaussian process.
    """

    def __init__(
        self,
        const_type: str,
        gamma: float | None = None,
        Gamma: float | None = None,
        kernel: Kernel | None = None,
    ) -> None:
        assert const_type in CONST_TYPES
        if const_type == "box":
            assert Gamma is not None and Gamma >= 1
        else:
            assert gamma is not None and gamma >= 0
        self.gamma = gamma if gamma is not None else 0.0
        self.Gamma = Gamma if Gamma is not None else 1.0
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
        self.Psi = get_orthogonal_functions(T, X, D, kernel)

        # For avoiding user warning about multiplication operator with `*` and `@`
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            w = cp.Variable(n)

            objective = cp.Minimize(cp.sum(r_np * w))

            constraints: List[cp.Constraint] = [np.zeros(n) <= w]
            constraints.extend(get_kernel_constraints(w, p_t_np, pi_np, self.Psi))
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
