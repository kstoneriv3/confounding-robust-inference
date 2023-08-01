from importlib import resources
from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from confounding_robust_inference.policies import BasePolicy
from confounding_robust_inference.utils.docs import WithDocstringsMeta
from confounding_robust_inference.utils.propensity import estimate_p_t_binary
from confounding_robust_inference.utils.types import as_tensor, as_tensors


class DataTuple(NamedTuple):
    Y: torch.Tensor
    T: torch.Tensor
    X: torch.Tensor
    U: torch.Tensor | None
    p_t_x: torch.Tensor
    p_t_xu: torch.Tensor | None


class BaseData(Protocol, metaclass=WithDocstringsMeta):
    """Base class for data used in the numerical experiments."""

    def sample(self, n: int) -> DataTuple:
        """Generate data.

        Args:
            n: Size of data

        Returns:
            Namedtuple that contains the generated data, whose attributes are
            Y, T, X, U, p_t_x, p_t_xu. Those attributes are of type Tensor.
        """
        raise NotImplementedError

    def evaluate_policy(self, policy: BasePolicy, n_mc: int = 1000) -> torch.Tensor:
        """Unbiased Monte Carlo estimator of policy value.

        Args:
            policy: Policy to be evaluated.
            n_mc: The number of Monte Carlo samples.

        Returns:
            Policy value estimator, which is a torch tensor differentiable w.r.t. policy.
        """
        raise NotImplementedError


@runtime_checkable
class BaseDataWithLowerBound(BaseData, Protocol):
    """Base class for data with known ground truth lower bound."""

    def evaluate_policy_lower_bound(
        self, policy: BasePolicy, Gamma: float, n_mc: int = 1000
    ) -> torch.Tensor:
        """Ground truth lower bound under box constraints.

        Unbiased Monte Carlo estimation is used to estimate the lower bound.

        Args:
            Gamma: Parameteer of box constraints.
            policy: Policy to be evaluated.
            n_mc: The number of Monte Carlo samples.

        Returns:
            Lower bound estimate, which is a torch tensor differentiable w.r.t. policy.
        """
        raise NotImplementedError


class SyntheticDataBinary(BaseDataWithLowerBound):
    """Synthetic data with witn binary action space and a known lower bound.

    This is synthetic data with binary action space, similar to Kallus and Zhou (2018).
    For this data, we know that analytic expression of the lower bound of Tan's marginal sensitivity
    model. Thus, this synthetic data can be is used to test the consistency and asypmtotic property
    of estimators.
    """

    def __init__(self) -> None:
        # We define constants in __init__ to avoid issues around different torch dtype.
        self.beta0 = 2.5
        self.beta0_t = -2
        self.beta_x = as_tensor([0, 0.5, -0.5, 0, 0])
        self.beta_x_t = as_tensor([-1.5, 1, -1.5, 1.0, 0.5])
        self.beta_xi = 1
        self.beta_xi_t = -2
        self.beta_p_t = as_tensor([0, 0.75, -0.5, 0, -1])
        self.mu_x = as_tensor([-1, 0.5, -1, 0, -1])

    def sample(self, n: int) -> DataTuple:
        xi = (torch.rand(n) > 0.5).int()  # noqa: F841
        X = self.mu_x[None, :] + torch.randn(n * 5).reshape(n, 5)
        p_t_x = torch.sigmoid(X @ self.beta_p_t)
        T = (torch.rand(n) < p_t_x).int()
        Y = (
            X @ self.beta_x
            + X @ self.beta_x_t * T
            # + xi * self.beta_xi
            # + xi * self.beta_xi * T
            + self.beta0
            + self.beta0_t * T
            + torch.randn(n)
        )
        p_t_x = (1 - p_t_x) * (1 - T) + p_t_x * T
        return DataTuple(Y, T, X, None, p_t_x, None)

    def evaluate_policy(self, policy: BasePolicy, n_mc: int = 1000) -> torch.Tensor:
        # xi = (torch.rand(n_mc) > 0.5).int()
        X = self.mu_x[None, :] + torch.randn(n_mc * 5).reshape(n_mc, 5)
        Y_po = torch.Tensor(size=(n_mc, 2))
        for t in (0, 1):
            Y_po[:, t] = (
                X @ self.beta_x
                + X @ self.beta_x_t * t
                # + xi * self.beta_xi
                # + xi * self.beta_xi * t
                + self.beta0
                + self.beta0_t * t
                + torch.randn(n_mc)
            )
        # We avoid sampling w.r.t. policy for its differentiatiability.
        pi = policy.prob(torch.zeros(n_mc), X)
        Y = Y_po[:, 0] * pi + Y_po[:, 1] * (1 - pi)
        return Y.mean()

    def evaluate_policy_lower_bound(
        self, policy: BasePolicy, Gamma: float, n_mc: int = 1000
    ) -> torch.Tensor:
        tau = 1 / (1 + Gamma)
        # See Proposition 2. of Dorn and Guo (2022).
        return self.evaluate_policy(policy, n_mc) - norm.pdf(norm.ppf(tau)) * (Gamma - 1 / Gamma)


class SyntheticDataContinuous(BaseDataWithLowerBound):
    """Synthetic data with with continuous action space and a known lower bound.

    This is synthetic data with continuous action space, similar to Kallus and Zhou (2018).
    For this data, we know that analytic expression of the lower bound of likelihood ratio
    sensitivity model. Thus, this synthetic data can be is used to test the consistency and
    asypmtotic property of estimators.
    """

    def __init__(self) -> None:
        # We define constants in __init__ to avoid issues around different torch dtype.
        self.beta0 = 2.5
        self.beta0_t = -2
        self.beta_x = as_tensor([0, 0.5, -0.5, 0, 0])
        self.beta_x_t = as_tensor([-1.5, 1, -1.5, 1.0, 0.5])
        self.beta_xi = 1
        self.beta_xi_t = -2
        self.beta_p_t = as_tensor([0, 0.75, -0.5, 0, -1])
        self.mu_x = as_tensor([-1, 0.5, -1, 0, -1])

    def sample(self, n: int) -> DataTuple:
        xi = (torch.rand(n) > 0.5).int()  # noqa: F841
        X = self.mu_x[None, :] + torch.randn(n * 5).reshape(n, 5)
        dist = torch.distributions.Normal(X @ self.beta_p_t, 1.0)  # type: ignore
        T = dist.sample()  # type: ignore
        p_t_x = dist.log_prob(T).exp()  # type: ignore
        Y = (
            X @ self.beta_x
            + X @ self.beta_x_t * T
            # + xi * self.beta_xi
            # + xi * self.beta_xi * T
            + self.beta0
            + self.beta0_t * T
            + torch.randn(n)
        )
        return DataTuple(Y, T, X, None, p_t_x, None)

    def evaluate_policy(self, policy: BasePolicy, n_mc: int = 1000) -> torch.Tensor:
        # xi = (torch.rand(n_mc) > 0.5).int()
        X = self.mu_x[None, :] + torch.randn(n_mc * 5).reshape(n_mc, 5)
        T = policy.sample(X)
        Y = (
            X @ self.beta_x
            + X @ self.beta_x_t * T
            # + xi * self.beta_xi
            # + xi * self.beta_xi * T
            + self.beta0
            + self.beta0_t * T
            + torch.randn(n_mc)
        )
        return Y.mean()

    def evaluate_policy_lower_bound(
        self, policy: BasePolicy, Gamma: float, n_mc: int = 1000
    ) -> torch.Tensor:
        tau = 1 / (1 + Gamma)
        # See Proposition 2. of Dorn and Guo (2022).
        return self.evaluate_policy(policy, n_mc) - norm.pdf(norm.ppf(tau)) * (Gamma - 1 / Gamma)


class SyntheticDataKallusZhou2018(BaseData):
    """Synthetic data from Kallus and Zhou (2018).

    For this synthetic data, we know that the Tan's box constraints with p_obs(t|x) = p_t_x and
    Gamma=1.5 is an uncertainty set that contains true propensity p(t|x, u). Thus, this synthetic
    data can be is used to test if the true policy value is actually included when a valid
    uncertainty set is given.
    """

    def __init__(self) -> None:
        # We define constants in __init__ to avoid issues around different torch dtype.
        self.beta0 = 2.5
        self.beta0_t = -2
        self.beta_x = as_tensor([0, 0.5, -0.5, 0, 0])
        self.beta_x_t = as_tensor([-1.5, 1, -1.5, 1.0, 0.5])
        self.beta_xi = 1
        self.beta_xi_t = -2
        self.beta_p_t = as_tensor([0, 0.75, -0.5, 0, -1])
        self.mu_x = as_tensor([-1, 0.5, -1, 0, -1])

    def sample(self, n: int) -> DataTuple:
        xi = (torch.rand(n) > 0.5).int()
        X = self.mu_x[None, :] + torch.randn(n * 5).reshape(n, 5)
        eps = [torch.randn(n) for t in (0, 1)]
        Y_po = torch.Tensor(size=(n, 2))  # potential outcome
        for t in (0, 1):
            Y_po[:, t] = (
                X @ (self.beta_x + self.beta_x_t * t)
                + xi * (self.beta_xi + self.beta_xi_t * t)
                + (self.beta0 + self.beta0_t * t)
                + eps[t]
            )
        U = (Y_po[:, 0] > Y_po[:, 1]).int()
        z = X @ self.beta_p_t
        p_t_x = torch.exp(z) / (1 + torch.exp(z))
        p_t_xu = (6 * p_t_x) / (4 + 5 * U + p_t_x * (2 - 5 * U))
        T = (torch.rand(n) < p_t_xu).int()
        Y = (1 - T) * Y_po[:, 0] + T * Y_po[:, 1]
        p_t_x = (1 - p_t_x) * (1 - T) + p_t_x * T
        p_t_xu = (1 - p_t_xu) * (1 - T) + p_t_xu * T
        return DataTuple(Y, T, X, U, p_t_x, p_t_xu)

    def evaluate_policy(self, policy: BasePolicy, n_mc: int = 1000) -> torch.Tensor:
        xi = (torch.rand(n_mc) > 0.5).int()
        X = self.mu_x[None, :] + torch.randn(n_mc * 5).reshape(n_mc, 5)
        eps = [torch.randn(n_mc) for t in (0, 1)]
        Y_po = torch.Tensor(size=(n_mc, 2))
        for t in (0, 1):
            Y_po[:, t] = (
                X @ (self.beta_x + self.beta_x_t * t)
                + xi * (self.beta_xi + self.beta_xi_t * t)
                + (self.beta0 + self.beta0_t * t)
                + eps[t]
            )
        # We avoid sampling w.r.t. policy for its differentiatiability.
        pi = policy.prob(torch.zeros(n_mc), X)
        Y = Y_po[:, 0] * pi + Y_po[:, 1] * (1 - pi)
        return Y.mean()


class SyntheticDataKallusZhou2018Continuous(BaseData):
    """Synthetic data with continuous action space similar to Kallus and Zhou (2018).

    For this synthetic data, we know that the Tan's box constraints with p_obs(t|x) = p_t_x and
    Gamma=1.5 is an uncertainty set that contains true propensity p(t|x, u). Thus, this synthetic
    data can be is used to test if the true policy value is actually included when a valid
    uncertainty set is given.
    """

    def __init__(self) -> None:
        # We define constants in __init__ to avoid issues around different torch dtype.
        self.beta0 = 2.5
        self.beta0_t = -2
        self.beta_x = as_tensor([0, 0.5, -0.5, 0, 0])
        self.beta_x_t = as_tensor([-1.5, 1, -1.5, 1.0, 0.5])
        self.beta_xi = 1
        self.beta_xi_t = -2
        self.beta_p_t = as_tensor([0, 0.75, -0.5, 0, -1])
        self.mu_x = as_tensor([-1, 0.5, -1, 0, -1])

    def sample(self, n: int) -> DataTuple:
        xi = (torch.rand(n) > 0.5).int()
        X = self.mu_x[None, :] + torch.randn(n * 5).reshape(n, 5)
        eps = [torch.randn(n) for t in (0, 1)]
        Y_po = torch.Tensor(size=(n, 2))
        for t in (0, 1):
            Y_po[:, t] = (
                X @ (self.beta_x + self.beta_x_t * t)
                + xi * (self.beta_xi + self.beta_xi_t * t)
                + (self.beta0 + self.beta0_t * t)
                + eps[t]
            )
        U = (Y_po[:, 0] > Y_po[:, 1]).int()
        Unif = torch.rand(n)
        T = torch.where(torch.rand(n) < 2 / 3, Unif, U * Unif**2 + (1 - U) * (1 - Unif) ** 2)
        Y = (1 - T) * Y_po[:, 0] + T * Y_po[:, 1]
        p_t_x = torch.ones(n)
        p_t_xu = (2 / 3) * (1 + U * Unif + (1 - U) * (1 - Unif))
        return DataTuple(Y, T, X, U, p_t_x, p_t_xu)

    def evaluate_policy(self, policy: BasePolicy, n_mc: int = 1000) -> torch.Tensor:
        xi = (torch.rand(n_mc) > 0.5).int()
        X = self.mu_x[None, :] + torch.randn(n_mc * 5).reshape(n_mc, 5)
        eps = [torch.randn(n_mc) for t in (0, 1)]
        T = policy.sample(X)
        Y_po = torch.Tensor(size=(n_mc, 2))
        for t in (0, 1):
            Y_po[:, t] = (
                X @ (self.beta_x + self.beta_x_t * t)
                + xi * (self.beta_xi + self.beta_xi_t * t)
                + (self.beta0 + self.beta0_t * t)
                + eps[t]
            )
        Y = (1 - T) * Y_po[:, 0] + T * Y_po[:, 1]
        return Y.mean()


class NLSDataDornGuo2022:
    """NLS data from Dorn and Guo (2022)."""

    def __init__(self) -> None:
        self.data: DataTuple | None = None

    def sample(self, n: int = 667) -> DataTuple:
        if n != 667:
            raise ValueError(
                f"n = {n} is given as the input but there are n = 667 data points in this dataset."
            )
        if self.data is None:
            self.data = self.load_and_prepare_data()
        return self.data

    def load_and_prepare_data(self) -> DataTuple:
        import confounding_robust_inference

        path = resources.files(confounding_robust_inference).joinpath("union1978.csv")
        df = pd.read_csv(path)
        df.columns = (
            "id",
            "age",
            "black",
            "educ76",
            "smsa",
            "south",
            "married",
            "enrolled",
            "educ78",
            "manufacturing",
            "occupation",
            "union",
            "wage",
        )

        df.age = df.age + 12  # age was recorded in 1966
        df["education"] = np.maximum(df.educ76, df.educ78)
        df.black = (df.black == 2).astype(int)
        df.married = np.logical_or(df.married == 1, df.married == 2).astype(int)
        df.smsa = np.logical_or(df.smsa == 1, df.smsa == 2).astype(int)
        df.manufacturing = np.logical_and(206 <= df.manufacturing, df.manufacturing <= 459).astype(
            int
        )

        def get_occupation_id(occ_number: int) -> int:
            if 401 <= occ_number <= 545:
                return 0  # craftsman
            elif 960 <= occ_number <= 985:
                return 1  # laborer
            else:
                return 2  # other

        df.occupation = df.occupation.apply(get_occupation_id)

        df = df[df.occupation != 2]
        df = df[df.enrolled == 0]
        df = df.drop(columns=["id", "educ76", "educ78", "enrolled"])

        # remove missing values
        missing = np.logical_or(df.to_numpy() == -4, df.to_numpy() == -5)
        df = df[~missing.any(axis=1)]

        # df.describe()

        Y = np.log(df.wage.to_numpy())
        T = df.union.to_numpy()
        X = df.drop(columns=["wage", "union"]).to_numpy().astype(float)

        Y, T, X = as_tensors(Y, T, X)
        p_t_x = estimate_p_t_binary(T, X)

        return DataTuple(Y, T, X, None, p_t_x, None)
