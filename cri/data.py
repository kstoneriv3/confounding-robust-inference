from collections import namedtuple
from typing import Callable

import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV

PolicyType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
DataType = namedtuple("DataType", ["Y", "T", "X", "U", "e_x", "e_xu"])

### Synthetic Data from Kallus and Zhou 2018, 2021
# This is used to test that the true policy value is included when valid uncertainty set is given

beta0 = 2.5
beta0_t = -2
beta_x = np.asarray([0, 0.5, -0.5, 0, 0])
beta_x_t = np.asarray([-1.5, 1, -1.5, 1.0, 0.5])
beta_xi = 1
beta_xi_t = -2
beta_e_x = np.asarray([0, 0.75, -0.5, 0, -1])
mu_x = np.asarray([-1, 0.5, -1, 0, -1])


def generate_data(n: int) -> DataType:
    """Generate data.

    Args:
        n: Size of data

    Returns:
        Data

    Example:
        .. doctest::

            >>> print("doctest")
            doctest
    """
    xi = (np.random.rand(n) > 0.5).astype(int)
    X = mu_x[None, :] + np.random.randn(n * 5).reshape(n, 5)
    eps = [np.random.randn(n) for t in (0, 1)]
    Y = np.array(
        [X @ (beta_x + beta_x_t * t) + (beta_xi + beta_xi_t * t) * xi + (beta0 + beta0_t * t) + eps[t] for t in (0, 1)]
    )
    U = (Y[0, :] > Y[1, :]).astype(int)
    z = X @ beta_e_x
    e_x = np.exp(z) / (1 + np.exp(z))
    e_xu = (6 * e_x) / (4 + 5 * U + e_x * (2 - 5 * U))
    T = (np.random.rand(n) < e_xu).astype(int)
    Y = Y[T, range(n)]
    e_x = e_x * T + (1 - e_x) * (1 - T)
    e_xu = e_xu * T + (1 - e_xu) * (1 - T)
    return DataType(Y, T, X, U, e_x, e_xu)


def evaluate_policy(policy: PolicyType, n: int = 1000) -> torch.Tensor:
    xi = (np.random.rand(n) > 0.5).astype(int)
    X = mu_x[None, :] + np.random.randn(n * 5).reshape(n, 5)
    eps = [np.random.randn(n) for t in (0, 1)]
    Y = torch.as_tensor(
        np.array(
            [
                X @ (beta_x + beta_x_t * t) + (beta_xi + beta_xi_t * t) * xi + (beta0 + beta0_t * t) + eps[t]
                for t in (0, 1)
            ]
        )
    )
    pi = policy(X, torch.zeros(n))
    Y = Y[0] * pi + Y[1] * (1 - pi)
    return Y.mean()


def estimate_p_t(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    model = LogisticRegressionCV().fit(X, T)
    p_t = model.predict_proba(X)[range(T.shape[0]), T]
    return p_t


### Synthetic Data for testing the tightness of the bounds
class BaseData:
    """Base class for data used in the numerical experiments."""

    def __init__(self) -> None:
        raise NotImplementedError

    def sample(self, n: int) -> DataType:
        raise NotImplementedError
        # return Y, T, X, p_t

    @property
    def default_policy(self) -> PolicyType:
        raise NotImplementedError

    def evaluate_policy(self, policy: PolicyType | None = None) -> torch.Tensor:
        raise NotImplementedError
        if policy:
            # logger.warn("Monte Carlo Sampling")
            pass
        else:
            # return true policy value of self.default_policy
            pass

    def ground_truth_box_bounds(self, Gamma: float, policy: PolicyType | None = None) -> torch.Tensor:
        raise NotImplementedError
        if policy:
            raise NotImplementedError
        else:
            pass
            # return true policy value of self.default_policy
        # return


class ContinuousSyntheticData(BaseData):
    def sample(self, n: int) -> DataType:
        """sampling data

        Args:
            n: Size of data

        Returns:
            Data
        """
        return DataType(*[None for _ in range(6)])

    pass
