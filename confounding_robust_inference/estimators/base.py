from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from confounding_robust_inference.policies import BasePolicy
from confounding_robust_inference.utils.docs import WithDocstringsMeta


class BaseEstimator(Protocol, metaclass=WithDocstringsMeta):
    def fit(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
        X: torch.Tensor,
        p_t: torch.Tensor,
        policy: BasePolicy,
    ) -> BaseEstimator:
        """Estimate the lower bound of the policy value.

        Args:
            Y: Outcome variable. It must be of shape [n_samples] and type
                confounding_robust_inference.types._DEFAULT_TORCH_FLOAT_DTYPE.
            T: Action variable (i.e. treatment). It must be of shape [n_samples].
            X: Context variable. It must be of shape [n_samples, n_dim] and type
                confounding_robust_inference.types._DEFAULT_TORCH_FLOAT_DTYPE.
            p_t: Nominal propensity p_obs(t|x). It must be of shape [n_samples] and type
                confounding_robust_inference.types._DEFAULT_TORCH_FLOAT_DTYPE.
            policy: Policy to be evaluated.
        """
        raise NotImplementedError

    def predict(self) -> torch.Tensor:
        """Calculate the lower bound obtained by fit method."""
        raise NotImplementedError


@runtime_checkable
class BaseKCMCEstimator(BaseEstimator, Protocol):
    def predict_dual(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
        X: torch.Tensor,
        p_t: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the dual objective value of the minimization problem for the lower bound
        for individual samples of given data.

        Args:
            Y: Outcome variable. It must be of shape [n_samples] and type
                confounding_robust_inference.types._DEFAULT_TORCH_FLOAT_DTYPE.
            T: Action variable (i.e. treatment). It must be of shape [n_samples].
            X: Context variable. It must be of shape [n_samples, n_dim] and type
                confounding_robust_inference.types._DEFAULT_TORCH_FLOAT_DTYPE.
            p_t: Nominal propensity p_obs(t|x). It must be of shape [n_samples] and type
                confounding_robust_inference.types._DEFAULT_TORCH_FLOAT_DTYPE.

        Returns:
            Lower bound estimate obtained by solving the dual problem
        """
        raise NotImplementedError
