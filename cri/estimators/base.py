import torch

from cri.policies import BasePolicy


class BaseEstimator:
    def fit(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
        X: torch.Tensor,
        p_t: torch.Tensor,
        policy: BasePolicy,
    ) -> "BaseEstimator":
        """Solve the minimization problem for obtaining the lower bound for the given data.

        Args:
            Y: Outcome variable. Its dtype must be cri.types._DEFAULT_TORCH_FLOAT_DTYPE.
            T: Action variable (i.e. treatment).
            X: Context variable. Its dtype must be cri.types._DEFAULT_TORCH_FLOAT_DTYPE.
            p_t: Nominal propensity p_obs(t|x). Its dtype must be
                cri.types._DEFAULT_TORCH_FLOAT_DTYPE.
            policy: Policy to be evaluated.
        """
        raise NotImplementedError

    def predict(self) -> torch.Tensor:
        """Calculate the lower bound obtained by fit method."""
        raise NotImplementedError

    def predict_dual(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
        X: torch.Tensor,
        p_t: torch.Tensor,
        policy: BasePolicy,
    ) -> torch.Tensor:
        """Calculate the dual objective value of the minimization problem for the lower bound
        for the given samples.

        Args:
            Y: Outcome variable. Its dtype must be cri.types._DEFAULT_TORCH_FLOAT_DTYPE.
            T: Action variable (i.e. treatment).
            X: Context variable. Its dtype must be cri.types._DEFAULT_TORCH_FLOAT_DTYPE.
            p_t: Nominal propensity p_obs(t|x). Its dtype must be
                cri.types._DEFAULT_TORCH_FLOAT_DTYPE.
            policy: Policy to be evaluated

        Returns:
            Lower bound estimate obtained by solving the dual problem
        """
        raise NotImplementedError
