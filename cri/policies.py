from typing import Any

import torch


class BasePolicy:
    """Template of policy used in CRI."""

    def sample(self, X: torch.Tensor) -> torch.Tensor:
        """Sample T given X from the policy.

        Args:
            X: tensor of shape [n_samples, n_dim]

        Returns:
            T: tensor of shape [n_samples]
        """
        raise NotImplementedError

    def prob(self, T: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Calculate the policy's probability of sampling T given X.

        Non-proper probability such as difference of probability is allowed here for applications
        such as average treatment effect (ATE) estimation and policy improvement by Kallus and
        Zhous (2018).

        Args:
            X: tensor of shape [n_samples, n_dim]

        Returns:
            T: tensor of shape [n_samples]
        """
        raise NotImplementedError


class OnehotPolicy(BasePolicy):
    """A policy for calculating the value of a treatment in case of discrete action space."""

    def __init__(self, treatment: Any):
        self.treatment = treatment

    def sample(self, X: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor([self.treatment] * X.shape[0])

    def prob(self, T: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        return torch.ones(X.shape[0])


class PolicyDifference(BasePolicy):
    """A non-proper policy class for calculating difference of the policy values."""

    def __init__(self, target_policy: BasePolicy, base_policy: BasePolicy):
        self.target_policy = target_policy
        self.base_policy = base_policy

    def prob(self, T: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        ret = torch.ones_like(T)
        ret[T == 0] = -1
        return ret


class ATEPolicy(PolicyDifference):
    """A non-proper policy class used for calculating treatment effect."""

    def __init__(self, target_treatment: Any, base_treatment: Any) -> None:
        target_policy = OnehotPolicy(target_treatment)
        base_policy = OnehotPolicy(base_treatment)
        super().__init__(target_policy, base_policy)


class GaussianPolicy(BasePolicy):
    """Continuous policy with conditional Gaussian density.

    The action follows Gaussian with mean beta^T x and standard deviation 0.5
    so that T|X=x ~ N(beta^T x, 0.25).

    Args:
        beta: Coefficients of the mean
    """

    def __init__(self, beta: torch.Tensor) -> None:
        assert len(beta.shape) == 1
        self.beta = beta

    def sample(self, X: torch.Tensor) -> torch.Tensor:
        mu_t_x = X @ self.beta
        T = mu_t_x + 0.5 * torch.randn(X.shape[0])
        return T

    def prob(self, T: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        mu_t_x = X @ self.beta
        p_t_x = torch.distributions.Normal(mu_t_x, 0.5).log_prob(T).exp()  # type: ignore
        return p_t_x


class LogisticPolicy(BasePolicy):
    """Binary policy with logistic probability.

    The action follows the probability T|X=x ~ Bin(sigmoid(beta^T x)).

    Args:
        beta: Coefficients of the logit
    """

    def __init__(self, beta: torch.Tensor) -> None:
        assert len(beta.shape) == 1
        self.beta = beta

    def sample(self, X: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(X @ self.beta)  # p(T=1|X)
        T = (torch.rand(X.shape[0]) < p).int()
        return T

    def prob(self, T: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(X @ self.beta)  # p(T=1|X)
        p_t_x = p * T + (1 - p) * (1 - T)
        return p_t_x
