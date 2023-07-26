from typing import Any, Protocol, Sequence

import torch

from confounding_robust_inference.utils.docs import WithDocstringsMeta


class BasePolicy(Protocol, metaclass=WithDocstringsMeta):
    """Template of policy used in confounding-robust-inference."""

    def sample(self, X: torch.Tensor) -> torch.Tensor:
        """Sample T given X from the policy.

        Args:
            X: A tensor of shape [n_samples, n_dim]

        Returns:
            T: A tensor of shape [n_samples]
        """
        raise NotImplementedError

    def prob(self, T: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Calculate the policy's probability of sampling T given X.

        Non-proper probability such as difference of probability is allowed here for applications
        such as average treatment effect (ATE) estimation and policy improvement by Kallus and
        Zhous (2018).

        Args:
            X: A tensor of shape [n_samples, n_dim]

        Returns:
            T: A tensor of shape [n_samples]
        """
        raise NotImplementedError


class OnehotPolicy(BasePolicy):
    """A policy for calculating the value of a treatment in case of discrete action space."""

    def __init__(self, treatment: Any):
        self.treatment = treatment

    def sample(self, X: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor([self.treatment] * X.shape[0])

    def prob(self, T: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        ret = torch.zeros(X.shape[0])
        ret[T == self.treatment] = 1.0
        return ret


class PolicyDifference(BasePolicy):
    """A non-proper policy class for calculating difference of the policy values."""

    def __init__(self, target_policy: BasePolicy, base_policy: BasePolicy):
        self.target_policy = target_policy
        self.base_policy = base_policy

    def prob(self, T: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        return self.target_policy.prob(T, X) - self.base_policy.prob(T, X)


class ATEPolicy(PolicyDifference):
    """A non-proper policy class used for calculating treatment effect."""

    def __init__(self, target_treatment: Any, base_treatment: Any) -> None:
        target_policy = OnehotPolicy(target_treatment)
        base_policy = OnehotPolicy(base_treatment)
        super().__init__(target_policy, base_policy)

    def sample(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


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


class MixedPolicy(BasePolicy):
    """A mixed policy (e.g. the policy learned by DualKCMCPolicyLearner).

    Args:
        policies: Policies to be mixed
        beta: Mixing probability of the policies
    """

    def __init__(self, policies: Sequence[BasePolicy], beta: torch.Tensor) -> None:
        self.policies = policies
        self.beta = beta

    def sample(self, X: torch.Tensor) -> torch.Tensor:
        ret = torch.zeros_like(self.policies[0].sample(X))
        mix_dist = torch.distributions.Categorical(self.beta)  # type: ignore
        assigned_policy = mix_dist.sample(X.shape[:1])  # type: ignore
        for i, policy in enumerate(self.policies):
            mask = assigned_policy == i
            if mask.sum() == 0:
                continue
            ret[mask] = policy.sample(X[mask])
        return ret

    def prob(self, T: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        probs = [policy.prob(T, X) for policy in self.policies]
        ret = torch.stack([p * b for p, b in zip(probs, self.beta)]).sum(dim=0)
        return ret
