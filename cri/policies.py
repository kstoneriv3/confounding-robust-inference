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
