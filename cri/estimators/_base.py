import torch
from torch import Tensor

from cri.policies import BasePolicy


class BaseEstimator:
    def fit(
        self,
        Y: Tensor,
        T: Tensor,
        X: Tensor,
        p_t: Tensor,
        policy: BasePolicy,
    ) -> None:
        raise NotImplementedError

    @property
    def lower_bound(self) -> torch.Tensor | None:
        """Lower bound obtained by fit method.
        return None
        """
        raise NotImplementedError

    def predict_lower_bound_dual(self) -> float | None:
        raise NotImplementedError
