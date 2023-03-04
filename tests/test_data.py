import pytest

import torch

from cri.data import (
    SyntheticDataKallusZhou2018,
    SyntheticDataKallusZhou2018Continuous,
    SyntheticDataBinary,
    SyntheticDataContinuous,
    NLSDataDornGuo2022,
)
from cri.policies import BasePolicy

DATA_CLASSES = [
    SyntheticDataKallusZhou2018,
    SyntheticDataKallusZhou2018Continuous,
    SyntheticDataBinary,
    SyntheticDataContinuous,
    NLSDataDornGuo2022,
]

class BinaryPolicy(BasePolicy):
    def sample(self, X: torch.Tensor) -> torch.Tensor:
        return (torch.rand(X.shape[0]) < 0.5).int()

    def prob(self, T: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.ones_like(T)


class ContinuousPolicy(BasePolicy):
    def sample(self, X: torch.Tensor) -> torch.Tensor:
        dist = torch.distributions.Normal(0., 1.)
        return dist.sample([X.shape[0]])

    def prob(self, T: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        dist = torch.distributions.Normal(0., 1.)
        return dist.log_prob(T).exp()


@pytest.mark.parametrize("data_cls", DATA_CLASSES)
def test_sample_shape(data_cls) -> None:
    if "DornGuo2022" in data_cls.__name__:
        n = 667
        d = 8
    else:
        n = 30
        d = 5

    Y, T, X, U, p_t_x, p_t_xu = data_cls().sample(n)

    assert Y.shape == (n,)
    assert X.shape == (n, d)
    assert T.shape == (n,)
    assert p_t_x.shape == (n,)

    if "KallusZhou2018" in data_cls.__name__:
        assert U.shape == (n,)
        assert p_t_xu.shape == (n,)
    else:
        assert U is None
        assert p_t_xu is None


@pytest.mark.parametrize("data_cls", DATA_CLASSES)
def test_evaluate_policy_shape(data_cls) -> None:
    if "DornGuo2022" in data_cls.__name__:
        pytest.skip("No evaluate_policy method implemeted.")
    n = 30
    d = 5
    data = data_cls()
    Y, T, X, U, p_t_x, p_t_xu = data.sample(n)
    if "Continuous" in data_cls.__name__:
        policy = ContinuousPolicy()
    else:
        policy = BinaryPolicy()

    assert data.evaluate_policy(policy, 10).shape == tuple()


@pytest.mark.parametrize("data_cls", DATA_CLASSES)
def test_evaluate_policy_lower_bound_shape(data_cls) -> None:
    if "DornGuo2022" in data_cls.__name__ or "KallusZhou2018" in data_cls.__name__:
        pytest.skip("No evaluate_policy_lower_bound method implemeted.")
    n = 30
    d = 5
    data = data_cls()
    Y, T, X, U, p_t_x, p_t_xu = data.sample(n)
    if "Continuous" in data_cls.__name__:
        policy = ContinuousPolicy()
    else:
        policy = BinaryPolicy()

    assert data.evaluate_policy_lower_bound(policy, 1.5, 10).shape == tuple()
