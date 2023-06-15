import torch

from cri.policies import ATEPolicy, GaussianPolicy, LogisticPolicy


def test_ate_policy() -> None:
    policy = ATEPolicy(1, 0)
    T = torch.arange(2)
    X = torch.arange(2).reshape(2, 1)
    prob = torch.as_tensor([-1, 1])
    assert all(policy.prob(T, X) == prob)


def test_gaussian_policy_shape() -> None:
    beta = torch.randn(2)
    policy = GaussianPolicy(beta)
    X = torch.randn(20).reshape(10, 2)
    T = policy.sample(X)
    p_t_x = policy.prob(T, X)
    assert T.shape == (10,)
    assert p_t_x.shape == (10,)


def test_logistic_policy_shape() -> None:
    beta = torch.randn(2).float()
    policy = LogisticPolicy(beta)
    X = torch.randn(20).reshape(10, 2)
    T = policy.sample(X)
    p_t_x = policy.prob(T, X)
    assert T.shape == (10,)
    assert p_t_x.shape == (10,)
    assert all(p_t_x <= 1)
