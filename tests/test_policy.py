import torch

from confounding_robust_inference.policies import (
    ATEPolicy,
    GaussianPolicy,
    LogisticPolicy,
    MixedPolicy,
    OnehotPolicy,
)


def test_ate_policy() -> None:
    policy = ATEPolicy(1, 0)
    T = torch.arange(2)
    X = torch.arange(2).reshape(2, 1)
    p_t_x = policy.prob(T, X)
    assert all(p_t_x == torch.as_tensor([-1, 1]))


def test_gaussian_policy() -> None:
    beta = torch.randn(2)
    policy = GaussianPolicy(beta)
    X = torch.randn(20).reshape(10, 2)
    T = policy.sample(X)
    p_t_x = policy.prob(T, X)
    assert T.dtype == torch.float32
    assert T.shape == (10,)
    assert p_t_x.dtype == torch.float32
    assert p_t_x.shape == (10,)


def test_logistic_policy() -> None:
    beta = torch.randn(2).float()
    policy = LogisticPolicy(beta)
    X = torch.randn(20).reshape(10, 2)
    T = policy.sample(X)
    p_t_x = policy.prob(T, X)
    assert T.dtype == torch.int32
    assert T.shape == (10,)
    assert p_t_x.dtype == torch.float32
    assert p_t_x.shape == (10,)
    assert all(p_t_x <= 1)


def test_mixed_policy() -> None:
    beta = torch.Tensor([0, 0.5, 0.5])
    policies = [OnehotPolicy(i) for i in range(3)]
    policy = MixedPolicy(policies, beta)
    X = torch.randn(20).reshape(10, 2)
    T = policy.sample(X)
    p_t_x = policy.prob(T, X)
    assert T.dtype == torch.int64
    assert T.shape == (10,)
    assert all(T != 0)
    assert all(p_t_x == 0.5)
