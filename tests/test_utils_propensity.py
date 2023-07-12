import torch

from confounding_robust_inference.data import SyntheticDataBinary, SyntheticDataContinuous
from confounding_robust_inference.utils.propensity import (
    estimate_p_t_binary,
    estimate_p_t_bounded_continuous,
    estimate_p_t_continuous,
)


def test_estimate_p_t_binary_shape() -> None:
    n = 30
    Y, T, X, U, p_t_x, p_t_xu = SyntheticDataBinary().sample(n)
    assert estimate_p_t_binary(T, X).shape == (n,)


def test_estimate_p_t_bounded_continuous_shape() -> None:
    n = 30
    Y, T, X, U, p_t_x, p_t_xu = SyntheticDataContinuous().sample(n)
    T = torch.sigmoid(T)
    assert estimate_p_t_bounded_continuous(T, X).shape == (n,)


def test_estimate_p_t_continuous_shape() -> None:
    n = 30
    Y, T, X, U, p_t_x, p_t_xu = SyntheticDataBinary().sample(n)
    assert estimate_p_t_continuous(T, X).shape == (n,)
