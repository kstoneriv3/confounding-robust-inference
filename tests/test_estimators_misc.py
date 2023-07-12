import numpy as np
import pytest
import torch

from confounding_robust_inference.estimators.misc import get_a_b, normalize_p_t


def test_normalized_p_t() -> None:
    T = torch.as_tensor([0] * 2 + [1] * 3 + [2] * 5)
    p_t = 1 / (1 + 4 * torch.rand(10))
    p_t_normalized = normalize_p_t(p_t, T)
    assert torch.isclose(torch.sum(1 / p_t_normalized[:2]), torch.as_tensor(10.0))
    assert torch.isclose(torch.sum(1 / p_t_normalized[2:5]), torch.as_tensor(10.0))
    assert torch.isclose(torch.sum(1 / p_t_normalized[5:]), torch.as_tensor(10.0))


@pytest.mark.parametrize("Gamma", [1.0, 1.5])
@pytest.mark.parametrize("const_type", ["Tan_box", "lr_box"])
def test_get_a_b(Gamma: float, const_type: str) -> None:
    p_t = np.random.rand(10)
    a, b = get_a_b(p_t, Gamma, const_type)
    assert all(0 < a)
    assert all(a <= b)
    if Gamma == 1.0:
        assert np.allclose(a, b)
