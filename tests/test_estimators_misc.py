import torch

from cri.estimators.misc import normalize_p_t


def test_normalized_p_t() -> None:
    T = torch.as_tensor([0] * 2 + [1] * 3 + [2] * 5)
    p_t = 1 / (1 + 4 * torch.rand(10))
    p_t_normalized = normalize_p_t(p_t, T)
    assert torch.isclose(torch.sum(1 / p_t_normalized[:2]), torch.as_tensor(10.0))
    assert torch.isclose(torch.sum(1 / p_t_normalized[2:5]), torch.as_tensor(10.0))
    assert torch.isclose(torch.sum(1 / p_t_normalized[5:]), torch.as_tensor(10.0))
