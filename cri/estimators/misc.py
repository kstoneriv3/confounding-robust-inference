import torch


def normalize_p_t(p_t: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Normalize p_t for individual treatment T when treatment space is discrete.

    The normlization enforces the constraint E[1 / p_t|T = t] = 1 for any t at the empirical level.
    """
    p_t_new = torch.zeros_like(p_t)
    for t in set(T):
        p_t_new[T == t] = p_t[T == t] * torch.mean((T == t) / p_t)
    return p_t_new


def assert_input(
    Y: torch.Tensor,
    T: torch.Tensor,
    X: torch.Tensor,
    p_t: torch.Tensor,
) -> None:
    assert len(Y.shape) == 1
    assert len(T.shape) == 1
    assert len(X.shape) == 2
    assert len(p_t.shape) == 1

    assert Y.shape[0] == T.shape[0] == X.shape[0]
    assert Y.dtype == X.dtype == X.dtype
