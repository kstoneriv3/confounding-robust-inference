import pytest
import torch

from cri.data import DataTuple, SyntheticDataBinary, SyntheticDataContinuous
from cri.estimators import (
    BaseEstimator,
    DualKCMCEstimator,
    DualNCMCEstimator,
    GPKCMCEstimator,
    HajekEstimator,
    IPWEstimator,
    KCMCEstimator,
    QBEstimator,
    ZSBEstimator,
)
from cri.estimators.misc import normalize_p_t
from cri.policies import BasePolicy, LogisticPolicy
from cri.utils.types import _DEFAULT_TORCH_FLOAT_DTYPE, as_tensor

N_SAMPLES = 100


@pytest.fixture(scope="session")
def toy_policy() -> BasePolicy:
    beta = as_tensor([0, 0.5, -0.5, 0, 0])
    return LogisticPolicy(beta)


@pytest.fixture(scope="session")
def binary_data() -> DataTuple:
    data = SyntheticDataBinary()
    return data.sample(N_SAMPLES)


@pytest.fixture(scope="session")
def continuous_data() -> DataTuple:
    data = SyntheticDataContinuous()
    return data.sample(N_SAMPLES)


zero_outcome_estimators = [
    DualKCMCEstimator("Tan_box", Gamma=1.5, D=5),
    DualNCMCEstimator("Tan_box", Gamma=1.5),
    GPKCMCEstimator("Tan_box", Gamma=1.5, D=5),
    HajekEstimator(),
    IPWEstimator(),
    KCMCEstimator("Tan_box", Gamma=1.5, D=5),
    QBEstimator("Tan_box", Gamma=1.5, D=5),
    ZSBEstimator(1.5),
]
zero_outcome_test_names = [estimator.__class__.__name__ for estimator in zero_outcome_estimators]


@pytest.mark.parametrize("estimator", zero_outcome_estimators, ids=zero_outcome_test_names)
def test_zero_outcome(
    binary_data: DataTuple, toy_policy: BasePolicy, estimator: BaseEstimator
) -> None:
    Y, T, X, _, p_t, _ = binary_data
    Y[:] = torch.zeros_like(Y)
    est = estimator.fit(Y, T, X, p_t, toy_policy).predict()
    zero = torch.zeros(1, dtype=_DEFAULT_TORCH_FLOAT_DTYPE)
    atol = 1e-2 if "Dual" in estimator.__class__.__name__ else 1e-6
    assert torch.isclose(est, zero, atol=atol)


def test_hajek_estimator(binary_data: DataTuple, toy_policy: BasePolicy) -> None:
    Y, T, X, _, p_t, _ = binary_data
    hajek = HajekEstimator().fit(Y, T, X, p_t, toy_policy).predict()
    p_t_normalized = normalize_p_t(p_t, T)
    ipw = IPWEstimator().fit(Y, T, X, p_t_normalized, toy_policy).predict()
    assert torch.isclose(hajek, ipw)
