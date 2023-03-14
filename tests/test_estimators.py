from typing import NamedTuple, Type

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
from cri.estimators.misc import CONSTRAINT_TYPES, DUAL_FEASIBLE_CONSTRAINT_TYPES, normalize_p_t
from cri.policies import BasePolicy, LogisticPolicy
from cri.utils.types import _DEFAULT_TORCH_FLOAT_DTYPE, as_tensor

N_SAMPLES = 50
D = 3


class EstimatorSpec(NamedTuple):
    estimator_cls: Type[BaseEstimator]
    parameters: list[str]
    valid_const_type: list[str]


ESTIMATOR_SPECS: list[EstimatorSpec] = [
    EstimatorSpec(
        estimator_cls=DualKCMCEstimator,
        parameters=["const_type", "gamma", "Gamma", "D"],
        valid_const_type=DUAL_FEASIBLE_CONSTRAINT_TYPES,
    ),
    EstimatorSpec(
        estimator_cls=DualNCMCEstimator,
        parameters=["const_type", "gamma", "Gamma"],
        valid_const_type=DUAL_FEASIBLE_CONSTRAINT_TYPES,
    ),
    EstimatorSpec(
        estimator_cls=GPKCMCEstimator,
        parameters=["const_type", "gamma", "Gamma", "D"],
        valid_const_type=CONSTRAINT_TYPES,
    ),
    EstimatorSpec(
        estimator_cls=HajekEstimator,
        parameters=[],
        valid_const_type=[],
    ),
    EstimatorSpec(
        estimator_cls=IPWEstimator,
        parameters=[],
        valid_const_type=[],
    ),
    EstimatorSpec(
        estimator_cls=KCMCEstimator,
        parameters=["const_type", "gamma", "Gamma", "D"],
        valid_const_type=CONSTRAINT_TYPES,
    ),
    EstimatorSpec(
        estimator_cls=QBEstimator,
        parameters=["const_type", "Gamma", "D"],
        valid_const_type=["Tan_box", "lr_box"],
    ),
    EstimatorSpec(
        estimator_cls=ZSBEstimator,
        parameters=["const_type", "Gamma"],
        valid_const_type=["Tan_box", "lr_box"],
    ),
]
ESTIMATOR_NAMES = [spec.estimator_cls.__name__ for spec in ESTIMATOR_SPECS]


@pytest.fixture(scope="function", autouse=True)
def set_seed() -> None:
    # all the random implementation uses torch
    torch.random.manual_seed(0)


@pytest.fixture(scope="session")
def toy_policy() -> BasePolicy:
    beta = as_tensor([0, 0.5, -0.5, 0, 0])
    return LogisticPolicy(beta)


@pytest.fixture(scope="session")
def binary_data() -> DataTuple:
    torch.random.manual_seed(0)
    data = SyntheticDataBinary()
    return data.sample(N_SAMPLES)


@pytest.fixture(scope="session")
def continuous_data() -> DataTuple:
    torch.random.manual_seed(0)
    data = SyntheticDataContinuous()
    return data.sample(N_SAMPLES)


def estimator_factory(
    spec: EstimatorSpec,
    const_type: str = "Tan_box",
    gamma: float = 0.0,
    Gamma: float = 1.0,
    D: int = 3,
) -> BaseEstimator:
    kwargs: dict[str, str | float | int] = {}
    if "const_type" in spec.parameters:
        kwargs["const_type"] = const_type
    # If IPW or Hajek, skip the test except when const_type == "Tan_box".
    elif const_type != "Tan_box":
        pytest.skip()
    if "Gamma" in spec.parameters:
        kwargs["Gamma"] = Gamma
    if "gamma" in spec.parameters:
        kwargs["gamma"] = gamma
    if "D" in spec.parameters:
        kwargs["D"] = D
    return spec.estimator_cls(**kwargs)


@pytest.mark.parametrize("const_type", CONSTRAINT_TYPES)
@pytest.mark.parametrize("spec", ESTIMATOR_SPECS, ids=ESTIMATOR_NAMES)
def test_zero_outcome(
    binary_data: DataTuple,
    toy_policy: BasePolicy,
    spec: EstimatorSpec,
    const_type: str,
) -> None:
    """When outcome Y == 0 everywhere, the lower bound must be 0."""
    if (
        spec.estimator_cls in (HajekEstimator, IPWEstimator)
        or const_type not in spec.valid_const_type
    ):
        pytest.skip()
    Y, T, X, _, p_t, _ = binary_data
    Y[:] = torch.zeros_like(Y)
    estimator = estimator_factory(spec, const_type, gamma=0.05, Gamma=1.5)
    est = estimator.fit(Y, T, X, p_t, toy_policy).predict()
    zero = torch.zeros(1, dtype=_DEFAULT_TORCH_FLOAT_DTYPE)
    atol = 1e-2 if "Dual" in estimator.__class__.__name__ else 1e-6
    assert torch.isclose(est, zero, atol=atol)


@pytest.mark.parametrize("const_type", CONSTRAINT_TYPES)
@pytest.mark.parametrize("spec", ESTIMATOR_SPECS, ids=ESTIMATOR_NAMES)
def test_singleton_uncertainty_set(
    binary_data: DataTuple,
    toy_policy: BasePolicy,
    spec: EstimatorSpec,
    const_type: str,
) -> None:
    """When the uncertainty set of w is a singleton {1 / p_t}, the lower bound must be equal to
    the IPW estimator.
    """
    if (
        spec.estimator_cls in (HajekEstimator, IPWEstimator)
        or const_type not in spec.valid_const_type
    ):
        pytest.skip()
    Y, T, X, _, p_t, _ = binary_data
    estimator = estimator_factory(spec, const_type, gamma=0.0, Gamma=1.0)
    est = estimator.fit(Y, T, X, p_t, toy_policy).predict()
    if spec.estimator_cls == ZSBEstimator:
        target = HajekEstimator().fit(Y, T, X, p_t, toy_policy).predict()
    else:
        target = IPWEstimator().fit(Y, T, X, p_t, toy_policy).predict()
    # atol = 1e-2 if "Dual" in estimator.__class__.__name__ else 1e-6
    # assert torch.isclose(est, target, atol=atol)
    assert torch.isclose(est, target)


def test_true_lower_bound() -> None:
    pass


def test_overfitting() -> None:
    # predict dual for out-of-fit data
    pass


def test_strong_duality() -> None:
    pass


def test_gic() -> None:
    pass


def test_ci() -> None:
    pass


def test_hajek_estimator(binary_data: DataTuple, toy_policy: BasePolicy) -> None:
    Y, T, X, _, p_t, _ = binary_data
    hajek = HajekEstimator().fit(Y, T, X, p_t, toy_policy).predict()
    p_t_normalized = normalize_p_t(p_t, T)
    ipw = IPWEstimator().fit(Y, T, X, p_t_normalized, toy_policy).predict()
    assert torch.isclose(hajek, ipw)


# how many parameters are there?
# - 6 different lower bound estimators
# - 9 different const_types
# - binary and continuous treatment space
