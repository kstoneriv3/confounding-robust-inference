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
from cri.policies import BasePolicy, LogisticPolicy, GaussianPolicy
from cri.utils.types import _DEFAULT_TORCH_FLOAT_DTYPE, as_tensor


torch.random.manual_seed(0)

N_SAMPLES = 50
D = 3
BETA = as_tensor([0, 0.5, -0.5, 0, 0])


class EstimatorSpec(NamedTuple):
    estimator_cls: Type[BaseEstimator]
    parameters: list[str]
    valid_const_type: list[str]
    supports_continuous_action_space: bool


ESTIMATOR_SPECS: list[EstimatorSpec] = [
    EstimatorSpec(
        estimator_cls=DualKCMCEstimator,
        parameters=["const_type", "gamma", "Gamma", "D"],
        valid_const_type=DUAL_FEASIBLE_CONSTRAINT_TYPES,
        supports_continuous_action_space=True,
    ),
    EstimatorSpec(
        estimator_cls=DualNCMCEstimator,
        parameters=["const_type", "gamma", "Gamma"],
        valid_const_type=DUAL_FEASIBLE_CONSTRAINT_TYPES,
        supports_continuous_action_space=True,
    ),
    EstimatorSpec(
        estimator_cls=GPKCMCEstimator,
        parameters=["const_type", "gamma", "Gamma", "D"],
        valid_const_type=CONSTRAINT_TYPES,
        supports_continuous_action_space=True,
    ),
    EstimatorSpec(
        estimator_cls=HajekEstimator,
        parameters=[],
        valid_const_type=[],
        supports_continuous_action_space=False,
    ),
    EstimatorSpec(
        estimator_cls=IPWEstimator,
        parameters=[],
        valid_const_type=[],
        supports_continuous_action_space=True,
    ),
    EstimatorSpec(
        estimator_cls=KCMCEstimator,
        parameters=["const_type", "gamma", "Gamma", "D"],
        valid_const_type=CONSTRAINT_TYPES,
        supports_continuous_action_space=True,
    ),
    EstimatorSpec(
        estimator_cls=QBEstimator,
        parameters=["const_type", "Gamma", "D"],
        valid_const_type=["Tan_box", "lr_box"],
        supports_continuous_action_space=True,
    ),
    EstimatorSpec(
        estimator_cls=ZSBEstimator,
        parameters=["const_type", "Gamma"],
        valid_const_type=["Tan_box", "lr_box"],
        supports_continuous_action_space=False,
    ),
]
ESTIMATOR_NAMES = [spec.estimator_cls.__name__ for spec in ESTIMATOR_SPECS]
DATA = {
    "binary": SyntheticDataBinary().sample(N_SAMPLES),
    "continuous": SyntheticDataContinuous().sample(N_SAMPLES),
}
POLICIES = {
    "binary": LogisticPolicy(BETA),
    "continuous": GaussianPolicy(BETA),
}
TRUE_LOWER_BOUND = {
    "binary": SyntheticDataBinary().evaluate_policy_lower_bound(
        POLICIES["binary"], Gamma=1.5, n_mc=100000
    ),
    "continuous": SyntheticDataContinuous().evaluate_policy_lower_bound(
        POLICIES["continuous"], Gamma=1.5, n_mc=100000
    ),
}

@pytest.fixture(scope="function", autouse=True)
def set_seed() -> None:
    # all the random implementation uses torch
    torch.random.manual_seed(0)


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


@pytest.mark.parametrize("data_and_policy_type", ["binary", "continuous"])
@pytest.mark.parametrize("const_type", CONSTRAINT_TYPES)
@pytest.mark.parametrize("spec", ESTIMATOR_SPECS, ids=ESTIMATOR_NAMES)
def test_zero_outcome(
    data_and_policy_type: str,
    spec: EstimatorSpec,
    const_type: str,
) -> None:
    """When outcome Y == 0 everywhere, the lower bound must be 0."""
    if (
        spec.estimator_cls in (HajekEstimator, IPWEstimator)
        or const_type not in spec.valid_const_type
        or (data_and_policy_type is "continuous" and not spec.supports_continuous_action_space)
    ):
        pytest.skip()

    Y, T, X, _, p_t, _ = DATA[data_and_policy_type]
    policy = POLICIES[data_and_policy_type]
    Y = torch.zeros_like(Y)
    estimator = estimator_factory(spec, const_type, gamma=0.05, Gamma=1.5)
    est = estimator.fit(Y, T, X, p_t, policy).predict()
    zero = torch.zeros(1, dtype=_DEFAULT_TORCH_FLOAT_DTYPE)
    is_dual_estimator = "Dual" in estimator.__class__.__name__
    atol = 1e-1 if is_dual_estimator else 5e-3

    assert est <= 0
    assert torch.isclose(est, zero, atol=atol) or is_dual_estimator
    # Refit Dual estimator if the learning parameters are not appropriate.
    if not torch.isclose(est, zero, atol=atol):
        estimator = estimator_factory(spec, const_type, gamma=0.0, Gamma=1.0)
        est = estimator.fit(Y, T, X, p_t, policy, lr=1e-1, n_steps=100).predict()
        assert est <= 0
        assert torch.isclose(est, zero, atol=atol)


@pytest.mark.parametrize("data_and_policy_type", ["binary", "continuous"])
@pytest.mark.parametrize("const_type", CONSTRAINT_TYPES)
@pytest.mark.parametrize("spec", ESTIMATOR_SPECS, ids=ESTIMATOR_NAMES)
def test_singleton_uncertainty_set(
    data_and_policy_type: str,
    spec: EstimatorSpec,
    const_type: str,
) -> None:
    """When the uncertainty set of w is a singleton {1 / p_t}, the lower bound must be equal to
    the IPW estimator.
    """
    if (
        spec.estimator_cls in (HajekEstimator, IPWEstimator)
        or const_type not in spec.valid_const_type
        or (data_and_policy_type is "continuous" and not spec.supports_continuous_action_space)
    ):
        pytest.skip()
    Y, T, X, _, p_t, _ = DATA[data_and_policy_type]
    policy = POLICIES[data_and_policy_type]
    estimator = estimator_factory(spec, const_type, gamma=0.0, Gamma=1.0)
    est = estimator.fit(Y, T, X, p_t, policy).predict()
    if spec.estimator_cls == ZSBEstimator:
        target = HajekEstimator().fit(Y, T, X, p_t, policy).predict()
    else:
        target = IPWEstimator().fit(Y, T, X, p_t, policy).predict()
    is_gp_estimator = isinstance(estimator, GPKCMCEstimator)
    is_dual_estimator = "Dual" in estimator.__class__.__name__
    atol = 1e-1 if is_dual_estimator else 5e-3

    assert est <= target + atol
    assert torch.isclose(est, target, atol=atol) or is_dual_estimator or is_gp_estimator
    # Refit Dual estimator if the learning parameters are not appropriate.
    if not torch.isclose(est, target, atol=atol) and is_dual_estimator:
        estimator = estimator_factory(spec, const_type, gamma=0.0, Gamma=1.0)
        est = estimator.fit(Y, T, X, p_t, policy, lr=1e-1, n_steps=100).predict()
        assert est <= target + atol
        assert torch.isclose(est, target, atol=atol)


@pytest.mark.parametrize("data_and_policy_type", ["binary", "continuous"])
@pytest.mark.parametrize("spec", ESTIMATOR_SPECS, ids=ESTIMATOR_NAMES)
def test_true_lower_bound(
    data_and_policy_type: str,
    spec: EstimatorSpec,
) -> None:
    """The out-of-fit prediction of the lower bound should be lower than the true lower bound."""
    pytest.skip()
    if (
        hasattr(spec.estimator_cls, "predict_dual")
        or const_type not in spec.valid_const_type
        or (data_and_policy_type is "continuous" and not spec.supports_continuous_action_space)
    ):
        pytest.skip()


    Y, T, X, _, p_t, _ = DATA[data_and_policy_type]
    policy = POLICIES[data_and_policy_type]
    estimator = estimator_factory(spec, const_type, gamma=0.05, Gamma=1.5)
    estimator.fit(Y[:30], T[:30], X[:30], p_t[:30], policy)
    out_of_fit_est = estimator.predict_dual(Y[30:], T[30:], X[30:], p_t[30:], policy)

    true_lower_bound = TRUE_LOWER_BOUND[data_and_policy_type]
    atol = 1e-1 if "Dual" in estimator.__class__.__name__ else 1e-5
    assert est <= 0


def test_strong_duality() -> None:
    pass


def test_constraints_dimensions() -> None:
    # The lower bound should get tighter as the number of coinstraints increases.
    pass


def test_gic() -> None:
    # GIC < lower bound estimator
    # GIC(D=n) < GIC(D=appropriate)
    pass


def test_ci() -> None:
    pass


def test_hajek_estimator() -> None:
    Y, T, X, _, p_t, _ = DATA["binary"]
    policy = POLICIES["binary"]
    hajek = HajekEstimator().fit(Y, T, X, p_t, policy).predict()
    p_t_normalized = normalize_p_t(p_t, T)
    ipw = IPWEstimator().fit(Y, T, X, p_t_normalized, policy).predict()
    assert torch.isclose(hajek, ipw)


# how many parameters are there?
# - 6 different lower bound estimators
# - 9 different const_types
# - binary and continuous treatment space
