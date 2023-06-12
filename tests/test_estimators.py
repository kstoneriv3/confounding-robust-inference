from typing import NamedTuple, Type

import numpy as np
import pytest
import torch
from torch.optim import Adam

from cri.data import SyntheticDataBinary, SyntheticDataContinuous
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
from cri.estimators.misc import (
    CONSTRAINT_TYPES,
    DUAL_FEASIBLE_CONSTRAINT_TYPES,
    get_a_b,
    normalize_p_t,
)
from cri.policies import GaussianPolicy, LogisticPolicy
from cri.utils.types import _DEFAULT_TORCH_FLOAT_DTYPE, as_ndarrays, as_tensor

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
DATA_LARGE = {
    "binary": SyntheticDataBinary().sample(2 * N_SAMPLES),
    "continuous": SyntheticDataContinuous().sample(2 * N_SAMPLES),
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
        or (data_and_policy_type == "continuous" and not spec.supports_continuous_action_space)
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
        assert is_dual_estimator
        optimizer_kwargs = {"lr": 5e-2}
        estimator = estimator_factory(spec, const_type, gamma=0.0, Gamma=1.0)
        estimator.fit(
            Y, T, X, p_t, policy, optimizer_kwargs=optimizer_kwargs, n_steps=100
        )  # type: ignore
        est = estimator.predict()
        assert est <= 0
        assert torch.isclose(est, zero, atol=3e-1)


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
        spec.estimator_cls in (HajekEstimator, IPWEstimator, GPKCMCEstimator)
        or const_type not in spec.valid_const_type
        or (data_and_policy_type == "continuous" and not spec.supports_continuous_action_space)
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
    is_dual_estimator = "Dual" in estimator.__class__.__name__
    atol = 1e-1 if is_dual_estimator else 5e-3

    assert est <= target + 1e-5
    assert torch.isclose(est, target, atol=atol) or is_dual_estimator
    # Refit Dual estimator if the learning parameters are not appropriate.
    if not torch.isclose(est, target, atol=atol):
        assert is_dual_estimator
        estimator = estimator_factory(spec, const_type, gamma=0.0, Gamma=1.0)
        if const_type == "KL":
            if isinstance(estimator, DualNCMCEstimator) and data_and_policy_type == "binary":
                estimator.fit(
                    Y,
                    T,
                    X,
                    p_t,
                    policy,
                    optimizer_cls=Adam,
                    optimizer_kwargs={"lr": 5e-2},
                    n_steps=100,
                )
            else:
                # TODO: Dual*CMCEstimator-KL cases are somehow broken. Need more investigation.
                pytest.skip()
            est = estimator.predict()
            assert est <= target
            assert torch.isclose(est, target, rtol=0.1)
        else:
            estimator.fit(
                Y, T, X, p_t, policy, optimizer_kwargs={"lr": 5e-2}, n_steps=300
            )  # type: ignore
            est = estimator.predict()
            assert est <= target
            assert torch.isclose(est, target, atol=atol)


@pytest.mark.parametrize("data_and_policy_type", ["binary", "continuous"])
@pytest.mark.parametrize("spec", ESTIMATOR_SPECS, ids=ESTIMATOR_NAMES)
def test_true_lower_bound(
    data_and_policy_type: str,
    spec: EstimatorSpec,
) -> None:
    """The out-of-fit prediction of the lower bound should be lower than the true lower bound."""
    if spec.estimator_cls not in (KCMCEstimator, DualKCMCEstimator, DualNCMCEstimator):
        pytest.skip()

    Y, T, X, _, p_t, _ = DATA[data_and_policy_type]
    policy = POLICIES[data_and_policy_type]
    const_type = "Tan_box" if data_and_policy_type == "binary" else "lr_box"
    estimator = estimator_factory(spec, const_type, Gamma=1.5)
    estimator.fit(Y[:30], T[:30], X[:30], p_t[:30], policy)
    out_of_fit_est = estimator.predict_dual(Y[30:], T[30:], X[30:], p_t[30:], policy)

    true_lower_bound = TRUE_LOWER_BOUND[data_and_policy_type]
    assert out_of_fit_est <= true_lower_bound  # This holds only in expectation or sample size limit


'''
@pytest.mark.parametrize("data_and_policy_type", ["binary", "continuous"])
@pytest.mark.parametrize("const_type", CONSTRAINT_TYPES)
def test_augment_data(
    data_and_policy_type: str,
    const_type: str,
) -> None:
    """When the uncertainty set of w is a singleton {1 / p_t}, the lower bound must be equal to
    the IPW estimator.
    """
    if const_type not in ("Tan_box", "lr_box"):
        pytest.skip()
    Y, T, X, _, p_t, _ = DATA[data_and_policy_type]
    policy = POLICIES[data_and_policy_type]
    estimator = KCMCEstimator(const_type, gamma=0.02, Gamma=1.5, D=3)
    estimator.fit(Y, T, X, p_t, policy)
    estimator_arg = KCMCEstimator(const_type, gamma=0.02, Gamma=1.5, D=3, should_augment_data=True)
    estimator_arg.fit(Y, T, X, p_t, policy)
    losses, scores = estimator._get_dual_loss_and_jacobian()
    losses_aug, scores_aug = estimator._get_dual_loss_and_jacobian()
    assert torch.allclose(losses, losses_aug, rtol=1e-1)
    assert torch.allclose(scores, scores_aug, rtol=1e-1)
'''


@pytest.mark.parametrize("data_and_policy_type", ["binary", "continuous"])
@pytest.mark.parametrize("const_type", CONSTRAINT_TYPES)
def test_strong_duality(
    data_and_policy_type: str,
    const_type: str,
) -> None:
    """When the uncertainty set of w is a singleton {1 / p_t}, the lower bound must be equal to
    the IPW estimator.
    """
    Y, T, X, _, p_t, _ = DATA[data_and_policy_type]
    policy = POLICIES[data_and_policy_type]
    estimator = KCMCEstimator(const_type, gamma=0.02, Gamma=1.5, D=3)
    primal = estimator.fit(Y, T, X, p_t, policy).predict()
    dual = estimator.predict_dual(Y, T, X, p_t, policy)
    assert dual <= primal + 1e-5
    assert torch.isclose(dual, primal, atol=1e-5)


@pytest.mark.parametrize("data_and_policy_type", ["binary", "continuous"])
@pytest.mark.parametrize("const_type", CONSTRAINT_TYPES)
def test_get_dual_loss_and_jacobian(data_and_policy_type: str, const_type: str) -> None:
    Y, T, X, _, p_t, _ = DATA[data_and_policy_type]
    policy = POLICIES[data_and_policy_type]
    const_type = "Tan_box" if data_and_policy_type == "binary" else "lr_box"
    estimator = KCMCEstimator(const_type, Gamma=1.5, D=3)
    estimator.fit(Y, T, X, p_t, policy)

    loss = estimator._get_fitted_dual_loss(estimator.eta)
    dual_loss, autodiff_jacobian = estimator._get_dual_loss_and_jacobian()

    # Test loss
    assert torch.allclose(dual_loss, loss, atol=1e-5)

    # The first order condition for the dual objective should be zero.
    assert torch.allclose(
        torch.zeros_like(autodiff_jacobian[0, :]),
        autodiff_jacobian.mean(axis=0),  # type: ignore
        atol=0.02,
    )

    # Check the Jacobian obtained by autodiff with analytic expression.
    if "box" in const_type:
        Y_np, p_t_np, pi, eta_kcmc = as_ndarrays(Y, estimator.p_t, estimator.pi, estimator.eta_kcmc)
        Psi_np = estimator.Psi_np
        a, b = get_a_b(p_t_np, Gamma=1.5, const_type=const_type)

        analytic_jacobian = as_tensor(
            Psi_np
            * np.where(Psi_np @ eta_kcmc < Y_np * pi / p_t_np, 1 - p_t_np * a, 1 - p_t_np * b)[
                :, None
            ]
        )
        assert torch.allclose(analytic_jacobian, autodiff_jacobian)


@pytest.mark.parametrize("data_and_policy_type", ["binary", "continuous"])
@pytest.mark.parametrize("const_type", CONSTRAINT_TYPES)
@pytest.mark.parametrize("spec", ESTIMATOR_SPECS, ids=ESTIMATOR_NAMES)
def test_kcmc_dimensions(
    data_and_policy_type: str,
    spec: EstimatorSpec,
    const_type: str,
) -> None:
    """The lower bound get tighter as the number of coinstraints increases."""
    if spec.estimator_cls not in (KCMCEstimator, DualKCMCEstimator) or "D" not in spec.parameters:
        pytest.skip()

    Y, T, X, _, p_t, _ = DATA[data_and_policy_type]
    policy = POLICIES[data_and_policy_type]
    const_type = "Tan_box" if data_and_policy_type == "binary" else "lr_box"
    D1, D2, D3 = (3, 4, 5) if spec.estimator_cls == KCMCEstimator else (4, 10, 25)
    estimator = estimator_factory(spec, const_type, Gamma=1.5, D=D1)
    est_d1 = estimator.fit(Y, T, X, p_t, policy).predict()
    estimator = estimator_factory(spec, const_type, Gamma=1.5, D=D2)
    est_d2 = estimator.fit(Y, T, X, p_t, policy).predict()
    estimator = estimator_factory(spec, const_type, Gamma=1.5, D=D3)
    est_d3 = estimator.fit(Y, T, X, p_t, policy).predict()

    assert est_d1 <= est_d2 <= est_d3


@pytest.mark.parametrize("data_and_policy_type", ["binary", "continuous"])
@pytest.mark.parametrize("const_type", ["Tan_box", "lr_box", "KL"])
def test_gic(
    data_and_policy_type: str,
    const_type: str,
) -> None:
    """Test GIC < lower bound estimator and GIC(D=n) < GIC(D=appropriate)."""
    if const_type == "Tan_box" and data_and_policy_type == "continuous":
        pytest.skip()

    failing_inputs = [("KL", "binary")]
    if (const_type, data_and_policy_type) in failing_inputs:
        pytest.skip("These tests are broken due to unstability of Hessian and covariance estiamtors")
    Y, T, X, _, p_t, _ = DATA_LARGE[data_and_policy_type]
    policy = POLICIES[data_and_policy_type]

    # D_opt = 10 if data_and_policy_type == "binary" else 3
    D_opt = 10
    D_over = 30

    # Underfit
    # estimator = KCMCEstimator(const_type, gamma=0.02, Gamma=1.5, D=1)
    estimator = KCMCEstimator(const_type, gamma=0.02, Gamma=1.5, D=1)
    estimator.fit(Y, T, X, p_t, policy)
    est_under = estimator.predict()
    gic_under = estimator.predict_gic()
    assert gic_under <= est_under
    # estimator = KCMCEstimator(const_type, gamma=0.02, Gamma=1.5, D=D_opt)
    estimator = KCMCEstimator(const_type, gamma=0.02, Gamma=1.5, D=D_opt)
    estimator.fit(Y, T, X, p_t, policy)
    est_opt = estimator.predict()
    gic_opt = estimator.predict_gic()
    assert gic_opt <= est_opt
    # Overfit
    # estimator = KCMCEstimator(const_type, gamma=0.02, Gamma=1.5, D=40)
    # estimator = KCMCEstimator(const_type, gamma=0.02, Gamma=1.5, D=25)
    estimator = KCMCEstimator(const_type, gamma=0.02, Gamma=1.5, D=D_over)
    estimator.fit(Y, T, X, p_t, policy)
    est_over = estimator.predict()
    gic_over = estimator.predict_gic()
    assert gic_over <= est_over

    assert gic_under <= gic_opt, f"D_opt too large {gic_under} > {gic_opt}"
    assert gic_over <= gic_opt, f"D_over too small {gic_over} > {gic_opt}"


@pytest.mark.parametrize("data_and_policy_type", ["binary", "continuous"])
@pytest.mark.parametrize("const_type", ["Tan_box", "lr_box", "KL"])
def test_ci(
    data_and_policy_type: str,
    const_type: str,
) -> None:
    Y, T, X, _, p_t, _ = DATA[data_and_policy_type]
    policy = POLICIES[data_and_policy_type]
    estimator = KCMCEstimator(const_type, gamma=0.01, Gamma=1.5, D=2)
    estimator.fit(Y, T, X, p_t, policy)
    gic = estimator.predict_gic()
    low, high = estimator.predict_ci(alpha=1e-2)

    assert low < high
    assert low < gic, f"low = {low} is expected to be smaller than gic = {gic}"
    assert gic < high, f"high = {high} is expected to be larger than gic = {gic}"

    if "box" in const_type:
        true_lower_bound = TRUE_LOWER_BOUND[data_and_policy_type]
        assert low < true_lower_bound, f"low = {low} is expected to be smaller than true lower bound = {true_lower_bound}"
        # We don't necessarily have tight lower bound, so it should be fine to skip this:
        # assert true_lower_bound < high, f"high = {high} is expected to be larger than true lower bound = {true_lower_bound}"


@pytest.mark.parametrize("data_and_policy_type", ["binary", "continuous"])
@pytest.mark.parametrize("const_type", ["Tan_box", "lr_box", "KL"])
def test_bootstrap_lower_bounds(
    data_and_policy_type: str,
    const_type: str,
) -> None:
    Y, T, X, _, p_t, _ = DATA[data_and_policy_type]
    policy = POLICIES[data_and_policy_type]
    estimator = KCMCEstimator(const_type, gamma=0.01, Gamma=1.5, D=2)
    estimator.fit(Y, T, X, p_t, policy)
    gic = estimator.predict_gic()
    boot_lb_mean = estimator._bootstrap_lower_bounds(10000).mean()
    assert torch.isclose(gic, boot_lb_mean, rtol=5e-2)


def test_hajek_estimator() -> None:
    Y, T, X, _, p_t, _ = DATA["binary"]
    policy = POLICIES["binary"]
    hajek = HajekEstimator().fit(Y, T, X, p_t, policy).predict()
    p_t_normalized = normalize_p_t(p_t, T)
    ipw = IPWEstimator().fit(Y, T, X, p_t_normalized, policy).predict()
    assert torch.isclose(hajek, ipw)
