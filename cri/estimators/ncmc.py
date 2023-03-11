import numpy as np
import torch
from torch.optim import SGD

from cri.estimators.base import BaseEstimator
from cri.estimators.misc import F_DIVERGENCES, assert_input, get_dual_objective
from cri.policies import BasePolicy

CONSTRAINT_TYPES = F_DIVERGENCES + ["Tan_box", "lr_box"]


class DualNCMCEstimator(BaseEstimator):
    """Dual Neural Conditional Moment Constraints (NCMC) Estimator.

    Args:
        const_type: Type of the constraint used. It must be one of "Tan_box", "lr_box", "KL",
            "inverse_KL", "Jensen_Shannon", "squared_Hellinger", "Pearson_chi_squared",
            "Neyman_chi_squared", and "total_variation".
        dim_tx: dim(T) + dim(X).
        gamma: Sensitivity parameter for f-divergence constraint satisfying Gamma >= 1.0.
            When gamma == 0.0, QB estimator is equivalent to the IPW estimator.
        Gamma: Sensitivity parameter for box constraints satisfying Gamma >= 1.0.
            When Gamma == 1.0, QB estimator is equivalent to the IPW estimator.
        n_layers: The number of layers used by the neural network.
        n_hidden: The number of nodes in hidden layers of the neural network.
    """

    def __init__(
        self,
        const_type: str,
        gamma: float | None = None,
        Gamma: float | None = None,
        n_layers: int = 3,
        n_hidden: int = 64,
    ) -> None:
        assert const_type in CONSTRAINT_TYPES
        if "box" in const_type:
            assert Gamma is not None and Gamma >= 1
        else:
            assert gamma is not None and gamma >= 0
        self.const_type = const_type
        self.gamma = gamma if gamma is not None else 0.0
        self.Gamma = Gamma if Gamma is not None else 1.0
        self.n_layers = n_layers
        self.n_hidden = n_hidden

    def fit(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
        X: torch.Tensor,
        p_t: torch.Tensor,
        policy: BasePolicy,
        n_steps: int = 200,
        batch_size: int = 1024,
        lr: float = 1e-2,
    ) -> "BaseEstimator":
        assert_input(Y, T, X, p_t)
        pi = policy.prob(T, X)
        TX = torch.concat([T[:, None], X], dim=1)
        self.eta_nn, self.eta_f = get_multipliers(TX.shape[0], self.n_hidden, self.n_layers)

        n = T.shape[0]

        optimizer = SGD(params=list(self.eta_nn.parameters()) + [self.eta_f], lr=lr)
        for i in range(n_steps):
            train_idx = torch.as_tensor(np.random.choice(n, batch_size))
            eta_cmc = self.eta_nn(TX[train_idx]) * pi[train_idx] / p_t[train_idx]
            objective = -get_dual_objective(
                Y[train_idx],
                p_t[train_idx],
                pi[train_idx],
                eta_cmc[train_idx],
                self.eta_f,
                self.gamma,
                self.Gamma,
                self.const_type,
            )
            objective.backward()  # type: ignore
            optimizer.step()
            optimizer.zero_grad()

        m = 1024
        lower_bounds = torch.zeros(n)
        for i in range((n + m - 1) // m):
            val_idx = slice(m * i, min(n, m * (i + 1)))
            eta_cmc = self.eta_nn(TX[val_idx]) * pi[val_idx] / p_t[val_idx]
            lower_bounds[val_idx] = get_dual_objective(
                Y[val_idx],
                p_t[val_idx],
                pi[val_idx],
                eta_cmc,
                self.eta_f,
                self.gamma,
                self.Gamma,
                self.const_type,
            )
        self.fitted_lower_bound = torch.mean(lower_bounds)
        return self

    def predict(self) -> torch.Tensor:
        return self.fitted_lower_bound

    def predict_dual(
        self,
        Y: torch.Tensor,
        T: torch.Tensor,
        X: torch.Tensor,
        p_t: torch.Tensor,
        policy: BasePolicy,
    ) -> torch.Tensor:
        assert hasattr(self, "fitted_lower_bound")
        assert_input(Y, T, X, p_t)
        pi = policy.prob(T, X)
        TX = torch.concat([T[:, None], X], dim=1)
        eta_cmc = self.eta_nn(TX) * pi / p_t
        dual = get_dual_objective(
            Y, p_t, pi, eta_cmc, self.eta_f, self.gamma, self.Gamma, self.const_type
        )
        return dual.mean()


def get_multipliers(
    input_dim: int, n_hidden: int = 64, n_layers: int = 3, seed: int = 0
) -> tuple[torch.nn.Module, torch.Tensor]:
    torch.manual_seed(seed)
    input_dims = [input_dim] + [n_hidden] * (n_layers - 1)
    output_dims = [n_hidden] * (n_layers - 1) + [1]
    layers = [
        layer
        for (d_in, d_out) in zip(input_dims, output_dims)
        for layer in (torch.nn.Linear(d_in, d_out), torch.nn.ReLU())
    ]
    eta_nn = torch.nn.Sequential(*layers[:-1])
    eta_nn[-1].weight.data.fill_(0.0)
    eta_nn[-1].bias.data.fill_(0.0)
    log_eta_f = torch.tensor(0.0, requires_grad=True)
    return eta_nn, log_eta_f
