import torch
import torchmin


class TorchQuantileRegressor:
    """Quantile regression model that uses gradient descent.

    This is a quantile regression model with fit and predict methods. This is used as a faster
    alternative to sklearn.linear_model.QuantileRegressor, which relies on slow (but more accurate,
    of course!) linear programming. The speed up becomes significant for n >= 10000, d >= 20.

    Args:
        quantile: quantile of the regressor

    Example:
        The basic interface is same as scikit-learn's QuantileRegressor.

        >>> import numpy as np
        >>> np.random.seed(0)
        >>> import torch
        >>>
        >>> n = 30
        >>> d = 5
        >>> X = np.random.randn(n * d).reshape(n, d)
        >>> beta = np.random.randn(d)
        >>> Y = X @ beta + np.random.randn(n)
        >>> tX = torch.as_tensor(X)
        >>> tY = torch.as_tensor(Y)
        >>>
        >>> from sklearn.linear_model import QuantileRegressor
        >>> from confounding_robust_inference.utils.quantile_regression import (
        ...     TorchQuantileRegressor
        ... )
        >>> qr = QuantileRegressor(
        ...     quantile=0.4, alpha=0., fit_intercept=False, solver='highs'
        ... ).fit(X, Y)
        >>> tqr = TorchQuantileRegressor(quantile=0.4).fit(tX, tY)
        >>> assert torch.allclose(torch.as_tensor(qr.coef_), tqr.coef_, atol=0.1)
    """

    def __init__(self, quantile: float) -> None:
        self.quantile = quantile

    def fit(self, X: torch.Tensor, Y: torch.Tensor, n_iter: int = 200) -> "TorchQuantileRegressor":
        beta_init = torch.zeros(X.shape[1], dtype=Y.dtype)
        beta_init[-1] = Y.mean()
        res = torchmin.bfgs._minimize_bfgs(
            lambda beta: self.quantile_loss(Y - X @ beta, self.quantile).mean(),
            beta_init,
            lr=0.5,
            max_iter=200,
        )
        self.coef_ = res.x
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.coef_

    @staticmethod
    def quantile_loss(error: torch.Tensor, q: float) -> torch.Tensor:
        loss = q * torch.relu(error) + (1 - q) * torch.relu(-error)
        return loss
