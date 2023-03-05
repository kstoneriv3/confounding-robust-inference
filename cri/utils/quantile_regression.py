import torch


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
        >>> from cri.utils.quantile_regression import TorchQuantileRegressor
        >>> qr = QuantileRegressor(
        ...     quantile=0.4, alpha=0., fit_intercept=False, solver='highs'
        ... ).fit(X, Y)
        >>> fqr = TorchQuantileRegressor(quantile=0.4).fit(tX, tY)
        >>> assert torch.allclose(torch.as_tensor(qr.coef_), fqr.coef_, atol=0.1)
    """

    def __init__(self, quantile: float) -> None:
        self.quantile = quantile

    def fit(self, X: torch.Tensor, Y: torch.Tensor, n_iter: int = 200) -> "TorchQuantileRegressor":
        Y_scale = Y.std()
        Y = Y / Y_scale
        L = max(self.quantile, 1 - self.quantile) * torch.linalg.norm(X, axis=1).mean()
        beta = torch.zeros(X.shape[1], requires_grad=True, dtype=float)  # type: ignore
        beta.data[-1] = Y.mean()
        optim = torch.optim.SGD([beta], lr=0.05 / L)
        for i in range(n_iter):
            loss = self.quantile_loss(Y - X @ beta, self.quantile).mean()
            loss.backward()  # type: ignore
            optim.step()
            optim.zero_grad()
        self.coef_ = beta.data * Y_scale
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.coef_

    @staticmethod
    def quantile_loss(error: torch.Tensor, q: float) -> torch.Tensor:
        loss = q * torch.relu(error) + (1 - q) * torch.relu(-error)
        return loss
