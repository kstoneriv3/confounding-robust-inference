import numpy as np
import torch
from sklearn.linear_model import QuantileRegressor

from confounding_robust_inference.utils.quantile_regression import TorchQuantileRegressor
from confounding_robust_inference.utils.types import as_tensor


def test_fast_quantile_regressor() -> None:
    np.random.seed(0)
    n = 30
    d = 5
    X = np.random.randn(n * d).reshape(n, d)
    beta = np.random.randn(d)
    Y = X @ beta + np.random.randn(n)
    tX = as_tensor(X)
    tY = as_tensor(Y)

    qr = QuantileRegressor(quantile=0.4, alpha=0.0, fit_intercept=False, solver="highs").fit(X, Y)
    fqr = TorchQuantileRegressor(quantile=0.4).fit(tX, tY)
    assert torch.allclose(as_tensor(qr.coef_), fqr.coef_, atol=0.1)
    Y_hat = qr.predict(X)
    tY_hat = fqr.predict(tX)
    assert torch.allclose(as_tensor(Y_hat), tY_hat, atol=0.2)
