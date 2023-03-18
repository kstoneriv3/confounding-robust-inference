import warnings

import numpy as np
import torch
from scipy.stats import norm
from sklearn.decomposition import KernelPCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from statsmodels.othermod.betareg import BetaModel

from cri.estimators.misc import DEFAULT_KERNEL
from cri.utils.types import as_ndarrays, as_tensors


def estimate_p_t_binary(T: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Estimate nominal propensity p_obs(t|x) from data when T = 0 or 1."""
    assert len(T.shape) == 1 and len(X.shape) == 2
    X_np, T_np = as_ndarrays(X, T)
    model = LogisticRegressionCV().fit(X_np, T_np)
    p_t_np = model.predict_proba(X_np)[range(T.shape[0]), T_np]
    (p_t,) = as_tensors(p_t_np)
    return p_t


def estimate_p_t_bounded_continuous(T: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Estimate nominal propensity p_obs(t|x) from data when 0 <= T <= 1."""
    assert len(T.shape) == 1 and len(X.shape) == 2
    X_np, T_np = as_ndarrays(X, T)
    X_np = StandardScaler().fit_transform(X_np)
    # Z: additional non-linear feature
    Z_np = KernelPCA(n_components=2, kernel="rbf", gamma=0.01).fit_transform(X_np)
    # catch warning to avoid user warning on multiplication operator with `*` and `@`
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = BetaModel(endog=T_np, exog=np.concatenate([Z_np, X_np], axis=1))
        params = model.fit().params
    p_t_np = np.exp(model.loglikeobs(params))
    (p_t,) = as_tensors(p_t_np)
    return p_t


def estimate_p_t_continuous(T: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Estimate nominal propensity p_obs(t|x) from data when T is continuous."""
    assert len(T.shape) == 1 and len(X.shape) == 2
    X_np, T_np = as_ndarrays(X, T)
    X_np = StandardScaler().fit_transform(X_np)
    kernel = WhiteKernel() + DEFAULT_KERNEL
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    model.fit(X_np[:1000], T_np[:1000])  # GP is slow for large sample size so truncate at n = 1000.
    t_mean, t_std = model.predict(X_np, return_std=True)
    p_t_np = norm.pdf(T_np, loc=t_mean, scale=t_std)
    (p_t,) = as_tensors(p_t_np)
    return p_t
