import numpy as np
from scipy.stats import beta

from sklearn.decomposition import KernelPCA
from statsmodels.othermod.betareg import BetaModel
import torch



beta0 = 2.5
beta0_t = -2
beta_x = np.asarray([0, .5, -0.5, 0, 0])
beta_x_t = np.asarray([-1.5, 1, -1.5, 1., 0.5])
beta_xi = 1
beta_xi_t = -2
beta_e_x = np.asarray([0, .75, -.5, 0, -1])
mu_x = np.asarray([-1, .5, -1, 0, -1]);

def generate_data(n):
    xi = (np.random.rand(n) > 0.5).astype(int)
    X = mu_x[None, :] + np.random.randn(n * 5).reshape(n, 5)
    eps = [np.random.randn(n) for t in (0, 1)]
    Y = np.array([
        X @ (beta_x + beta_x_t * t) + (beta_xi + beta_xi_t * t) * xi + (beta0 + beta0_t * t) + eps[t]
        for t in (0, 1)
    ])
    U = (Y[0, :] > Y[1, :]).astype(int)
    z = X @ beta_e_x
    e_x = np.exp(z) / (1 + np.exp(z))
    mu_xu = (6 * e_x) / (4 + 5 * U + e_x * (2 - 5 * U))
    mu_xu = np.clip(mu_xu, 1e-6, 1 - 1e-6)
    #unif = np.random.rand(n)
    a, b = 4 * mu_xu + 1, 4 * (1 - mu_xu) + 1
    T = beta.rvs(a, b)
    e_xu = beta.pdf(T, a, b)
    Y = (1 - T) * Y[0, :] + T * Y[1, :]
    Y = torch.as_tensor(Y)
    return Y, T, X, U, e_x, e_xu

def evaluate_policy(policy, n=1000, requires_grad=False):
    xi = (np.random.rand(n) > 0.5).astype(int)
    X = mu_x[None, :] + np.random.randn(n * 5).reshape(n, 5)
    eps = [np.random.randn(n) for t in (0, 1)]
    Y = np.array([
        X @ (beta_x + beta_x_t * t) + (beta_xi + beta_xi_t * t) * xi + (beta0 + beta0_t * t) + eps[t]
        for t in (0, 1)
    ])
    T = policy(X, return_sample=True, requires_grad=requires_grad)
    Y = torch.as_tensor(Y)
    Y = (1 - T) * Y[0, :] + T * Y[1, :]
    return Y.mean()

def estimate_p_t(X, T):
    Z = KernelPCA(n_components=2, kernel='rbf', gamma=0.01).fit_transform(X)
    model = BetaModel(endog=T, exog=np.concatenate([Z, X], axis=1))
    params = model.fit().params
    p_t = np.exp(model.loglikeobs(params))
    return p_t
