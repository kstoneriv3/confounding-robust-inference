import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV



### Synthetic Data from Kallus and Zhou 2018, 2021

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
    e_xu = (6 * e_x) / (4 + 5 * U + e_x * (2 - 5 * U))
    T = (np.random.rand(n) < e_xu).astype(int)
    Y = Y[T, range(n)]
    e_x = e_x * T + (1 - e_x) * (1 - T)
    e_xu = e_xu * T + (1 - e_xu) * (1 - T)
    return torch.tensor(Y), T, X, U, e_x, e_xu

def evaluate_policy(policy, n=1000):
    xi = (np.random.rand(n) > 0.5).astype(int)
    X = mu_x[None, :] + np.random.randn(n * 5).reshape(n, 5)
    eps = [np.random.randn(n) for t in (0, 1)]
    Y = np.array([
        X @ (beta_x + beta_x_t * t) + (beta_xi + beta_xi_t * t) * xi + (beta0 + beta0_t * t) + eps[t]
        for t in (0, 1)
    ])
    Y = torch.as_tensor(Y)
    pi = policy(X, torch.zeros(n))
    Y = Y[0] * pi + Y[1] * (1 - pi)
    return Y.mean()

def estimate_p_t(X, T):
    model = LogisticRegressionCV().fit(X, T)
    p_t = model.predict_proba(X)[range(T.shape[0]), T]
    return p_t
