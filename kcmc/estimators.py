import warnings
import numpy as np
import cvxpy as cp
import torch
from scipy.linalg import eigh
from scipy.stats import chi2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, RBF
from sklearn.linear_model import QuantileRegressor


f_divergences = [
    'KL', 'inverse KL', 'Jensen-Shannon', 'squared Hellinger',
    'Pearson chi squared', 'Neyman chi squared', 'total variation'
]

def ipw(Y, T, X, p_t, policy):
    n = p_t.shape[0]
    r = Y * policy(X)[range(n), T]
    est = torch.mean(r / torch.as_tensor(p_t))
    return est

def hajek(Y, T, X, p_t, policy, return_w=False):
    n = p_t.shape[0]
    r = Y * policy(X)[range(n), T]
    p_t_new = np.empty_like(p_t)
    for t in set(T):
        p_t_new[T==t] = p_t[T==t] * np.mean((T==t) / p_t)
    est = torch.mean(r / torch.as_tensor(p_t_new))
    return est

def confounding_robust_estimator(
    Y, T, X, p_t, policy,
    D=200,
    lambd=1.5, 
    gamma=0.5,
    alpha=0.05,
    kernel=RBF(),
    sigma2=1.0,
    hard_kernel_const=False,
    f_divergence='KL', 
    hajek_const=False,
    kernel_const=False,
    quantile_const=False,
    tan_box_const=False,
    lr_box_const=False,
    f_const=False,
    return_w=False,
):
    n = T.shape[0]
    pi = policy(X)[range(n), T] 
    r = Y * pi
    Y_np, r_np, pi_np = map(lambda tensor: tensor.data.numpy(), (Y, r, pi))
    
    p_t_original = p_t
    p_t = normalize_p_t(p_t, T)  # normalization for simply guaranteeing the feasibility for Hajek constraints
        
    with warnings.catch_warnings():  # to avoid user warning about multiplication operator with `*` and `@`
        warnings.simplefilter("ignore")
        w = cp.Variable(n)
        constraints = [np.zeros(n) <= w]
        if hajek_const:
            constraints.extend(get_hajek_constraint(w, T, p_t))
        if kernel_const:
            constraints.extend(get_kernel_constraint(w, T, X, p_t, alpha, sigma2, kernel, D, hard_kernel_const))
        if quantile_const:
            assert f_const == False, "quantile constraint is only for box constraints"
            constraints.extend(get_quantile_constraint(w, Y_np, pi_np, T, X, p_t, lambd))
        if tan_box_const:
            constraints.extend(get_tan_box_constraint(w, p_t, p_t_original, lambd))
        if lr_box_const:
            constraints.extend(get_likelihood_ratio_box_constraint(w, p_t, lambd))
        if f_const:
            assert f_divergence in f_divergences, f"Supported f-divergences are {f_divergences}."
            constraints.extend(get_f_constraint(w, p_t, gamma, f_divergence))
            constraints.append(cp.sum(-cp.log(w * p_t)) <= 0.1)
        objective = cp.Minimize(cp.sum(r_np * w))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK if f_const else cp.ECOS)

    if problem.status != 'optimal':
        raise ValueError(f"The optimizer found the associated convex programming to be {problem.status}.")

    w = w.value
    est = torch.mean(torch.as_tensor(w) * r)
    return (est, w) if return_w else est

def normalize_p_t(p_t, T):
    p_t_new = np.empty_like(p_t)
    for t in set(T):
        p_t_new[T==t] = p_t[T==t] * np.mean((T==t) / p_t)
    return p_t_new

def fit_gp_kernel(Y, T, X):
    TX = np.concatenate([T[:, None], X], axis=1)
    TX /= TX.std(axis=0)[None, :]
    kernel = WhiteKernel() + ConstantKernel() * RBF()
    model = GaussianProcessRegressor(kernel=kernel).fit(TX[:1000], Y[:1000])
    return model.kernel_

def get_hajek_constraint(w, T, p_t):
    n = T.shape[0]
    constraints = []
    for t in set(T):
        constraints.append(cp.sum(w[T==t]) == n)
    return constraints

def get_kernel_constraint(w, T, X, p_t, alpha, sigma2, kernel, D, hard_kernel_const):
    n = T.shape[0]
    D = min(D, n)
    M, u, Cov_z, D = get_gpqc(T, X, p_t, D, sigma2, kernel)
    chi2_bound = chi2(df=D).ppf(1 - alpha)
    
    z = cp.Variable(D)
    if hard_kernel_const:
        constraints = [np.zeros(D) ==  M @ w - u]
    else:
        constraints = [
            z == M @ w - u,
            cp.sum(z ** 2 / Cov_z) <= chi2_bound,
        ]
    return constraints

def get_gpqc(T, X, p_t, D, sigma2, kernel):
    n = T.shape[0]
    TX = np.concatenate([T[:, None], X], axis=1)
    TX /= TX.std(axis=0)[None, :]
    K = kernel(TX, TX)
    S, V = eigh(K, subset_by_index=[n - D, n-1])
    S, V = cutoff_neg_eigvals(S, V)
    
    M = np.diag(S / (S + sigma2)) @ V.T @ np.diag(p_t)
    u = np.diag(S / (S + sigma2)) @ V.T @ np.ones(n)
    Cov_z = S - S ** 2 / (S + sigma2)

    return M, u, Cov_z, u.shape[0]

def cutoff_neg_eigvals(S, V):
    V = V[:, S > 1e-6]
    S = S[S > 1e-6]
    return S, V

def get_quantile_constraint(w, Y, pi, T, X, p_t, lambd):
    USE_KERNEL = False
    n = T.shape[0]
    TX = np.concatenate([T[:, None], X], axis=1)
    TX /= TX.std(axis=0)[None, :]
    if USE_KERNEL:
        kernel = fit_gp_kernel(Y, T, X)
        TX = KernelPCA(30).fit_transform(TX)
    Q = QuantileRegressor(quantile=1. / (lambd + 1), alpha=0.).fit(TX, Y).predict(TX) # any regressor will do, 
    ### Carveat: np.ones(n) * w is NOT the element-wise product in cvxpy!!!
    return [cp.scalar_product(pi * Q, w) == np.sum(pi * Q / p_t)]

def get_tan_box_constraint(w, p_t, p_t_original, lambd):
    # p_t does not always satisfy p_t < 1, therefore, we use p_t_original
    # to construct box constraint for w_original and rescale it for w.
    a_original = 1 + 1 / lambd * (1 / p_t_original - 1)
    b_original = 1 + lambd * (1 / p_t_original - 1)
    a = a_original * p_t_original / p_t
    b = b_original * p_t_original / p_t
    return [a <= w, w <= b]

def get_likelihood_ratio_box_constraint(w, p_t, lambd):
    a = 1 / (lambd * p_t)
    b = lambd / p_t
    return [a <= w, w <= b]

def get_f_constraint(w, p_t, gamma, f):
    EPS = 1e-4
    n = p_t.shape[0]
    f = {
        'KL': lambda u: - cp.entr(u),
        'inverse KL': lambda u: - cp.log(u),
        'Jensen-Shannon': lambda u: - 0.5 * (cp.entr(u) + cp.log(u)),
        'squared Hellinger': lambda u: u - 2 * cp.sqrt(u) + 1,
        'Pearson chi squared': lambda u: cp.square(u) - 1,
        'Neyman chi squared': lambda u: cp.inv_pos(u) - 1,
        'total variation': lambda u: 0.5 * cp.abs(u - 1),
    }[f]
    ### Carveat: np.ones(n) * w is NOT the element-wise product in cvxpy!!!
    constraints = [
        cp.sum(f(cp.multiply(w, p_t))) <= gamma * n,
        #cp.sum(-cp.log(cp.multiply(w, p_t))) <= 0.1,
        cp.scalar_product(w, p_t) == n,
        #EPS * np.ones(n) <= w,
    ]
    return constraints
