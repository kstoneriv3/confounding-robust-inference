#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
plt.style.use("seaborn-whitegrid")
from matplotlib.cm import tab10
from tqdm.auto import tqdm

import kcmc
from kcmc.estimators import confounding_robust_estimator, hajek, ipw
from kcmc.experiment_policy_evaluation import run_policy_evaluation_experiment



MAX_GAMMA = {
    'KL': 0.02,
    'inverse KL': 0.04,
    'Jensen-Shannon': 0.03,
    'squared Hellinger': 0.01,
    'Pearson chi squared':0.06,
    'Neyman chi squared':0.03,
    'total variation': 0.015,
}

def main(F_DIVERGENCE):


    from kcmc.data_binary import evaluate_policy, generate_data, estimate_p_t


    # In[59]:


    beta_e_x = np.asarray([0, .75, -.5, 0, -1])

    def toy_policy(X, T):
        n = X.shape[0]
        T = torch.as_tensor(T)
        z = torch.as_tensor(X) @ torch.as_tensor(beta_e_x)
        e_x = torch.exp(z) / (1 + torch.exp(z))
        return (1. - T) * e_x + T * (1. - e_x)


    # In[60]:


    # Guessing kernel with approximate solution
    Y, T, X, _, _, _ = generate_data(1000)
    p_t = estimate_p_t(X, T)
    _, w_guess = confounding_robust_estimator(
        Y, T, X, p_t, toy_policy, gamma=MAX_GAMMA[F_DIVERGENCE] * 0.5,
        hajek_const=True, return_w=True, normalize_p_t=True,
        f_const=True, f_divergence=F_DIVERGENCE,
    )
    e_guess = p_t * w_guess - 1
    gp_kernel = kcmc.estimators.fit_gp_kernel(e_guess, T, X)
    kernel = gp_kernel.k2


    # In[61]:


    hajek(Y, T, X, p_t, toy_policy)


    # ### Experiment of changing sensitivity parameter $\gamma$ 

    # In[62]:


    def update_base_method(**new_params):
        ret = kcmc.experiment_policy_evaluation.EXAMPLE_PARAMS.copy()
        ret.update(new_params)
        return ret

    grid_methods = {
        'ZSB': update_base_method(
            hajek_const=True,
            normalize_p_t=True,
            f_const=True, 
            f_divergence=F_DIVERGENCE,
        ),
        'GP_KCMC': update_base_method(
            hajek_const=True,
            normalize_p_t=True,
            f_const=True, 
            f_divergence=F_DIVERGENCE,
            kernel=kernel,
            kernel_const=True,
            D=100,
        ),
        'hard_KCMC': update_base_method(
            hajek_const=True,
            normalize_p_t=True,
            f_const=True, 
            f_divergence=F_DIVERGENCE,
            kernel=kernel,
            kernel_const=True,
            hard_kernel_const=True,
            D=100,
        ),
        'quantile': update_base_method(
            hajek_const=True,
            normalize_p_t=True,
            f_const=True, 
            f_divergence=F_DIVERGENCE,
            quantile_const=True,
        ),
    }

    grid_gamma = [MAX_GAMMA[F_DIVERGENCE] * (0.01 + 0.1 * i) for i in range(11)]


    # In[ ]:


    log_file=f'logs/policy_evaluation_synthetic_binary_changing_gamma_{F_DIVERGENCE}.csv'

    pbar = tqdm(total=len(grid_methods) * len(grid_gamma))
    for method_name, params in grid_methods.items():
        for gamma in grid_gamma:
            params['gamma'] = gamma
            run_policy_evaluation_experiment(
                log_file, params, toy_policy, data_type='synthetic binary', 
                n_seeds=10, sample_size=500, log_info=method_name
            )
            pbar.update(1)
    pbar.close()


    # In[ ]:


    df = pd.read_csv(f'logs/policy_evaluation_synthetic_binary_changing_gamma_{F_DIVERGENCE}.csv')


    # In[ ]:


    df.head()


    # In[ ]:


    # due to the numerical rounding, some of the original values are invalid indices
    grid_gamma = df.gamma.unique()  


    # In[ ]:


    df_grouped = df.groupby(by=['log_info', 'gamma'])['lower_bound', 'upper_bound']
    values_mean = df_grouped.mean()
    values_std = df_grouped.std()


    # In[ ]:


    colors = {method: tab10((0.5 + i) / 10) for i, method in enumerate(grid_methods.keys())}
    legend_targets = []
    legend_tags = ["ZSB", "low-rank GP KCMC ($D=100$)", "hard KCMC ($D=100$)", "QB"]

    for method_name in grid_methods.keys():
        upper = np.array([values_mean.loc[(method_name, gamma)]['upper_bound'] for gamma in grid_gamma])
        lower = np.array([values_mean.loc[(method_name, gamma)]['lower_bound'] for gamma in grid_gamma])
        dupper = np.array([values_std.loc[(method_name, gamma)]['upper_bound'] for gamma in grid_gamma])
        dlower = np.array([values_std.loc[(method_name, gamma)]['lower_bound'] for gamma in grid_gamma])
        c = colors[method_name]
        upper_line = plt.plot(grid_gamma, upper, c=c)[0]
        lower_line = plt.plot(grid_gamma, lower, c=c)[0]
        upper_band = plt.fill_between(grid_gamma, upper + dupper, upper - dupper, color=c, alpha=0.1)
        lower_band = plt.fill_between(grid_gamma, lower + dlower, lower - dlower, color=c, alpha=0.1)
        legend_targets.append((upper_line, lower_line, upper_band, lower_band))

    plt.legend(legend_targets, legend_tags)
    plt.xlabel(r"Sensitivity parameter $\gamma$")
    plt.ylabel(r"Upper/lower bounds of policy value")
    plt.xlim([1.0, 2.0])
    plt.savefig(f'logs/policy_evaluation_synthetic_binary_changing_gamma_{F_DIVERGENCE}.pdf')


    # #### The interpretation of the above plot:
    # - Since the upper/lower bound's tightness is independent of the sample size by definition, the width of interval should not change significantly for different sample size
    # - However, the 

    # # Continuous Synthetic Data

    # In[ ]:


    from kcmc.data_continuous import evaluate_policy, generate_data, estimate_p_t


    # In[ ]:


    def wrap_continuous_policy(policy):
        def wrapped_policy(X, T=None, return_sample=False, requires_grad=False): 
            policy_dist = policy(X)
            if return_sample:
                return policy_dist.rsample() if requires_grad else policy_dist.sample()
            else:
                return torch.exp(policy_dist.log_prob(torch.as_tensor(T)))
        return wrapped_policy

    beta_e_x = np.asarray([0, .75, -.5, 0, -1])

    @wrap_continuous_policy
    def toy_policy(X):
        z = torch.as_tensor(X) @ torch.as_tensor(beta_e_x)
        mu_t = torch.exp(z) / (1 + torch.exp(z))
        a, b = 3 * mu_t + 1, 3 * (1 - mu_t) + 1
        return torch.distributions.beta.Beta(a, b)


    # In[ ]:


    beta_e_x = np.asarray([0, .75, -.5, 0, -1])

    def toy_policy(X, T):
        n = X.shape[0]
        T = torch.as_tensor(T)
        z = torch.as_tensor(X) @ torch.as_tensor(beta_e_x)
        e_x = torch.exp(z) / (1 + torch.exp(z))
        return (1. - T) * e_x + T * (1. - e_x)


    # ### Experiment of changing sensitivity parameter $\gamma$ 

    # In[ ]:


    def update_base_method(**new_params):
        ret = kcmc.experiment_policy_evaluation.EXAMPLE_PARAMS.copy()
        ret.update(new_params)
        return ret

    grid_methods = {
        'GP_KCMC': update_base_method(
            f_const=True, 
            f_divergence=F_DIVERGENCE,
            kernel=kernel,
            kernel_const=True,
            D=100,
        ),
        'hard_KCMC': update_base_method(
            f_const=True, 
            f_divergence=F_DIVERGENCE,
            kernel=kernel,
            kernel_const=True,
            hard_kernel_const=True,
            D=100,
        ),
        'quantile': update_base_method(
            f_const=True, 
            f_divergence=F_DIVERGENCE,
            quantile_const=True,
        ),
    }

    grid_gamma = [MAX_GAMMA[F_DIVERGENCE] * (0.01 + 0.1 * i) for i in range(11)]


    # In[ ]:


    log_file=f'logs/policy_evaluation_synthetic_continuous_changing_gamma_{F_DIVERGENCE}.csv'

    pbar = tqdm(total=len(grid_methods) * len(grid_gamma))
    for method_name, params in grid_methods.items():
        for gamma in grid_gamma:
            params['gamma'] = gamma
            run_policy_evaluation_experiment(
                log_file, params, toy_policy, data_type='synthetic continuous', 
                n_seeds=10, sample_size=500, log_info=method_name
            )
            pbar.update(1)
    pbar.close()


    # In[ ]:


    df = pd.read_csv(f'logs/policy_evaluation_synthetic_continuous_changing_gamma_{F_DIVERGENCE}.csv')


    # In[ ]:


    df.head()


    # In[ ]:


    # due to the numerical rounding, some of the original values are invalid indices
    grid_gamma = df.gamma.unique()  


    # In[ ]:


    df_grouped = df.groupby(by=['log_info', 'gamma'])['lower_bound', 'upper_bound']
    values_mean = df_grouped.mean()
    values_std = df_grouped.std()


    # In[ ]:


    colors = {method: tab10((0.5 + i) / 10) for i, method in enumerate(grid_methods.keys())}
    legend_targets = []
    legend_tags = ["ZSB", "low-rank GP KCMC ($D=100$)", "hard KCMC ($D=100$)", "QB"]

    for method_name in grid_methods.keys():
        upper = np.array([values_mean.loc[(method_name, gamma)]['upper_bound'] for gamma in grid_gamma])
        lower = np.array([values_mean.loc[(method_name, gamma)]['lower_bound'] for gamma in grid_gamma])
        dupper = np.array([values_std.loc[(method_name, gamma)]['upper_bound'] for gamma in grid_gamma])
        dlower = np.array([values_std.loc[(method_name, gamma)]['lower_bound'] for gamma in grid_gamma])
        c = colors[method_name]
        upper_line = plt.plot(grid_gamma, upper, c=c)[0]
        lower_line = plt.plot(grid_gamma, lower, c=c)[0]
        upper_band = plt.fill_between(grid_gamma, upper + dupper, upper - dupper, color=c, alpha=0.1)
        lower_band = plt.fill_between(grid_gamma, lower + dlower, lower - dlower, color=c, alpha=0.1)
        legend_targets.append((upper_line, lower_line, upper_band, lower_band))

    plt.legend(legend_targets, legend_tags)
    plt.xlabel(r"Sensitivity parameter $\gamma$")
    plt.ylabel(r"Upper/lower bounds of policy value")
    plt.xlim([1.0, 2.0])
    plt.savefig(f'logs/policy_evaluation_synthetic_continuous_changing_gamma_{F_DIVERGENCE}.pdf')



if __name__ == '__main__':
    import fire
    fire.Fire(main)


