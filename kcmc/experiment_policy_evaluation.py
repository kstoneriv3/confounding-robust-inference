import csv 
import os.path
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from kcmc.estimators import confounding_robust_estimator


EXAMPLE_PARAMS = {
    'D': 200,
    'Gamma': 1.5, 
    'gamma': 0.01,
    'alpha': 0.05,
    'sigma2': 0.01,
    'kernel': RBF(),
    'hard_kernel_const': False,
    'rescale_kernel': False,
    'normalize_p_t': False,
    'f_divergence': 'total variation', 
    'hajek_const': False,
    'kernel_const': False,
    'quantile_const': False,
    'regressor_const': False,
    'tan_box_const': False,
    'lr_box_const': False,
    'f_const': False,
}

def run_policy_evaluation_experiment(
    log_file, params, policy, data_type='synthetic binary', sample_size=1000, n_seeds=1, seed0=0, log_info="",
):
    assert data_type in ['synthetic binary', 'synthetic continuous', 'real binary']
    assert set(params.keys()) == set(EXAMPLE_PARAMS.keys())
    for seed in range(seed0, seed0 + n_seeds):
        Y, T, X, p_t = get_data(data_type, sample_size, seed)
        lower_bound = confounding_robust_estimator(Y, T, X, p_t, policy, **params).data.numpy()
        try:
            lower_bound = confounding_robust_estimator(Y, T, X, p_t, policy, **params).data.numpy()
            upper_bound = -confounding_robust_estimator(-Y, T, X, p_t, policy, **params).data.numpy()
        except:
            print(f"Encountered error for data_type={data_type}, sample_size={sample_size}, params={params}. Skipping the experiment.")
            continue
        log_csv(log_file, data_type, policy.__name__, lower_bound, upper_bound, params, min(sample_size, T.shape[0]), seed, log_info)

def get_data(data_type, sample_size, seed):
    if 'synthetic' in data_type:
        if 'binary' in data_type:
            from kcmc.data_binary import generate_data, evaluate_policy, estimate_p_t
        elif 'continuous' in data_type:
            from kcmc.data_continuous import generate_data, evaluate_policy, estimate_p_t
        else:
            raise ValueError
        np.random.seed(seed)
        Y, T, X, U, e_x, e_xu = generate_data(sample_size)
        p_t = estimate_p_t(X, T)
    elif 'real' in data_type:
        from kcmc.data_real import generate_data, estimate_p_t
        Y, T, X = generate_data()
        p_t = estimate_p_t(X, T)
    return Y, T, X, p_t

def log_csv(log_file, data_type, policy_name, lower_bound, upper_bound, params, sample_size, seed, log_info):
    if not os.path.exists(log_file):
        # make a column name
        columns = ['log_info', 'data_type', 'policy_name', 'sample_size', 'seed', 'lower_bound', 'upper_bound', *params.keys()]
        with open(log_file, 'a') as file:
            writer = csv.writer(file)
            writer.writerow(columns)  
    # log data by appending to the csv file
    fields=[log_info, data_type, policy_name, sample_size, seed, lower_bound, upper_bound, *params.values()]
    with open(log_file, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(fields)
