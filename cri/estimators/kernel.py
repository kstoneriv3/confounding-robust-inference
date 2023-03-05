import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Kernel, WhiteKernel


def select_kernel(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
) -> Kernel:
    """Learn the kernel parameters by fitting Gaussian process regression of Y on (T, X)."""
    TX = np.concatenate([T[:, None], X], axis=1)
    TX /= TX.std(axis=0)[None, :]
    kernel = WhiteKernel() + ConstantKernel() * RBF()
    # As GP gets slower for large sample size, we truncate data by n=1000.
    model = GaussianProcessRegressor(kernel=kernel).fit(TX[:1000], Y[:1000])
    return model.kernel_
