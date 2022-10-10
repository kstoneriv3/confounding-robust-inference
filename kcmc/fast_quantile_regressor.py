import numpy as np
import torch

class QuantileRegressor:
    def __init__(self, quantile, alpha=None):
        self.quantile = quantile
    
    def fit(self, X, Y, n_iter=200):
        self.dim = X.shape[1]
        self.beta = torch.zeros(self.dim + 1, requires_grad=True, dtype=float)
        Y = torch.as_tensor(Y)
        Y_scale = Y.std()
        Y = torch.as_tensor(Y) / Y_scale
        X = torch.as_tensor(np.concatenate([X, np.ones((X.shape[0], 1))], axis=1))
        L = max(self.quantile, 1 - self.quantile) * torch.linalg.norm(X, axis=1).mean()
        optim = torch.optim.SGD([self.beta], lr=0.1 / L)
        for i in range(n_iter):
            loss = self.quantile_loss(Y - X @ self.beta, self.quantile).mean()
            loss.backward()
            optim.step()
            optim.zero_grad()
        self.beta.data = self.beta.data * Y_scale
        return self
            
    def predict(self, X):
        X = torch.as_tensor(np.concatenate([X, np.ones((X.shape[0], 1))], axis=1))
        return (X @ self.beta).data.numpy()
        
    @staticmethod
    def quantile_loss(error, q):
        loss = q * torch.relu(error) + (1 - q) * torch.relu(-error)
        return loss

