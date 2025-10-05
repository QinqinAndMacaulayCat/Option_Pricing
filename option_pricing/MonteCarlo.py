"""
Option pricing models using monte carlo simulation and Heston model.
"""
import numpy as np

def MonteCarlowithHeston(S0: float,
                         v0: float,
                         kappa: float,
                         theta: float,
                         sigma: float,
                         rho: float,
                         r: float,
                         T: float,
                         n_steps: int,
                         n_paths: int) -> np.ndarray:
    """
    Simulate asset price paths using the Heston model and Monte Carlo simulation.

    Args:
        S0 (float): initial asset price
        v0 (float): initial variance
        kappa (float): rate of mean reversion of variance
        theta (float): long-term variance
        sigma (float): volatility of variance
        rho (float): correlation between asset and variance
        r (float): risk-free interest rate
        T (float): time to maturity
        n_steps (int): number of time steps
        n_paths (int): number of simulation paths
    Returns:
        np.ndarray: simulated asset price paths of shape (n_paths, n_steps + 1)
    """
    dt = T / n_steps
    S = np.zeros((n_paths, n_steps + 1))
    v = np.zeros((n_paths, n_steps + 1))

    S[:, 0] = S0
    v[:, 0] = v0

    z1 = np.random.normal(size=(n_paths // 2 + 1, n_steps))
    z1 = np.concatenate((z1, -z1), axis=0)[:n_paths, :]
    z2 = np.random.normal(size=(n_paths // 2 + 1, n_steps))
    z2 = np.concatenate((z2, -z2), axis=0)[:n_paths, :]
    w1 = z1
    w2 = rho * z1 + np.sqrt(1 - rho ** 2) * z2
    for i in range(1, n_steps + 1):
        v[:, i] = v[:, i - 1] + kappa * (theta - v[:, i - 1]) * dt + sigma * np.sqrt(v[:, i - 1] * dt) * w2[:, i - 1]
        v[:, i] = np.maximum(v[:, i], 0)
        S[:, i] = S[:, i - 1] * np.exp((r - 0.5 * v[:, i - 1]) * dt + np.sqrt(v[:, i - 1] * dt) * w1[:, i - 1])
    return S



