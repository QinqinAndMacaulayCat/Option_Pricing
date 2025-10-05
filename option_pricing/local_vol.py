"""
This module implements the Local Volatility Model for option pricing.
"""
import numpy as np

def local_volatilities(log_moneyness: np.ndarray,
                              maturities: np.ndarray,
                              W: np.ndarray) -> np.ndarray:
    """
    Compute local volatilities from implied volatilities using Dupire's formula.

    Args:
        log_moneyness (np.ndarray): Array of log moneyness values.
        maturities (np.ndarray): Array of maturities in years.
        W (np.ndarray): 2D array of total implied variances (implied vol^2 * T). Shape: (len(maturities), len(log_moneyness))

    Returns:
        np.ndarray: 2D array of local volatilities.
    """

    dw_dk = np.gradient(W, log_moneyness, axis=1)
    dw_dT = np.gradient(W, maturities, axis=0)
    d2w_dk2 = np.gradient(dw_dk, log_moneyness, axis=1)
    W = np.clip(W, 1e-6, None)  # Prevent division by zero or negative variances
    denom = 1 - (log_moneyness[None, :] / W) * dw_dk + 0.25 * (-0.25 - 1 / W + log_moneyness[None, :]**2 / W**2) * dw_dk**2 + 0.5 * d2w_dk2
    local_var = dw_dT / denom

    return np.sqrt(np.clip(local_var, 1e-6, None))

