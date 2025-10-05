

"""
Heston calibration from a (k, T) implied vol surface.

Inputs:
- k_grid: 1D array of log-moneyness k = ln(K / F(T))
- T_grid: 1D array of maturities (in years)
- iv_matrix: 2D array, shape (nT, nK), implied vol at (T_i, k_j)
- F_T: array-like or callable, forward price for each T_i (e.g., futures price)  --> already embeds dividends q
- DF_T: array-like or callable, discount factor for each T_i = exp(-r(T)*T)

Output:
- Calibrated Heston params: (kappa, theta, sigma, rho, v0)

"""

import numpy as np
from typing import Tuple
from scipy.optimize import minimize

from option_pricing.european import EuropeanOption


def softplus(x: np.ndarray) -> np.ndarray:
    """
    Softplus function to map R -> (0, inf)
    This is a smooth approximation to ReLU: log(1 + exp(x)) which is stable for large |x|. 
    Compared to simple bound-constrained optimization, this often helps with convergence in practice.
    """
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

def unpack_params(z: np.ndarray) -> Tuple[float,float,float,float,float]:
    """
    unpack unconstrained params to Heston params with constraints.
    Returns (kappa, theta, sigma, rho, v0)
    """
    a,b,c,d,e = z
    kappa = float(softplus(a))
    theta = float(softplus(b))
    sigma = float(softplus(c))
    v0    = float(softplus(d))
    rho   = float(np.tanh(e))     # (-1, 1)
    return kappa, theta, sigma, rho, v0


def _heston_phi(u: np.ndarray, T: float,
               kappa: float, theta: float, sigma: float, rho: float, v0: float) -> np.ndarray:
    iu = 1j*u
    d = np.sqrt((rho*sigma*iu - kappa)**2 + sigma**2*(iu + u*u))
    g = (kappa - rho*sigma*iu - d) / (kappa - rho*sigma*iu + d)
    g = np.where(np.abs(g) < 1e308, g, 1e308+0j)
    exp_dt = np.exp(-d*T)
    D = ((kappa - rho*sigma*iu - d)/(sigma**2)) * ((1 - exp_dt)/(1 - g*exp_dt))
    C = (kappa*theta/(sigma**2)) * ((kappa - rho*sigma*iu - d)*T - 2.0*np.log((1 - g*exp_dt)/(1 - g)))
    return np.exp(C + D*v0)

def heston_call_forward(k: float, T: float, 
                        params: Tuple[float,float,float,float,float],
                        U: float, du: float = 0.1) -> float:

    """
    Heston model European call price on forward, via characteristic function integration.

    Assume F = 1 and the price is not discounted. 
    To match the market price, should use the same assumption when calculating price from vol surface
    Args:
        k (float): log-moneyness k = ln(K / F)
        T (float): time to maturity
        params (tuple): Heston parameters (kappa, theta, sigma, rho, v
        U (float): upper limit of integration
        du (float): integration step size
    Returns:
        float: call price
    """

    kappa, theta, sigma, rho, v0 = params
    if T <= 0:
        return max(1 - np.exp(k), 0.0)
    u = np.arange(du, U+du, du, dtype=np.float64)
    phi1 = _heston_phi(u - 1j, T, kappa, theta, sigma, rho, v0)
    phi2 = _heston_phi(u, T, kappa, theta, sigma, rho, v0)
    integrand1 = np.real(np.exp(-1j*u*k) * phi1 / (1j*u))
    integrand2 = np.real(np.exp(-1j*u*k) * phi2 / (1j*u))
    P1 = 0.5 + (1/np.pi) * np.trapezoid(integrand1, u)
    P2 = 0.5 + (1/np.pi) * np.trapezoid(integrand2, u)
    price = P1 - np.exp(k) * P2
    return price    


def build_points_from_surface(k_grid: np.ndarray,
                              T_grid: np.ndarray,
                              iv_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turn a smooth vol surface into pseudo price samples for calibration.
    Args:
        k_grid (np.ndarray): 1D array of log-moneyness k = ln(K / F)
        T_grid (np.ndarray): 1D array of maturities (in years)
        iv_matrix (np.ndarray): 2D array, shape (nT, nK), implied vol at (T_i, k_j)
    """

    prices = np.zeros_like(iv_matrix)
    vegas = np.zeros_like(iv_matrix)
    for i, T in enumerate(T_grid):
        for j, k in enumerate(k_grid):
            sigma = iv_matrix[i, j]
            prices[i, j] = EuropeanOption.black_scholes(
                sigma=sigma,
                F=1.0, K=np.exp(k), T=T, r=0.0, option_type='C'
            )
            vegas[i, j] = EuropeanOption.vega(
                F=1.0, K=np.exp(k), T=T, sigma=sigma
            )
    return prices, vegas



def loss_function(
                  k_grid: np.ndarray,
              T_grid: np.ndarray,
              call_prices: np.ndarray,
              vegas: np.ndarray,
              vega_floor: float = 1e-4,
              lambda_feller: float = 1e3,
              U: float = 240.0,
              du: float = 0.4):

    """
    objective function for Heston calibration: vega-weighted least squares on call prices
    with penalty for Feller condition violation and rho bounds.

    """
    def loss_(z):
        kappa, theta, sigma, rho, v0 = unpack_params(z)
        penalty = 0.0
        # Feller condition penalty
        feller_violation = sigma*sigma - 2.0*kappa*theta
        if feller_violation > 0:
            penalty += lambda_feller * (feller_violation**2)
        # rho bounds penalty
        if abs(rho) > 0.995:
            penalty += 1e3 * (abs(rho) - 0.995)**2
        params = (kappa, theta, sigma, rho, v0)
        model_prices = np.zeros_like(call_prices)
        w = np.zeros_like(call_prices)
        for i, T in enumerate(T_grid):
            for j, k in enumerate(k_grid):
                model_prices[i, j] = heston_call_forward(k, T, params, U=U, du=du)
                w[i, j] = 1.0 / max(vegas[i, j], vega_floor)
        
        sse = np.sum(np.sum(w * (model_prices - call_prices)**2))
        return sse + penalty
    return loss_

def calibrate_heston_from_surface(k_grid: np.ndarray,
                                  T_grid: np.ndarray,
                                  iv_matrix: np.ndarray,
                                  method: str = "L-BFGS-B",
                                  maxiter: int = 2000,
                                  U: float = 240.0,
                                  du: float = 0.4):
    """
    Calibrate Heston model parameters from a (k, T) implied vol surface.

    """
    call_prices, vegas = build_points_from_surface(k_grid, T_grid, iv_matrix)

    # Initial guess for Heston params
    iv_guess = np.median(iv_matrix) if iv_matrix.size > 0 else 0.2
    v0_init = max(1e-4, iv_guess**2)

    def inv_softplus(y: float) -> float:
        y = max(y, 1e-12)
        return np.log(np.expm1(y))

    x0 = np.array([
        inv_softplus(1.5),         # kappa ~ 1.5
        inv_softplus(v0_init),     # theta ~ v0_init
        inv_softplus(0.5),         # sigma ~ 0.5
            inv_softplus(v0_init),     # v0 ~ v0_init
            np.arctanh(np.clip(-0.5, -0.99, 0.99))  # rho ~ -0.5
        ], dtype=float)

    loss = loss_function(k_grid, T_grid, call_prices, vegas, U=U, du=du)
    res  = minimize(loss, x0, method=method, options={"maxiter": maxiter, "ftol": 1e-9})
    params = unpack_params(res.x)
    return params, res



