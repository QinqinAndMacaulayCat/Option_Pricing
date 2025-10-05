
"""
This module contains functions to compute the implied volatility of options using the Black-Scholes model.
"""
from scipy.optimize import brentq
from option_pricing.european import EuropeanOption


def implied_volatility(market_price: float,
                       F: float, 
                       K: float,
                       T: float,
                       r: float,
                       option_type: str = 'C',
                      ) -> float:
    """
    Implied volatility using Newton-Raphson method

    Args:
        market_price (float): market price of the option
        F (float): forward price
        K (float): strike price
        T (float): time to maturity in years
        r (float): risk-free interest rate
        option_type (str, optional): option type. Defaults to 'C'.
        max_iterations (int, optional): maximum number of iterations. Defaults to 100.

    Returns:
        float: implied volatility
    """
    
    def objective_function(sigma):
        price = EuropeanOption.black_scholes(sigma, F, K, T, r, option_type)
        return price - market_price

    try: 
        root, result = brentq(objective_function, -5, 10, full_output=True) # In case of failure to converge
        if not result.converged:
            print("Warning: Implied volatility calculation did not converge for market price:", market_price, 'F:', F, 'K:', K, 'T:', T, 'r:', r, 'option_type:', option_type)
        return max(root, 1e-6)  # Ensure non-negative volatility
    except ValueError:
        print("Warning: Implied volatility calculation failed for market price:", market_price, 'F:', F, 'K:', K, 'T:', T, 'r:', r, 'option_type:', option_type)
        return 1e-6  # Return a small positive value if no solution is found

