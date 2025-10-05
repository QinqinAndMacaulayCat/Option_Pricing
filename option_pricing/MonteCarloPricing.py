"""
This module contains functions for pricing exotic options.
"""

import numpy as np

def price_european(S_paths: np.ndarray,
                   K: float,
                   r: float,
                   T: float,
                   option_type: str = 'C') -> float:
    """
    Price a European option using Monte Carlo simulated asset paths.

    Args:
        S_paths (np.ndarray): simulated asset price paths of shape (n_paths, )
        K (float): strike price
        r (float): risk-free interest rate
        T (float): time to maturity
        option_type (str): 'C' for call, 'P' for put
    Returns:
        float: estimated option price
    """
    if option_type not in ['C', 'P']:
        raise ValueError("option_type must be 'C' for call or 'P' for put")
    
    if option_type == 'C':
        payoffs = np.maximum(S_paths - K, 0)
    else:
        payoffs = np.maximum(K - S_paths, 0)

    discounted_payoff = np.exp(-r * T) * payoffs
    return np.mean(discounted_payoff)

def price_asian(S_paths: np.ndarray,
                K: float,
                r: float,
                T: float,
                option_type: str = 'C') -> float:
    """
    Price an Asian option using Monte Carlo simulated asset paths.

    Args:
        S_paths (np.ndarray): simulated asset price paths of shape (n_paths, n_steps + 1)
        K (float): strike price
        r (float): risk-free interest rate
        T (float): time to maturity
        option_type (str): 'C' for call, 'P' for put
    Returns:
        float: estimated option price
    """
    if option_type not in ['C', 'P']:
        raise ValueError("option_type must be 'C' for call or 'P' for put")
    
    average_prices = np.mean(S_paths, axis=1)
    
    if option_type == 'C':
        payoffs = np.maximum(average_prices - K, 0)
    else:
        payoffs = np.maximum(K - average_prices, 0)

    discounted_payoff = np.exp(-r * T) * payoffs
    return np.mean(discounted_payoff)

def price_barrier(S_paths: np.ndarray,
                    K: float,
                    r: float,
                    T: float,
                    barrier: float,
                    barrier_type: str = 'up-and-out',
                    option_type: str = 'C') -> float:
        """
        Price a Barrier option using Monte Carlo simulated asset paths.
    
        Args:
            S_paths (np.ndarray): simulated asset price paths of shape (n_paths, n_steps + 1)
            K (float): strike price
            r (float): risk-free interest rate
            T (float): time to maturity
            barrier (float): barrier level
            barrier_type (str): 'up-and-out', 'down-and-out', 'up-and-in', 'down-and-in'
            option_type (str): 'C' for call, 'P' for put
        Returns:
            float: estimated option price
        """
        if option_type not in ['C', 'P']:
            raise ValueError("option_type must be 'C' for call or 'P' for put")
        
        if barrier_type not in ['up-and-out', 'down-and-out', 'up-and-in', 'down-and-in']:
            raise ValueError("barrier_type must be one of 'up-and-out', 'down-and-out', 'up-and-in', 'down-and-in'")
        
    
        if barrier_type in ['up-and-out', 'up-and-in']:
            breached = np.any(S_paths >= barrier, axis=1)
        else:
            breached = np.any(S_paths <= barrier, axis=1)
    
        if barrier_type in ['up-and-out', 'down-and-out']:
            valid_paths = ~breached
        else:
            valid_paths = breached
    
        final_prices = S_paths[:, -1]
        
        if option_type == 'C':
            payoffs = np.maximum(final_prices - K, 0)
        else:
            payoffs = np.maximum(K - final_prices, 0)
    
        payoffs *= valid_paths
        discounted_payoff = np.exp(-r * T) * payoffs
        return np.mean(discounted_payoff)

def price_lookback(S_paths: np.ndarray,
                     r: float,
                     T: float,
                     option_type: str = 'C') -> float:
     """
     Price a Lookback option using Monte Carlo simulated asset paths.
    
     Args:
          S_paths (np.ndarray): simulated asset price paths of shape (n_paths, n_steps + 1)
          r (float): risk-free interest rate
          T (float): time to maturity
          option_type (str): 'C' for call, 'P' for put
     Returns:
          float: estimated option price
     """
     if option_type not in ['C', 'P']:
          raise ValueError("option_type must be 'C' for call or 'P' for put")
     
     if option_type == 'C':
          payoffs = np.maximum(S_paths[:, -1] - np.min(S_paths, axis=1), 0)
     else:
          payoffs = np.maximum(np.max(S_paths, axis=1) - S_paths[:, -1], 0)
    
     discounted_payoff = np.exp(-r * T) * payoffs
     return np.mean(discounted_payoff)

