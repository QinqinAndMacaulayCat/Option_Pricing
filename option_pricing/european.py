"""
This module contains functions for pricing European options using the Black-Scholes formula.
"""
from math import log, sqrt, exp
from scipy.stats import norm

class EuropeanOption:

    @staticmethod
    def __d1(sigma: float,
             F: float, 
             K: float,
             T: float,
           ) -> float:
        """
        d1 function in Black Scholes formula

        Args:
            sigma (float): volatility
            F (float): forward price
            K (float): strike price
            T (float): time to maturity
        Returns:
            float: d1 value
        """
    
        return (log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt(T))

    @staticmethod
    def __d2(sigma: float,
             F: float, 
             K: float,
             T: float,
             ) -> float:
        """
        d2 function in Black Scholes formula

        Args:
            sigma (float): volatility
            F (float): forward price
            K (float): strike price
            T (float): time to maturity
        Returns:
            float: d2 value
        """
    
        return EuropeanOption.__d1(sigma, F, K, T) - sigma * sqrt(T)


    @staticmethod
    def black_scholes(sigma: float,
                        F: float, 
                        K: float,
                        T: float,
                        r: float,
                        option_type: str = 'C') -> float:
        """
        Black Scholes formula

        Args:
            sigma (float): volatility
            F (float): forward price
            K (float): strike price
            T (float): time to maturity
            r (float): risk-free interest rate
            option (str, optional): option type. Defaults to 'C'.
        Returns:
            float: option price
        """

        d1 = EuropeanOption.__d1(sigma, F, K, T)
        d2 = EuropeanOption.__d2(sigma, F, K, T)

        if option_type == "C":
            price = F * norm.cdf(d1) - K * norm.cdf(d2)
            price *= exp(-r * T)
        elif option_type == "P":
            price = K * norm.cdf(-d2) - F * norm.cdf(-d1)
            price *= exp(-r * T)
        else:
            raise ValueError("option_type must be 'C' or 'P'")
        return price

    @staticmethod
    def vega(sigma: float,
             F: float, 
             K: float,
             T: float,
           ) -> float:
        """
        Vega function in Black Scholes formula

        Args:
            sigma (float): volatility
            F (float): forward price
            K (float): strike price
            T (float): time to maturity
        Returns:
            float: vega value
        """
    
        d1 = EuropeanOption.__d1(sigma, F, K, T)
        return F * norm.pdf(d1) * sqrt(T)

