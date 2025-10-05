"""
This module contains functions for bootstrapping volatility and building volatility surface from Vanilla options
"""

import pandas as pd
from math import sqrt
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline, interp1d

class VolSurface:

    def __init__(self, option_data: pd.DataFrame):
        """
        VolSurface class

        Args:
            option_data (pd.DataFrame): option data, with columns ['maturity', 'log_moneyness', 'implied_vol']
        """
        if not all(col in option_data.columns for col in ['maturity', 'log_moneyness', 'implied_vol']):
            raise ValueError("option_data must contain columns ['maturity', 'log_moneyness', 'implied_vol']")

        # Insert total variance column
        option_data.loc[:, 'total_variance'] = option_data['implied_vol'] ** 2 * option_data['maturity']
        self.option_data = option_data
        self.option_data.sort_values(by=['maturity', 'log_moneyness'], inplace=True)
        self.option_data.drop_duplicates(subset=['maturity', 'log_moneyness'], inplace=True)
         
        self.__vol_surface = None
    
    @property
    def vol_surface(self) -> pd.DataFrame:
        """
        Get volatility surface

        Returns:
            pd.DataFrame: volatility surface
        """
        if self.__vol_surface is None:
            print("Volatility surface not built yet. Call build_surface() first.")
        else:
            if 'implied_vol' not in self.__vol_surface.columns and 'total_variance' in self.__vol_surface.columns:
                self.__vol_surface.loc[:, 'implied_vol'] = (self.__vol_surface['total_variance'] / self.__vol_surface['maturity']) ** 0.5
        return self.__vol_surface

        
    def __cubic_spline(self, 
                       x: pd.Series,
                       y: pd.Series,
                       x_new: pd.Series

                       ) -> pd.Series:
        """
        Build volatility surface using cubic spline interpolation
        Args:
            x (pd.Series): x values which can be maturities or log-moneyness
            y (pd.Series): y values which are the total variances
            x_new (pd.Series): new x values to interpolate
        Returns:
        """

        cs = CubicSpline(x, y, extrapolate=True)
        y_new = cs(x_new)
        return pd.Series(y_new)
    
    @staticmethod
    def __svi(a: float,
              b: float, 
              rho: float,
              m: float, 
              sigma: float, 
              k: float | pd.Series) -> float | pd.Series:
        """
        SVI total variance function
        Args:
            a (float): SVI parameter
            b (float): SVI parameter
            rho (float): SVI parameter
            m (float): SVI parameter
            sigma (float): SVI parameter
            k (float): log-moneyness
        Returns:
            float: total variance
        """
        if isinstance(k, pd.Series):
            return a + b * (rho * (k - m) + ((k - m) ** 2 + sigma ** 2).apply(sqrt))
        else:
            return a + b * (rho * (k - m) + sqrt((k - m) ** 2 + sigma ** 2))
    
    
    def __svi_fit(self, 
                     x: pd.Series,
                     y: pd.Series,
                     neighbor_params: dict | None = None,
                     lambda_: float = 0
                     ) -> dict:
        """
        Build volatility surface using SVI interpolation
        Args:
            x (pd.Series): x values which can be maturities or log-moneyness
            y (pd.Series): y values which are the total variances

        Returns:
            dict: SVI parameters
        """
        # Calibrate SVI parameters
        def svi_objective(params):
            a, b, rho, m, sigma = params
            y_pred = VolSurface.__svi(a, b, rho, m, sigma, x)
            if neighbor_params is None:
                return ((y - y_pred) ** 2).sum()
            else:
                return ((y - y_pred) ** 2).sum() + lambda_ * ((a - neighbor_params['a'])**2 \
                                                          + (b - neighbor_params['b'])**2\
                                                          + (rho - neighbor_params['rho'])**2\
                                                          + (m - neighbor_params['m'])**2
                                                          + (sigma - neighbor_params['sigma'])**2)
        
        initial_params = [0.1, 0.1, 0.0, 0.0, 0.1]

        bounds = [(-float('inf'), float('inf')),
                  (1e-6, float('inf')),
                  (-1, 1),
                  (-float('inf'), float('inf')),
                  (1e-6, float('inf'))]
        result = minimize(svi_objective, initial_params, bounds=bounds)
        if result.success:
            a, b, rho, m, sigma = result.x
            return {'a': a, 'b': b, 'rho': rho, 'm': m, 'sigma': sigma}
        else:
            print("SVI calibration failed")
            return {}
    


    def build_svi_surface(self,
                          maturity_grid: pd.Series,
                          logmoneyness_grid: pd.Series,
                          apply_penalty: bool = True,
                          penalty_lambda_: float = 0
                          ) -> pd.DataFrame:
        """
        Build volatility surface using SVI interpolation
        
        Args:
            maturity_grid (pd.Series): series of maturities to interpolate.
            logmoneyness_grid (pd.Series): series of logmoneynesss to interpolate. We recommend using log-moneyness grid.
            smooth_params (bool, optional): whether to smooth SVI parameters across maturities. Defaults to True.
        """

        surface = pd.DataFrame(columns=['maturity', 'log_moneyness', 'total_variance'])
        params = pd.DataFrame(columns=['a', 'b', 'rho', 'm', 'sigma'], index=self.option_data['maturity'].unique())

        param_dict = {}
        # Cubic or SVI for logmoneyness interpolation with fixed maturity
        for maturity in self.option_data['maturity'].unique():
            subset = self.option_data[self.option_data['maturity'] == maturity]

            if len(subset) < 2:
                 # Skip maturities with less than 2 data points as we cannot interpolate
                continue

            if len(param_dict) > 0 and apply_penalty:
                neighbor_params = param_dict.copy()
            else:
                neighbor_params = None
            param_dict = self.__svi_fit(subset['log_moneyness'],
                                                    subset['total_variance'],
                                                    neighbor_params=neighbor_params,
                                                    lambda_= penalty_lambda_)
            params.loc[maturity, ['a', 'b', 'rho', 'm', 'sigma']] = pd.Series(param_dict)

        print("Nan params: ", params.isna().sum(), "Total params: ", len(params)
              )

        params.ffill(inplace=True)
        params.bfill(inplace=True)
        print("params", params)
        for maturity in params.index:
            vols = VolSurface.__svi(params.loc[maturity, 'a'],
                                    params.loc[maturity, 'b'],
                                    params.loc[maturity, 'rho'],
                                    params.loc[maturity, 'm'],
                                    params.loc[maturity, 'sigma'],
                                    logmoneyness_grid)


            
            if len(surface) == 0:
                surface = pd.DataFrame({'maturity': maturity,
                                        'log_moneyness': logmoneyness_grid,
                                        'total_variance': vols})
            else:
                surface = pd.concat([surface, pd.DataFrame({'maturity': maturity,
                                                       'log_moneyness': logmoneyness_grid,
                                                       'total_variance': vols})],
                                                       axis=0,
                                                        ignore_index=True)
        for logmoneyness in logmoneyness_grid:
            subset = surface[surface['log_moneyness'] == logmoneyness]
            if len(subset) < 2:
                continue
            vols = self.__cubic_spline(subset['maturity'],
                                                     subset['total_variance'],
                                                     maturity_grid)
            surface.drop(subset.index, inplace=True)
            surface = pd.concat([surface, pd.DataFrame({'maturity': maturity_grid,
                                                       'log_moneyness': logmoneyness,
                                                       'total_variance': vols})], 
                                                       axis=0,
                                                        ignore_index=True)
        self.__vol_surface = surface.reset_index(drop=True)
        print(params)



        
    def build_ssvi_surface(self,
                           maturity_grid: pd.Series,
                           logmoneyness_grid: pd.Series):
        """
        Build volatility surface using SSVI interpolation
        
        Args:
            maturity_grid (pd.Series): series of maturities to interpolate.
            logmoneyness_grid (pd.Series): series of logmoneynesss to interpolate. We recommend using log-moneyness grid.
        """
        self.__vol_surface = self.__ssvi_surface(self.option_data, logmoneyness_grid, maturity_grid)

    @staticmethod
    def __ssvi_point(rho: float,
                eta: float,
                gamma: float,
                k: float,
                theta: float) -> float:
        """
        SSVI total variance function
        Args:
            rho (float): SSVI parameter
            eta (float): SSVI parameter
            gamma (float): SSVI parameter
            k (float): log-moneyness
            theta (float): total variance at-the-money
        Returns:
            float: total variance
        """
        phi = eta / (theta ** gamma)
        return (theta / 2) * (1 + rho * phi * k + sqrt((phi * k + rho) ** 2 + (1 - rho ** 2)))
    
    @staticmethod
    def __ssvi(rho: float,
                        eta: float,
                        gamma: float,
                        k: pd.Series,
                        theta: pd.Series,
                        Ts: pd.Series) -> pd.DataFrame:
        """
        Build volatility surface using SSVI interpolation
        Args:
            k (pd.Series): logmoneyness prices
            theta (pd.Series): total variances at-the-money
            Ts (pd.Series): maturities corresponding to theta
            rho (float): SSVI parameter
            eta (float): SSVI parameter
            gamma (float): SSVI parameter
        Returns:
            pd.Series: total variances
        """
        surface = pd.DataFrame(0, columns=['total_variance'], 
                               index=pd.MultiIndex.from_product([Ts, k],
                               names=['maturity', 'log_moneyness']), dtype=float)
        surface.sort_index(inplace=True)
    
        for T, theta in zip(Ts, theta):
            for logmoneyness in k:
                surface.loc[(T, logmoneyness), 'total_variance'] = VolSurface.__ssvi_point(rho, eta, gamma, logmoneyness, theta)
        surface = surface.reset_index()
        return surface


    def __ssvi_surface(self,
                       option_data: pd.DataFrame,
                       logmoneyness_grid: pd.Series,
                       maturity_grid: pd.Series
                       ) -> pd.DataFrame:
        """
        Build volatility surface using SSVI interpolation
        Args:
            logmoneyness (pd.Series): logmoneyness prices
            maturity (pd.Series): maturities
            implied_vol (pd.Series): implied volatilities
            logmoneyness_grid (pd.Series): new logmoneyness prices to interpolate
            maturity_grid (pd.Series): new maturities to interpolate
        Returns:
            pd.DataFrame: volatility surface
        """

        atm = option_data[option_data['log_moneyness'].abs() < 1e-2].copy()
        atm.sort_values(by='maturity', inplace=True)
        atm.drop_duplicates(subset=['maturity'], inplace=True)
        if len(atm) < 2:
            raise ValueError("Not enough ATM data points for SSVI calibration")
        # Calibrate SSVI parameters
        def ssvi_objective(params):
            rho, eta, gamma = params
            surface = VolSurface.__ssvi(rho, eta, gamma,
                                                option_data['log_moneyness'],
                                                atm['total_variance'],
                                                atm['maturity'])
            valid_data = option_data[option_data['maturity'].isin(atm['maturity'])]
            return ((valid_data['total_variance'] - surface['total_variance']) ** 2).sum()

        initial_params = [0.0, 0.1, 0.5]
        bounds = [(-0.999, 0.999),
                  (1e-6, float('inf')),
                  (1e-6, 1.0)]
        result = minimize(ssvi_objective, initial_params, bounds=bounds, method="L-BFGS-B")
        if result.success:
            rho, eta, gamma = result.x
        else:
            print("SSVI calibration failed")
            return pd.DataFrame()
        
        print("SSVI parameters: rho = {}, eta = {}, gamma = {}".format(rho, eta, gamma))

        # Interpolate ATM

        thetas = interp1d(atm['maturity'], atm['total_variance'], 
                          kind='linear',
                          fill_value="extrapolate")(maturity_grid)
        print(thetas)
        thetas = thetas.clip(min=1e-6)  # Ensure positivity

        # Build surface
        surface = VolSurface.__ssvi(rho, eta, gamma, logmoneyness_grid, thetas, maturity_grid)
        return surface


    def plot_surface(self, data: pd.DataFrame):
        """
        Draw volatility surface

        Args:
            data (pd.DataFrame): volatility surface data, with columns ['maturity', 'log_moneyness', 'implied_vol']
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        X = data['maturity'].values
        Y = data['log_moneyness'].values
        Z = data['implied_vol'].values

        surf = ax.plot_trisurf(X, Y, Z, cmap=cm.viridis, linewidth=0.2)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        ax.set_xlabel('Maturity')
        ax.set_ylabel('log_moneyness')
        ax.set_zlabel('Implied Volatility')

        plt.show()
