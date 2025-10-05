
# Option Pricing Model Documentation

## Model Overview

The process of the project is:

1. Bootstrapping the implied volatilities from market data.
2. Interpolating the implied volatilities to create a smooth volatility surface using the SVI model or cubic splines.
3. Calibrating the Heston model parameters to fit the volatility surface.
4. Using the calibrated Heston model to price exotic options.
5. Validating the model by comparing the prices of vanilla options from the Heston model with market prices. (as no exotic options market data is available)

## Data

The market data is sourced from the Wharton Research Data Services (WRDS) database, specifically from the OptionMetrics IvyDB US dataset. This dataset provides comprehensive historical option prices and implied volatilities for a wide range of underlying assets.

I used SPX options data from 2022-01-03 to 2023-08-31 for testing and demonstration purposes. The data includes daily closing prices, strike prices, expiration dates, and implied volatilities for both call and put options.

### Introduction to SPX Options
SPX options are European-style options based on the S&P 500 index, which is a widely followed benchmark for the U.S. stock market. The choice of SPX options is due to their liquidity and the availability of a wide range of strike prices and maturities.

Types:
- standard options (SPX): Monthly expirations (3rd Friday of each month) up to 12 months
- weekly options (SPXW): Expire every weekday in next 5 weeks
- end-of-month(EOM) options: Expire on the last trading day of each month.
- end-of-quarter(EOQ) options: Expire on the last trading day of each quarter.
- Long-term options (LEAPS): Expire in 12 to 180 months, with up to 10 contracts available at any time.

Premiums and Multipliers:
- SPX options are cash-settled, meaning that upon exercise, the holder receives a cash amount rather than the underlying asset.
- The multiplier for SPX options is 100, meaning that the option price is multiplied by 100 to determine the total premium. For example, if an SPX call option is priced at 5.00, the total premium would be $500 (5.00 x 100). When exercising the option, the payoff is also multiplied by 100.

Strike Prices:
- SPX options have a wide range of strike prices, typically in increments of 5 for near-the-money options and larger increments for deep in-the-money or out-of-the-money options. (5, 10, 25, 50, 100, etc.)
- The shorter the time to expiration, the higher granularity of strike prices available.

Since both call and put options are available but the implied volatilities may differ due to market imperfections, we use the following approach to bootstrap the implied volatilities:

1. **Data Selection**: For each underlying asset, we select options that are near-the-money (NTM) and have sufficient trading volume to ensure liquidity.

## Implied Volatility Calculation

In this part, we should use `brentq` method from `scipy.optimize` to find the root of the objective function, which is the difference between the market price and the Black-Scholes price. The `brentq` method is a robust root-finding algorithm that combines bisection, secant, and inverse quadratic interpolation methods. It is particularly effective for finding roots of continuous functions.

However, `brentq` requires the function to have different signs at the endpoints of the interval. Therefore, we need to ensure that our objective function meets this requirement. We can do this by checking the values of the objective function at the endpoints and adjusting them if necessary.

We should not use minimize error methods like `minimize` from `scipy.optimize` because they are not guaranteed to find the root of the function. They may converge to a local minimum or fail to converge altogether, especially if the function is not well-behaved. In contrast, `brentq` is specifically designed for root-finding and is more reliable in this context especially when the function is continuous and monotonic.

`minimize` methods are more suitable for optimization problems where we want to find the minimum or maximum of a function, rather than finding the root of a function. And the function should be complex and not flat.



## Interpolation Methods


### Data Preparation

We should interpolate the total implied variance $w(k, T) = \sigma_{imp}^2(K, T) T$ instead of the implied volatility $\sigma_{imp}(K, T)$ directly. This is because the total implied variance is more stable and behaves more linearly with respect to log-moneyness and time to maturity. Also, it helps to avoid arbitrage opportunities in the volatility surface such as calendar spread arbitrage and butterfly arbitrage. For calendar spread arbitrage, we need to ensure that $w(k, T)$ is increasing in $T$ for all $k$. For butterfly arbitrage, we need to ensure that $w(k, T)$ is convex in $k$ for all $T$.

I also filtered the data based on:
- the $k \in [-1.5, 1.5]$ to avoid extreme values of log-moneyness which may lead to unreliable interpolation results.
- maturity: The extremely short-term options (less than 7 days to maturity) are also excluded from the calibration process, as they can introduce noise and instability in the volatility surface due to their high sensitivity to market movements.
- volume: Options with very low trading volume may have unreliable prices and implied volatilities, which can distort the interpolation results. Therefore, we set a minimum volume threshold to filter out such options.

### SVI Model
SVI (Stochastic Volatility Inspired) is a parametric model used to fit the implied volatility surface of options. It is particularly useful for capturing the smile and skew patterns observed in market data. 
(It's called "stochastic volatility inspired" because it was originally derived from a stochastic volatility model, but it is not a stochastic volatility model itself.)

The SVI model expresses the total implied variance as a function of log-moneyness and time to maturity. The formula for the SVI model is given by:

$$
w(k, T) = \sigma_{imp}^2(K, T) T
$$

where:

- $k=\log(K/F)$ is the log-moneyness, with $K$ being the strike price and $F$ the forward price of the underlying asset. $F_T = S_0 e^{(r-q)T}$. 


The pros of using the SVI model include:
- **Flexibility**: SVI can capture a wide range of shapes in the volatility surface
- **Parsimony**: It uses a relatively small number of parameters to describe the entire surface
- **Arbitrage-Free**: When properly calibrated, SVI can ensure that the implied volatility surface is free from arbitrage opportunities.



#### Raw SVI model formula:


Raw SVI model is a special case of the general SVI model where the parameters are not time-dependent. The 5 parameters of the Raw SVI model are $(a, b, \rho, m, \sigma)$.

Raw SVI model formula:

$$
w(k) = a + b \left( \rho (k - m) + \sqrt{(k - m)^2 + \sigma^2} \right)
$$

where:
- $w(k)$ is the total implied variance at log-moneyness $k$.
- $a$ is the baseline variance level.
- $b > 0$ controls the overall slope of the volatility smile. (or the wing)
- $\rho \in [-1, 1]$ determines the skewness of the smile. (for equity options, $\rho$ is typically negative, reflecting the common market observation that out-of-the-money puts are more expensive than out-of-the-money calls)
- $m$ is the log-moneyness at which the minimum variance occurs.(can be interpreted as the "center" of the smile)
- $\sigma > 0$ controls the curvature of the peak of the smile.

#### SSVI model formula:

SSVI (Surface SVI) is an extension of the Raw SVI model that introduces time-dependency to the parameters, allowing for a more dynamic representation of the implied volatility surface over different maturities. The SSVI model is particularly useful for capturing the evolution of the volatility surface as time to maturity changes. The 5 parameters of the SSVI model are $(\theta, \phi, \rho, m, \sigma)$.

SSVI model formula:

$$
w(k, \theta) = \frac{\theta}{2} \left( 1 + \rho \phi(\theta) k + \sqrt{(\phi(\theta) k + \rho)^2 + (1 - \rho^2)} \right)
$$

where:
- $w(k, \theta)$ is the total implied variance at log-moneyness $k$ and total variance $\theta$.
- $\theta = w(0, T)$ is the total implied variance at-the-money for maturity $T$. This is estimated from market data rather than being a free parameter.
- $\phi(\theta)$ is a function that describes how the slope of the volatility smile changes with total variance $\theta$. A common choice is $\phi(\theta) = \eta \theta^{-\gamma}$, where $\eta > 0$ and $\gamma \in (0, 1)$ are additional parameters to be calibrated.
- $\rho \in [-1, 1]$ determines the skewness of the smile.
- w(k, T) should be increasing in T for all k to avoid calendar spread arbitrage.
- w(k, T) should be convex in k for all T to avoid butterfly arbitrage.

Choice of $\phi(\theta)$:
1. Power-law form: 

$$
\phi(\theta) = \eta \theta^{-\gamma}
$$

where 
- $\eta > 0$ controls the skewness of the volatility smile.
- $\gamma \in (0, 1)$ controls the decay speed of the skew as maturity increases.

This is the most commonly used form in practice. It's simple and easy to calibrate.

2. Heston-like form:

$$
\phi(\theta) = \frac{2 \eta}{1 + \epsilon \theta}
$$

where
- $\eta > 0$ controls the skewness of the volatility smile.
- $\epsilon > 0$ controls the long-term behavior of the skew. (longer maturities, the skew flattens out)

It's closer to the behavior of the Heston model, which is a popular stochastic volatility model.

3. Staircase form

#### How to calibrate the SVI model

Target function to minimize:

$$
\text{Objective} = \sum_{i=1}^{N} w_i \left( \sigma_{imp}^{market}(K_i, T_i) - \sigma_{imp}^{SVI}(K_i, T_i; \text{params}) \right)^2
$$

The weights $w_i$ can be chosen based on vega. 

#### Comparison between Raw SVI and SSVI

Raw SVI:
-  pros: Simpler and easier to implement.
- cons: not continuous across maturities, which can lead to arbitrage opportunities.

SSVI:   
- pros: Avoids arbitrage opportunities across maturities, more realistic for practical applications.
- cons: Less flexible and the shape of the volatility smile is constrained by the choice of $\phi(\theta)$.



### Cubic Splines

Cubic splines are a type of piecewise polynomial function that can be used to interpolate data points. In the context of implied volatility surfaces, cubic splines can be used to create a smooth surface that fits the observed market data.


The main idea behind cubic splines is to divide the range of the data into intervals and fit a cubic polynomial to each interval. The polynomials are chosen such that they are continuous and have continuous first and second derivatives at the interval boundaries (knots). This ensures a smooth transition between the polynomials.

### Application details

Cubic splines can be applied to the implied volatility surface by interpolating the total implied variance $w(k, T)$ across different maturities and strikes. This allows for a smooth and continuous surface that can better capture the dynamics of the market.

The steps to implement cubic splines for implied volatility surface interpolation are as follows:
1. **Data Preparation**: Collect the market data for implied volatilities, strikes, and maturities. Convert the strikes to log-moneyness $k = \log(K/F)$ and calculate the total implied variance $w(k, T) = \sigma_{imp}^2(K, T) T$.
2. **Knot Selection**: Choose the knots for the cubic splines. This can be done by selecting a set of maturities and strikes that cover the range of the data.
3. **SVI Calibration**: For each maturity, fit the SVI model to the total implied variance data to obtain a smooth curve for each maturity.
4. **Cubic Spline Interpolation**: Use cubic splines to interpolate the SVI-fitted curves across different maturities. This will create a smooth surface that captures the dynamics of the market.
5. **Surface Construction**: Construct the implied volatility surface by evaluating the cubic splines at the desired strikes and maturities.

## Volatility model

### Heston Model

The Heston model is the most classic stochastic volatility model used in option pricing. It was introduced by Steven Heston in 1993 as a way to capture the observed market phenomena of volatility smiles and skews, which are not accounted for in the Black-Scholes model.

The Heston model assumes that the dynamics of the underlying asset price $S_t$ and its variance $v_t$ follow the stochastic differential equations (SDEs):

$$
dS_t = \mu S_t dt + \sqrt{v_t} S_t dW_t^S
$$ 

$$
dv_t = \kappa (\theta - v_t) dt + \sigma \sqrt{v_t} dW_t^v
$$  

where:
- $\kappa$ is the rate at which the variance reverts to the long-term mean $\theta$.
- $\theta$ is the long-term mean of the variance.
- $\sigma$ is the volatility of the variance process (vol of vol).
- $W_t^S$ and $W_t^v$ are two correlated Wiener processes with correlation $\rho$.

The Heston model does not have a closed-form solution for option prices like the Black-Scholes model, but it can be solved using numerical methods such as the Fourier transform or Monte Carlo simulation.

Pros:
- Captures the stochastic nature of volatility and can model volatility smiles and skews.
- More realistic dynamics of asset prices compared to constant volatility models.
- The volatility process follows CIR process, ensuring non-negativity of variance.

Cons:
- Difficult to calibrate due to the number of parameters.
- Can't fit the volatility surface exactly, leading to potential mispricing of options.

#### Calibration of Heston Model


1. Data
- Collect market data for vanilla options, including prices, strikes, maturities, and implied volatilities.

2. Objective Function

There is no closed-form option price formula under the Heston model, but we can use the semi-analytical solution based on the characteristic function of the Heston model. The characteristic function $\phi(u; T)$ of the log asset price under the Heston model is given by the following formula which is closed-form:

$$
\phi(u; T) = \mathbb{E} \left[ e^{i u \log(S_T)} \right]
= \exp \left( C(u; T) + D(u; T) v_0 + i u \log(S_0) \right)
$$

where $C(u; T)$ and $D(u; T)$ are complex-valued functions defined as:

$$
d(u) = \sqrt{(\rho \sigma i u - \kappa)^2 + \sigma^2 (i u + u^2)}
$$

$$
g(u) = \frac{\kappa - \rho \sigma i u - d(u)}{\kappa - \rho \sigma i u + d(u)}
$$

$$
D(T, u) = \frac{\kappa - \rho \sigma i u - d(u)}{\sigma^2} \left( \frac{1 - e^{-d(u) T}}{1 - g(u) e^{-d(u) T}} \right)
$$

$$
C(T, u) = \frac{\kappa \theta}{\sigma^2} \left( (\kappa - \rho \sigma i u - d(u)) T - 2 \log \left( \frac{1 - g(u) e^{-d(u) T}}{1 - g(u)} \right) \right)
$$


The price of a European call option under the Heston model can be computed using the following formula:


$$
C(k, T) = e^{-rT} F_T (P_1(k, T) - e^k P_2(k, T))
$$

where $P_1$ and $P_2$ are the risk-neutral probabilities that can be computed using the characteristic function of the Heston model.

$$
P_j(k, T) = \frac{1}{2} + \frac{1}{\pi} \int_0^{\infty} \text{Re} \left( \frac{e^{-i u k} \phi(u - i (j - 1); T)}{i u} \right) du
, j = 1, 2
$$

Re presents the real part of a complex number.

The objective function to minimize is the sum of squared differences between market prices and model prices:

$$
\text{Objective} = \sum_{i=1}^{N} w_i \left( C^{market}(K_i, T_i) - C^{Heston}(K_i, T_i; \text{params}) \right)^2
$$

The weights $w_i$ can be chosen as the inverse of the option's vega to give less weight to options with high vega.

Since there is no closed-form solution for this integral, we call it semi-analytical rather than analytical.

3. Calibration

The initial guess:

$$
\kappa = 1.0, \theta = ATM \text{ variance}, \sigma = 0.5, \rho = -0.5, v_0 = ATM \text{ variance}
$$

Constrants:
$$
\kappa > 0, \theta > 0, \sigma > 0, -1 < \rho < 1, v_0 > 0
$$

Use optimization algorithms such as L-BFGS-B or Nelder-Mead to minimize the objective function and find the optimal parameters.

More advanced techniques include GNNs.

When calibrating, we use the call option prices calculated from the volatility surface generated by the SVI model or cubic splines as the market prices to fit the Heston model. That's because:

- The market prices of vanilla options may be noisy and contain bid-ask spreads, which can lead to instability in the calibration process.
- Heston model is not convex, so the optimization may converge to local minima. Using smoothed prices can help to mitigate this issue.


4. Validation

- After calibration, validate the model by comparing the prices of vanilla options generated by the Heston model with the market prices. Calculate metrics such as root mean square error (RMSE) or mean absolute error (MAE) to assess the accuracy of the model.
- Check if the Feller condition is satisfied: $2 \kappa \theta > \sigma^2$. If not, the variance process may hit zero, which can lead to numerical instability in option pricing. Feller condition is not strictly necessary for the Heston model to be valid, but it helps to ensure that the variance remains positive and avoids potential issues in numerical methods.



### Local Volatility Model

Local volatility model is a type of stochastic volatility model where the volatility of the underlying asset is assumed to be a deterministic function of the asset price and time. The local volatility model was introduced by Bruno Dupire in 1994 as a way to capture the observed market prices of options more accurately than the Black-Scholes model.

The function $\sigma_{loc}(S, t)$ represents the local volatility at asset price $S$ and time $t$. The local volatility model assumes that the dynamics of the underlying asset price $S_t$ follow the stochastic differential equation (SDE):

$$
dS_t = \mu S_t dt + \sigma_{loc}(S_t, t) S_t dW_t
$$


The local volatility function can be derived from the market prices of European options using the Dupire formula:

$$
\sigma_{loc}^2(K, T) = \frac{\frac{\partial C}{\partial T} + r K \frac{\partial C}{\partial K}}{\frac{1}{2} K^2 \frac{\partial^2 C}{\partial K^2}}
$$

Or derived from the implied volatility surface:

$$

\sigma_{\text{Dup}}^2(k, T) =
\frac{
\frac{\partial w}{\partial T}
}{
1
- \frac{k}{w} \frac{\partial w}{\partial k}
+ \frac{1}{4}
\left(
- \frac{1}{4}
- \frac{1}{w}
+ \frac{k^2}{w^2}
\right)
\left(
\frac{\partial w}{\partial k}
\right)^2
+ \frac{1}{2} \frac{\partial^2 w}{\partial k^2}
}

$$


Pros: 
- Fits market prices of vanilla options / vol surface exactly.
- Simple to implement and calibrate thus widely used in risk management

Cons:
- Assumes volatility is a deterministic function, which may not capture all market dynamics.
- Can lead to unrealistic local volatility values, especially for deep in-the-money or out-of-the-money options.
- The derivatives in the Dupire formula may exaggerate market noise.

