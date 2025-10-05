# Option Pricing Library

A comprehensive Python library for option pricing, volatility surface construction, and quantitative finance. This package provides analytical and numerical methods for pricing various types of options and modeling financial derivatives.

## Features

- **European Options**: Black-Scholes analytical pricing
- **Implied Volatility**: Newton-Raphson method for volatility calculation
- **Volatility Surface**: Construction using cubic spline and SVI interpolation
- **Monte Carlo Simulation**: Heston model implementation
- **Exotic Options**: Asian, Barrier, and Lookback options pricing
- **Local Volatility**: Dupire's formula implementation

## Installation

```bash
pip install -e .
```

For development dependencies:
```bash
pip install -e ".[dev]"
```

## Quick Start

### European Option Pricing

```python
from option_pricing import EuropeanOption

# Create option instance
option = EuropeanOption()

# Price a call option
price = option.black_scholes(
    sigma=0.2,    # volatility
    S=100,        # spot price
    K=105,        # strike price
    T=1.0,        # time to maturity (years)
    r=0.05,       # risk-free rate
    q=0.02,       # dividend yield
    option_type='C'  # 'C' for call, 'P' for put
)
print(f"Option price: ${price:.2f}")
```

### Implied Volatility Calculation

```python
from option_pricing import implied_volatility

# Calculate implied volatility from market price
market_price = 8.50
impl_vol = implied_volatility(
    market_price=market_price,
    S=100, K=105, T=1.0, r=0.05, q=0.02,
    option_type='C'
)
print(f"Implied volatility: {impl_vol:.4f}")
```

### Volatility Surface Construction

```python
import pandas as pd
from option_pricing import VolSurface

# Prepare option data
option_data = pd.DataFrame({
    'maturity': [0.25, 0.5, 1.0, 0.25, 0.5, 1.0],
    'strike': [-0.1, -0.1, -0.1, 0.0, 0.0, 0.0],  # log-moneyness
    'implied_vol': [0.22, 0.20, 0.18, 0.20, 0.18, 0.16]
})

# Build volatility surface
vol_surface = VolSurface(option_data)
surface = vol_surface.build_surface(
    maturity_grid=pd.Series([0.5, 1.0, 1.5]),
    strike_grid=pd.Series([-0.2, -0.1, 0.0, 0.1, 0.2]),
    strike_method='cubic'  # or 'svi'
)

print(vol_surface.vol_surface)
```

### Monte Carlo Simulation with Heston Model

```python
import numpy as np
from option_pricing import MonteCarlowithHeston, price_european

# Simulate asset paths using Heston model
S_paths = MonteCarlowithHeston(
    S0=100,          # initial stock price
    v0=0.04,         # initial variance
    kappa=2.0,       # mean reversion speed
    theta=0.04,      # long-term variance
    sigma=0.3,       # volatility of volatility
    rho=-0.7,        # correlation
    r=0.05,          # risk-free rate
    T=1.0,           # time to maturity
    n_steps=252,     # number of time steps
    n_paths=10000    # number of simulation paths
)

# Price European option using simulated paths
option_price = price_european(
    S_paths=S_paths[:, -1],  # terminal prices
    K=105,                   # strike price
    r=0.05,                  # risk-free rate
    T=1.0,                   # time to maturity
    option_type='C'          # call option
)
print(f"Monte Carlo option price: ${option_price:.2f}")
```

### Exotic Options Pricing

```python
from option_pricing import price_asian, price_barrier, price_lookback

# Asian option (average price)
asian_price = price_asian(
    S_paths=S_paths,
    K=100,
    r=0.05,
    T=1.0,
    option_type='C'
)

# Barrier option (knock-out)
barrier_price = price_barrier(
    S_paths=S_paths,
    K=100,
    barrier=120,
    r=0.05,
    T=1.0,
    barrier_type='up-and-out',
    option_type='C'
)

# Lookback option
lookback_price = price_lookback(
    S_paths=S_paths,
    r=0.05,
    T=1.0,
    option_type='C'
)

print(f"Asian option: ${asian_price:.2f}")
print(f"Barrier option: ${barrier_price:.2f}")
print(f"Lookback option: ${lookback_price:.2f}")
```

### Local Volatility Model

```python
import numpy as np
from option_pricing import local_volatilities

# Define grid
log_moneyness = np.linspace(-0.3, 0.3, 21)
maturities = np.array([0.25, 0.5, 1.0, 2.0])

# Total implied variance matrix (example)
W = np.array([
    [0.05, 0.045, 0.04, 0.045, 0.05],  # T=0.25
    [0.04, 0.036, 0.032, 0.036, 0.04], # T=0.5
    [0.032, 0.029, 0.025, 0.029, 0.032], # T=1.0
    [0.025, 0.023, 0.020, 0.023, 0.025]  # T=2.0
])

# Calculate local volatilities
local_vols = local_volatilities(
    log_moneyness[:5],  # subset for example
    maturities,
    W
)
print("Local volatilities:")
print(local_vols)
```

## Module Overview

### Core Modules

- **`european.py`**: Black-Scholes pricing for European options
- **`implied_vol.py`**: Implied volatility calculation using optimization
- **`vol_surface.py`**: Volatility surface construction and interpolation
- **`MonteCarlo.py`**: Monte Carlo simulation with Heston stochastic volatility
- **`MonteCarloPricing.py`**: Exotic option pricing using Monte Carlo methods
- **`local_vol.py`**: Local volatility model implementation

### Data Processing

- **`scripts/sample_data_fetcher.py`**: Sample financial data retrieval
- **`scripts/data_to_sql.py`**: Data processing and storage utilities


## Project Structure

```
option_pricing/
├── __init__.py              # Package initialization
├── european.py              # European option pricing
├── implied_vol.py           # Implied volatility calculation
├── vol_surface.py           # Volatility surface construction
├── MonteCarlo.py            # Heston Monte Carlo simulation
├── MonteCarloPricing.py     # Exotic options pricing
└── local_vol.py             # Local volatility model

scripts/
├── sample_data_fetcher.py   # Sample data utilities
└── data_to_sql.py           # Data processing

data/
├── option_data.db           # Sample option data (SQLite)
├── vol_surface.csv          # Volatility surface data
└── *.csv                    # Additional market data
```

## Mathematical Models
See the [Mathematical Models Documentation](docs/models.md) for detailed explanations of the models implemented in this library.

## Requirements

- Python >= 3.8
- pandas
- numpy
- scipy


## Authors

- QinqinAndMacaulayCat

## Acknowledgments

- Black-Scholes-Merton option pricing model
- Heston stochastic volatility model
- SVI volatility surface parameterization
- Dupire local volatility model
