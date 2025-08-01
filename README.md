# Monte Carlo Simulation for Quantitative Finance

A comprehensive Python implementation of Monte Carlo simulations for quantitative finance applications, including stock price simulation, option pricing, and portfolio optimization.

## Features

### 1. Stock Price Simulation
- **Geometric Brownian Motion (GBM)** implementation
- Multiple simulation paths with confidence intervals
- Visual representation of price evolution
- Statistical analysis of final price distributions

### 2. European Option Pricing
- Monte Carlo pricing for both call and put options
- Confidence intervals for price estimates
- Risk-neutral valuation framework
- Support for various option parameters

### 3. Portfolio Optimization
- Monte Carlo portfolio weight optimization
- Efficient frontier analysis
- Maximum Sharpe ratio portfolio
- Minimum volatility portfolio
- Risk-return analysis

### 4. Risk Analysis
- Value at Risk (VaR) calculation
- Conditional Value at Risk (CVaR)
- Portfolio return distribution analysis
- Correlation and covariance modeling

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the complete simulation:

```bash
python main.py
```

This will execute all four simulation types with sample data and generate visualizations.

### Custom Usage

```python
from main import MonteCarloSimulation

# Initialize simulation
mc = MonteCarloSimulation(seed=42)

# Stock price simulation
S, t = mc.simulate_stock_price(
    S0=100,      # Initial price
    mu=0.08,     # Expected return (8%)
    sigma=0.2,   # Volatility (20%)
    T=1.0,       # Time horizon (1 year)
    n_steps=252, # Daily steps
    n_simulations=1000
)

# Option pricing
call_price = mc.price_european_option(
    S0=100,      # Current stock price
    K=100,       # Strike price
    T=1.0,       # Time to maturity
    r=0.05,      # Risk-free rate
    sigma=0.2,   # Volatility
    option_type='call',
    n_simulations=50000
)

# Portfolio optimization
returns_data = your_returns_data  # Shape: (n_assets, n_periods)
optimization_results = mc.optimize_portfolio_monte_carlo(
    returns_data,
    n_portfolios=10000,
    risk_free_rate=0.02
)
```

## Mathematical Background

### Geometric Brownian Motion
The stock price follows the stochastic differential equation:
```
dS = μS dt + σS dW
```
where:
- S = Stock price
- μ = Drift (expected return)
- σ = Volatility
- W = Wiener process

### Option Pricing
European options are priced using risk-neutral valuation:
```
C = e^(-rT) * E[max(S_T - K, 0)]  # Call option
P = e^(-rT) * E[max(K - S_T, 0)]  # Put option
```

### Portfolio Optimization
The Sharpe ratio is maximized:
```
Sharpe Ratio = (E[R_p] - R_f) / σ_p
```
where:
- R_p = Portfolio return
- R_f = Risk-free rate
- σ_p = Portfolio volatility

## Output Examples

### Stock Price Simulation
- Multiple price paths with confidence intervals
- Statistical summary of final prices
- Probability of loss calculation

### Option Pricing
- Option prices with confidence intervals
- Comparison between call and put options
- Error estimation

### Portfolio Optimization
- Efficient frontier visualization
- Optimal portfolio weights
- Risk-return characteristics

### Risk Analysis
- VaR and CVaR calculations
- Return distribution histograms
- Risk metrics summary

## Customization

### Parameters
- **Simulation parameters**: Number of paths, time steps, seed
- **Market parameters**: Volatility, drift, risk-free rate
- **Option parameters**: Strike price, maturity, option type
- **Portfolio parameters**: Asset weights, correlation structure

### Extensions
The code can be extended for:
- American options pricing
- Exotic options (barriers, Asians, etc.)
- Multi-asset portfolios
- Different stochastic processes (Heston, etc.)
- Real-time data integration

## Dependencies

- **numpy**: Numerical computations
- **matplotlib**: Plotting and visualization
- **pandas**: Data manipulation
- **scipy**: Statistical functions
- **seaborn**: Enhanced plotting

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to contribute by:
- Adding new simulation types
- Improving visualization
- Optimizing performance
- Adding more documentation
- Reporting bugs or issues 