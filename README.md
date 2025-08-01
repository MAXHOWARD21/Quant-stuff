# Quantitative Finance Tools

A comprehensive Python implementation of quantitative finance tools including Monte Carlo simulations and multi-factor stock rating models with real-time data fetching capabilities.

## Features

### 1. Live Multi-Factor Stock Rating Model (`stock_rater_live.py`)
- **Real-time stock rating system** (1.0-10.0 scale) for any ticker symbol
- **Live data fetching** from Yahoo Finance API
- **Six factor categories**: Value, Quality, Growth, Momentum, Risk, Sentiment
- **Customizable factor weights** for different investment strategies
- **Instant analysis** - just enter any stock ticker and get ratings
- **Comprehensive factor breakdown** with detailed recommendations
- **Current market data** including price, P/E ratio, market cap, and more

### 2. Multi-Factor Stock Rating Model (`Max.py`)
- **Comprehensive stock rating system** (1.0-10.0 scale)
- **Six factor categories**: Value, Quality, Growth, Momentum, Risk, Sentiment
- **Customizable factor weights** for different investment strategies
- **Sector analysis** and comparison
- **Visual factor breakdown** with radar charts and heatmaps
- **Risk-adjusted ratings** and performance metrics

### 3. Monte Carlo Simulation (`main.py`)
- **Geometric Brownian Motion (GBM)** implementation
- Multiple simulation paths with confidence intervals
- Visual representation of price evolution
- Statistical analysis of final price distributions

### 4. European Option Pricing
- Monte Carlo pricing for both call and put options
- Confidence intervals for price estimates
- Risk-neutral valuation framework
- Support for various option parameters

### 5. Portfolio Optimization
- Monte Carlo portfolio weight optimization
- Efficient frontier analysis
- Maximum Sharpe ratio portfolio
- Minimum volatility portfolio
- Risk-return analysis

### 6. Risk Analysis
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

### Live Stock Rater (Recommended)

The easiest way to rate any stock with real-time data:

```bash
python3 stock_rater_live.py
```

**How it works:**
1. Run the program
2. Choose option 1 (Rate a stock)
3. Enter any stock ticker symbol (e.g., "AAPL", "TSLA", "MSFT", "GOOGL")
4. Get instant rating and analysis with live market data

**Example output:**
```
🎯 LIVE STOCK RATER
==================================================
Rate any stock using real-time market data!
==================================================

Options:
1. Rate a stock
2. Exit

Enter your choice (1-2): 1

Enter stock ticker symbol: AMZN

📊 Fetching live data for AMZN...
✅ Successfully fetched data for AMZN
============================================================
STOCK RATING: AMZN
============================================================
OVERALL RATING: 5.67/10
============================================================
FACTOR BREAKDOWN:
----------------------------------------
Value      : 6.07/10 (Weight: 25.0%)
Quality    : 2.76/10 (Weight: 20.0%)
Growth     : 8.00/10 (Weight: 20.0%)
Momentum   : 5.99/10 (Weight: 15.0%)
Risk       : 5.67/10 (Weight: 10.0%)
Sentiment  : 5.35/10 (Weight: 10.0%)
RECOMMENDATION:
----------------------------------------
  WEAK - Consider selling

STOCK DETAILS:
----------------------------------------
Current Price: $234.11
Market Cap: $2485.41B
P/E Ratio: 38.07
P/B Ratio: 8.12
ROE: 0.25%
ROA: 0.08%
Revenue Growth: 0.09%
Earnings Growth: 0.62%
Beta: 1.34
Volatility: 0.34%
Sector: Consumer Cyclical
```

### Multi-Factor Stock Rating Model

Run the stock rating model with sample data:

```bash
python Max.py
```

This will analyze sample stocks and provide comprehensive ratings with visualizations.

### Monte Carlo Simulation

Run the complete simulation:

```bash
python main.py
```

This will execute all simulation types with sample data and generate visualizations.

### Example Usage

Run the example usage script:

```bash
python example_usage.py
```

This demonstrates custom stock data, different factor weights, and sector analysis.

### Multi-Factor Model Usage

```python
from Max import MultiFactorModel, StockData

# Initialize model
model = MultiFactorModel()

# Create stock data
stock = StockData(
    ticker="AAPL",
    price=150.0,
    market_cap=2.5e12,
    pe_ratio=25.0,
    pb_ratio=15.0,
    debt_to_equity=1.2,
    current_ratio=1.8,
    roe=0.25,
    roa=0.15,
    revenue_growth=0.08,
    earnings_growth=0.12,
    dividend_yield=0.005,
    beta=1.1,
    volatility=0.20,
    sharpe_ratio=1.2,
    momentum=0.15,
    volume=1e8,
    institutional_ownership=0.75,
    analyst_rating=4.2,
    sector="Technology"
)

# Rate the stock
rating = model.rate_stock(stock)
print(f"Overall Rating: {rating['overall_rating']:.2f}/10")

# Custom factor weights
growth_weights = {
    'value': 0.10,
    'quality': 0.15,
    'growth': 0.40,
    'momentum': 0.20,
    'risk': 0.10,
    'sentiment': 0.05
}

growth_model = MultiFactorModel(growth_weights)
growth_rating = growth_model.rate_stock(stock)
```

### Monte Carlo Simulation Usage

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

### Multi-Factor Model
The overall stock rating is calculated as a weighted average of factor scores:
```
Overall Rating = Σ(w_i × FactorScore_i)
```
where:
- w_i = Weight for factor i
- FactorScore_i = Score (0-10) for factor i

#### Factor Categories:
1. **Value**: P/E ratio, P/B ratio, market cap, dividend yield
2. **Quality**: ROE, ROA, current ratio, debt-to-equity
3. **Growth**: Revenue growth, earnings growth
4. **Momentum**: Price momentum, trading volume
5. **Risk**: Beta, volatility, Sharpe ratio
6. **Sentiment**: Analyst ratings, institutional ownership

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

### Live Stock Rating
- Real-time stock ratings (1.0-10.0 scale) with factor breakdowns
- Current market data and financial metrics
- Buy/hold/sell recommendations
- Detailed factor analysis for any ticker symbol

### Multi-Factor Stock Rating
- Overall ratings (1.0-10.0 scale) with factor breakdowns
- Radar charts showing factor performance
- Heatmaps comparing multiple stocks
- Sector analysis and rankings
- Risk-adjusted performance metrics

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

### Multi-Factor Model Parameters
- **Factor weights**: Customize importance of each factor category
- **Scoring thresholds**: Adjust scoring criteria for different market conditions
- **Sector-specific weights**: Different weights for different sectors
- **Time periods**: Adjust for different analysis periods

### Monte Carlo Parameters
- **Simulation parameters**: Number of paths, time steps, seed
- **Market parameters**: Volatility, drift, risk-free rate
- **Option parameters**: Strike price, maturity, option type
- **Portfolio parameters**: Asset weights, correlation structure

### Extensions
The code can be extended for:
- **Multi-Factor Model**: Additional factors, machine learning integration, real-time data feeds
- **Monte Carlo**: American options pricing, exotic options (barriers, Asians, etc.), multi-asset portfolios, different stochastic processes (Heston, etc.)
- **General**: Real-time data integration, backtesting frameworks, risk management tools

## Dependencies

- **numpy**: Numerical computations
- **matplotlib**: Plotting and visualization
- **pandas**: Data manipulation
- **scipy**: Statistical functions
- **seaborn**: Enhanced plotting
- **yfinance**: Real-time stock data fetching
- **requests**: HTTP requests for data APIs

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to contribute by:
- Adding new simulation types
- Improving visualization
- Optimizing performance
- Adding more documentation
- Reporting bugs or issues 
