# Quantitative Finance Stuff

## Features so far

- **Live Stock Rater** - Real-time stock ratings (1-10 scale) for any ticker using Yahoo Finance
- **Multi-Factor Rating Model** - Comprehensive stock analysis across 6 factor categories
- **Monte Carlo Simulation** - Stock price evolution and option pricing
- **Portfolio Optimization** - Efficient frontier and risk analysis

## Quick Start

Install dependencies:
```bash
pip install -r requirements.txt
```

Rate any stock with live data:
```bash
python3 stock_rater_live.py
```

Run full analysis:
```bash
python Max.py  
python main.py 
```

## Live Stock Rater Example

```
Enter stock ticker symbol: AAPL

STOCK RATING: AAPL
OVERALL RATING: 7.45/10

FACTOR BREAKDOWN:
Value      : 6.2/10 (Weight: 25%)
Quality    : 8.1/10 (Weight: 20%)
Growth     : 7.8/10 (Weight: 20%)
Momentum   : 6.9/10 (Weight: 15%)
Risk       : 7.3/10 (Weight: 10%)
Sentiment  : 8.5/10 (Weight: 10%)

RECOMMENDATION: BUY - Strong fundamentals
```

## Multi-Factor Model

Six factor categories with customizable weights:
- **Value**: P/E, P/B ratios, market cap, dividend yield
- **Quality**: ROE, ROA, debt ratios, current ratio
- **Growth**: Revenue and earnings growth rates
- **Momentum**: Price momentum, trading volume
- **Risk**: Beta, volatility, Sharpe ratio
- **Sentiment**: Analyst ratings, institutional ownership

## Usage Examples

```python
# Stock rating with custom weights
from Max import MultiFactorModel

growth_weights = {'growth': 0.4, 'quality': 0.3, 'value': 0.3}
model = MultiFactorModel(growth_weights)
rating = model.rate_stock(stock_data)

# Monte Carlo simulation
from main import MonteCarloSimulation

mc = MonteCarloSimulation()
prices, times = mc.simulate_stock_price(S0=100, mu=0.08, sigma=0.2, T=1)
option_price = mc.price_european_option(S0=100, K=100, T=1, r=0.05, sigma=0.2)
```

## Key Dependencies

- numpy, pandas, matplotlib, scipy
- yfinance (real-time data)
- seaborn (visualizations)

## Extensions

- Custom factor weights for different strategies
- Real-time data integration
- Portfolio optimization and risk analysis
- Option pricing and Monte Carlo methods
