import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

@dataclass
class StockData:
    """Data class to hold stock financial metrics."""
    ticker: str
    price: float
    market_cap: float
    pe_ratio: float
    pb_ratio: float
    debt_to_equity: float
    current_ratio: float
    roe: float  # Return on Equity
    roa: float  # Return on Assets
    revenue_growth: float
    earnings_growth: float
    dividend_yield: float
    beta: float
    volatility: float
    sharpe_ratio: float
    momentum: float  # Price momentum (6-month)
    volume: float
    institutional_ownership: float
    analyst_rating: float  # Average analyst rating (1-5 scale)
    sector: str

class MultiFactorModel:
    """
    A comprehensive multi-factor model for stock rating and analysis.
    Rates stocks on a scale of 1.0 to 10.0 based on multiple financial factors.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the multi-factor model with factor weights.
        
        Args:
            weights (Dict[str, float]): Dictionary of factor weights (optional)
        """
        # Default factor weights (can be customized)
        self.default_weights = {
            'value': 0.25,      # Value factors (P/E, P/B, etc.)
            'quality': 0.20,    # Quality factors (ROE, ROA, etc.)
            'growth': 0.20,     # Growth factors (revenue, earnings growth)
            'momentum': 0.15,   # Momentum factors (price momentum, volume)
            'risk': 0.10,       # Risk factors (volatility, beta, debt)
            'sentiment': 0.10   # Sentiment factors (analyst ratings, institutional ownership)
        }
        
        self.weights = weights if weights else self.default_weights
        self.scaler = StandardScaler()
        self.factor_scores = {}
        self.overall_scores = {}
        
    def calculate_value_score(self, stock_data: StockData, industry_avg: Dict[str, float]) -> float:
        """
        Calculate value factor score based on valuation metrics.
        
        Args:
            stock_data: Stock financial data
            industry_avg: Industry average metrics
            
        Returns:
            float: Value score (0-10)
        """
        scores = []
        
        # P/E Ratio Score (lower is better)
        if stock_data.pe_ratio > 0 and stock_data.pe_ratio < 50:
            pe_score = max(0, 10 - (stock_data.pe_ratio / 5))
            scores.append(pe_score)
        
        # P/B Ratio Score (lower is better)
        if stock_data.pb_ratio > 0 and stock_data.pb_ratio < 10:
            pb_score = max(0, 10 - (stock_data.pb_ratio / 1))
            scores.append(pb_score)
        
        # Market Cap Score (larger companies get higher scores)
        if stock_data.market_cap > 0:
            if stock_data.market_cap >= 10e9:  # Large cap
                cap_score = 10
            elif stock_data.market_cap >= 2e9:  # Mid cap
                cap_score = 8
            elif stock_data.market_cap >= 300e6:  # Small cap
                cap_score = 6
            else:  # Micro cap
                cap_score = 4
            scores.append(cap_score)
        
        # Dividend Yield Score (higher is better, but not too high)
        if stock_data.dividend_yield > 0:
            if 0.02 <= stock_data.dividend_yield <= 0.06:  # Sweet spot
                div_score = 10
            elif stock_data.dividend_yield > 0.06:
                div_score = max(5, 10 - (stock_data.dividend_yield - 0.06) * 50)
            else:
                div_score = stock_data.dividend_yield * 500
            scores.append(div_score)
        
        return np.mean(scores) if scores else 5.0
    
    def calculate_quality_score(self, stock_data: StockData) -> float:
        """
        Calculate quality factor score based on profitability and efficiency metrics.
        
        Args:
            stock_data: Stock financial data
            
        Returns:
            float: Quality score (0-10)
        """
        scores = []
        
        # ROE Score (higher is better)
        if stock_data.roe > 0:
            roe_score = min(10, stock_data.roe * 10)  # 10% ROE = 10 points
            scores.append(roe_score)
        
        # ROA Score (higher is better)
        if stock_data.roa > 0:
            roa_score = min(10, stock_data.roa * 20)  # 5% ROA = 10 points
            scores.append(roa_score)
        
        # Current Ratio Score (1.5-3.0 is ideal)
        if stock_data.current_ratio > 0:
            if 1.5 <= stock_data.current_ratio <= 3.0:
                cr_score = 10
            elif stock_data.current_ratio > 3.0:
                cr_score = max(5, 10 - (stock_data.current_ratio - 3.0) * 2)
            else:
                cr_score = max(0, stock_data.current_ratio * 6.67)
            scores.append(cr_score)
        
        # Debt-to-Equity Score (lower is better)
        if stock_data.debt_to_equity >= 0:
            if stock_data.debt_to_equity <= 0.5:
                de_score = 10
            elif stock_data.debt_to_equity <= 1.0:
                de_score = 8
            elif stock_data.debt_to_equity <= 2.0:
                de_score = 5
            else:
                de_score = max(0, 10 - stock_data.debt_to_equity * 2)
            scores.append(de_score)
        
        return np.mean(scores) if scores else 5.0
    
    def calculate_growth_score(self, stock_data: StockData) -> float:
        """
        Calculate growth factor score based on growth metrics.
        
        Args:
            stock_data: Stock financial data
            
        Returns:
            float: Growth score (0-10)
        """
        scores = []
        
        # Revenue Growth Score
        if stock_data.revenue_growth > -1:  # Allow for negative growth
            if stock_data.revenue_growth >= 0.20:  # 20%+ growth
                rev_score = 10
            elif stock_data.revenue_growth >= 0.10:  # 10-20% growth
                rev_score = 8
            elif stock_data.revenue_growth >= 0.05:  # 5-10% growth
                rev_score = 6
            elif stock_data.revenue_growth >= 0:  # 0-5% growth
                rev_score = 4
            else:  # Negative growth
                rev_score = max(0, 2 + stock_data.revenue_growth * 10)
            scores.append(rev_score)
        
        # Earnings Growth Score
        if stock_data.earnings_growth > -1:
            if stock_data.earnings_growth >= 0.25:  # 25%+ growth
                earn_score = 10
            elif stock_data.earnings_growth >= 0.15:  # 15-25% growth
                earn_score = 8
            elif stock_data.earnings_growth >= 0.08:  # 8-15% growth
                earn_score = 6
            elif stock_data.earnings_growth >= 0:  # 0-8% growth
                earn_score = 4
            else:  # Negative growth
                earn_score = max(0, 2 + stock_data.earnings_growth * 10)
            scores.append(earn_score)
        
        return np.mean(scores) if scores else 5.0
    
    def calculate_momentum_score(self, stock_data: StockData) -> float:
        """
        Calculate momentum factor score based on price and volume momentum.
        
        Args:
            stock_data: Stock financial data
            
        Returns:
            float: Momentum score (0-10)
        """
        scores = []
        
        # Price Momentum Score (6-month)
        if stock_data.momentum > -1:
            if stock_data.momentum >= 0.30:  # 30%+ momentum
                mom_score = 10
            elif stock_data.momentum >= 0.20:  # 20-30% momentum
                mom_score = 8
            elif stock_data.momentum >= 0.10:  # 10-20% momentum
                mom_score = 6
            elif stock_data.momentum >= 0:  # 0-10% momentum
                mom_score = 4
            else:  # Negative momentum
                mom_score = max(0, 2 + stock_data.momentum * 10)
            scores.append(mom_score)
        
        # Volume Score (higher volume = higher score)
        if stock_data.volume > 0:
            # Normalize volume (assuming typical range)
            vol_score = min(10, stock_data.volume / 1e6)  # 1M volume = 10 points
            scores.append(vol_score)
        
        return np.mean(scores) if scores else 5.0
    
    def calculate_risk_score(self, stock_data: StockData) -> float:
        """
        Calculate risk factor score (inverse - lower risk = higher score).
        
        Args:
            stock_data: Stock financial data
            
        Returns:
            float: Risk score (0-10, higher = lower risk)
        """
        scores = []
        
        # Beta Score (lower beta = lower risk)
        if stock_data.beta > 0:
            if stock_data.beta <= 0.8:  # Low beta
                beta_score = 10
            elif stock_data.beta <= 1.2:  # Market beta
                beta_score = 8
            elif stock_data.beta <= 1.5:  # High beta
                beta_score = 5
            else:  # Very high beta
                beta_score = max(0, 10 - stock_data.beta * 2)
            scores.append(beta_score)
        
        # Volatility Score (lower volatility = lower risk)
        if stock_data.volatility > 0:
            if stock_data.volatility <= 0.15:  # Low volatility
                vol_score = 10
            elif stock_data.volatility <= 0.25:  # Medium volatility
                vol_score = 7
            elif stock_data.volatility <= 0.35:  # High volatility
                vol_score = 4
            else:  # Very high volatility
                vol_score = max(0, 10 - stock_data.volatility * 20)
            scores.append(vol_score)
        
        # Sharpe Ratio Score (higher = better risk-adjusted returns)
        if stock_data.sharpe_ratio > -10:  # Allow for negative Sharpe
            if stock_data.sharpe_ratio >= 1.0:  # Excellent
                sharpe_score = 10
            elif stock_data.sharpe_ratio >= 0.5:  # Good
                sharpe_score = 8
            elif stock_data.sharpe_ratio >= 0:  # Positive
                sharpe_score = 6
            else:  # Negative
                sharpe_score = max(0, 3 + stock_data.sharpe_ratio * 3)
            scores.append(sharpe_score)
        
        return np.mean(scores) if scores else 5.0
    
    def calculate_sentiment_score(self, stock_data: StockData) -> float:
        """
        Calculate sentiment factor score based on analyst ratings and institutional ownership.
        
        Args:
            stock_data: Stock financial data
            
        Returns:
            float: Sentiment score (0-10)
        """
        scores = []
        
        # Analyst Rating Score (1-5 scale, convert to 0-10)
        if 1 <= stock_data.analyst_rating <= 5:
            analyst_score = stock_data.analyst_rating * 2
            scores.append(analyst_score)
        
        # Institutional Ownership Score
        if stock_data.institutional_ownership > 0:
            if stock_data.institutional_ownership >= 0.80:  # 80%+ institutional
                inst_score = 10
            elif stock_data.institutional_ownership >= 0.60:  # 60-80%
                inst_score = 8
            elif stock_data.institutional_ownership >= 0.40:  # 40-60%
                inst_score = 6
            else:  # <40%
                inst_score = stock_data.institutional_ownership * 15
            scores.append(inst_score)
        
        return np.mean(scores) if scores else 5.0
    
    def rate_stock(self, stock_data: StockData, industry_avg: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Rate a stock on a scale of 1.0 to 10.0 based on all factors.
        
        Args:
            stock_data: Stock financial data
            industry_avg: Industry average metrics (optional)
            
        Returns:
            Dict[str, float]: Dictionary with factor scores and overall rating
        """
        if industry_avg is None:
            industry_avg = {}
        
        # Calculate individual factor scores
        value_score = self.calculate_value_score(stock_data, industry_avg)
        quality_score = self.calculate_quality_score(stock_data)
        growth_score = self.calculate_growth_score(stock_data)
        momentum_score = self.calculate_momentum_score(stock_data)
        risk_score = self.calculate_risk_score(stock_data)
        sentiment_score = self.calculate_sentiment_score(stock_data)
        
        # Calculate weighted overall score
        overall_score = (
            value_score * self.weights['value'] +
            quality_score * self.weights['quality'] +
            growth_score * self.weights['growth'] +
            momentum_score * self.weights['momentum'] +
            risk_score * self.weights['risk'] +
            sentiment_score * self.weights['sentiment']
        )
        
        # Ensure score is within 1.0-10.0 range
        overall_score = max(1.0, min(10.0, overall_score))
        
        return {
            'ticker': stock_data.ticker,
            'overall_rating': overall_score,
            'value_score': value_score,
            'quality_score': quality_score,
            'growth_score': growth_score,
            'momentum_score': momentum_score,
            'risk_score': risk_score,
            'sentiment_score': sentiment_score,
            'factor_weights': self.weights.copy()
        }
    
    def rate_multiple_stocks(self, stocks_data: List[StockData], 
                           industry_avgs: Optional[Dict[str, Dict[str, float]]] = None) -> pd.DataFrame:
        """
        Rate multiple stocks and return results as a DataFrame.
        
        Args:
            stocks_data: List of stock data objects
            industry_avgs: Dictionary of industry averages by sector
            
        Returns:
            pd.DataFrame: DataFrame with ratings for all stocks
        """
        if industry_avgs is None:
            industry_avgs = {}
        
        results = []
        for stock in stocks_data:
            industry_avg = industry_avgs.get(stock.sector, {})
            rating = self.rate_stock(stock, industry_avg)
            results.append(rating)
        
        df = pd.DataFrame(results)
        df = df.sort_values('overall_rating', ascending=False)
        
        return df
    
    def plot_factor_breakdown(self, stock_rating: Dict[str, float]):
        """
        Plot the factor breakdown for a stock rating.
        
        Args:
            stock_rating: Stock rating dictionary
        """
        factors = ['value_score', 'quality_score', 'growth_score', 
                  'momentum_score', 'risk_score', 'sentiment_score']
        scores = [stock_rating[factor] for factor in factors]
        factor_names = ['Value', 'Quality', 'Growth', 'Momentum', 'Risk', 'Sentiment']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Radar chart
        angles = np.linspace(0, 2 * np.pi, len(factors), endpoint=False).tolist()
        scores += scores[:1]  # Complete the circle
        angles += angles[:1]
        
        ax1.plot(angles, scores, 'o-', linewidth=2, label=stock_rating['ticker'])
        ax1.fill(angles, scores, alpha=0.25)
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(factor_names)
        ax1.set_ylim(0, 10)
        ax1.set_title(f'Factor Breakdown - {stock_rating["ticker"]}')
        ax1.grid(True)
        
        # Bar chart
        bars = ax2.bar(factor_names, scores[:-1], color='skyblue', alpha=0.7)
        ax2.set_ylabel('Score (0-10)')
        ax2.set_title(f'Factor Scores - Overall Rating: {stock_rating["overall_rating"]:.2f}')
        ax2.set_ylim(0, 10)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores[:-1]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{score:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def plot_comparison(self, ratings_df: pd.DataFrame, top_n: int = 10):
        """
        Plot comparison of top-rated stocks.
        
        Args:
            ratings_df: DataFrame with stock ratings
            top_n: Number of top stocks to display
        """
        top_stocks = ratings_df.head(top_n)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Overall ratings
        bars1 = ax1.barh(top_stocks['ticker'], top_stocks['overall_rating'], 
                        color='lightgreen', alpha=0.7)
        ax1.set_xlabel('Overall Rating')
        ax1.set_title(f'Top {top_n} Stocks - Overall Ratings')
        ax1.set_xlim(0, 10)
        
        # Add value labels
        for bar, rating in zip(bars1, top_stocks['overall_rating']):
            ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{rating:.2f}', ha='left', va='center')
        
        # Factor comparison heatmap
        factor_cols = ['value_score', 'quality_score', 'growth_score', 
                      'momentum_score', 'risk_score', 'sentiment_score']
        factor_names = ['Value', 'Quality', 'Growth', 'Momentum', 'Risk', 'Sentiment']
        
        heatmap_data = top_stocks[factor_cols].values
        im = ax2.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=10)
        
        ax2.set_xticks(range(len(factor_names)))
        ax2.set_xticklabels(factor_names, rotation=45)
        ax2.set_yticks(range(len(top_stocks)))
        ax2.set_yticklabels(top_stocks['ticker'])
        ax2.set_title('Factor Scores Heatmap')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Score (0-10)')
        
        plt.tight_layout()
        plt.show()

def create_sample_stocks() -> List[StockData]:
    """Create sample stock data for demonstration."""
    stocks = [
        # Tech stocks
        StockData("AAPL", 150.0, 2.5e12, 25.0, 15.0, 1.2, 1.8, 0.25, 0.15, 0.08, 0.12, 0.005, 1.1, 0.20, 1.2, 0.15, 1e8, 0.75, 4.2, "Technology"),
        StockData("MSFT", 300.0, 2.2e12, 30.0, 12.0, 0.8, 2.1, 0.30, 0.18, 0.12, 0.18, 0.008, 0.9, 0.18, 1.5, 0.20, 8e7, 0.80, 4.5, "Technology"),
        StockData("GOOGL", 2800.0, 1.8e12, 28.0, 6.0, 0.3, 3.2, 0.22, 0.12, 0.15, 0.20, 0.000, 1.0, 0.22, 1.1, 0.25, 6e7, 0.85, 4.3, "Technology"),
        
        # Financial stocks
        StockData("JPM", 140.0, 400e9, 12.0, 1.5, 1.5, 1.3, 0.12, 0.008, 0.05, 0.08, 0.025, 1.2, 0.25, 0.8, 0.05, 5e7, 0.70, 4.0, "Financial"),
        StockData("BAC", 35.0, 280e9, 10.0, 1.2, 1.8, 1.1, 0.10, 0.006, 0.03, 0.06, 0.030, 1.3, 0.28, 0.6, 0.03, 4e7, 0.65, 3.8, "Financial"),
        
        # Healthcare stocks
        StockData("JNJ", 160.0, 400e9, 18.0, 4.0, 0.4, 1.6, 0.20, 0.10, 0.06, 0.08, 0.025, 0.7, 0.16, 1.3, 0.08, 3e7, 0.75, 4.1, "Healthcare"),
        StockData("PFE", 45.0, 250e9, 15.0, 2.5, 0.6, 1.4, 0.15, 0.08, 0.04, 0.05, 0.035, 0.8, 0.20, 1.0, 0.02, 2e7, 0.70, 3.9, "Healthcare"),
        
        # Consumer stocks
        StockData("KO", 55.0, 240e9, 22.0, 10.0, 1.0, 1.2, 0.35, 0.12, 0.04, 0.06, 0.030, 0.6, 0.18, 1.4, 0.05, 2e7, 0.65, 4.2, "Consumer"),
        StockData("PG", 140.0, 330e9, 20.0, 4.5, 0.5, 1.5, 0.25, 0.10, 0.03, 0.05, 0.025, 0.5, 0.16, 1.6, 0.03, 1e7, 0.60, 4.0, "Consumer"),
        
        # Energy stocks
        StockData("XOM", 90.0, 350e9, 8.0, 1.8, 0.3, 1.8, 0.18, 0.08, 0.15, 0.20, 0.040, 1.1, 0.30, 0.9, 0.10, 3e7, 0.75, 3.7, "Energy"),
        StockData("CVX", 150.0, 280e9, 10.0, 1.5, 0.4, 1.6, 0.15, 0.07, 0.12, 0.15, 0.035, 1.0, 0.25, 1.0, 0.08, 2e7, 0.70, 3.8, "Energy"),
        
        # Growth stocks
        StockData("TSLA", 800.0, 800e9, 150.0, 25.0, 0.2, 1.3, 0.08, 0.05, 0.50, 0.80, 0.000, 2.0, 0.60, 0.5, 0.40, 1e8, 0.60, 3.5, "Automotive"),
        StockData("NVDA", 500.0, 1.2e12, 80.0, 35.0, 0.1, 2.5, 0.25, 0.15, 0.40, 0.60, 0.002, 1.8, 0.50, 0.8, 0.35, 9e7, 0.80, 4.4, "Technology"),
        
        # Value stocks
        StockData("BRK.A", 450000.0, 650e9, 8.0, 1.2, 0.2, 2.0, 0.12, 0.08, 0.02, 0.03, 0.000, 0.9, 0.20, 1.2, 0.02, 1e6, 0.85, 4.1, "Financial"),
        StockData("WMT", 140.0, 380e9, 18.0, 3.5, 0.8, 1.4, 0.15, 0.08, 0.04, 0.06, 0.015, 0.6, 0.18, 1.3, 0.03, 2e7, 0.70, 4.0, "Consumer")
    ]
    
    return stocks

def main():
    """Main function to demonstrate the multi-factor model."""
    print("=== Multi-Factor Stock Rating Model ===\n")
    
    # Initialize the model
    model = MultiFactorModel()
    
    # Create sample stock data
    stocks = create_sample_stocks()
    
    print(f"Analyzing {len(stocks)} stocks...\n")
    
    # Rate all stocks
    ratings_df = model.rate_multiple_stocks(stocks)
    
    # Display results
    print("Top 10 Rated Stocks:")
    print("=" * 80)
    print(f"{'Rank':<4} {'Ticker':<8} {'Rating':<8} {'Value':<6} {'Quality':<8} {'Growth':<7} {'Momentum':<9} {'Risk':<6} {'Sentiment':<9}")
    print("-" * 80)
    
    for i, (_, row) in enumerate(ratings_df.head(10).iterrows(), 1):
        print(f"{i:<4} {row['ticker']:<8} {row['overall_rating']:<8.2f} {row['value_score']:<6.1f} "
              f"{row['quality_score']:<8.1f} {row['growth_score']:<7.1f} {row['momentum_score']:<9.1f} "
              f"{row['risk_score']:<6.1f} {row['sentiment_score']:<9.1f}")
    
    print("\n" + "=" * 80)
    
    # Show detailed analysis for top stock
    top_stock = ratings_df.iloc[0]
    print(f"\nDetailed Analysis - {top_stock['ticker']} (Rating: {top_stock['overall_rating']:.2f})")
    print("-" * 50)
    
    factor_weights = top_stock['factor_weights']
    print("Factor Breakdown:")
    for factor, weight in factor_weights.items():
        score = top_stock[f'{factor}_score']
        print(f"  {factor.title():<10}: {score:.2f}/10 (Weight: {weight:.1%})")
    
    # Plot factor breakdown for top stock
    model.plot_factor_breakdown(top_stock.to_dict())
    
    # Plot comparison of top stocks
    model.plot_comparison(ratings_df, top_n=10)
    
    # Sector analysis
    print("\nSector Analysis:")
    print("-" * 30)
    sector_avg = ratings_df.groupby('sector')['overall_rating'].agg(['mean', 'count']).round(2)
    sector_avg = sector_avg.sort_values('mean', ascending=False)
    
    for sector, (avg_rating, count) in sector_avg.iterrows():
        print(f"{sector:<15}: {avg_rating:.2f}/10 ({count} stocks)")
    
    # Risk-return analysis
    print("\nRisk-Return Analysis:")
    print("-" * 30)
    
    # Calculate risk-adjusted ratings (rating / risk_score)
    ratings_df['risk_adjusted_rating'] = ratings_df['overall_rating'] / ratings_df['risk_score']
    risk_adjusted_top = ratings_df.nlargest(5, 'risk_adjusted_rating')
    
    print("Top 5 Risk-Adjusted Stocks:")
    for i, (_, row) in enumerate(risk_adjusted_top.iterrows(), 1):
        print(f"{i}. {row['ticker']}: {row['overall_rating']:.2f}/10 (Risk Score: {row['risk_score']:.1f})")
    
    print("\n=== Analysis Complete ===")
    
    return ratings_df

if __name__ == "__main__":
    results = main()
