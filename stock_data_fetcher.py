#!/usr/bin/env python3
"""
Real-time Stock Data Fetcher

This module provides functionality to fetch real-time stock data from various sources.
Note: Some APIs require API keys for real-time data access.
"""

import requests
import time
import json
from typing import Dict, Optional, List
from Max import StockData
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class StockDataFetcher:
    """
    A class to fetch real-time stock data from various sources.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the stock data fetcher.
        
        Args:
            api_key (str, optional): API key for premium data sources
        """
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_from_yahoo_finance(self, ticker: str) -> Optional[StockData]:
        """
        Fetch stock data from Yahoo Finance using yfinance library.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            StockData: Stock data object or None if failed
        """
        try:
            # Get stock info
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get historical data for calculations
            hist = stock.history(period="1y")
            
            if hist.empty:
                return None
            
            # Calculate derived metrics
            current_price = info.get('currentPrice', hist['Close'].iloc[-1])
            market_cap = info.get('marketCap', 0)
            
            # Calculate volatility (annualized)
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * (252 ** 0.5)
            
            # Calculate momentum (6-month return)
            if len(hist) >= 126:  # 6 months of trading days
                momentum = (hist['Close'].iloc[-1] / hist['Close'].iloc[-126]) - 1
            else:
                momentum = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1
            
            # Calculate Sharpe ratio (simplified)
            risk_free_rate = 0.02  # Assume 2% risk-free rate
            excess_returns = returns - risk_free_rate/252
            sharpe_ratio = excess_returns.mean() / returns.std() * (252 ** 0.5)
            
            # Get volume
            avg_volume = hist['Volume'].mean()
            
            # Create StockData object with available data
            stock_data = StockData(
                ticker=ticker.upper(),
                price=current_price,
                market_cap=market_cap,
                pe_ratio=info.get('trailingPE', 20.0),
                pb_ratio=info.get('priceToBook', 3.0),
                debt_to_equity=info.get('debtToEquity', 0.5),
                current_ratio=info.get('currentRatio', 1.5),
                roe=info.get('returnOnEquity', 0.15),
                roa=info.get('returnOnAssets', 0.08),
                revenue_growth=info.get('revenueGrowth', 0.10),
                earnings_growth=info.get('earningsGrowth', 0.12),
                dividend_yield=info.get('dividendYield', 0.02),
                beta=info.get('beta', 1.0),
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                momentum=momentum,
                volume=avg_volume,
                institutional_ownership=info.get('heldPercentInstitutions', 0.70),
                analyst_rating=info.get('recommendationMean', 4.0),
                sector=info.get('sector', 'Unknown')
            )
            
            return stock_data
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def fetch_from_alpha_vantage(self, ticker: str) -> Optional[StockData]:
        """
        Fetch stock data from Alpha Vantage API (requires API key).
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            StockData: Stock data object or None if failed
        """
        if not self.api_key:
            print("Alpha Vantage API key required for this method")
            return None
        
        try:
            # Get company overview
            overview_url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={self.api_key}"
            overview_response = self.session.get(overview_url)
            overview_data = overview_response.json()
            
            if 'Error Message' in overview_data:
                print(f"Error: {overview_data['Error Message']}")
                return None
            
            # Get quote
            quote_url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={self.api_key}"
            quote_response = self.session.get(quote_url)
            quote_data = quote_response.json()
            
            if 'Error Message' in quote_data:
                print(f"Error: {quote_data['Error Message']}")
                return None
            
            quote = quote_data.get('Global Quote', {})
            
            # Create StockData object
            stock_data = StockData(
                ticker=ticker.upper(),
                price=float(quote.get('05. price', 100.0)),
                market_cap=float(overview_data.get('MarketCapitalization', 0)),
                pe_ratio=float(overview_data.get('PERatio', 20.0)),
                pb_ratio=float(overview_data.get('PriceToBookRatio', 3.0)),
                debt_to_equity=float(overview_data.get('DebtToEquityRatio', 0.5)),
                current_ratio=float(overview_data.get('CurrentRatio', 1.5)),
                roe=float(overview_data.get('ReturnOnEquityTTM', 0.15)),
                roa=float(overview_data.get('ReturnOnAssetsTTM', 0.08)),
                revenue_growth=float(overview_data.get('QuarterlyRevenueGrowthYOY', 0.10)),
                earnings_growth=float(overview_data.get('QuarterlyEarningsGrowthYOY', 0.12)),
                dividend_yield=float(overview_data.get('DividendYield', 0.02)),
                beta=float(overview_data.get('Beta', 1.0)),
                volatility=0.20,  # Would need additional API calls to calculate
                sharpe_ratio=1.0,  # Would need additional API calls to calculate
                momentum=0.10,  # Would need additional API calls to calculate
                volume=float(quote.get('06. volume', 1e7)),
                institutional_ownership=0.70,  # Not available in free API
                analyst_rating=4.0,  # Not available in free API
                sector=overview_data.get('Sector', 'Unknown')
            )
            
            return stock_data
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def fetch_from_finnhub(self, ticker: str) -> Optional[StockData]:
        """
        Fetch stock data from Finnhub API (requires API key).
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            StockData: Stock data object or None if failed
        """
        if not self.api_key:
            print("Finnhub API key required for this method")
            return None
        
        try:
            # Get company profile
            profile_url = f"https://finnhub.io/api/v1/stock/profile2?symbol={ticker}&token={self.api_key}"
            profile_response = self.session.get(profile_url)
            profile_data = profile_response.json()
            
            # Get quote
            quote_url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={self.api_key}"
            quote_response = self.session.get(quote_url)
            quote_data = quote_response.json()
            
            # Get financial metrics
            metrics_url = f"https://finnhub.io/api/v1/stock/metric?symbol={ticker}&metric=all&token={self.api_key}"
            metrics_response = self.session.get(metrics_url)
            metrics_data = metrics_response.json()
            
            if not profile_data or not quote_data:
                return None
            
            # Create StockData object
            stock_data = StockData(
                ticker=ticker.upper(),
                price=quote_data.get('c', 100.0),
                market_cap=profile_data.get('marketCapitalization', 0),
                pe_ratio=metrics_data.get('peBasicExclExtraTTM', 20.0),
                pb_ratio=metrics_data.get('pbAnnual', 3.0),
                debt_to_equity=metrics_data.get('totalDebtToEquityAnnual', 0.5),
                current_ratio=metrics_data.get('currentRatioAnnual', 1.5),
                roe=metrics_data.get('roeRfy', 0.15),
                roa=metrics_data.get('roaRfy', 0.08),
                revenue_growth=metrics_data.get('revenueGrowth', 0.10),
                earnings_growth=metrics_data.get('epsGrowth', 0.12),
                dividend_yield=metrics_data.get('dividendYieldIndicatedAnnual', 0.02),
                beta=metrics_data.get('beta', 1.0),
                volatility=0.20,  # Would need additional API calls
                sharpe_ratio=1.0,  # Would need additional API calls
                momentum=0.10,  # Would need additional API calls
                volume=quote_data.get('v', 1e7),
                institutional_ownership=0.70,  # Not available in free API
                analyst_rating=4.0,  # Not available in free API
                sector=profile_data.get('finnhubIndustry', 'Unknown')
            )
            
            return stock_data
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def fetch_stock_data(self, ticker: str, source: str = 'yahoo') -> Optional[StockData]:
        """
        Fetch stock data from the specified source.
        
        Args:
            ticker (str): Stock ticker symbol
            source (str): Data source ('yahoo', 'alpha_vantage', 'finnhub')
            
        Returns:
            StockData: Stock data object or None if failed
        """
        ticker = ticker.upper()
        
        if source == 'yahoo':
            return self.fetch_from_yahoo_finance(ticker)
        elif source == 'alpha_vantage':
            return self.fetch_from_alpha_vantage(ticker)
        elif source == 'finnhub':
            return self.fetch_from_finnhub(ticker)
        else:
            print(f"Unknown data source: {source}")
            return None
    
    def fetch_multiple_stocks(self, tickers: List[str], source: str = 'yahoo') -> List[StockData]:
        """
        Fetch data for multiple stocks.
        
        Args:
            tickers (List[str]): List of stock ticker symbols
            source (str): Data source
            
        Returns:
            List[StockData]: List of stock data objects
        """
        stocks_data = []
        
        for ticker in tickers:
            print(f"Fetching data for {ticker}...")
            stock_data = self.fetch_stock_data(ticker, source)
            if stock_data:
                stocks_data.append(stock_data)
            else:
                print(f"Failed to fetch data for {ticker}")
            
            # Add delay to avoid rate limiting
            time.sleep(1)
        
        return stocks_data

def main():
    """Example usage of the stock data fetcher."""
    print("=== Stock Data Fetcher Example ===")
    
    # Initialize fetcher
    fetcher = StockDataFetcher()
    
    # Example tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    print(f"Fetching data for: {', '.join(tickers)}")
    
    # Fetch data using Yahoo Finance (free, no API key required)
    stocks_data = fetcher.fetch_multiple_stocks(tickers, source='yahoo')
    
    if stocks_data:
        print(f"\nSuccessfully fetched data for {len(stocks_data)} stocks:")
        for stock in stocks_data:
            print(f"{stock.ticker}: ${stock.price:.2f}, Market Cap: ${stock.market_cap/1e9:.1f}B")
    else:
        print("Failed to fetch any stock data")

if __name__ == "__main__":
    main() 