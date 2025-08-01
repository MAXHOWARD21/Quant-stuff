#!/usr/bin/env python3
"""
Live Stock Rater - Terminal Interface
Rate any stock using real-time data from Yahoo Finance
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Max import MultiFactorModel, StockData
from stock_data_fetcher import StockDataFetcher

def display_rating(ticker, stock_data, model):
    """Display the stock rating in a formatted way."""
    print("=" * 60)
    print(f"STOCK RATING: {ticker.upper()}")
    print("=" * 60)
    
    # Rate the stock
    rating_result = model.rate_stock(stock_data)
    overall_score = rating_result['overall_rating']
    
    # Create factor scores dictionary
    factor_scores = {
        'value': rating_result['value_score'],
        'quality': rating_result['quality_score'],
        'growth': rating_result['growth_score'],
        'momentum': rating_result['momentum_score'],
        'risk': rating_result['risk_score'],
        'sentiment': rating_result['sentiment_score']
    }
    
    print(f"OVERALL RATING: {overall_score:.2f}/10")
    print("=" * 60)
    
    print("FACTOR BREAKDOWN:")
    print("-" * 40)
    for factor, score in factor_scores.items():
        weight = model.weights[factor] * 100
        print(f"{factor.capitalize():<10} : {score:.2f}/10 (Weight: {weight:.1f}%)")
    
    print("RECOMMENDATION:")
    print("-" * 40)
    
    if overall_score >= 8.0:
        print(" EXCELLENT - Strong buy recommendation")
    elif overall_score >= 7.0:
        print(" GOOD - Buy recommendation")
    elif overall_score >= 6.0:
        print(" FAIR - Hold recommendation")
    elif overall_score >= 5.0:
        print("  WEAK - Consider selling")
    else:
        print(" POOR - Strong sell recommendation")
    
    print("\nSTOCK DETAILS:")
    print("-" * 40)
    print(f"Current Price: ${stock_data.price:.2f}")
    print(f"Market Cap: ${stock_data.market_cap/1e9:.2f}B")
    print(f"P/E Ratio: {stock_data.pe_ratio:.2f}")
    print(f"P/B Ratio: {stock_data.pb_ratio:.2f}")
    print(f"ROE: {stock_data.roe:.2f}%")
    print(f"ROA: {stock_data.roa:.2f}%")
    print(f"Revenue Growth: {stock_data.revenue_growth:.2f}%")
    print(f"Earnings Growth: {stock_data.earnings_growth:.2f}%")
    print(f"Beta: {stock_data.beta:.2f}")
    print(f"Volatility: {stock_data.volatility:.2f}%")
    print(f"Sector: {stock_data.sector}")

def main():
    """Main function for the live stock rater."""
    print(" LIVE STOCK RATER")
    print("=" * 50)
    print("Rate any stock using real-time market data!")
    print("=" * 50)
    
    # Initialize the model and data fetcher
    model = MultiFactorModel()
    fetcher = StockDataFetcher()
    
    while True:
        print("\nOptions:")
        print("1. Rate a stock")
        print("2. Exit")
        
        choice = input("\nEnter your choice (1-2): ").strip()
        
        if choice == '1':
            ticker = input("\nEnter stock ticker symbol: ").strip().upper()
            
            if not ticker:
                print(" Please enter a valid ticker symbol.")
                continue
            
            print(f"\n Fetching live data for {ticker}...")
            
            try:
                # Fetch live data
                stock_data = fetcher.fetch_stock_data(ticker, 'yahoo')
                
                if stock_data:
                    print(f" Successfully fetched data for {ticker}")
                    display_rating(ticker, stock_data, model)
                else:
                    print(f" Could not fetch data for {ticker}")
                    print("   Please check the ticker symbol and try again.")
                    
            except Exception as e:
                print(f" Error fetching data for {ticker}: {str(e)}")
                print("   Please check your internet connection and try again.")
        
        elif choice == '2':
            print("\n Thanks for using Live Stock Rater!")
            break
        
        else:
            print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main() 