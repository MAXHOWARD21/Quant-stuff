import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from typing import List, Tuple, Dict
import seaborn as sns
from datetime import datetime, timedelta

class MonteCarloSimulation:
    """
    A comprehensive Monte Carlo simulation class for quantitative finance applications.
    Includes stock price simulation, option pricing, and portfolio optimization.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the Monte Carlo simulation with a random seed.
        
        Args:
            seed (int): Random seed for reproducibility
        """
        np.random.seed(seed)
        self.seed = seed
    
    def simulate_stock_price(self, 
                           S0: float, 
                           mu: float, 
                           sigma: float, 
                           T: float, 
                           n_steps: int, 
                           n_simulations: int) -> np.ndarray:
        """
        Simulate stock prices using Geometric Brownian Motion.
        
        Args:
            S0 (float): Initial stock price
            mu (float): Drift (expected return)
            sigma (float): Volatility
            T (float): Time horizon in years
            n_steps (int): Number of time steps
            n_simulations (int): Number of simulation paths
            
        Returns:
            np.ndarray: Array of simulated stock prices (n_simulations x n_steps)
        """
        dt = T / n_steps
        t = np.linspace(0, T, n_steps)
        
        # Generate random normal variables
        Z = np.random.normal(0, 1, (n_simulations, n_steps))
        
        # Calculate stock prices using GBM formula
        S = np.zeros((n_simulations, n_steps))
        S[:, 0] = S0
        
        for i in range(1, n_steps):
            S[:, i] = S[:, i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, i])
        
        return S, t
    
    def price_european_option(self, 
                             S0: float, 
                             K: float, 
                             T: float, 
                             r: float, 
                             sigma: float, 
                             option_type: str = 'call',
                             n_simulations: int = 10000) -> Dict:
        """
        Price European options using Monte Carlo simulation.
        
        Args:
            S0 (float): Current stock price
            K (float): Strike price
            T (float): Time to maturity in years
            r (float): Risk-free rate
            sigma (float): Volatility
            option_type (str): 'call' or 'put'
            n_simulations (int): Number of simulations
            
        Returns:
            Dict: Dictionary containing option price and confidence interval
        """
        # Simulate final stock prices
        Z = np.random.normal(0, 1, n_simulations)
        ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:  # put
            payoffs = np.maximum(K - ST, 0)
        
        # Discount payoffs
        option_price = np.exp(-r * T) * np.mean(payoffs)
        
        # Calculate confidence interval
        std_error = np.std(payoffs) / np.sqrt(n_simulations)
        confidence_interval = 1.96 * std_error  # 95% confidence
        
        return {
            'price': option_price,
            'confidence_interval': confidence_interval,
            'std_error': std_error,
            'simulations': n_simulations
        }
    
    def simulate_portfolio_returns(self, 
                                  weights: List[float], 
                                  returns_data: np.ndarray, 
                                  n_simulations: int = 10000) -> Dict:
        """
        Simulate portfolio returns using historical data and Monte Carlo.
        
        Args:
            weights (List[float]): Portfolio weights
            returns_data (np.ndarray): Historical returns data (n_assets x n_periods)
            n_simulations (int): Number of simulations
            
        Returns:
            Dict: Dictionary containing portfolio statistics
        """
        n_assets = len(weights)
        
        # Calculate mean returns and covariance matrix
        mean_returns = np.mean(returns_data, axis=1)
        cov_matrix = np.cov(returns_data)
        
        # Generate correlated random returns
        correlated_returns = np.random.multivariate_normal(
            mean_returns, cov_matrix, n_simulations
        )
        
        # Calculate portfolio returns
        portfolio_returns = np.dot(correlated_returns, weights)
        
        # Calculate statistics
        mean_return = np.mean(portfolio_returns)
        volatility = np.std(portfolio_returns)
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0
        
        # Calculate VaR and CVaR (95% confidence)
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])
        
        return {
            'mean_return': mean_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'returns': portfolio_returns
        }
    
    def optimize_portfolio_monte_carlo(self, 
                                     returns_data: np.ndarray, 
                                     n_portfolios: int = 10000,
                                     risk_free_rate: float = 0.02) -> Dict:
        """
        Optimize portfolio weights using Monte Carlo simulation.
        
        Args:
            returns_data (np.ndarray): Historical returns data
            n_portfolios (int): Number of random portfolios to generate
            risk_free_rate (float): Risk-free rate for Sharpe ratio calculation
            
        Returns:
            Dict: Dictionary containing optimal portfolio and efficient frontier
        """
        n_assets = returns_data.shape[0]
        
        # Generate random portfolio weights
        portfolios = []
        for _ in range(n_portfolios):
            weights = np.random.random(n_assets)
            weights = weights / np.sum(weights)  # Normalize to sum to 1
            portfolios.append(weights)
        
        portfolios = np.array(portfolios)
        
        # Calculate portfolio statistics
        mean_returns = np.mean(returns_data, axis=1)
        cov_matrix = np.cov(returns_data)
        
        portfolio_returns = np.dot(portfolios, mean_returns)
        portfolio_volatilities = np.sqrt(np.diag(np.dot(np.dot(portfolios, cov_matrix), portfolios.T)))
        sharpe_ratios = (portfolio_returns - risk_free_rate) / portfolio_volatilities
        
        # Find optimal portfolios
        max_sharpe_idx = np.argmax(sharpe_ratios)
        min_vol_idx = np.argmin(portfolio_volatilities)
        
        return {
            'max_sharpe_weights': portfolios[max_sharpe_idx],
            'min_vol_weights': portfolios[min_vol_idx],
            'max_sharpe_return': portfolio_returns[max_sharpe_idx],
            'max_sharpe_vol': portfolio_volatilities[max_sharpe_idx],
            'max_sharpe_ratio': sharpe_ratios[max_sharpe_idx],
            'min_vol_return': portfolio_returns[min_vol_idx],
            'min_vol_vol': portfolio_volatilities[min_vol_idx],
            'all_returns': portfolio_returns,
            'all_volatilities': portfolio_volatilities,
            'all_sharpe_ratios': sharpe_ratios
        }
    
    def plot_stock_simulation(self, S: np.ndarray, t: np.ndarray, title: str = "Stock Price Simulation"):
        """Plot stock price simulation results."""
        plt.figure(figsize=(12, 8))
        
        # Plot all simulation paths
        for i in range(min(100, S.shape[0])):  # Plot first 100 paths
            plt.plot(t, S[i, :], alpha=0.1, color='blue')
        
        # Plot mean path
        mean_path = np.mean(S, axis=0)
        plt.plot(t, mean_path, 'r-', linewidth=2, label='Mean Path')
        
        # Plot confidence intervals
        percentile_95 = np.percentile(S, 95, axis=0)
        percentile_5 = np.percentile(S, 5, axis=0)
        plt.fill_between(t, percentile_5, percentile_95, alpha=0.3, color='gray', label='90% Confidence Interval')
        
        plt.xlabel('Time (years)')
        plt.ylabel('Stock Price')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_portfolio_optimization(self, optimization_results: Dict):
        """Plot portfolio optimization results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Efficient frontier
        ax1.scatter(optimization_results['all_volatilities'], 
                   optimization_results['all_returns'], 
                   c=optimization_results['all_sharpe_ratios'], 
                   cmap='viridis', alpha=0.6)
        
        # Mark optimal portfolios
        ax1.scatter(optimization_results['max_sharpe_vol'], 
                   optimization_results['max_sharpe_return'], 
                   color='red', s=100, marker='*', label='Max Sharpe Ratio')
        ax1.scatter(optimization_results['min_vol_vol'], 
                   optimization_results['min_vol_return'], 
                   color='green', s=100, marker='*', label='Min Volatility')
        
        ax1.set_xlabel('Volatility')
        ax1.set_ylabel('Expected Return')
        ax1.set_title('Efficient Frontier')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Sharpe ratio distribution
        ax2.hist(optimization_results['all_sharpe_ratios'], bins=50, alpha=0.7, color='skyblue')
        ax2.axvline(optimization_results['max_sharpe_ratio'], color='red', linestyle='--', 
                   label=f'Max Sharpe: {optimization_results["max_sharpe_ratio"]:.3f}')
        ax2.set_xlabel('Sharpe Ratio')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Sharpe Ratio Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to demonstrate Monte Carlo simulations."""
    print("=== Monte Carlo Simulation for Quantitative Finance ===\n")
    
    # Initialize simulation
    mc = MonteCarloSimulation(seed=42)
    
    # 1. Stock Price Simulation
    print("1. Stock Price Simulation")
    print("-" * 30)
    
    S0 = 100  # Initial price
    mu = 0.08  # Expected return (8%)
    sigma = 0.2  # Volatility (20%)
    T = 1.0  # 1 year
    n_steps = 252  # Daily steps
    n_simulations = 1000
    
    S, t = mc.simulate_stock_price(S0, mu, sigma, T, n_steps, n_simulations)
    
    final_prices = S[:, -1]
    print(f"Initial Price: ${S0}")
    print(f"Expected Final Price: ${np.mean(final_prices):.2f}")
    print(f"Price Range: ${np.min(final_prices):.2f} - ${np.max(final_prices):.2f}")
    print(f"Probability of Loss: {np.mean(final_prices < S0)*100:.1f}%")
    
    # Plot stock simulation
    mc.plot_stock_simulation(S, t, "Stock Price Simulation (GBM)")
    
    # 2. Option Pricing
    print("\n2. European Option Pricing")
    print("-" * 30)
    
    K = 100  # Strike price
    r = 0.05  # Risk-free rate
    
    # Price call option
    call_result = mc.price_european_option(S0, K, T, r, sigma, 'call', 50000)
    print(f"Call Option Price: ${call_result['price']:.4f}")
    print(f"95% Confidence Interval: ±${call_result['confidence_interval']:.4f}")
    
    # Price put option
    put_result = mc.price_european_option(S0, K, T, r, sigma, 'put', 50000)
    print(f"Put Option Price: ${put_result['price']:.4f}")
    print(f"95% Confidence Interval: ±${put_result['confidence_interval']:.4f}")
    
    # 3. Portfolio Optimization
    print("\n3. Portfolio Optimization")
    print("-" * 30)
    
    # Generate sample returns data for 3 assets
    np.random.seed(42)
    n_periods = 252  # Daily data for 1 year
    
    # Asset 1: High return, high volatility
    returns1 = np.random.normal(0.001, 0.02, n_periods)
    
    # Asset 2: Medium return, medium volatility
    returns2 = np.random.normal(0.0005, 0.015, n_periods)
    
    # Asset 3: Low return, low volatility
    returns3 = np.random.normal(0.0002, 0.01, n_periods)
    
    returns_data = np.vstack([returns1, returns2, returns3])
    
    # Optimize portfolio
    optimization_results = mc.optimize_portfolio_monte_carlo(returns_data, 10000)
    
    print("Optimal Portfolio Weights (Max Sharpe):")
    for i, weight in enumerate(optimization_results['max_sharpe_weights']):
        print(f"  Asset {i+1}: {weight:.3f}")
    
    print(f"\nMax Sharpe Portfolio:")
    print(f"  Expected Return: {optimization_results['max_sharpe_return']*252:.2%}")
    print(f"  Volatility: {optimization_results['max_sharpe_vol']*np.sqrt(252):.2%}")
    print(f"  Sharpe Ratio: {optimization_results['max_sharpe_ratio']:.3f}")
    
    print(f"\nMin Volatility Portfolio:")
    print(f"  Expected Return: {optimization_results['min_vol_return']*252:.2%}")
    print(f"  Volatility: {optimization_results['min_vol_vol']*np.sqrt(252):.2%}")
    
    # Plot portfolio optimization results
    mc.plot_portfolio_optimization(optimization_results)
    
    # 4. Portfolio Risk Analysis
    print("\n4. Portfolio Risk Analysis")
    print("-" * 30)
    
    # Use optimal weights for risk analysis
    optimal_weights = optimization_results['max_sharpe_weights']
    portfolio_stats = mc.simulate_portfolio_returns(optimal_weights, returns_data, 10000)
    
    print(f"Portfolio Statistics (Annualized):")
    print(f"  Expected Return: {portfolio_stats['mean_return']*252:.2%}")
    print(f"  Volatility: {portfolio_stats['volatility']*np.sqrt(252):.2%}")
    print(f"  Sharpe Ratio: {portfolio_stats['sharpe_ratio']:.3f}")
    print(f"  VaR (95%): {portfolio_stats['var_95']*252:.2%}")
    print(f"  CVaR (95%): {portfolio_stats['cvar_95']*252:.2%}")
    
    # Plot portfolio returns distribution
    plt.figure(figsize=(10, 6))
    plt.hist(portfolio_stats['returns']*252, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    plt.axvline(portfolio_stats['var_95']*252, color='red', linestyle='--', 
               label=f'VaR (95%): {portfolio_stats["var_95"]*252:.2%}')
    plt.axvline(portfolio_stats['cvar_95']*252, color='orange', linestyle='--', 
               label=f'CVaR (95%): {portfolio_stats["cvar_95"]*252:.2%}')
    plt.xlabel('Annual Return')
    plt.ylabel('Frequency')
    plt.title('Portfolio Returns Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\n=== Simulation Complete ===")

if __name__ == "__main__":
    main()
