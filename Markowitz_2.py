"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        # Advanced Momentum-Volatility Strategy
        # Combines momentum, volatility adjustment, and sector rotation
        
        # Use shorter lookback for more responsive strategy
        momentum_window = 20  # 1 month momentum
        volatility_window = 60  # 3 month volatility
        min_periods = max(momentum_window, volatility_window)
        
        for i in range(min_periods, len(self.price)):
            # Get current date
            current_date = self.price.index[i]
            
            # Calculate momentum scores (price momentum + return momentum)
            price_window = self.price[assets].iloc[i-momentum_window:i+1]
            returns_window = self.returns[assets].iloc[i-momentum_window:i+1]
            
            # Price momentum: current price / average price
            avg_prices = price_window.mean()
            current_prices = self.price[assets].iloc[i]
            price_momentum = (current_prices / avg_prices) - 1
            
            # Return momentum: cumulative returns
            return_momentum = returns_window.sum()
            
            # Combined momentum score
            momentum_score = 0.6 * price_momentum + 0.4 * return_momentum
            
            # Calculate volatility for risk adjustment
            volatility = self.returns[assets].iloc[i-volatility_window:i].std() * np.sqrt(252)
            volatility = volatility.replace(0, volatility.mean())  # Handle zero volatility
            
            # Risk-adjusted momentum (momentum per unit of risk)
            risk_adjusted_momentum = momentum_score / volatility
            
            # Additional factor: recent performance (last 5 days)
            recent_returns = self.returns[assets].iloc[i-5:i].sum()
            
            # Combined score with momentum, volatility adjustment, and recent performance
            combined_score = (0.5 * risk_adjusted_momentum + 
                            0.3 * momentum_score + 
                            0.2 * recent_returns)
            
            # Select top assets (top 60% of assets by score)
            n_select = max(3, int(len(assets) * 0.6))  # At least 3 assets, max 60% of total
            top_assets = combined_score.nlargest(n_select).index
            
            # Calculate weights using modified risk parity for selected assets
            selected_volatility = volatility[top_assets]
            
            # Inverse volatility weights with momentum boost
            inv_vol_weights = 1.0 / selected_volatility
            momentum_boost = 1 + (combined_score[top_assets] - combined_score[top_assets].min()) / (combined_score[top_assets].max() - combined_score[top_assets].min() + 1e-8)
            
            # Apply momentum boost to weights
            adjusted_weights = inv_vol_weights * momentum_boost
            
            # Normalize weights to sum to 1
            final_weights = adjusted_weights / adjusted_weights.sum()
            
            # Assign weights
            for asset in assets:
                if asset in top_assets:
                    self.portfolio_weights.loc[current_date, asset] = final_weights[asset]
                else:
                    self.portfolio_weights.loc[current_date, asset] = 0.0
            
            # Set excluded asset to 0
            self.portfolio_weights.loc[current_date, self.exclude] = 0.0
        
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
