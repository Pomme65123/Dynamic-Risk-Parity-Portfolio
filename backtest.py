import pandas as pd
import numpy as np

def evaluate_strategy(weights, actual_returns):

    portfolio_weights = weights.shift(1).dropna()
    aligned_returns = actual_returns.loc[portfolio_weights.index]

    strategy_daily = (portfolio_weights * aligned_returns).sum(axis=1)

    equal_weights = 1.0 / len(actual_returns.columns)
    benchmark_daily = (aligned_returns * equal_weights).sum(axis=1)

    strategy_cum = (1 + strategy_daily).cumprod()
    benchmark_cum = (1 + benchmark_daily).cumprod()

    strategy_sharpe = (strategy_daily.mean() / strategy_daily.std()) * np.sqrt(252)
    benchmark_sharpe = (benchmark_daily.mean() / benchmark_daily.std()) * np.sqrt(252)

    return strategy_sharpe, benchmark_sharpe, strategy_cum, benchmark_cum