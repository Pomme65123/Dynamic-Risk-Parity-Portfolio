# Dynamic Risk Parity Portfolio Backtest

## Overview
This project implements a dynamic portfolio allocation strategy based on historical volatility. It calculates the 30-day rolling standard deviation of daily returns for a basket of five ETFs (SPY, TLT, GLD, USO, VNQ).


**1. Inverse Volatility Calculation**
Let $\sigma_i$ be the realized 30-day rolling volatility of an asset $i$. The inverse volatility $IV_i$ is calculated to prioritize lower-risk assets:
$$IV_i = \frac{1}{\sigma_i}$$

**2. Weight Normalization**
To ensure the portfolio is exactly 100% invested daily, the allocation weight $w_i$ for each asset is normalized across the $N$ assets:
$$w_i = \frac{IV_i}{\sum_{j=1}^{N} IV_j}$$

**3. Look-Ahead Bias Prevention and Portfolio Returns**
To simulate realistic trading, weights calculated at the market close of day $t-1$ are used to capture the returns $R$ on day $t$. The daily portfolio return $R_{p, t}$ is:
$$R_{p, t} = \sum_{i=1}^{N} w_{i, t-1} R_{i, t}$$

**4. Performance Evaluation (Annualized Sharpe Ratio)**
The strategy's performance is evaluated using the Sharpe Ratio. Assuming a risk-free rate of 0, the daily mean return ($\mu_{daily}$) divided by the daily return standard deviation ($\sigma_{daily}$) is annualized using 252 trading days:
$$Sharpe = \frac{\mu_{daily}}{\sigma_{daily}} \times \sqrt{252}$$

Cumulative returns are calculated over time using $\prod (1 + R_{p, t})$.# ML-Portfolio
