# ML-Driven Dynamic Risk Parity Portfolio Backtest

## Overview
This project implements a dynamic portfolio allocation strategy based on machine learning predictions of future asset volatility, transitioning from a historical 30-day rolling standard deviation baseline for a basket of five ETFs (SPY, TLT, GLD, USO, VNQ). It trains predictive models to forecast short-term forward volatility and dynamically adjusts capital allocations to maintain risk parity.

## Machine Learning Architecture
The strategy simulates time-series features to predict the forward 5-day standard deviation of daily returns.

**1. Feature & Target Setup**
*   **Features:** 15-day and 30-day rolling volatility alongside 15-day and 30-day rolling momentum.
*   **Target Label:** The realized 5-day volatility shifted backwards, predicting future risk using present features.

**2. Model Evolution & Cross-Validation**
*   **Baseline Model (Main Branch):** A Ridge Regression model evaluated across an expanding `TimeSeriesSplit` window.
*   **Advanced Model (Feature Branch):** A `RandomForestRegressor` utilizing Nested Time-Series Cross-Validation. An outer loop evaluates out-of-sample performance, while an inner loop dynamically tune hyperparameters (`max_depth`, `min_samples_leaf`, `n_estimators`) on expanding training data.

## Portfolio Allocation Methodology

**1. Inverse Volatility Calculation**
Let $\sigma_i$ be the predicted rolling volatility of an asset $i$. The inverse volatility $IV_i$ is calculated to prioritize lower-risk assets:
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

Cumulative returns are calculated over time using $\prod (1 + R_{p, t})$.

## Branch Performance & Output Comparison

The models were backtested from January 2020 through October 2026 against a static equal-weight benchmark.

### Baseline: Ridge Regression (`main` branch)
*   **Strategy Sharpe Ratio:** 1.40
*   **Strategy Cumulative Returns:** $1.2497
*   **Benchmark Sharpe Ratio:** 2.27
*   **Benchmark Cumulative Returns:** $1.3758

### Advanced: Random Forest (`feature` branch)
*   **Strategy Sharpe Ratio:** 1.89
*   **Strategy Cumulative Returns:** $1.2333
*   **Benchmark Sharpe Ratio:** 2.27
*   **Benchmark Cumulative Returns:** $1.3758

**Analysis:**
The transition from a linear Ridge regression to a non-linear Random Forest significantly improved the strategy's risk-adjusted returns, increasing the Sharpe Ratio from 1.40 to 1.89. While the advanced model experienced a marginal decrease in absolute cumulative returns ($1.2333 vs $1.2497), the higher Sharpe ratio indicates the Random Forest was much more effective at actively reducing portfolio volatility during the test period.