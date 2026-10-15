import pandas as pd
import yfinance as yf

# custom
from risk_parity import get_risk_parity_weights
from backtest import evaluate_strategy

def main():
    tickers = ['SPY', 'TLT', 'GLD', 'USO', 'VNQ']

    df = yf.download(tickers, start='2023-01-01', end='2026-10-14')['Close']

    print(df)

    actual_returns = df.pct_change().dropna()

    predicted_vol = actual_returns.rolling(window=30).std().dropna()
    actual_returns = actual_returns.loc[predicted_vol.index]

    weights = get_risk_parity_weights(predicted_vol)

    # print(weights)

    s_sharpe, b_sharpe, s_cum, b_cum = evaluate_strategy(weights, actual_returns)

    print(f"Sharpe Ratio: {s_sharpe:.2f}")
    print(f"Benchmark Sharpe Ratio: {b_sharpe:.2f}")
    print(f"Returns:   ${s_cum.iloc[-1]:.4f}")
    print(f"Benchmark Returns:  ${b_cum.iloc[-1]:.4f}")

if __name__ == "__main__":
    main()