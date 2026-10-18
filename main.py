import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# custom modules
from risk_parity import get_risk_parity_weights
from backtest import evaluate_strategy
from visualization import plot_performance

def build_features_and_target(prices):

    df = pd.DataFrame({'Close': prices})
    df['Returns'] = df['Close'].pct_change()

    df['Vol_15d'] = df['Returns'].rolling(window=15).std()
    df['Vol_30d'] = df['Returns'].rolling(window=30).std()
    df['Mom_15d'] = df['Returns'].rolling(window=15).sum()
    df['Mom_30d'] = df['Returns'].rolling(window=30).sum()

    df['Target_Vol'] = df['Returns'].rolling(window=5).std().shift(-5)

    return df.dropna()

def main():
    tickers = ['SPY', 'TLT', 'GLD', 'USO', 'VNQ']
    prices_df = yf.download(tickers, start='2020-01-01', end='2026-10-14')['Close']

    all_predictions = {}

    tscv = TimeSeriesSplit(n_splits=6, test_size=60)

    for ticker in tqdm(tickers, desc="Training ETFs", unit="model"):
        df_ticker = build_features_and_target(prices_df[ticker])

        feature_cols = ['Vol_15d', 'Vol_30d', 'Mom_15d', 'Mom_30d']
        X = df_ticker[feature_cols]
        y = df_ticker['Target_Vol']

        ticker_preds = pd.Series(dtype=float)

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            tscv_inner = TimeSeriesSplit(n_splits=3, test_size=20)

            rf = RandomForestRegressor(random_state=42)

            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [3, 5, 7],
                'min_samples_leaf': [5, 15]
            }

            gsearch = GridSearchCV(
                estimator=rf,
                param_grid=param_grid,
                cv=tscv_inner,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )

            gsearch.fit(X_train, y_train)

            best_model = gsearch.best_estimator_
            preds = best_model.predict(X_test)

            preds = np.maximum(preds, 0.001)

            ticker_preds = pd.concat([ticker_preds, pd.Series(preds, index=X_test.index)])

        all_predictions[ticker] = ticker_preds


    predicted_vol = pd.DataFrame(all_predictions)

    weights = get_risk_parity_weights(predicted_vol)


    actual_returns = prices_df.pct_change().dropna()
    actual_returns = actual_returns.loc[weights.index]



    s_sharpe, b_sharpe, s_cum, b_cum = evaluate_strategy(weights, actual_returns)

    print(f"Sharpe Ratio: {s_sharpe:.2f}")
    print(f"Benchmark Sharpe Ratio: {b_sharpe:.2f}")
    print(f"Returns:   ${s_cum.iloc[-1]:.4f}")
    print(f"Benchmark Returns:  ${b_cum.iloc[-1]:.4f}")
    plot_performance(weights, s_cum, b_cum)

if __name__ == "__main__":
    main()