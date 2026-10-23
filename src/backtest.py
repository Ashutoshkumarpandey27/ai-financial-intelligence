
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# 1. CONFIGURATION
# =========================================================
INITIAL_CAPITAL = 10000.0
TRANSACTION_COST = 0.001  # Assumed combined cost: 0.1% per trade
TEST_RATIO = 0.80
TRADING_DAYS_PER_YEAR = 252

# =========================================================
# 2. LOAD DATA AND TRAINED MODEL
# =========================================================
raw_data = pd.read_csv("data/reliance_stock_data.csv")
model_data = pd.read_csv("data/model_ready_dataset.csv")
model = joblib.load("models/logistic_regression.pkl")

required_raw = {"Date", "Open", "Close"}

feature_columns = [
    "Close",
    "Daily_Return",
    "Previous_Day_Return",
    "Return_2_Days_Ago",
    "MA_5",
    "Volume_Change",
]

required_model = set(feature_columns + ["Date", "Target"])

if not required_raw.issubset(raw_data.columns):
    raise ValueError(
        f"Raw dataset missing columns: "
        f"{required_raw - set(raw_data.columns)}"
    )

if not required_model.issubset(model_data.columns):
    raise ValueError(
        f"Model dataset missing columns: "
        f"{required_model - set(model_data.columns)}"
    )

if raw_data.empty or model_data.empty:
    raise ValueError("One or both datasets are empty.")

raw_data["Date"] = pd.to_datetime(raw_data["Date"], errors="raise")
model_data["Date"] = pd.to_datetime(model_data["Date"], errors="raise")

raw_data = raw_data.sort_values("Date").reset_index(drop=True)
model_data = model_data.sort_values("Date").reset_index(drop=True)

if raw_data["Date"].duplicated().any():
    raise ValueError("Raw dataset contains duplicate dates.")

if model_data["Date"].duplicated().any():
    raise ValueError("Model-ready dataset contains duplicate dates.")

if not np.isfinite(
    raw_data[["Open", "Close"]].to_numpy(dtype=float)
).all():
    raise ValueError("Raw prices contain non-finite values.")

if (raw_data[["Open", "Close"]] <= 0).any().any():
    raise ValueError("Raw prices must be positive.")

if not 0 <= TRANSACTION_COST < 1:
    raise ValueError("Transaction cost must be between 0 and 1.")

# =========================================================
# 3. CHRONOLOGICAL TEST SPLIT
# =========================================================
split_index = int(len(model_data) * TEST_RATIO)

if split_index <= 0 or split_index >= len(model_data):
    raise ValueError("Not enough data for the selected split.")

test_data = model_data.iloc[split_index:].copy()

if test_data[feature_columns].isna().any().any():
    raise ValueError("Missing values found in test features.")

if not np.isfinite(
    test_data[feature_columns].to_numpy(dtype=float)
).all():
    raise ValueError("Non-finite values found in test features.")

# The model uses features from the signal date.
test_data["Prediction"] = model.predict(
    test_data[feature_columns]
)

# =========================================================
# 4. PREPARE NEXT SESSION'S EXECUTION PRICES
# =========================================================
# The signal is generated using information from date t.
# Execution takes place during the next trading session.

raw_data["Next_Open"] = raw_data["Open"].shift(-1)
raw_data["Next_Close"] = raw_data["Close"].shift(-1)

execution_prices = raw_data[
    ["Date", "Next_Open", "Next_Close"]
].copy()

# IMPORTANT FIX:
# The model-ready CSV may already contain Next_Open/Next_Close.
# Remove them before merging to avoid duplicate column suffixes.
test_data = test_data.drop(
    columns=["Next_Open", "Next_Close"],
    errors="ignore",
)

test_data = test_data.merge(
    execution_prices,
    on="Date",
    how="left",
    validate="one_to_one",
)

print("Test rows before price validation:", len(test_data))

if not test_data.empty:
    print(
        "Signal period:",
        test_data["Date"].min().date(),
        "to",
        test_data["Date"].max().date(),
    )

missing_prices = test_data[
    ["Next_Open", "Next_Close"]
].isna().any(axis=1).sum()

print("Rows missing execution prices:", int(missing_prices))

test_data = test_data.dropna(
    subset=["Next_Open", "Next_Close"]
).copy()

if test_data.empty:
    raise ValueError("No rows available for backtesting.")

if (
    (test_data["Next_Open"] <= 0).any()
    or (test_data["Next_Close"] <= 0).any()
):
    raise ValueError("Execution prices must be positive.")

# =========================================================
# 5. CALCULATE RETURNS
# =========================================================
# Prediction == 1 means UP signal.
# For an UP signal, buy at next session's Open and sell
# at that same session's Close.
# Prediction == 0 means stay out of the market.

test_data["Market_Return"] = (
    test_data["Next_Close"] / test_data["Next_Open"]
) - 1

test_data["Is_Trade"] = (
    test_data["Prediction"] == 1
).astype(int)

test_data["Strategy_Return_Before_Cost"] = np.where(
    test_data["Is_Trade"] == 1,
    test_data["Market_Return"],
    0.0,
)

# Apply the assumed combined transaction cost to each trade.
test_data["Strategy_Return"] = np.where(
    test_data["Is_Trade"] == 1,
    (
        (1 + test_data["Market_Return"])
        * (1 - TRANSACTION_COST)
    ) - 1,
    0.0,
)

# =========================================================
# 6. MATCHED-PERIOD BENCHMARK
# =========================================================
# Benchmark: earn each test session's Open-to-Close return.
# This is NOT continuous buy-and-hold.

test_data["Benchmark_Return"] = test_data["Market_Return"]

strategy_total_return = (
    (1 + test_data["Strategy_Return"]).prod()
) - 1

benchmark_total_return = (
    (1 + test_data["Benchmark_Return"]).prod()
) - 1

trades = test_data.loc[
    test_data["Is_Trade"] == 1,
    "Strategy_Return",
]

winning_trades = int((trades > 0).sum())
total_trades = int(test_data["Is_Trade"].sum())

# =========================================================
# 7. EQUITY CURVES AND DRAWDOWN
# =========================================================
test_data["Strategy_Equity"] = (
    INITIAL_CAPITAL
    * (1 + test_data["Strategy_Return"]).cumprod()
)

test_data["Benchmark_Equity"] = (
    INITIAL_CAPITAL
    * (1 + test_data["Benchmark_Return"]).cumprod()
)

# Include initial capital when calculating running peaks.
equity_with_start = pd.concat(
    [
        pd.Series([INITIAL_CAPITAL]),
        test_data["Strategy_Equity"].reset_index(drop=True),
    ],
    ignore_index=True,
)

running_peaks = equity_with_start.cummax()

test_data["Running_Peak"] = (
    running_peaks.iloc[1:].to_numpy()
)

test_data["Drawdown"] = (
    test_data["Strategy_Equity"]
    / test_data["Running_Peak"]
) - 1

max_drawdown = float(test_data["Drawdown"].min())

# =========================================================
# 8. RISK-ADJUSTED PERFORMANCE
# =========================================================
daily_returns = test_data["Strategy_Return"]
daily_volatility = daily_returns.std(ddof=1)

if (
    len(daily_returns) > 1
    and pd.notna(daily_volatility)
    and daily_volatility > 0
):
    sharpe_ratio = (
        daily_returns.mean() / daily_volatility
    ) * np.sqrt(TRADING_DAYS_PER_YEAR)
else:
    sharpe_ratio = np.nan

# =========================================================
# 9. PRINT RESULTS
# =========================================================
print("\n--- BACKTEST RESULTS ---")
print("Trading sessions:", len(test_data))
print("UP signals / completed trades:", total_trades)
print(f"Assumed combined cost per trade: {TRANSACTION_COST:.2%}")
print(f"ML Strategy Return: {strategy_total_return:.2%}")
print(f"Matched-period benchmark return: {benchmark_total_return:.2%}")
print(
    f"Final strategy equity: "
    f"₹{test_data['Strategy_Equity'].iloc[-1]:,.2f}"
)
print(
    f"Final benchmark equity: "
    f"₹{test_data['Benchmark_Equity'].iloc[-1]:,.2f}"
)

print("\n--- TRADE ANALYSIS ---")
print("Winning trades:", winning_trades)

if not trades.empty:
    print(f"Win rate: {winning_trades / len(trades):.2%}")
    print(f"Average return per trade: {trades.mean():.2%}")
    print(f"Best trade: {trades.max():.2%}")
    print(f"Worst trade: {trades.min():.2%}")
else:
    print("No trades were executed.")

print("\n--- RISK ANALYSIS ---")
print(f"Maximum drawdown: {max_drawdown:.2%}")

if pd.notna(sharpe_ratio):
    print(f"Annualized Sharpe ratio: {sharpe_ratio:.2f}")
else:
    print("Annualized Sharpe ratio: Not available")

# =========================================================
# 10. EXPORT REPORTS
# =========================================================
os.makedirs("reports", exist_ok=True)

report_columns = [
    "Date",
    "Prediction",
    "Is_Trade",
    "Next_Open",
    "Next_Close",
    "Market_Return",
    "Strategy_Return_Before_Cost",
    "Strategy_Return",
    "Benchmark_Return",
    "Strategy_Equity",
    "Benchmark_Equity",
    "Running_Peak",
    "Drawdown",
]

test_data[report_columns].to_csv(
    "reports/backtest_results.csv",
    index=False,
)

summary = pd.DataFrame(
    {
        "Metric": [
            "Trading Sessions",
            "UP Signals / Completed Trades",
            "Combined Transaction Cost per Trade",
            "ML Strategy Return",
            "Matched-Period Benchmark Return",
            "Maximum Drawdown",
            "Annualized Sharpe Ratio",
            "Winning Trades",
            "Win Rate",
            "Average Return per Trade",
            "Best Trade",
            "Worst Trade",
            "Initial Capital",
            "Final Strategy Equity",
            "Final Benchmark Equity",
        ],
        "Value": [
            len(test_data),
            total_trades,
            TRANSACTION_COST,
            strategy_total_return,
            benchmark_total_return,
            max_drawdown,
            sharpe_ratio,
            winning_trades,
            winning_trades / len(trades) if not trades.empty else np.nan,
            trades.mean() if not trades.empty else np.nan,
            trades.max() if not trades.empty else np.nan,
            trades.min() if not trades.empty else np.nan,
            INITIAL_CAPITAL,
            test_data["Strategy_Equity"].iloc[-1],
            test_data["Benchmark_Equity"].iloc[-1],
        ],
    }
)

summary.to_csv(
    "reports/backtest_summary.csv",
    index=False,
)

print("\n--- REPORTS SAVED ---")
print("reports/backtest_results.csv")
print("reports/backtest_summary.csv")

# =========================================================
# 11. PLOT EQUITY CURVES
# =========================================================
plt.figure(figsize=(12, 6))

plt.plot(
    test_data["Date"],
    test_data["Strategy_Equity"],
    label="ML Strategy",
)

plt.plot(
    test_data["Date"],
    test_data["Benchmark_Equity"],
    label="Matched-period benchmark",
)

plt.title("Backtesting: ML Strategy vs Matched-Period Benchmark")
plt.xlabel("Signal Date")
plt.ylabel("Portfolio Value (₹)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
