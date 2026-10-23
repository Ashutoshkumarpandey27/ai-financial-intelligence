
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# 1. CONFIGURATION
# =========================================================
INITIAL_CAPITAL = 10000.0
TRANSACTION_COST = 0.001  # Assumed combined round-trip cost: 0.1%
TEST_RATIO = 0.80
TRADING_DAYS_PER_YEAR = 252

DATA_PATH = "data/model_ready_dataset.csv"
MODEL_PATH = "models/logistic_regression.pkl"
REPORT_DIR = "reports"

FEATURE_COLUMNS = [
    "Close",
    "Daily_Return",
    "Previous_Day_Return",
    "Return_2_Days_Ago",
    "MA_5",
    "Volume_Change",
]

# =========================================================
# 2. LOAD DATA AND MODEL
# =========================================================
model_data = pd.read_csv(DATA_PATH, parse_dates=["Date"])
model = joblib.load(MODEL_PATH)

required_columns = set(
    FEATURE_COLUMNS
    + ["Date", "Target", "Next_Open", "Next_Close"]
)

missing_columns = required_columns - set(model_data.columns)

if missing_columns:
    raise ValueError(
        f"Model-ready dataset missing columns: {missing_columns}"
    )

if model_data.empty:
    raise ValueError("Model-ready dataset is empty.")

model_data = (
    model_data.sort_values("Date")
    .reset_index(drop=True)
)

if model_data["Date"].duplicated().any():
    raise ValueError("Duplicate dates found in model-ready dataset.")

if not 0 <= TRANSACTION_COST < 1:
    raise ValueError("TRANSACTION_COST must be between 0 and 1.")

if INITIAL_CAPITAL <= 0:
    raise ValueError("INITIAL_CAPITAL must be positive.")

# =========================================================
# 3. VALIDATE TARGET AND EXECUTION PRICES
# =========================================================
numeric_columns = FEATURE_COLUMNS + [
    "Target",
    "Next_Open",
    "Next_Close",
]

for column in numeric_columns:
    model_data[column] = pd.to_numeric(
        model_data[column],
        errors="raise",
    )

if model_data[numeric_columns].isna().any().any():
    raise ValueError("Missing values found in required columns.")

if not np.isfinite(
    model_data[numeric_columns].to_numpy(dtype=float)
).all():
    raise ValueError("Infinite or non-finite values found.")

if not model_data["Target"].isin([0, 1]).all():
    raise ValueError("Target must contain only 0 and 1.")

if (
    (model_data["Next_Open"] <= 0).any()
    or (model_data["Next_Close"] <= 0).any()
):
    raise ValueError("Next-session execution prices must be positive.")

# Target 1 means next session Close > next session Open.
expected_target = (
    model_data["Next_Close"] > model_data["Next_Open"]
).astype(int)

if not np.array_equal(
    model_data["Target"].to_numpy(),
    expected_target.to_numpy(),
):
    raise ValueError(
        "Target does not match Next_Close > Next_Open. "
        "Run and verify feature_engineering.py first."
    )

# =========================================================
# 4. CHRONOLOGICAL SPLIT
#    Match train_model.py exactly:
#    train = [:split_index - 1]
#    test  = [split_index:]
# =========================================================
split_index = int(len(model_data) * TEST_RATIO)

if split_index <= 1 or split_index >= len(model_data):
    raise ValueError("Not enough data for the selected split.")

train_data = model_data.iloc[:split_index - 1].copy()
test_data = model_data.iloc[split_index:].copy()

if train_data.empty or test_data.empty:
    raise ValueError("Training or test dataset is empty.")

if test_data[FEATURE_COLUMNS].isna().any().any():
    raise ValueError("Missing values found in test features.")

if not np.isfinite(
    test_data[FEATURE_COLUMNS].to_numpy(dtype=float)
).all():
    raise ValueError("Non-finite values found in test features.")

# =========================================================
# 5. VERIFY TRAINED MODEL'S TEST PERIOD
# =========================================================
# The saved model should have been trained using the same
# chronological split and the same feature columns.

if hasattr(model, "n_features_in_"):
    if model.n_features_in_ != len(FEATURE_COLUMNS):
        raise ValueError(
            "Saved model expects a different number of features. "
            "Retrain the model using train_model.py."
        )

metadata_path = "models/logistic_regression_metadata.pkl"

if os.path.exists(metadata_path):
    metadata = joblib.load(metadata_path)

    saved_features = metadata.get("feature_columns")

    if saved_features is not None:
        if list(saved_features) != FEATURE_COLUMNS:
            raise ValueError(
                "Model metadata feature columns do not match "
                "the backtest feature columns. Retrain the model."
            )

# Predict using signal-date features only.
# Next_Open and Next_Close are NOT passed to the model.
test_data["Prediction"] = model.predict(
    test_data[FEATURE_COLUMNS]
).astype(int)

# =========================================================
# 6. CALCULATE NEXT-SESSION RETURNS
# =========================================================
# Date = signal date t.
# Next_Open and Next_Close = execution prices on session t+1.
#
# Prediction 1: buy at next session Open, sell at Close.
# Prediction 0: stay out of the market.

test_data["Market_Return"] = (
    test_data["Next_Close"] / test_data["Next_Open"]
) - 1.0

test_data["Is_Trade"] = (
    test_data["Prediction"] == 1
).astype(int)

test_data["Strategy_Return_Before_Cost"] = np.where(
    test_data["Is_Trade"] == 1,
    test_data["Market_Return"],
    0.0,
)

# Apply the assumed combined round-trip cost only to trades.
test_data["Strategy_Return"] = np.where(
    test_data["Is_Trade"] == 1,
    (
        (1.0 + test_data["Market_Return"])
        * (1.0 - TRANSACTION_COST)
    ) - 1.0,
    0.0,
)

# =========================================================
# 7. MATCHED-PERIOD BENCHMARK
# =========================================================
# Benchmark invests at every test session's Open and exits
# at that session's Close. It is not continuous buy-and-hold.

test_data["Benchmark_Return"] = test_data["Market_Return"]

strategy_total_return = (
    (1.0 + test_data["Strategy_Return"]).prod()
) - 1.0

benchmark_total_return = (
    (1.0 + test_data["Benchmark_Return"]).prod()
) - 1.0

trades = test_data.loc[
    test_data["Is_Trade"] == 1,
    "Strategy_Return",
]

total_trades = len(trades)
winning_trades = int((trades > 0).sum())
losing_trades = int((trades < 0).sum())

win_rate = (
    winning_trades / total_trades
    if total_trades > 0
    else np.nan
)

# =========================================================
# 8. EQUITY CURVES AND DRAWDOWN
# =========================================================
test_data["Strategy_Equity"] = (
    INITIAL_CAPITAL
    * (1.0 + test_data["Strategy_Return"]).cumprod()
)

test_data["Benchmark_Equity"] = (
    INITIAL_CAPITAL
    * (1.0 + test_data["Benchmark_Return"]).cumprod()
)

# Include initial capital when finding the running peak.
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
) - 1.0

max_drawdown = float(test_data["Drawdown"].min())

# =========================================================
# 9. RISK METRICS
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

final_strategy_equity = float(
    test_data["Strategy_Equity"].iloc[-1]
)

final_benchmark_equity = float(
    test_data["Benchmark_Equity"].iloc[-1]
)

# =========================================================
# 10. PRINT RESULTS
# =========================================================
print("\n--- DATA AND SPLIT ---")
print("Total model-ready rows:", len(model_data))
print("Training rows:", len(train_data))
print("Testing rows:", len(test_data))
print(
    "Training signal dates:",
    train_data["Date"].min().date(),
    "to",
    train_data["Date"].max().date(),
)
print(
    "Test signal dates:",
    test_data["Date"].min().date(),
    "to",
    test_data["Date"].max().date(),
)
print("Features used:", len(FEATURE_COLUMNS))
print("Target: Next session Close > Next session Open")

print("\n--- BACKTEST RESULTS ---")
print("Trading sessions:", len(test_data))
print("UP signals / completed trades:", total_trades)
print(f"Assumed combined cost per trade: {TRANSACTION_COST:.2%}")
print(f"ML strategy total return: {strategy_total_return:.2%}")
print(f"Benchmark total return: {benchmark_total_return:.2%}")
print(f"Final strategy equity: ₹{final_strategy_equity:,.2f}")
print(f"Final benchmark equity: ₹{final_benchmark_equity:,.2f}")

print("\n--- TRADE ANALYSIS ---")
print("Winning trades:", winning_trades)
print("Losing trades:", losing_trades)

if total_trades > 0:
    print(f"Win rate: {win_rate:.2%}")
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
# 11. EXPORT REPORTS
# =========================================================
os.makedirs(REPORT_DIR, exist_ok=True)

report_columns = [
    "Date",
    "Prediction",
    "Target",
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

results_path = os.path.join(
    REPORT_DIR,
    "backtest_results.csv",
)

test_data[report_columns].to_csv(
    results_path,
    index=False,
)

summary = pd.DataFrame(
    {
        "Metric": [
            "Trading Sessions",
            "UP Signals / Completed Trades",
            "Winning Trades",
            "Losing Trades",
            "Win Rate",
            "Combined Transaction Cost per Trade",
            "ML Strategy Return",
            "Matched-Period Benchmark Return",
            "Maximum Drawdown",
            "Annualized Sharpe Ratio",
            "Initial Capital",
            "Final Strategy Equity",
            "Final Benchmark Equity",
        ],
        "Value": [
            len(test_data),
            total_trades,
            winning_trades,
            losing_trades,
            win_rate,
            TRANSACTION_COST,
            strategy_total_return,
            benchmark_total_return,
            max_drawdown,
            sharpe_ratio,
            INITIAL_CAPITAL,
            final_strategy_equity,
            final_benchmark_equity,
        ],
    }
)

summary_path = os.path.join(
    REPORT_DIR,
    "backtest_summary.csv",
)

summary.to_csv(summary_path, index=False)

print("\n--- REPORTS SAVED ---")
print(results_path)
print(summary_path)

# =========================================================
# 12. PLOT EQUITY CURVES
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
    label="Matched-period intraday benchmark",
)

plt.title("ML Strategy vs Matched-Period Intraday Benchmark")
plt.xlabel("Signal Date")
plt.ylabel("Portfolio Value (₹)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
