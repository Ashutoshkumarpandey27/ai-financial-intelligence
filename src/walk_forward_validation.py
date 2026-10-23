
import os
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# =========================================================
# 1. CONFIGURATION
# =========================================================
DATA_PATH = "data/model_ready_dataset.csv"
REPORT_DIR = "reports"

INITIAL_CAPITAL = 10000.0
TRANSACTION_COST = 0.001  # Assumed combined round-trip cost: 0.1%
TRADING_DAYS_PER_YEAR = 252

INITIAL_TRAIN_RATIO = 0.60
TEST_WINDOW_RATIO = 0.10

FEATURE_COLUMNS = [
    "Close",
    "Daily_Return",
    "Previous_Day_Return",
    "Return_2_Days_Ago",
    "MA_5",
    "Volume_Change",
]

# =========================================================
# 2. LOAD AND VALIDATE DATA
# =========================================================
data = pd.read_csv(DATA_PATH, parse_dates=["Date"])

required_columns = FEATURE_COLUMNS + [
    "Date",
    "Target",
    "Next_Open",
    "Next_Close",
]

missing_columns = set(required_columns) - set(data.columns)

if missing_columns:
    raise ValueError(
        f"Dataset is missing columns: {missing_columns}"
    )

if data.empty:
    raise ValueError("Dataset is empty.")

data = data.sort_values("Date").reset_index(drop=True)

if data["Date"].duplicated().any():
    raise ValueError("Duplicate dates found.")

numeric_columns = FEATURE_COLUMNS + [
    "Target",
    "Next_Open",
    "Next_Close",
]

for column in numeric_columns:
    data[column] = pd.to_numeric(
        data[column],
        errors="raise",
    )

if data[numeric_columns].isna().any().any():
    raise ValueError("Missing values found in required columns.")

if not np.isfinite(
    data[numeric_columns].to_numpy(dtype=float)
).all():
    raise ValueError("Non-finite values found in required columns.")

if not data["Target"].isin([0, 1]).all():
    raise ValueError("Target must contain only 0 and 1.")

if (
    (data["Next_Open"] <= 0).any()
    or (data["Next_Close"] <= 0).any()
):
    raise ValueError("Execution prices must be positive.")

# Verify that Target matches the next session's intraday direction.
expected_target = (
    data["Next_Close"] > data["Next_Open"]
).astype(int)

if not np.array_equal(
    data["Target"].to_numpy(),
    expected_target.to_numpy(),
):
    raise ValueError(
        "Target alignment is incorrect. "
        "Verify feature_engineering.py first."
    )

if not 0 <= TRANSACTION_COST < 1:
    raise ValueError("TRANSACTION_COST must be between 0 and 1.")

if INITIAL_CAPITAL <= 0:
    raise ValueError("INITIAL_CAPITAL must be positive.")

X = data[FEATURE_COLUMNS]
y = data["Target"].astype(int)

# =========================================================
# 3. DEFINE MODEL
# =========================================================
base_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000)),
])

# =========================================================
# 4. WALK-FORWARD SETTINGS
# =========================================================
initial_train_size = int(
    len(data) * INITIAL_TRAIN_RATIO
)

test_window = int(
    len(data) * TEST_WINDOW_RATIO
)

if test_window < 1:
    raise ValueError("Test window is too small.")

# Each round:
# - Train on historical rows.
# - Leave one signal row as a gap.
# - Test on the next chronological window.
#
# The gap helps prevent adjacent target/execution periods
# from overlapping across the train/test boundary.

results = []
round_number = 1
train_end = initial_train_size

while train_end + 1 + test_window <= len(data):

    test_start = train_end + 1
    test_end = test_start + test_window

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_test = X.iloc[test_start:test_end]
    y_test = y.iloc[test_start:test_end]

    test_data = data.iloc[test_start:test_end].copy()

    current_model = clone(base_model)
    current_model.fit(X_train, y_train)

    predictions = current_model.predict(X_test)

    # =====================================================
    # 5. CLASSIFICATION METRICS
    # =====================================================
    accuracy = accuracy_score(y_test, predictions)

    precision_up = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall_up = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1_up = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    # =====================================================
    # 6. SIMULATE TRADES
    # =====================================================
    test_data["Prediction"] = predictions

    # Only predictions are used to decide whether to trade.
    # Execution prices are used only to calculate returns.
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

    test_data["Strategy_Return"] = np.where(
        test_data["Is_Trade"] == 1,
        (
            (1.0 + test_data["Market_Return"])
            * (1.0 - TRANSACTION_COST)
        ) - 1.0,
        0.0,
    )

    # Benchmark invests in every session's Open-to-Close move.
    test_data["Benchmark_Return"] = test_data["Market_Return"]

    strategy_return = (
        (1.0 + test_data["Strategy_Return"]).prod()
    ) - 1.0

    benchmark_return = (
        (1.0 + test_data["Benchmark_Return"]).prod()
    ) - 1.0

    # =====================================================
    # 7. TRADE AND RISK METRICS
    # =====================================================
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

    test_data["Strategy_Equity"] = (
        INITIAL_CAPITAL
        * (1.0 + test_data["Strategy_Return"]).cumprod()
    )

    test_data["Benchmark_Equity"] = (
        INITIAL_CAPITAL
        * (1.0 + test_data["Benchmark_Return"]).cumprod()
    )

    # Include starting capital when calculating drawdown.
    equity_with_start = pd.concat(
        [
            pd.Series([INITIAL_CAPITAL]),
            test_data["Strategy_Equity"].reset_index(drop=True),
        ],
        ignore_index=True,
    )

    running_peak = equity_with_start.cummax()

    test_data["Running_Peak"] = (
        running_peak.iloc[1:].to_numpy()
    )

    test_data["Drawdown"] = (
        test_data["Strategy_Equity"]
        / test_data["Running_Peak"]
    ) - 1.0

    max_drawdown = float(test_data["Drawdown"].min())

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

    # =====================================================
    # 8. SAVE ROUND RESULTS
    # =====================================================
    results.append({
        "Round": round_number,
        "Train_End_Date": data["Date"].iloc[
            train_end - 1
        ].date(),
        "Test_Start_Date": test_data["Date"].iloc[0].date(),
        "Test_End_Date": test_data["Date"].iloc[-1].date(),
        "Train_Rows": len(X_train),
        "Test_Rows": len(X_test),
        "Accuracy": accuracy,
        "Precision_UP": precision_up,
        "Recall_UP": recall_up,
        "F1_UP": f1_up,
        "Trades": total_trades,
        "Winning_Trades": winning_trades,
        "Losing_Trades": losing_trades,
        "Win_Rate": win_rate,
        "Strategy_Return": strategy_return,
        "Benchmark_Return": benchmark_return,
        "Excess_Return": strategy_return - benchmark_return,
        "Max_Drawdown": max_drawdown,
        "Sharpe_Ratio": sharpe_ratio,
        "Final_Strategy_Equity": float(
            test_data["Strategy_Equity"].iloc[-1]
        ),
        "Final_Benchmark_Equity": float(
            test_data["Benchmark_Equity"].iloc[-1]
        ),
    })

    # Move forward in time. The next round trains on more data.
    train_end = test_end
    round_number += 1

# =========================================================
# 9. DISPLAY RESULTS
# =========================================================
results_df = pd.DataFrame(results)

if results_df.empty:
    raise ValueError(
        "No walk-forward rounds were completed. "
        "Check dataset size and validation settings."
    )

print("\nWALK-FORWARD VALIDATION RESULTS\n")

print(
    results_df[
        [
            "Round",
            "Train_End_Date",
            "Test_Start_Date",
            "Test_End_Date",
            "Train_Rows",
            "Test_Rows",
            "Accuracy",
            "Precision_UP",
            "Recall_UP",
            "F1_UP",
        ]
    ].round(4).to_string(index=False)
)

print("\n--- CLASSIFICATION AVERAGES ---")

print(
    results_df[
        [
            "Accuracy",
            "Precision_UP",
            "Recall_UP",
            "F1_UP",
        ]
    ].mean().round(4)
)

print("\n--- TRADING RESULTS BY ROUND ---")

print(
    results_df[
        [
            "Round",
            "Trades",
            "Win_Rate",
            "Strategy_Return",
            "Benchmark_Return",
            "Excess_Return",
            "Max_Drawdown",
            "Sharpe_Ratio",
        ]
    ].round(4).to_string(index=False)
)

print("\n--- AGGREGATE TRADING SUMMARY ---")

# These are sums/averages of separate round-level results.
# They are not one continuous compounded portfolio.

print(
    "Total trades across rounds:",
    int(results_df["Trades"].sum()),
)

print(
    "Average strategy return per round:",
    f"{results_df['Strategy_Return'].mean():.2%}",
)

print(
    "Average benchmark return per round:",
    f"{results_df['Benchmark_Return'].mean():.2%}",
)

print(
    "Average excess return per round:",
    f"{results_df['Excess_Return'].mean():.2%}",
)

print(
    "Average round win rate:",
    f"{results_df['Win_Rate'].mean():.2%}",
)

print(
    "Average round maximum drawdown:",
    f"{results_df['Max_Drawdown'].mean():.2%}",
)

# =========================================================
# 10. SAVE REPORT
# =========================================================
os.makedirs(REPORT_DIR, exist_ok=True)

results_path = os.path.join(
    REPORT_DIR,
    "walk_forward_results.csv",
)

results_df.to_csv(results_path, index=False)

print("\nResults saved to", results_path)
