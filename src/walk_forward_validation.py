
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
TRANSACTION_COST = 0.001  # Assumed round-trip cost: 0.1%
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
        f"Dataset is missing columns: {sorted(missing_columns)}"
    )

if data.empty:
    raise ValueError("Dataset is empty.")

if data["Date"].isna().any():
    raise ValueError("Missing or invalid dates found.")

data = data.sort_values("Date").reset_index(drop=True)

if data["Date"].duplicated().any():
    raise ValueError("Duplicate dates found.")

numeric_columns = FEATURE_COLUMNS + [
    "Target",
    "Next_Open",
    "Next_Close",
]

for column in numeric_columns:
    data[column] = pd.to_numeric(data[column], errors="raise")

if data[numeric_columns].isna().any().any():
    raise ValueError("Missing values found in required columns.")

if not np.isfinite(data[numeric_columns].to_numpy(dtype=float)).all():
    raise ValueError("Non-finite values found in required columns.")

if not data["Target"].isin([0, 1]).all():
    raise ValueError("Target must contain only 0 and 1.")

if (data[["Next_Open", "Next_Close"]] <= 0).any().any():
    raise ValueError("Execution prices must be positive.")

# Verify that the target matches the next session's intraday direction.
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

if not 0 < INITIAL_TRAIN_RATIO < 1:
    raise ValueError("INITIAL_TRAIN_RATIO must be between 0 and 1.")

if not 0 < TEST_WINDOW_RATIO < 1:
    raise ValueError("TEST_WINDOW_RATIO must be between 0 and 1.")

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
initial_train_size = int(len(data) * INITIAL_TRAIN_RATIO)
test_window = int(len(data) * TEST_WINDOW_RATIO)

if initial_train_size < 2:
    raise ValueError("Initial training set is too small.")

if test_window < 1:
    raise ValueError("Test window is too small.")

results = []
round_number = 1
train_end = initial_train_size

# Each round trains on historical rows, leaves one row as a gap,
# and evaluates on the next chronological test window.
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
        y_test, predictions, zero_division=0
    )

    recall_up = recall_score(
        y_test, predictions, zero_division=0
    )

    f1_up = f1_score(
        y_test, predictions, zero_division=0
    )

    # =====================================================
    # 6. SIMULATE STRATEGY AND BENCHMARK
    # =====================================================
    test_data["Prediction"] = predictions

    # Next-session Open-to-Close market return.
    test_data["Market_Return"] = (
        test_data["Next_Close"] / test_data["Next_Open"]
    ) - 1.0

    # Trade only when the model predicts UP.
    test_data["Is_Trade"] = (
        test_data["Prediction"] == 1
    ).astype(int)

    # Strategy return before transaction costs.
    test_data["Strategy_Return_Before_Cost"] = np.where(
        test_data["Is_Trade"] == 1,
        test_data["Market_Return"],
        0.0,
    )

    # Strategy pays transaction costs only when it trades.
    test_data["Strategy_Return"] = np.where(
        test_data["Is_Trade"] == 1,
        (
            (1.0 + test_data["Market_Return"])
            * (1.0 - TRANSACTION_COST)
        ) - 1.0,
        0.0,
    )

    # Gross benchmark: trades every test session, before costs.
    test_data["Benchmark_Return"] = test_data["Market_Return"]

    # Cost-adjusted benchmark: assumes a trade every session.
    test_data["Benchmark_Return_After_Cost"] = (
        (1.0 + test_data["Market_Return"])
        * (1.0 - TRANSACTION_COST)
    ) - 1.0

    # Compound returns within each test round.
    strategy_return = (
        (1.0 + test_data["Strategy_Return"]).prod()
    ) - 1.0

    benchmark_return = (
        (1.0 + test_data["Benchmark_Return"]).prod()
    ) - 1.0

    benchmark_return_after_cost = (
        (1.0 + test_data["Benchmark_Return_After_Cost"]).prod()
    ) - 1.0

    # These comparisons have different trading exposure.
    excess_return = strategy_return - benchmark_return

    excess_return_after_cost = (
        strategy_return - benchmark_return_after_cost
    )

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

    test_data["Benchmark_Equity_After_Cost"] = (
        INITIAL_CAPITAL
        * (1.0 + test_data["Benchmark_Return_After_Cost"]).cumprod()
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
        "Train_End_Date": data["Date"].iloc[train_end - 1].date(),
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
        "Excess_Return": excess_return,
        "Benchmark_Return_After_Cost": benchmark_return_after_cost,
        "Excess_Return_After_Cost": excess_return_after_cost,
        "Max_Drawdown": max_drawdown,
        "Sharpe_Ratio": sharpe_ratio,
        "Final_Strategy_Equity": float(
            test_data["Strategy_Equity"].iloc[-1]
        ),
        "Final_Benchmark_Equity": float(
            test_data["Benchmark_Equity"].iloc[-1]
        ),
        "Final_Benchmark_Equity_After_Cost": float(
            test_data["Benchmark_Equity_After_Cost"].iloc[-1]
        ),
    })

    # Advance chronologically.
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

print("\n--- WALK-FORWARD CLASSIFICATION RESULTS ---\n")

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

print("\n--- CLASSIFICATION AVERAGES ---\n")

print(
    results_df[
        ["Accuracy", "Precision_UP", "Recall_UP", "F1_UP"]
    ].mean().round(4)
)

print("\n--- TRADING RESULTS BY ROUND ---\n")

print(
    results_df[
        [
            "Round",
            "Trades",
            "Win_Rate",
            "Strategy_Return",
            "Benchmark_Return",
            "Excess_Return",
            "Benchmark_Return_After_Cost",
            "Excess_Return_After_Cost",
            "Max_Drawdown",
            "Sharpe_Ratio",
        ]
    ].round(4).to_string(index=False)
)

print("\n--- AVERAGE PERFORMANCE PER ROUND ---\n")

summary_columns = [
    "Accuracy",
    "Trades",
    "Win_Rate",
    "Strategy_Return",
    "Benchmark_Return",
    "Excess_Return",
    "Benchmark_Return_After_Cost",
    "Excess_Return_After_Cost",
    "Max_Drawdown",
    "Sharpe_Ratio",
]

print(results_df[summary_columns].mean(numeric_only=True).round(4))

print("\nTotal trades across rounds:", int(results_df["Trades"].sum()))

print(
    "Assumed round-trip transaction cost:",
    f"{TRANSACTION_COST:.2%}",
)

print(
    "Note: Gross benchmark trades every test session. "
    "The cost-adjusted benchmark also assumes a trade every session. "
    "Model strategy trades only on UP predictions; exposures differ."
)

print(
    "Note: Average round returns summarize separate test windows; "
    "they are not one continuous compounded portfolio."
)


# =========================================================
# 10. SAVE REPORT
# =========================================================
os.makedirs(REPORT_DIR, exist_ok=True)

results_path = os.path.join(
    REPORT_DIR,
    "walk_forward_results.csv",
)

summary_path = os.path.join(
    REPORT_DIR,
    "walk_forward_summary.csv",
)

results_df.to_csv(results_path, index=False)

summary_df = results_df[summary_columns].mean(
    numeric_only=True
).to_frame(name="Average_Per_Round")

summary_df.to_csv(summary_path)

print("\nReports saved:")
print(results_path)
print(summary_path)
