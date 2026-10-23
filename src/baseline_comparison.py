
import os
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================================================
# 1. CONFIGURATION
# =========================================================
DATA_PATH = "data/model_ready_dataset.csv"
REPORT_DIR = "reports"

INITIAL_CAPITAL = 10000.0
TRANSACTION_COST = 0.001
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
data = data.sort_values("Date").reset_index(drop=True)

required = FEATURE_COLUMNS + [
    "Target",
    "Next_Open",
    "Next_Close",
]

missing = set(required) - set(data.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

if data.empty or data["Date"].duplicated().any():
    raise ValueError("Dataset is empty or contains duplicate dates.")

for column in required:
    data[column] = pd.to_numeric(data[column], errors="raise")

if data[required].isna().any().any():
    raise ValueError("Missing values found.")

if not np.isfinite(data[required].to_numpy(dtype=float)).all():
    raise ValueError("Non-finite values found.")

if not data["Target"].isin([0, 1]).all():
    raise ValueError("Target must contain only 0 and 1.")

expected_target = (
    data["Next_Close"] > data["Next_Open"]
).astype(int)

if not np.array_equal(data["Target"].to_numpy(), expected_target.to_numpy()):
    raise ValueError("Target alignment check failed.")

if (data[["Next_Open", "Next_Close"]] <= 0).any().any():
    raise ValueError("Execution prices must be positive.")

# =========================================================
# 3. MODEL AND WALK-FORWARD SETTINGS
# =========================================================
X = data[FEATURE_COLUMNS]
y = data["Target"].astype(int)

base_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000)),
])

initial_train_size = int(len(data) * INITIAL_TRAIN_RATIO)
test_window = int(len(data) * TEST_WINDOW_RATIO)

if test_window < 1:
    raise ValueError("Test window is too small.")

# Collect out-of-sample predictions for every round.
all_predictions = []
train_end = initial_train_size
round_number = 1

while train_end + 1 + test_window <= len(data):

    test_start = train_end + 1
    test_end = test_start + test_window

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_test = X.iloc[test_start:test_end]
    y_test = y.iloc[test_start:test_end]

    test_data = data.iloc[test_start:test_end].copy()

    model = clone(base_model)
    model.fit(X_train, y_train)

    model_predictions = model.predict(X_test)

    # Majority class is determined using training data only.
    majority_class = int(y_train.mean() >= 0.5)

    # Baselines predict on the same unseen test rows.
    test_data["Model_Prediction"] = model_predictions
    test_data["Majority_Prediction"] = majority_class
    test_data["Always_UP_Prediction"] = 1

    test_data["Round"] = round_number

    all_predictions.append(test_data)

    train_end = test_end
    round_number += 1

if not all_predictions:
    raise ValueError("No walk-forward test rounds completed.")

oos = pd.concat(all_predictions, ignore_index=True)

# =========================================================
# 4. EVALUATE EACH STRATEGY
# =========================================================
def evaluate_strategy(frame, prediction_column, strategy_name):
    result = frame.copy()

    predictions = result[prediction_column].astype(int)

    result["Market_Return"] = (
        result["Next_Close"] / result["Next_Open"]
    ) - 1.0

    result["Is_Trade"] = (predictions == 1).astype(int)

    result["Strategy_Return"] = np.where(
        result["Is_Trade"] == 1,
        (
            (1.0 + result["Market_Return"])
            * (1.0 - TRANSACTION_COST)
        ) - 1.0,
        0.0,
    )

    # Evaluate each round separately because each round has
    # a separately trained model and starts with fresh capital.
    round_results = []

    for round_id, group in result.groupby("Round", sort=True):
        actual = group["Target"].astype(int)
        predicted = group[prediction_column].astype(int)

        returns = group["Strategy_Return"]
        trades = group.loc[group["Is_Trade"] == 1, "Strategy_Return"]

        equity = INITIAL_CAPITAL * (1.0 + returns).cumprod()

        equity_with_start = pd.concat(
            [
                pd.Series([INITIAL_CAPITAL]),
                equity.reset_index(drop=True),
            ],
            ignore_index=True,
        )

        peaks = equity_with_start.cummax().iloc[1:].to_numpy()

        drawdown = (equity.to_numpy() / peaks) - 1.0

        volatility = returns.std(ddof=1)

        if pd.notna(volatility) and volatility > 0:
            sharpe = (
                returns.mean() / volatility
            ) * np.sqrt(252)
        else:
            sharpe = np.nan

        round_results.append({
            "Strategy": strategy_name,
            "Round": int(round_id),
            "Test_Start": group["Date"].min().date(),
            "Test_End": group["Date"].max().date(),
            "Accuracy": accuracy_score(actual, predicted),
            "Trades": int(group["Is_Trade"].sum()),
            "Winning_Trades": int((trades > 0).sum()),
            "Win_Rate": (
                float((trades > 0).mean())
                if len(trades) > 0 else np.nan
            ),
            "Strategy_Return": float(
                (1.0 + returns).prod() - 1.0
            ),
            "Benchmark_Return": float(
                (1.0 + group["Market_Return"]).prod() - 1.0
            ),
            "Excess_Return": float(
                ((1.0 + returns).prod() - 1.0)
                - ((1.0 + group["Market_Return"]).prod() - 1.0)
            ),
            "Max_Drawdown": float(drawdown.min()),
            "Sharpe_Ratio": float(sharpe),
            "Final_Equity": float(equity.iloc[-1]),
        })

    return round_results


# =========================================================
# 5. RUN ALL THREE STRATEGIES
# =========================================================
strategies = [
    ("Model_Prediction", "Logistic Regression"),
    ("Majority_Prediction", "Majority-Class Baseline"),
    ("Always_UP_Prediction", "Always-UP Baseline"),
]

all_results = []

for prediction_column, strategy_name in strategies:
    all_results.extend(
        evaluate_strategy(
            oos,
            prediction_column,
            strategy_name,
        )
    )

results_df = pd.DataFrame(all_results)

# =========================================================
# 6. DISPLAY RESULTS
# =========================================================
print("\n--- WALK-FORWARD BASELINE COMPARISON ---\n")

print(
    results_df[
        [
            "Strategy",
            "Round",
            "Accuracy",
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

print("\n--- AVERAGE PERFORMANCE BY STRATEGY ---\n")

summary = (
    results_df.groupby("Strategy")
    .agg(
        Average_Accuracy=("Accuracy", "mean"),
        Average_Trades=("Trades", "mean"),
        Average_Win_Rate=("Win_Rate", "mean"),
        Average_Return_Per_Round=("Strategy_Return", "mean"),
        Average_Benchmark_Return=("Benchmark_Return", "mean"),
        Average_Excess_Return=("Excess_Return", "mean"),
        Average_Max_Drawdown=("Max_Drawdown", "mean"),
        Average_Sharpe=("Sharpe_Ratio", "mean"),
    )
    .reset_index()
)

print(summary.round(4).to_string(index=False))

# =========================================================
# 7. SAVE REPORTS
# =========================================================
os.makedirs(REPORT_DIR, exist_ok=True)

round_report = os.path.join(
    REPORT_DIR,
    "baseline_comparison_results.csv",
)

summary_report = os.path.join(
    REPORT_DIR,
    "baseline_comparison_summary.csv",
)

results_df.to_csv(round_report, index=False)
summary.to_csv(summary_report, index=False)

print("\nReports saved:")
print(round_report)
print(summary_report)
