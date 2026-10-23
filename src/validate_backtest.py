
import pandas as pd
import os

# 1. Define report paths
results_path = "reports/backtest_results.csv"
summary_path = "reports/backtest_summary.csv"

# 2. Confirm that both files exist
for path in [results_path, summary_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Report not found: {path}")

# 3. Load the saved reports
results = pd.read_csv(results_path)
summary = pd.read_csv(summary_path)

# 4. Check whether reports contain data
if results.empty:
    raise ValueError("Detailed backtest report is empty.")

if summary.empty:
    raise ValueError("Backtest summary is empty.")

# 5. Check required columns
required_results = [
    "Date",
    "Prediction",
    "Strategy_Return",
    "Strategy_Equity",
    "Benchmark_Equity",
    "Drawdown",
]

missing_columns = [
    col for col in required_results
    if col not in results.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# 6. Check for missing values in important columns
check_columns = [
    "Date",
    "Prediction",
    "Strategy_Return",
    "Strategy_Equity",
    "Benchmark_Equity",
    "Drawdown",
]

print("Detailed report rows:", len(results))
print("\nMissing values:")
print(results[check_columns].isna().sum())

# 7. Compare saved final equity with summary
summary_values = summary.set_index("Metric")["Value"]

saved_final_equity = float(
    summary_values["Final Strategy Equity"]
)

actual_final_equity = float(
    results["Strategy_Equity"].iloc[-1]
)

print("\nFinal strategy equity in summary:",
      round(saved_final_equity, 2))

print("Final strategy equity in detailed report:",
      round(actual_final_equity, 2))

if abs(saved_final_equity - actual_final_equity) < 0.01:
    print("\nValidation passed: final equity matches.")
else:
    raise ValueError("Final equity does not match.")

print("\nBacktest reports validated successfully!")
