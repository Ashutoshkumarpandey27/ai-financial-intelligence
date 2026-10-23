
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -------------------- CONFIGURATION --------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "model_ready_dataset.csv"
REPORT_DIR = ROOT_DIR / "reports"

FEATURE_COLUMNS = [
    "Close",
    "Daily_Return",
    "Previous_Day_Return",
    "Return_2_Days_Ago",
    "MA_5",
    "Volume_Change",
]

THRESHOLDS = [0.50, 0.52, 0.55, 0.58, 0.60, 0.65]

INITIAL_CAPITAL = 10_000
TRANSACTION_COST = 0.001  # 0.1% per completed trade

INITIAL_TRAIN_RATIO = 0.60
TEST_RATIO = 0.10
MIN_VALIDATION_TRADES = 5


# -------------------- MODEL --------------------

def create_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression(max_iter=1000)),
    ])


# -------------------- STRATEGY METRICS --------------------

def calculate_metrics(probabilities, prices, threshold, round_number,
                      strategy_name):
    """
    Enter at next session's Open and exit at that session's Close.
    No trade means the strategy remains in cash.
    """

    signals = (probabilities >= threshold).astype(int)

    open_prices = prices["Next_Open"].to_numpy(dtype=float)
    close_prices = prices["Next_Close"].to_numpy(dtype=float)

    # Gross intraday return for each session.
    market_returns = close_prices / open_prices - 1

    # Cost is deducted on each completed round-trip trade.
    strategy_returns = np.where(
        signals == 1,
        market_returns - TRANSACTION_COST,
        0.0,
    )

    # Equity curve starts at 1.0.
    equity = np.concatenate([
        [1.0],
        np.cumprod(1.0 + strategy_returns),
    ])

    running_peak = np.maximum.accumulate(equity)
    drawdowns = equity / running_peak - 1.0
    max_drawdown = float(drawdowns.min())

    trades = int(signals.sum())

    if trades > 0:
        trade_returns = strategy_returns[signals == 1]
        win_rate = float((trade_returns > 0).mean())
        average_trade_return = float(trade_returns.mean())
    else:
        win_rate = np.nan
        average_trade_return = np.nan

    total_return = float(equity[-1] - 1.0)

    benchmark_equity = np.prod(1.0 + market_returns)
    benchmark_return = float(benchmark_equity - 1.0)

    daily_std = float(np.std(strategy_returns, ddof=1))

    if daily_std > 0 and len(strategy_returns) > 1:
        sharpe = float(
            np.mean(strategy_returns) / daily_std * np.sqrt(252)
        )
    else:
        sharpe = np.nan

    return {
        "Round": round_number,
        "Strategy": strategy_name,
        "Threshold": threshold,
        "Test_Start": prices["Date"].iloc[0].date(),
        "Test_End": prices["Date"].iloc[-1].date(),
        "Test_Rows": len(prices),
        "Trades": trades,
        "Win_Rate": win_rate,
        "Average_Trade_Return": average_trade_return,
        "Strategy_Return": total_return,
        "Benchmark_Return": benchmark_return,
        "Excess_Return": total_return - benchmark_return,
        "Max_Drawdown": max_drawdown,
        "Sharpe_Ratio": sharpe,
        "Final_Equity": INITIAL_CAPITAL * (1 + total_return),
    }


# -------------------- DATA VALIDATION --------------------

def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    required_columns = (
        ["Date"]
        + FEATURE_COLUMNS
        + ["Next_Open", "Next_Close", "Target"]
    )

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_columns = FEATURE_COLUMNS + [
        "Next_Open", "Next_Close", "Target"
    ]

    df[numeric_columns] = df[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    if df[required_columns].isna().any().any():
        raise ValueError(
            "Required data contains missing or invalid values. "
            "Clean the dataset before running this experiment."
        )

    if not np.isfinite(df[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains infinite values.")

    if not df["Target"].isin([0, 1]).all():
        raise ValueError("Target must contain only 0 and 1.")

    if (df["Next_Open"] <= 0).any() or (df["Next_Close"] <= 0).any():
        raise ValueError("Next_Open and Next_Close must be positive.")

    expected_target = (
        df["Next_Close"] > df["Next_Open"]
    ).astype(int)

    if not np.array_equal(
        expected_target.to_numpy(),
        df["Target"].to_numpy(dtype=int),
    ):
        raise ValueError(
            "Target does not match Next_Close > Next_Open. "
            "Check target construction before continuing."
        )

    return df


# -------------------- THRESHOLD SELECTION --------------------

def choose_threshold(train_df, round_number):
    """
    Select a threshold using a chronological inner validation split.
    The outer test window is never accessed here.
    """

    n = len(train_df)

    # Reserve the latest 20% of outer training data for validation.
    fit_end = int(n * 0.80)

    # Skip one row to prevent adjacent next-session labels overlapping.
    validation_start = fit_end + 1

    fit_df = train_df.iloc[:fit_end].copy()
    validation_df = train_df.iloc[validation_start:].copy()

    if len(fit_df) < 50 or len(validation_df) < 10:
        raise ValueError(
            f"Not enough data for inner validation in round {round_number}."
        )

    X_fit = fit_df[FEATURE_COLUMNS]
    y_fit = fit_df["Target"].astype(int)

    X_validation = validation_df[FEATURE_COLUMNS]

    if y_fit.nunique() < 2:
        raise ValueError(
            f"Training data in round {round_number} has only one class."
        )

    validation_model = create_model()
    validation_model.fit(X_fit, y_fit)

    probabilities = validation_model.predict_proba(
        X_validation
    )[:, 1]

    validation_results = []

    for threshold in THRESHOLDS:
        metrics = calculate_metrics(
            probabilities=probabilities,
            prices=validation_df,
            threshold=threshold,
            round_number=round_number,
            strategy_name="Validation",
        )

        validation_results.append(metrics)

    validation_results = pd.DataFrame(validation_results)

    # Avoid selecting a threshold based on very few trades.
    eligible = validation_results[
        validation_results["Trades"] >= MIN_VALIDATION_TRADES
    ].copy()

    if eligible.empty:
        # If no threshold reaches the minimum, use 0.50 as a
        # conservative fallback rather than optimizing a tiny sample.
        selected_threshold = 0.50
        print(
            f"Round {round_number}: no threshold generated at least "
            f"{MIN_VALIDATION_TRADES} validation trades; "
            "using threshold 0.50."
        )
    else:
        # Select using validation net strategy return only.
        # In a tie, prefer the threshold with fewer trades.
        eligible = eligible.sort_values(
            ["Strategy_Return", "Trades", "Threshold"],
            ascending=[False, True, True],
        )

        selected_threshold = float(
            eligible.iloc[0]["Threshold"]
        )

    return selected_threshold, validation_results


# -------------------- WALK-FORWARD TESTING --------------------

def main():
    df = load_dataset()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    n = len(df)
    initial_train_end = int(n * INITIAL_TRAIN_RATIO)
    test_size = max(1, int(n * TEST_RATIO))

    results = []
    validation_reports = []

    train_end = initial_train_end
    round_number = 1

    while train_end + 1 < n:
        # Skip one row between training and outer test.
        test_start = train_end + 1
        test_end = min(test_start + test_size, n)

        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()

        if len(test_df) == 0:
            break

        if train_df["Target"].nunique() < 2:
            raise ValueError(
                f"Outer training data in round {round_number} "
                "contains only one class."
            )

        selected_threshold, validation_results = choose_threshold(
            train_df, round_number
        )

        validation_results["Selected_Threshold"] = selected_threshold
        validation_reports.append(validation_results)

        # Refit using all outer training rows.
        # The selected threshold remains fixed for the test window.
        final_model = create_model()

        final_model.fit(
            train_df[FEATURE_COLUMNS],
            train_df["Target"].astype(int),
        )

        test_probabilities = final_model.predict_proba(
            test_df[FEATURE_COLUMNS]
        )[:, 1]

        # Compare selected threshold with 0.50 on the SAME test window.
        results.append(
            calculate_metrics(
                test_probabilities,
                test_df,
                selected_threshold,
                round_number,
                "Tuned Threshold",
            )
        )

        results.append(
            calculate_metrics(
                test_probabilities,
                test_df,
                0.50,
                round_number,
                "Fixed 0.50",
            )
        )

        print(f"\n{'=' * 65}")
        print(f"ROUND {round_number}")
        print(f"Training rows: {len(train_df)}")
        print(f"Training ends: {train_df['Date'].iloc[-1].date()}")
        print(
            f"Test period: {test_df['Date'].iloc[0].date()} to "
            f"{test_df['Date'].iloc[-1].date()}"
        )
        print(f"Selected threshold: {selected_threshold:.2f}")

        print("\nValidation returns by threshold:")
        print(
            validation_results[
                ["Threshold", "Trades", "Strategy_Return", "Max_Drawdown"]
            ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )

        round_results = pd.DataFrame(results)
        print("\nOuter test comparison:")
        print(
            round_results[round_results["Round"] == round_number][
                [
                    "Strategy", "Threshold", "Trades", "Win_Rate",
                    "Strategy_Return", "Benchmark_Return",
                    "Excess_Return", "Max_Drawdown", "Sharpe_Ratio",
                ]
            ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )

        # Expand training history for the next round.
        train_end = test_end
        round_number += 1

    if not results:
        raise ValueError("No walk-forward test rounds were produced.")

    results_df = pd.DataFrame(results)
    validation_df = pd.concat(validation_reports, ignore_index=True)

    results_path = REPORT_DIR / "threshold_tuning_results.csv"
    validation_path = REPORT_DIR / "threshold_validation_results.csv"
    summary_path = REPORT_DIR / "threshold_tuning_summary.csv"

    results_df.to_csv(results_path, index=False)
    validation_df.to_csv(validation_path, index=False)

    # Average returns across independent test rounds, not a single
    # compounded portfolio result.
    summary_df = (
        results_df.groupby("Strategy", as_index=False)
        .agg(
            Rounds=("Round", "nunique"),
            Average_Threshold=("Threshold", "mean"),
            Average_Trades=("Trades", "mean"),
            Average_Win_Rate=("Win_Rate", "mean"),
            Average_Strategy_Return=("Strategy_Return", "mean"),
            Average_Benchmark_Return=("Benchmark_Return", "mean"),
            Average_Excess_Return=("Excess_Return", "mean"),
            Average_Max_Drawdown=("Max_Drawdown", "mean"),
            Average_Sharpe=("Sharpe_Ratio", "mean"),
        )
    )

    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'=' * 65}")
    print("FINAL WALK-FORWARD SUMMARY")
    print(summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    ))

    print("\nReports saved:")
    print(results_path)
    print(validation_path)
    print(summary_path)

    print(
        "\nImportant: these are historical simulations, not guarantees "
        "of future profitability."
    )


if __name__ == "__main__":
    main()
