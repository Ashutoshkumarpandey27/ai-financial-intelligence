
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "model_ready_dataset.csv"
REPORT_DIR = ROOT / "reports"

FEATURES = [
    "Close",
    "Daily_Return",
    "Previous_Day_Return",
    "Return_2_Days_Ago",
    "MA_5",
    "Volume_Change",
]

THRESHOLDS = [0.50, 0.52, 0.55, 0.58, 0.60, 0.65]
COSTS = [0.001, 0.002, 0.003]

INITIAL_CAPITAL = 10_000
INITIAL_TRAIN_RATIO = 0.60
TEST_RATIO = 0.10

MIN_TOTAL_VALIDATION_TRADES = 10
MIN_ACTIVE_VALIDATION_FOLDS = 2
MIN_TRADES_PER_ACTIVE_FOLD = 3


def create_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000)),
    ])


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    required = FEATURES + [
        "Date", "Target", "Next_Open", "Next_Close"
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    numeric_cols = FEATURES + ["Target", "Next_Open", "Next_Close"]
    df[numeric_cols] = df[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    if df[required].isna().any().any():
        raise ValueError("Required columns contain missing values.")

    if not np.isfinite(df[numeric_cols].to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains infinite values.")

    if not df["Target"].isin([0, 1]).all():
        raise ValueError("Target must contain only 0 and 1.")

    if (df[["Next_Open", "Next_Close"]] <= 0).any().any():
        raise ValueError("Execution prices must be positive.")

    expected = (df["Next_Close"] > df["Next_Open"]).astype(int)

    if not np.array_equal(expected.to_numpy(), df["Target"].to_numpy()):
        raise ValueError("Target alignment check failed.")

    return df


def strategy_metrics(probabilities, prices, threshold, cost):
    signals = (probabilities >= threshold).astype(int)

    open_prices = prices["Next_Open"].to_numpy(dtype=float)
    close_prices = prices["Next_Close"].to_numpy(dtype=float)

    market_returns = close_prices / open_prices - 1.0

    # Flat position when signal is 0.
    # Cost is deducted for every completed trade.
    returns = np.where(
        signals == 1,
        market_returns - cost,
        0.0,
    )

    equity = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1.0

    std = np.std(returns, ddof=1) if len(returns) > 1 else 0.0

    sharpe = (
        np.mean(returns) / std * np.sqrt(252)
        if std > 0 else np.nan
    )

    trades = int(signals.sum())
    trade_returns = returns[signals == 1]

    return {
        "Threshold": threshold,
        "Cost": cost,
        "Trades": trades,
        "Win_Rate": (
            float((trade_returns > 0).mean())
            if trades else np.nan
        ),
        "Strategy_Return": float(equity[-1] - 1.0),
        "Benchmark_Return": float(
            np.prod(1.0 + market_returns) - 1.0
        ),
        "Excess_Return": float(
            equity[-1] - 1.0 - (np.prod(1.0 + market_returns) - 1.0)
        ),
        "Max_Drawdown": float(drawdowns.min()),
        "Sharpe_Ratio": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "Final_Equity": float(INITIAL_CAPITAL * equity[-1]),
    }


def select_threshold(outer_train, round_number):
    """
    Select a threshold from three chronological validation windows.
    Each window uses only preceding rows for fitting, with a one-row gap.
    """

    n = len(outer_train)
    validation_size = max(30, int(n * 0.10))

    # Starts at 60%, 70%, and 80% of the outer training history.
    starts = [
        int(n * 0.60),
        int(n * 0.70),
        int(n * 0.80),
    ]

    fold_records = []

    for fold_number, start in enumerate(starts, start=1):
        end = min(start + validation_size, n)

        # Row immediately before validation is left out of fitting.
        fit_df = outer_train.iloc[:start - 1].copy()
        val_df = outer_train.iloc[start:end].copy()

        if len(fit_df) < 50 or len(val_df) < 10:
            continue

        if fit_df["Target"].nunique() < 2:
            continue

        model = create_model()
        model.fit(fit_df[FEATURES], fit_df["Target"].astype(int))

        probabilities = model.predict_proba(val_df[FEATURES])[:, 1]

        for threshold in THRESHOLDS:
            metrics = strategy_metrics(
                probabilities, val_df, threshold, cost=0.001
            )

            fold_records.append({
                "Round": round_number,
                "Validation_Fold": fold_number,
                "Validation_Start": val_df["Date"].iloc[0].date(),
                "Validation_End": val_df["Date"].iloc[-1].date(),
                **metrics,
            })

    validation = pd.DataFrame(fold_records)

    if validation.empty:
        raise ValueError(
            f"No valid inner validation windows in round {round_number}."
        )

    # A threshold must have enough trades overall and activity in
    # multiple folds. This reduces selection based on tiny samples.
    summary = (
        validation.groupby("Threshold")
        .agg(
            Total_Trades=("Trades", "sum"),
            Active_Folds=(
                "Trades",
                lambda x: int((x >= MIN_TRADES_PER_ACTIVE_FOLD).sum()),
            ),
            Mean_Validation_Return=("Strategy_Return", "mean"),
            Mean_Validation_Drawdown=("Max_Drawdown", "mean"),
        )
        .reset_index()
    )

    eligible = summary[
        (summary["Total_Trades"] >= MIN_TOTAL_VALIDATION_TRADES)
        & (summary["Active_Folds"] >= MIN_ACTIVE_VALIDATION_FOLDS)
    ].copy()

    if eligible.empty:
        selected = 0.50
        print(
            f"Round {round_number}: no threshold met validation "
            "trade requirements; fallback to 0.50."
        )
    else:
        eligible = eligible.sort_values(
            ["Mean_Validation_Return", "Mean_Validation_Drawdown"],
            ascending=[False, False],
        )
        selected = float(eligible.iloc[0]["Threshold"])

    summary["Selected_Threshold"] = selected

    return selected, validation, summary


def main():
    df = load_data()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    n = len(df)
    train_end = int(n * INITIAL_TRAIN_RATIO)
    test_size = max(1, int(n * TEST_RATIO))

    test_results = []
    validation_results = []
    selection_summaries = []

    round_number = 1

    while train_end + 1 < n:
        test_start = train_end + 1
        test_end = min(test_start + test_size, n)

        outer_train = df.iloc[:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()

        if test_df.empty:
            break

        if outer_train["Target"].nunique() < 2:
            raise ValueError(
                f"Outer training round {round_number} has one class."
            )

        selected, validation, selection_summary = select_threshold(
            outer_train, round_number
        )

        validation_results.append(validation)
        selection_summaries.append(selection_summary)

        # Refit on all outer training rows; test labels remain untouched.
        final_model = create_model()
        final_model.fit(
            outer_train[FEATURES],
            outer_train["Target"].astype(int),
        )

        probabilities = final_model.predict_proba(test_df[FEATURES])[:, 1]

        print(f"\n{'=' * 70}")
        print(f"ROUND {round_number}")
        print(
            f"Test: {test_df['Date'].iloc[0].date()} to "
            f"{test_df['Date'].iloc[-1].date()}"
        )
        print(f"Selected threshold: {selected:.2f}")

        print("\nInner validation summary:")
        print(
            selection_summary.to_string(
                index=False, float_format=lambda x: f"{x:.4f}"
            )
        )

        for threshold_name, threshold in [
            ("Tuned", selected),
            ("Fixed_0.50", 0.50),
        ]:
            for cost in COSTS:
                metrics = strategy_metrics(
                    probabilities, test_df, threshold, cost
                )

                record = {
                    "Round": round_number,
                    "Strategy": threshold_name,
                    "Selected_Threshold": selected,
                    "Threshold_Used": threshold,
                    "Test_Start": test_df["Date"].iloc[0].date(),
                    "Test_End": test_df["Date"].iloc[-1].date(),
                    **metrics,
                }

                test_results.append(record)

        current = pd.DataFrame(test_results)
        current = current[current["Round"] == round_number]

        print("\nUntouched test comparison:")
        print(
            current[
                [
                    "Strategy", "Threshold_Used", "Cost", "Trades",
                    "Strategy_Return", "Benchmark_Return",
                    "Excess_Return", "Max_Drawdown", "Sharpe_Ratio",
                ]
            ].to_string(
                index=False, float_format=lambda x: f"{x:.4f}"
            )
        )

        train_end = test_end
        round_number += 1

    test_df = pd.DataFrame(test_results)
    validation_df = pd.concat(validation_results, ignore_index=True)
    selection_df = pd.concat(selection_summaries, ignore_index=True)

    test_df.to_csv(
        REPORT_DIR / "robustness_test_results.csv", index=False
    )
    validation_df.to_csv(
        REPORT_DIR / "robustness_validation_folds.csv", index=False
    )
    selection_df.to_csv(
        REPORT_DIR / "robustness_threshold_selection.csv", index=False
    )

    summary = (
        test_df.groupby(["Strategy", "Cost"], as_index=False)
        .agg(
            Rounds=("Round", "nunique"),
            Average_Threshold=("Threshold_Used", "mean"),
            Average_Trades=("Trades", "mean"),
            Average_Win_Rate=("Win_Rate", "mean"),
            Average_Strategy_Return=("Strategy_Return", "mean"),
            Average_Benchmark_Return=("Benchmark_Return", "mean"),
            Average_Excess_Return=("Excess_Return", "mean"),
            Average_Max_Drawdown=("Max_Drawdown", "mean"),
            Average_Sharpe=("Sharpe_Ratio", "mean"),
        )
    )

    summary.to_csv(
        REPORT_DIR / "robustness_summary.csv", index=False
    )

    print(f"\n{'=' * 70}")
    print("ROBUSTNESS SUMMARY")
    print(
        summary.to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nReports saved in:", REPORT_DIR)
    print(
        "\nNote: average round returns are not compounded portfolio returns. "
        "Costs are illustrative assumptions, not broker-specific estimates."
    )


if __name__ == "__main__":
    main()
