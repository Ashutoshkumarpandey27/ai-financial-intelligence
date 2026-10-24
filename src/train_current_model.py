
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "model_ready_dataset.csv"
RAW_PATH = ROOT / "data" / "reliance_stock_data.csv"

MODEL_PATH = ROOT / "models" / "logistic_regression_current.pkl"
METADATA_PATH = (
    ROOT / "models" / "logistic_regression_current_metadata.pkl"
)

FEATURES = [
    "Close",
    "Daily_Return",
    "Previous_Day_Return",
    "Return_2_Days_Ago",
    "MA_5",
    "Volume_Change",
]

TARGET = "Target"


def create_model():
    """Create a fresh Logistic Regression pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])


def main():

    # ========================================================
    # 2. CHECK FILES
    # ========================================================

    for path in [DATA_PATH, RAW_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    raw = pd.read_csv(RAW_PATH, parse_dates=["Date"])

    required = ["Date", TARGET, *FEATURES]
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    if df.empty or raw.empty:
        raise ValueError("Raw or model-ready dataset is empty.")

    if df["Date"].isna().any() or raw["Date"].isna().any():
        raise ValueError("A dataset contains invalid dates.")

    # ========================================================
    # 3. VALIDATE AND SORT DATES
    # ========================================================

    df = df.sort_values("Date").reset_index(drop=True)
    raw = raw.sort_values("Date").reset_index(drop=True)

    if df["Date"].duplicated().any():
        raise ValueError("Duplicate dates in model-ready dataset.")

    if raw["Date"].duplicated().any():
        raise ValueError("Duplicate dates in raw dataset.")

    latest_raw_date = raw["Date"].max()

    # A row's target uses the NEXT raw trading session's prices.
    # Find that next session date to check label availability.
    raw_dates = raw[["Date"]].copy()
    raw_dates["Label_Available_Date"] = raw_dates["Date"].shift(-1)

    df = df.merge(
        raw_dates,
        on="Date",
        how="left",
        validate="one_to_one",
    )

    # Keep only rows whose next-session target can be observed
    # using data available through the latest raw-data date.
     # Only use labels known strictly before the signal date.
    signal_date = latest_raw_date

    df = df[
        df["Label_Available_Date"].notna()
        & (df["Label_Available_Date"] < signal_date)
    ].copy()
    df = df.dropna(subset=required).reset_index(drop=True)

    if len(df) < 100:
        raise ValueError("Not enough eligible rows for model training.")

    for column in FEATURES + [TARGET]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)

    feature_values = df[FEATURES].to_numpy(dtype=float)

    if not np.isfinite(feature_values).all():
        raise ValueError("Features contain infinite or invalid values.")

    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("Target must contain only 0 and 1.")

    df[TARGET] = df[TARGET].astype(int)

    if df[TARGET].nunique() != 2:
        raise ValueError("Training data must contain both target classes.")

    # ========================================================
    # 4. CHRONOLOGICAL HOLDOUT EVALUATION
    # ========================================================

    split = int(len(df) * 0.80)

    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    if train[TARGET].nunique() != 2:
        raise ValueError(
            "Training portion must contain both target classes."
        )

    X_train = train[FEATURES].astype(float)
    y_train = train[TARGET]

    X_test = test[FEATURES].astype(float)
    y_test = test[TARGET]

    evaluation_model = create_model()
    evaluation_model.fit(X_train, y_train)

    predictions = evaluation_model.predict(X_test)

    metrics = {
        "holdout_accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "baseline_accuracy": float(
            max(y_test.mean(), 1 - y_test.mean())
        ),
        "precision_up": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall_up": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_up": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
    }

    # ========================================================
    # 5. TRAIN SEPARATE CURRENT MODEL
    # ========================================================

    # Evaluation above uses only the first 80% for fitting.
    # This separate model uses all eligible labeled rows for
    # predictions made AFTER those labels became available.
    current_model = create_model()

    current_model.fit(
        df[FEATURES].astype(float),
        df[TARGET],
    )

    # ========================================================
    # 6. SAVE MODEL AND METADATA SEPARATELY
    # ========================================================

    metadata = {
        "target_definition": "Next session Close > Next session Open",
        "feature_columns": FEATURES,
        "training_start": str(df["Date"].min().date()),
        "training_end": str(df["Date"].max().date()),
        "training_labels_available_through": str(
            df["Label_Available_Date"].max().date()
        ),
        "latest_raw_data_date": str(latest_raw_date.date()),
        "training_rows": int(len(df)),
        "holdout_start": str(test["Date"].min().date()),
        "holdout_end": str(test["Date"].max().date()),
        "holdout_rows": int(len(test)),
        **metrics,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(current_model, MODEL_PATH)
    joblib.dump(metadata, METADATA_PATH)

    # ========================================================
    # 7. DISPLAY REPORT
    # ========================================================

    print("\n--- CURRENT MODEL TRAINING REPORT ---")

    for key, value in metadata.items():
        print(f"{key}: {value}")

    print("\nSaved current model:", MODEL_PATH)
    print("Saved current metadata:", METADATA_PATH)

    print("\nOriginal model and historical reports were not overwritten.")
    print("Holdout metrics evaluate the earlier 80/20 chronological split.")
    print("The saved current model is fitted on all eligible labeled rows.")


if __name__ == "__main__":
    main()
