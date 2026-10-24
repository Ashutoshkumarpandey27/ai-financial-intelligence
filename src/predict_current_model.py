
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "reliance_stock_data.csv"
MODEL_PATH = ROOT / "models" / "logistic_regression_current.pkl"
META_PATH = ROOT / "models" / "logistic_regression_current_metadata.pkl"

FEATURES = [
    "Close",
    "Daily_Return",
    "Previous_Day_Return",
    "Return_2_Days_Ago",
    "MA_5",
    "Volume_Change",
]


def main():
    for path in [RAW_PATH, MODEL_PATH, META_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    data = pd.read_csv(RAW_PATH, parse_dates=["Date"])
    required = ["Date", "Open", "Close", "Volume"]
    missing = set(required) - set(data.columns)

    if missing:
        raise ValueError(f"Raw data missing columns: {sorted(missing)}")

    if data.empty or data["Date"].isna().any():
        raise ValueError("Raw data is empty or has invalid dates.")

    if data["Date"].duplicated().any():
        raise ValueError("Duplicate dates found.")

    data = data.sort_values("Date").reset_index(drop=True)

    for column in ["Open", "Close", "Volume"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data[["Open", "Close", "Volume"]].isna().any().any():
        raise ValueError("Raw data contains missing prices or volume.")

    if not np.isfinite(
        data[["Open", "Close", "Volume"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("Raw data contains invalid numeric values.")

    # Features available at the close of each session.
    data["Daily_Return"] = data["Close"].pct_change()
    data["Previous_Day_Return"] = data["Daily_Return"].shift(1)
    data["Return_2_Days_Ago"] = data["Daily_Return"].shift(2)
    data["MA_5"] = data["Close"].rolling(window=5).mean()
    data["Volume_Change"] = data["Volume"].pct_change()

    data["Volume_Change"] = data["Volume_Change"].replace(
        [np.inf, -np.inf], np.nan
    )

    latest = data.iloc[[-1]].copy()

    if latest[FEATURES].isna().any().any():
        raise ValueError("Latest row has missing feature values.")

    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(META_PATH)

    if metadata.get("feature_columns") != FEATURES:
        raise ValueError("Model feature order does not match metadata.")

    if model.n_features_in_ != len(FEATURES):
        raise ValueError("Model feature count does not match.")

    signal_date = latest["Date"].iloc[0].date()
    label_available_through = pd.Timestamp(
        metadata["training_labels_available_through"]
    ).date()

    # A model must not use labels that were unavailable on the signal date.
    if label_available_through >= signal_date:
        raise ValueError(
            f"Unsafe prediction: model labels are available through "
            f"{label_available_through}, but signal date is {signal_date}. "
            "Use a model trained only on labels known before this date."
        )

    X = latest[FEATURES].astype(float)
    prediction = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]
    classes = list(model.classes_)

    prob_up = float(probabilities[classes.index(1)])
    prob_not_up = float(probabilities[classes.index(0)])

    print("\n--- CURRENT MODEL PREDICTION ---")
    print("Signal date:", signal_date)
    print("Target: next session Close vs Open")
    print("Prediction:", "UP" if prediction == 1 else "NOT UP")
    print(f"Probability of UP: {prob_up:.2%}")
    print(f"Probability of NOT UP: {prob_not_up:.2%}")
    print("Training labels available through:", label_available_through)
    print("\nProbabilities are model estimates, not guarantees.")


if __name__ == "__main__":
    main()
