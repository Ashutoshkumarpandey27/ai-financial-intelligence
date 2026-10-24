import os
import joblib
import pandas as pd

# =========================================================
# 1. CONFIGURATION
# =========================================================

MODEL_PATH = "models/logistic_regression.pkl"
DATA_PATH = "data/model_ready_dataset.csv"

FEATURE_COLUMNS = [
    "Close",
    "Daily_Return",
    "Previous_Day_Return",
    "Return_2_Days_Ago",
    "MA_5",
    "Volume_Change",
]

REQUIRED_COLUMNS = ["Date", *FEATURE_COLUMNS]


# =========================================================
# 2. CHECK FILES
# =========================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")


# =========================================================
# 3. LOAD AND VALIDATE DATA
# =========================================================

model = joblib.load(MODEL_PATH)

data = pd.read_csv(DATA_PATH, parse_dates=["Date"])

missing_columns = set(REQUIRED_COLUMNS) - set(data.columns)

if missing_columns:
    raise ValueError(
        f"Dataset is missing columns: {sorted(missing_columns)}"
    )

if data.empty:
    raise ValueError("Dataset is empty.")

data = data.sort_values("Date").reset_index(drop=True)

if data["Date"].isna().any():
    raise ValueError("Date column contains missing values.")

if data["Date"].duplicated().any():
    raise ValueError("Duplicate dates found in the dataset.")

latest_data = data.iloc[[-1]].copy()

if latest_data[FEATURE_COLUMNS].isna().any().any():
    raise ValueError("Latest row contains missing feature values.")

if not latest_data[FEATURE_COLUMNS].apply(
    pd.to_numeric, errors="coerce"
).notna().all().all():
    raise ValueError("Latest row contains invalid feature values.")

latest_features = latest_data[FEATURE_COLUMNS].astype(float)

if model.n_features_in_ != len(FEATURE_COLUMNS):
    raise ValueError("Model feature count does not match input features.")


# =========================================================
# 4. GENERATE PREDICTION
# =========================================================

prediction = int(model.predict(latest_features)[0])
probabilities = model.predict_proba(latest_features)[0]

probability_not_up = float(probabilities[0])
probability_up = float(probabilities[1])

latest_date = latest_data["Date"].iloc[0].date()


# =========================================================
# 5. DISPLAY RESULTS
# =========================================================

print("\n--- AI FINANCIAL INTELLIGENCE ENGINE ---")
print("Signal date:", latest_date)
print("Prediction target: Next trading session Close vs Open")

print(
    "Predicted direction:",
    "UP" if prediction == 1 else "NOT UP"
)

print(f"Probability of NOT UP: {probability_not_up:.2%}")
print(f"Probability of UP:     {probability_up:.2%}")

print("\nNote: Probabilities are model estimates, not guarantees.")
print("This prediction is for research and educational purposes.")
