
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# ============================================================
# 1. LOAD AND VALIDATE DATA
# ============================================================

data_path = "data/model_ready_dataset.csv"
model_path = "models/logistic_regression.pkl"

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found: {data_path}")

data = pd.read_csv(data_path)

required_columns = [
    "Date",
    "Close",
    "Daily_Return",
    "Previous_Day_Return",
    "Return_2_Days_Ago",
    "MA_5",
    "Volume_Change",
    "Target",
]

missing = set(required_columns) - set(data.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

if data.empty:
    raise ValueError("Dataset is empty.")

data["Date"] = pd.to_datetime(data["Date"], errors="raise")
data = data.sort_values("Date").reset_index(drop=True)

if data["Date"].duplicated().any():
    raise ValueError("Duplicate dates found.")

feature_columns = [
    "Close",
    "Daily_Return",
    "Previous_Day_Return",
    "Return_2_Days_Ago",
    "MA_5",
    "Volume_Change",
]

if data[feature_columns + ["Target"]].isna().any().any():
    raise ValueError("Missing features or target values found.")

if not np.isfinite(
    data[feature_columns].to_numpy(dtype=float)
).all():
    raise ValueError("Features contain infinite or invalid values.")

if not data["Target"].isin([0, 1]).all():
    raise ValueError("Target must contain only 0 and 1.")

data["Target"] = data["Target"].astype(int)

# ============================================================
# 2. CHRONOLOGICAL TRAIN/TEST SPLIT
# ============================================================

split_index = int(len(data) * 0.80)

if split_index < 2 or split_index >= len(data):
    raise ValueError("Insufficient observations for splitting.")

# Target at row t describes the price movement from t to t+1.
# The final training row's target uses the first test row's close.
# Remove that boundary row from training to avoid overlap.
train_data = data.iloc[:split_index - 1].copy()
test_data = data.iloc[split_index:].copy()

if train_data.empty or test_data.empty:
    raise ValueError("Training or test dataset is empty.")

X_train = train_data[feature_columns]
y_train = train_data["Target"]

X_test = test_data[feature_columns]
y_test = test_data["Target"]

print("Dataset shape:", data.shape)
print("Training rows:", len(train_data))
print("Testing rows:", len(test_data))

print(
    "\nTraining feature dates:",
    train_data["Date"].min().date(),
    "to",
    train_data["Date"].max().date(),
)

print(
    "Testing feature dates:",
    test_data["Date"].min().date(),
    "to",
    test_data["Date"].max().date(),
)

# ============================================================
# 3. TRAIN MODEL
# ============================================================

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000)),
])

model.fit(X_train, y_train)

print("\nModel training completed.")

# ============================================================
# 4. EVALUATE ON THE HELD-OUT TEST PERIOD
# ============================================================

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, predictions)

# Use the training majority class as a simple baseline.
baseline_class = y_train.mode().iloc[0]
baseline_predictions = np.full(
    len(y_test),
    baseline_class,
    dtype=int,
)
baseline_accuracy = accuracy_score(y_test, baseline_predictions)

print(f"\nModel test accuracy: {accuracy:.2%}")
print(f"Baseline test accuracy: {baseline_accuracy:.2%}")

print("\nConfusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Not Up", "Up"],
    zero_division=0,
))

print("\nFirst 5 test predictions:")
print(pd.DataFrame({
    "Date": test_data["Date"].iloc[:5].dt.date.to_numpy(),
    "Actual": y_test.iloc[:5].to_numpy(),
    "Predicted": predictions[:5],
    "Probability_Not_Up": probabilities[:5, 0],
    "Probability_Up": probabilities[:5, 1],
}).to_string(index=False))

# ============================================================
# 5. DISPLAY MODEL COEFFICIENTS
# ============================================================

classifier = model.named_steps["classifier"]

coefficient_table = pd.DataFrame({
    "Feature": feature_columns,
    "Coefficient": classifier.coef_[0],
})

print("\nModel intercept:", classifier.intercept_[0])
print("\nFeature coefficients:")
print(coefficient_table.to_string(index=False))

# ============================================================
# 6. SAVE MODEL AND TEST METADATA
# ============================================================

os.makedirs("models", exist_ok=True)
joblib.dump(model, model_path)

metadata = {
    "feature_columns": feature_columns,
    "train_start": str(train_data["Date"].min().date()),
    "train_end": str(train_data["Date"].max().date()),
    "test_start": str(test_data["Date"].min().date()),
    "test_end": str(test_data["Date"].max().date()),
    "train_rows": len(train_data),
    "test_rows": len(test_data),
    "test_accuracy": float(accuracy),
    "baseline_accuracy": float(baseline_accuracy),
}

joblib.dump(metadata, "models/logistic_regression_metadata.pkl")

print(f"\nModel saved: {model_path}")
print("Model metadata saved: models/logistic_regression_metadata.pkl")
print("\nTraining and evaluation completed successfully.")
