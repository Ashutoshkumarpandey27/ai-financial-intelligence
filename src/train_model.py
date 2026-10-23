
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
    precision_score,
    recall_score,
    f1_score,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

data_path = "data/model_ready_dataset.csv"
model_path = "models/logistic_regression.pkl"
metadata_path = "models/logistic_regression_metadata.pkl"

feature_columns = [
    "Close",
    "Daily_Return",
    "Previous_Day_Return",
    "Return_2_Days_Ago",
    "MA_5",
    "Volume_Change",
]

required_columns = [
    "Date",
    *feature_columns,
    "Next_Open",
    "Next_Close",
    "Target",
]


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found: {data_path}")

data = pd.read_csv(data_path, parse_dates=["Date"])

missing_columns = set(required_columns) - set(data.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

if data.empty:
    raise ValueError("Dataset is empty.")

data = data.sort_values("Date").reset_index(drop=True)

if data["Date"].isna().any():
    raise ValueError("Date column contains missing values.")

if data["Date"].duplicated().any():
    raise ValueError("Duplicate dates found.")

if data[required_columns].isna().any().any():
    raise ValueError("Missing values found in required columns.")

if not np.isfinite(
    data[feature_columns].to_numpy(dtype=float)
).all():
    raise ValueError("Features contain infinite or invalid values.")

if not np.isfinite(
    data[["Next_Open", "Next_Close"]].to_numpy(dtype=float)
).all():
    raise ValueError("Execution prices contain invalid values.")

if not data["Target"].isin([0, 1]).all():
    raise ValueError("Target must contain only 0 and 1.")

# Verify that Target means:
# 1 = next session Close > next session Open
# 0 = next session Close <= next session Open

expected_target = (
    data["Next_Close"] > data["Next_Open"]
).astype(int)

if not np.array_equal(
    data["Target"].to_numpy(dtype=int),
    expected_target.to_numpy(dtype=int),
):
    raise ValueError("Target alignment verification failed.")

data["Target"] = data["Target"].astype(int)

print("Dataset loaded and validated successfully!")
print("Dataset shape:", data.shape)
print("Number of features:", len(feature_columns))
print("Target: Next session Close > Next session Open")


# ============================================================
# 3. CHRONOLOGICAL TRAIN/TEST SPLIT
# ============================================================

split_index = int(len(data) * 0.80)

if split_index < 2 or split_index >= len(data):
    raise ValueError("Insufficient observations for splitting.")

# The target on row t refers to the next session's prices.
# Keep the final pre-test row out of training so its target
# does not use the first test session's closing price.

train_data = data.iloc[:split_index - 1].copy()
test_data = data.iloc[split_index:].copy()

if train_data.empty or test_data.empty:
    raise ValueError("Training or testing data is empty.")

X_train = train_data[feature_columns]
y_train = train_data["Target"]

X_test = test_data[feature_columns]
y_test = test_data["Target"]

print("\n--- TRAIN/TEST SPLIT ---")
print("Training rows:", len(train_data))
print("Testing rows:", len(test_data))

print(
    "Training dates:",
    train_data["Date"].min().date(),
    "to",
    train_data["Date"].max().date(),
)

print(
    "Testing dates:",
    test_data["Date"].min().date(),
    "to",
    test_data["Date"].max().date(),
)

print("\nTraining target distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting target distribution:")
print(y_test.value_counts().sort_index())


# ============================================================
# 4. TRAIN LOGISTIC REGRESSION
# ============================================================

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000)),
])

model.fit(X_train, y_train)

print("\nModel training completed successfully!")


# ============================================================
# 5. EVALUATE ON THE HELD-OUT TEST PERIOD
# ============================================================

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, predictions)

# Baseline: always predict the most frequent training class.
baseline_class = int(y_train.mode().iloc[0])

baseline_predictions = np.full(
    len(y_test),
    baseline_class,
    dtype=int,
)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions,
)

precision_up = precision_score(
    y_test,
    predictions,
    pos_label=1,
    zero_division=0,
)

recall_up = recall_score(
    y_test,
    predictions,
    pos_label=1,
    zero_division=0,
)

f1_up = f1_score(
    y_test,
    predictions,
    pos_label=1,
    zero_division=0,
)

print("\n--- TEST PERFORMANCE ---")
print(f"Model accuracy:       {accuracy:.2%}")
print(f"Baseline accuracy:    {baseline_accuracy:.2%}")
print(f"Precision (UP):       {precision_up:.2%}")
print(f"Recall (UP):          {recall_up:.2%}")
print(f"F1-score (UP):        {f1_up:.4f}")

print("\nConfusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Not Up", "Up"],
        zero_division=0,
    )
)


# ============================================================
# 6. DISPLAY SAMPLE PREDICTIONS
# ============================================================

sample_predictions = pd.DataFrame({
    "Date": test_data["Date"].dt.date.to_numpy(),
    "Next_Open": test_data["Next_Open"].to_numpy(),
    "Next_Close": test_data["Next_Close"].to_numpy(),
    "Actual": y_test.to_numpy(),
    "Predicted": predictions,
    "Probability_Not_Up": probabilities[:, 0],
    "Probability_Up": probabilities[:, 1],
})

print("\nFirst 10 test predictions:")
print(sample_predictions.head(10).to_string(index=False))


# ============================================================
# 7. DISPLAY MODEL COEFFICIENTS
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
# 8. SAVE MODEL AND METADATA
# ============================================================

os.makedirs("models", exist_ok=True)

joblib.dump(model, model_path)

metadata = {
    "target_definition": "Next session Close > Next session Open",
    "feature_columns": feature_columns,
    "train_start": str(train_data["Date"].min().date()),
    "train_end": str(train_data["Date"].max().date()),
    "test_start": str(test_data["Date"].min().date()),
    "test_end": str(test_data["Date"].max().date()),
    "train_rows": int(len(train_data)),
    "test_rows": int(len(test_data)),
    "test_accuracy": float(accuracy),
    "baseline_accuracy": float(baseline_accuracy),
    "precision_up": float(precision_up),
    "recall_up": float(recall_up),
    "f1_up": float(f1_up),
}

joblib.dump(metadata, metadata_path)

print(f"\nModel saved: {model_path}")
print(f"Metadata saved: {metadata_path}")
print("\nTraining and evaluation completed successfully!")
