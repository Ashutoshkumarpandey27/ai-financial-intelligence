
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 1. Load the model-ready dataset
data = pd.read_csv("data/model_ready_dataset.csv")

print("Dataset shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())


# 2. Convert Date to datetime and sort chronologically
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date").reset_index(drop=True)


# 3. Select input features
feature_columns = [
    "Close",
    "Daily_Return",
    "Previous_Day_Return",
    "Return_2_Days_Ago",
    "MA_5",
    "Volume_Change",
]


# 4. Validate required columns
required_columns = feature_columns + ["Target"]

missing_columns = [
    column for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

if not np.isfinite(data[feature_columns].to_numpy(dtype=float)).all():
    raise ValueError("Infinite or invalid feature values found.")

if not data["Target"].isin([0, 1]).all():
    raise ValueError("Target must contain only 0 and 1.")


# 5. Split chronologically: first 80% for training, last 20% for testing
split_index = int(len(data) * 0.80)

train_data = data.iloc[:split_index].copy()
test_data = data.iloc[split_index:].copy()

if train_data.empty or test_data.empty:
    raise ValueError("Not enough observations for train-test split.")


# 6. Separate features (X) and target (y)
X_train = train_data[feature_columns]
y_train = train_data["Target"].astype(int)

X_test = test_data[feature_columns]
y_test = test_data["Target"].astype(int)


# 7. Display dataset sizes and date ranges
print("\nX_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

print("\nTraining dates:")
print(train_data["Date"].min(), "to", train_data["Date"].max())

print("\nTesting dates:")
print(test_data["Date"].min(), "to", test_data["Date"].max())


# 8. Create the scaling + Logistic Regression pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])


# 9. Train the model using training data only
model.fit(X_train, y_train)

print("\nModel training completed successfully!")


# 10. Predict classes for unseen testing data
predictions = model.predict(X_test)


# 11. Calculate test accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"\nTest Accuracy: {accuracy:.2%}")


# 12. Display the first 10 predictions and actual targets
print("\nFirst 10 predictions:")
print(predictions[:10])

print("\nFirst 10 actual targets:")
print(y_test.iloc[:10].to_numpy())


# Baseline: always predict the most frequent training class
baseline_class = y_train.mode()[0]

baseline_predictions = [baseline_class] * len(y_test)

baseline_accuracy = accuracy_score(y_test, baseline_predictions)

print(f"\nBaseline class: {baseline_class}")
print(f"Baseline Test Accuracy: {baseline_accuracy:.2%}")
print(f"Logistic Regression Accuracy: {accuracy:.2%}")


# 13. Display the confusion matrix
matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

print("\nConfusion Matrix (rows = actual, columns = predicted):")
print(matrix)


# 14. Display precision, recall, and F1-score
print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Target 0 (Not Up)", "Target 1 (Up)"],
    zero_division=0
))
