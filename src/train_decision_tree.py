
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# 1. Load and sort data chronologically
data = pd.read_csv("data/model_ready_dataset.csv")
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date").reset_index(drop=True)

feature_columns = [
    "Close",
    "Daily_Return",
    "Previous_Day_Return",
    "Return_2_Days_Ago",
    "MA_5",
    "Volume_Change",
]

# 2. Split: 80% development data, 20% final test data
split_index = int(len(data) * 0.80)

development_data = data.iloc[:split_index].copy()
test_data = data.iloc[split_index:].copy()

# 3. Split development data: 80% train, 20% validation
validation_index = int(len(development_data) * 0.80)

train_data = development_data.iloc[:validation_index].copy()
validation_data = development_data.iloc[validation_index:].copy()

# 4. Separate features (X) and target (y)
X_train = train_data[feature_columns]
y_train = train_data["Target"].astype(int)

X_val = validation_data[feature_columns]
y_val = validation_data["Target"].astype(int)

X_test = test_data[feature_columns]
y_test = test_data["Target"].astype(int)

# 5. Compare max_depth values using validation data
depth_values = [3, 5, 7, 10, 15]

best_depth = None
best_validation_accuracy = -1

print("VALIDATION RESULTS")
print("-" * 40)

for depth in depth_values:
    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)

    print(
        f"Depth: {depth:2} | "
        f"Validation Accuracy: {val_accuracy:.2%}"
    )

    if val_accuracy > best_validation_accuracy:
        best_validation_accuracy = val_accuracy
        best_depth = depth

# 6. Train the selected model on all development data
print("\nSELECTED MODEL")
print("-" * 40)
print(f"Best max_depth: {best_depth}")
print(f"Best validation accuracy: {best_validation_accuracy:.2%}")

X_development = development_data[feature_columns]
y_development = development_data["Target"].astype(int)

best_model = DecisionTreeClassifier(
    max_depth=best_depth,
    random_state=42
)

best_model.fit(X_development, y_development)

# 7. Evaluate on the untouched final test set
test_predictions = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)

print("\nFINAL TEST RESULTS")
print("-" * 40)
print(f"Test rows: {len(test_data)}")
print(f"Test period: {test_data['Date'].min().date()} "
      f"to {test_data['Date'].max().date()}")
print(f"Decision Tree Test Accuracy: {test_accuracy:.2%}")


print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, test_predictions, labels=[0, 1]))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    test_predictions,
    labels=[0, 1],
    target_names=["Not Up (0)", "Up (1)"],
    zero_division=0
))