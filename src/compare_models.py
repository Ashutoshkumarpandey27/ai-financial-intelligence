
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
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

# 2. Keep the same final test period
split_index = int(len(data) * 0.80)

development_data = data.iloc[:split_index]
test_data = data.iloc[split_index:]

X_development = development_data[feature_columns]
y_development = development_data["Target"].astype(int)

X_test = test_data[feature_columns]
y_test = test_data["Target"].astype(int)

# 3. Define both models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        random_state=42
    )
}

# 4. Train and evaluate both models
results = []

for name, model in models.items():
    model.fit(X_development, y_development)
    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision (Up)": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall (Up)": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1-score (Up)": f1_score(
            y_test, predictions, zero_division=0
        )
    })

# 5. Display comparison
comparison = pd.DataFrame(results)

print("MODEL COMPARISON")
print(comparison.round(4).to_string(index=False))
