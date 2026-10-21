import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("data/model_ready_dataset.csv")

feature_columns = [
    "Close",
    "Daily_Return",
    "Previous_Day_Return",
    "Return_2_Days_Ago",
    "MA_5",
    "Volume_Change",
]

split_index = int(len(data) * 0.80)

train_data = data.iloc[:split_index]
test_data = data.iloc[split_index:]

X_train = train_data[feature_columns]
y_train = train_data["Target"].astype(int)

X_test = test_data[feature_columns]
y_test = test_data["Target"].astype(int)

model = DecisionTreeClassifier(
    max_depth=7,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Decision Tree Test Accuracy: {accuracy:.2%}")