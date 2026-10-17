
import pandas as pd

# 1. Load our prepared dataset
data = pd.read_csv("data/model_ready_dataset.csv")

# 2. Ensure observations are in chronological order
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date").reset_index(drop=True)

# 3. Choose the split point: first 80%, last 20%
split_index = int(len(data) * 0.80)

train_data = data.iloc[:split_index].copy()
test_data = data.iloc[split_index:].copy()

# 4. Separate features (X) and target (y)
feature_columns = [
    "Close",
    "Daily_Return",
    "Previous_Day_Return",
    "Return_2_Days_Ago",
    "MA_5",
    "Volume_Change",
]

X_train = train_data[feature_columns]
y_train = train_data["Target"].astype(int)

X_test = test_data[feature_columns]
y_test = test_data["Target"].astype(int)

# 5. Inspect the split
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nTraining date range:")
print(train_data["Date"].min(), "to", train_data["Date"].max())

print("\nTesting date range:")
print(test_data["Date"].min(), "to", test_data["Date"].max())

# 6. Verify the split
assert len(X_train) + len(X_test) == len(data)
assert train_data["Date"].max() < test_data["Date"].min()
assert len(X_train) == len(y_train)
assert len(X_test) == len(y_test)

print("\nChronological split successful!")
