
import pandas as pd
import numpy as np


# ============================================================
# 1. LOAD AND PREPARE THE DATASET
# ============================================================

data = pd.read_csv("data/reliance_stock_data.csv")

# Convert Date column into datetime format
data["Date"] = pd.to_datetime(data["Date"])

# Arrange data from oldest to newest
data = data.sort_values("Date").reset_index(drop=True)

print("Dataset loaded successfully!")
print("Original dataset shape:", data.shape)
print("\nFirst 5 observations:")
print(data.head())


# ============================================================
# 2. CREATE RETURN FEATURES
# ============================================================

# Daily percentage return
data["Daily_Return"] = data["Close"].pct_change()

# Previous trading observation's return
data["Previous_Day_Return"] = data["Daily_Return"].shift(1)

# Return from two trading observations ago
data["Return_2_Days_Ago"] = data["Daily_Return"].shift(2)

print("\nReturn features:")
print(
    data[
        [
            "Date",
            "Close",
            "Daily_Return",
            "Previous_Day_Return",
            "Return_2_Days_Ago",
        ]
    ].head(10)
)


# ============================================================
# 3. CREATE MOVING AVERAGE FEATURE
# ============================================================

# Average closing price over the latest 5 observations,
# including the current observation
data["MA_5"] = data["Close"].rolling(window=5).mean()

print("\n5-observation moving average:")
print(data[["Date", "Close", "MA_5"]].head(10))


# ============================================================
# 4. CREATE VOLUME CHANGE FEATURE
# ============================================================

# Percentage change in trading volume
data["Volume_Change"] = data["Volume"].pct_change()

print("\nVolume change:")
print(data[["Date", "Volume", "Volume_Change"]].head(10))


# ============================================================
# 5. CHECK DATA QUALITY
# ============================================================

feature_columns = [
    "Close",
    "Daily_Return",
    "Previous_Day_Return",
    "Return_2_Days_Ago",
    "MA_5",
    "Volume_Change",
]

print("\nMissing values before cleaning:")
print(data[feature_columns].isnull().sum())

print("\nInfinite values before cleaning:")
print(np.isinf(data[feature_columns]).sum())

# Investigate observations with infinite volume changes
infinite_rows = data[np.isinf(data["Volume_Change"])]

print("\nRows with infinite Volume_Change:")
print(infinite_rows[["Date", "Volume", "Volume_Change"]])

# Display the previous observation for investigation
for i in infinite_rows.index:
    if i > 0:
        print("\nPrevious observation:")
        print(data.loc[i - 1, ["Date", "Volume"]])

# Replace positive and negative infinity with missing values
data["Volume_Change"] = data["Volume_Change"].replace(
    [np.inf, -np.inf],
    np.nan,
)

print("\nInfinite values after cleaning:")
print(np.isinf(data[["Volume_Change"]]).sum())

print("\nMissing Volume_Change values after cleaning:")
print(data["Volume_Change"].isnull().sum())


# ============================================================
# 6. CREATE THE TARGET VARIABLE
# ============================================================

# Next trading observation's closing price
data["Next_Close"] = data["Close"].shift(-1)

# Target:
# 1 = next closing price is higher
# 0 = next closing price is equal to or lower
data["Target"] = (
    data["Next_Close"] > data["Close"]
).astype("Int64")

# The final observation has no known next closing price
data.loc[data["Next_Close"].isna(), "Target"] = pd.NA

print("\nTarget variable (last 10 observations):")
print(data[["Date", "Close", "Next_Close", "Target"]].tail(10))


# ============================================================
# 7. CREATE THE MODEL-READY DATASET
# ============================================================

# Only require the selected features and target to be present
required_columns = feature_columns + ["Target"]

model_data = data.dropna(subset=required_columns).copy()

print("\nModel-ready dataset:")
print("Original rows:", len(data))
print("Rows after cleaning:", len(model_data))
print("Rows removed:", len(data) - len(model_data))

print("\nFirst 5 model-ready observations:")
print(model_data[["Date", "Close", "MA_5", "Target"]].head())


# ============================================================
# 8. SEPARATE FEATURES (X) AND TARGET (y)
# ============================================================

# X contains inputs used by the model
X = model_data[feature_columns].copy()

# y contains the answer the model learns to predict
y = model_data["Target"].astype(int).copy()

print("\nFeatures (X) shape:", X.shape)
print("Target (y) shape:", y.shape)

print("\nFirst 5 feature rows:")
print(X.head())

print("\nFirst 5 target values:")
print(y.head())


# ============================================================
# 9. FINAL VALIDATION
# ============================================================

print("\nFinal validation:")

print("X shape:", X.shape)
print("y shape:", y.shape)

print("Missing values in X:", X.isnull().sum().sum())
print("Infinite values in X:", np.isinf(X).sum().sum())
print("Missing values in y:", y.isnull().sum())

print("\nTarget class distribution:")
print(y.value_counts().sort_index())

# Verify the dataset is ready for the next stage
assert len(X) == len(y), "X and y have different row counts!"
assert X.isnull().sum().sum() == 0, "X contains missing values!"
assert np.isfinite(X.to_numpy()).all(), "X contains infinite or invalid values!"
assert y.notna().all(), "y contains missing values!"

print("\nAll validation checks passed!")
print("Feature engineering completed successfully.")



model_data.to_csv(
    "data/model_ready_dataset.csv",
    index=False
)

print("Model-ready dataset saved successfully!")


