
import os
import numpy as np
import pandas as pd


# ============================================================
# 1. LOAD AND VALIDATE RAW DATA
# ============================================================

input_path = "data/reliance_stock_data.csv"
output_path = "data/model_ready_dataset.csv"

if not os.path.exists(input_path):
    raise FileNotFoundError(f"Dataset not found: {input_path}")

data = pd.read_csv(input_path)

required_raw_columns = {"Date", "Open", "High", "Low", "Close", "Volume"}

missing_columns = required_raw_columns - set(data.columns)

if missing_columns:
    raise ValueError(f"Missing raw columns: {sorted(missing_columns)}")

if data.empty:
    raise ValueError("Raw dataset is empty.")

data["Date"] = pd.to_datetime(data["Date"], errors="raise")

if data["Date"].isna().any():
    raise ValueError("Date column contains missing values.")

# Ensure price and volume columns contain numeric values.
numeric_columns = ["Open", "High", "Low", "Close", "Volume"]

for column in numeric_columns:
    data[column] = pd.to_numeric(data[column], errors="raise")

if data[numeric_columns].isna().any().any():
    raise ValueError("Raw price or volume columns contain missing values.")

if not np.isfinite(data[numeric_columns].to_numpy(dtype=float)).all():
    raise ValueError("Raw data contains infinite or invalid numeric values.")

if (data[["Open", "High", "Low", "Close"]] <= 0).any().any():
    raise ValueError("Stock prices must be positive.")

if (data["Volume"] < 0).any():
    raise ValueError("Volume cannot be negative.")

data = data.sort_values("Date").reset_index(drop=True)

if data["Date"].duplicated().any():
    raise ValueError("Duplicate dates found in raw dataset.")

print("Dataset loaded successfully!")
print("Original dataset shape:", data.shape)
print("Date range:", data["Date"].min().date(), "to", data["Date"].max().date())


# ============================================================
# 2. CREATE RETURN FEATURES
# ============================================================

# Return from the previous trading observation to the current one.
data["Daily_Return"] = data["Close"].pct_change()

# These features use only current or past observations.
data["Previous_Day_Return"] = data["Daily_Return"].shift(1)
data["Return_2_Days_Ago"] = data["Daily_Return"].shift(2)

print("\nReturn features created.")


# ============================================================
# 3. CREATE MOVING AVERAGE
# ============================================================

# Includes today's closing price and the previous four observations.
data["MA_5"] = data["Close"].rolling(window=5).mean()

print("MA_5 created.")


# ============================================================
# 4. CREATE VOLUME CHANGE
# ============================================================

# A previous volume of zero can produce infinity.
data["Volume_Change"] = data["Volume"].pct_change()

data["Volume_Change"] = data["Volume_Change"].replace(
    [np.inf, -np.inf],
    np.nan
)

print("Volume_Change created.")


# ============================================================
# 5. CREATE NEXT-OBSERVATION TARGET
# ============================================================

# Next trading observation's closing price.
data["Next_Close"] = data["Close"].shift(-1)

# Use nullable integer labels so the final unknown target stays missing.
data["Target"] = pd.Series(
    pd.NA,
    index=data.index,
    dtype="Int64"
)

known_target = data["Next_Close"].notna()

data.loc[known_target, "Target"] = (
    data.loc[known_target, "Next_Close"]
    > data.loc[known_target, "Close"]
).astype(int)

# Target definition:
# 1 = next close is higher than current close
# 0 = next close is equal to or lower than current close

print("\nTarget created.")
print(data[["Date", "Close", "Next_Close", "Target"]].tail())


# ============================================================
# 6. CLEAN THE MODEL-READY DATASET
# ============================================================

feature_columns = [
    "Close",
    "Daily_Return",
    "Previous_Day_Return",
    "Return_2_Days_Ago",
    "MA_5",
    "Volume_Change",
]

required_model_columns = feature_columns + ["Target"]

# Drop rows where features or the target are unavailable.
model_data = data.dropna(
    subset=required_model_columns
).copy()

# Convert target from nullable integer to standard integer.
model_data["Target"] = model_data["Target"].astype(int)

# Validate features and labels.
feature_values = model_data[feature_columns].to_numpy(dtype=float)

if model_data.empty:
    raise ValueError("No rows remain after cleaning.")

if not np.isfinite(feature_values).all():
    raise ValueError("Model features contain invalid or infinite values.")

if not model_data["Target"].isin([0, 1]).all():
    raise ValueError("Target must contain only 0 and 1.")

if model_data["Date"].duplicated().any():
    raise ValueError("Duplicate dates found in model-ready dataset.")

model_data = model_data.sort_values("Date").reset_index(drop=True)

print("\n--- DATA CLEANING SUMMARY ---")
print("Original rows:", len(data))
print("Model-ready rows:", len(model_data))
print("Rows removed:", len(data) - len(model_data))
print(
    "Model-ready date range:",
    model_data["Date"].min().date(),
    "to",
    model_data["Date"].max().date()
)


# ============================================================
# 7. DISPLAY TARGET DISTRIBUTION
# ============================================================

print("\nTarget class distribution:")
print(model_data["Target"].value_counts().sort_index())

print("\nTarget proportions:")
print(
    model_data["Target"]
    .value_counts(normalize=True)
    .sort_index()
    .round(4)
)


# ============================================================
# 8. SAVE MODEL-READY DATASET
# ============================================================

os.makedirs("data", exist_ok=True)

model_data.to_csv(output_path, index=False)

print(f"\nModel-ready dataset saved to: {output_path}")


# ============================================================
# 9. RELOAD AND VERIFY SAVED FILE
# ============================================================

saved_data = pd.read_csv(output_path)

if saved_data.empty:
    raise ValueError("Saved model-ready dataset is empty.")

if saved_data[required_model_columns].isna().any().any():
    raise ValueError("Saved dataset contains missing features or targets.")

saved_features = saved_data[feature_columns].to_numpy(dtype=float)

if not np.isfinite(saved_features).all():
    raise ValueError("Saved dataset contains infinite feature values.")

if not saved_data["Target"].isin([0, 1]).all():
    raise ValueError("Saved target contains invalid labels.")

print("\n--- FINAL VALIDATION ---")
print("Saved dataset shape:", saved_data.shape)
print("Missing values in required columns:",
      int(saved_data[required_model_columns].isna().sum().sum()))
print("Infinite values in features:",
      int(np.isinf(saved_features).sum()))

print("\nFeature engineering completed successfully!")
