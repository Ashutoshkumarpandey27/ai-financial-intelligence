
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

required_raw_columns = [
    "Date", "Open", "High", "Low", "Close", "Volume"
]

missing_columns = set(required_raw_columns) - set(data.columns)

if missing_columns:
    raise ValueError(f"Missing raw columns: {sorted(missing_columns)}")

if data.empty:
    raise ValueError("Raw dataset is empty.")

data["Date"] = pd.to_datetime(data["Date"], errors="raise")

numeric_columns = ["Open", "High", "Low", "Close", "Volume"]

for column in numeric_columns:
    data[column] = pd.to_numeric(data[column], errors="raise")

if data["Date"].isna().any():
    raise ValueError("Date column contains missing values.")

if data[numeric_columns].isna().any().any():
    raise ValueError("Raw data contains missing numeric values.")

if not np.isfinite(data[numeric_columns].to_numpy(dtype=float)).all():
    raise ValueError("Raw data contains infinite or invalid values.")

if (data[["Open", "High", "Low", "Close"]] <= 0).any().any():
    raise ValueError("Stock prices must be positive.")

if (data["Volume"] < 0).any():
    raise ValueError("Volume cannot be negative.")

data = data.sort_values("Date").reset_index(drop=True)

if data["Date"].duplicated().any():
    raise ValueError("Duplicate dates found in raw dataset.")

if (data["High"] < data["Low"]).any():
    raise ValueError("High cannot be lower than Low.")

if (
    data["High"] < data[["Open", "Close"]].max(axis=1)
).any():
    raise ValueError("High is inconsistent with Open or Close.")

if (
    data["Low"] > data[["Open", "Close"]].min(axis=1)
).any():
    raise ValueError("Low is inconsistent with Open or Close.")

print("Dataset loaded and validated successfully!")
print("Original dataset shape:", data.shape)
print(
    "Date range:",
    data["Date"].min().date(),
    "to",
    data["Date"].max().date()
)


# ============================================================
# 2. CREATE FEATURES USING INFORMATION AVAILABLE AT CLOSE OF t
# ============================================================

data["Daily_Return"] = data["Close"].pct_change()

data["Previous_Day_Return"] = data["Daily_Return"].shift(1)

data["Return_2_Days_Ago"] = data["Daily_Return"].shift(2)

data["MA_5"] = data["Close"].rolling(window=5).mean()

data["Volume_Change"] = data["Volume"].pct_change()

data["Volume_Change"] = data["Volume_Change"].replace(
    [np.inf, -np.inf], np.nan
)

print("\nFeature engineering completed.")


# ============================================================
# 3. CREATE NEXT-TRADING-SESSION PRICES AND TARGET
# ============================================================

# At the close of day t, predict the direction of day t+1.
data["Next_Open"] = data["Open"].shift(-1)
data["Next_Close"] = data["Close"].shift(-1)

# Target = 1 when the next session closes above its open.
# Target = 0 when the next session closes at or below its open.
data["Target"] = np.where(
    data["Next_Open"].notna() & data["Next_Close"].notna(),
    (data["Next_Close"] > data["Next_Open"]).astype(int),
    np.nan
)

print("\nNext-session intraday target created.")

print(
    data[
        ["Date", "Close", "Next_Open", "Next_Close", "Target"]
    ].tail(5).to_string(index=False)
)


# ============================================================
# 4. PREPARE MODEL-READY DATASET
# ============================================================

# Only these six columns are used as model inputs.
# Next_Open and Next_Close are execution/validation columns,
# NOT model input features.

feature_columns = [
    "Close",
    "Daily_Return",
    "Previous_Day_Return",
    "Return_2_Days_Ago",
    "MA_5",
    "Volume_Change"
]

required_model_columns = feature_columns + ["Target"]

# Remove rows with missing features or unknown target.
model_data = data.dropna(
    subset=required_model_columns
).copy()

if model_data.empty:
    raise ValueError("No rows remain after cleaning.")

model_data["Target"] = model_data["Target"].astype(int)

feature_values = model_data[feature_columns].to_numpy(dtype=float)

if not np.isfinite(feature_values).all():
    raise ValueError("Model features contain invalid or infinite values.")

if not model_data["Target"].isin([0, 1]).all():
    raise ValueError("Target must contain only 0 and 1.")

if model_data["Date"].duplicated().any():
    raise ValueError("Duplicate dates found in model-ready data.")

model_data = (
    model_data.sort_values("Date")
    .reset_index(drop=True)
)

# Ensure the execution prices are available for every saved row.
if model_data[["Next_Open", "Next_Close"]].isna().any().any():
    raise ValueError("Execution prices are missing from model-ready data.")

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
# 5. DISPLAY TARGET DISTRIBUTION
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
# 6. SAVE MODEL-READY DATASET
# ============================================================

os.makedirs("data", exist_ok=True)

model_data.to_csv(output_path, index=False)

print(f"\nModel-ready dataset saved to: {output_path}")


# ============================================================
# 7. RELOAD AND VERIFY SAVED DATASET
# ============================================================

saved_data = pd.read_csv(output_path, parse_dates=["Date"])

required_saved_columns = (
    ["Date"] + feature_columns
    + ["Next_Open", "Next_Close", "Target"]
)

missing_saved_columns = (
    set(required_saved_columns) - set(saved_data.columns)
)

if missing_saved_columns:
    raise ValueError(
        f"Saved dataset is missing columns: {sorted(missing_saved_columns)}"
    )

if saved_data.empty:
    raise ValueError("Saved model-ready dataset is empty.")

if saved_data[required_saved_columns].isna().any().any():
    raise ValueError("Saved dataset contains missing required values.")

saved_features = saved_data[feature_columns].to_numpy(dtype=float)

if not np.isfinite(saved_features).all():
    raise ValueError("Saved dataset contains infinite feature values.")

if not saved_data["Target"].isin([0, 1]).all():
    raise ValueError("Saved target contains invalid labels.")

# Verify that Target matches the next session's Open-to-Close direction.
expected_target = (
    saved_data["Next_Close"] > saved_data["Next_Open"]
).astype(int)

if not np.array_equal(
    saved_data["Target"].to_numpy(dtype=int),
    expected_target.to_numpy(dtype=int)
):
    raise ValueError("Target alignment verification failed.")

# Verify that execution prices really correspond to the next raw row.
raw_check = data[["Date", "Open", "Close"]].copy()

raw_check["Expected_Next_Open"] = raw_check["Open"].shift(-1)
raw_check["Expected_Next_Close"] = raw_check["Close"].shift(-1)

verified = saved_data.merge(
    raw_check[
        ["Date", "Expected_Next_Open", "Expected_Next_Close"]
    ],
    on="Date",
    how="left",
    validate="one_to_one"
)

if not np.allclose(
    verified["Next_Open"].to_numpy(dtype=float),
    verified["Expected_Next_Open"].to_numpy(dtype=float),
    rtol=1e-9,
    atol=1e-8
):
    raise ValueError("Next_Open does not match the next trading session.")

if not np.allclose(
    verified["Next_Close"].to_numpy(dtype=float),
    verified["Expected_Next_Close"].to_numpy(dtype=float),
    rtol=1e-9,
    atol=1e-8
):
    raise ValueError("Next_Close does not match the next trading session.")

print("\n--- FINAL VALIDATION ---")
print("Saved dataset shape:", saved_data.shape)
print("Saved columns:", saved_data.columns.tolist())
print("Missing values in required columns:", int(
    saved_data[required_saved_columns].isna().sum().sum()
))
print("Infinite values in features:", int(np.isinf(saved_features).sum()))
print("Target alignment verified successfully!")
print("Next-session execution prices verified successfully!")
print("\nFeature engineering completed successfully!")
