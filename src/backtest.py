
import pandas as pd

# Load datasets
raw_data = pd.read_csv("data/reliance_stock_data.csv")
model_data = pd.read_csv("data/model_ready_dataset.csv")

# Convert dates
raw_data["Date"] = pd.to_datetime(raw_data["Date"])
model_data["Date"] = pd.to_datetime(model_data["Date"])

# Validate required columns
required_raw = {"Date", "Open", "Close"}
required_model = {"Date", "Target"}

if not required_raw.issubset(raw_data.columns):
    raise ValueError("Raw data is missing required columns.")

if not required_model.issubset(model_data.columns):
    raise ValueError("Model data is missing required columns.")

# Sort chronologically
raw_data = raw_data.sort_values("Date").reset_index(drop=True)
model_data = model_data.sort_values("Date").reset_index(drop=True)

# Use the same 80/20 chronological split
split_index = int(len(model_data) * 0.8)
test_data = model_data.iloc[split_index:].copy()

test_start = test_data["Date"].min()

print("Model-ready rows:", len(model_data))
print("Test rows:", len(test_data))
print("Test period:", test_data["Date"].min().date(),
      "to", test_data["Date"].max().date())

# Inspect raw prices for the test period
test_prices = raw_data.loc[
    raw_data["Date"] >= test_start,
    ["Date", "Open", "Close"]
]

print(test_prices.head())
