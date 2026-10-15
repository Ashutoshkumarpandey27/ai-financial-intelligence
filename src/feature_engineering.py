
import pandas as pd

# 1. Load the stock dataset
data = pd.read_csv("data/reliance_stock_data.csv")

# 2. Convert Date column into datetime format
data["Date"] = pd.to_datetime(data["Date"])

# 3. Arrange data from oldest to newest
data = data.sort_values("Date").reset_index(drop=True)

# 4. Calculate daily returns
data["Daily_Return"] = data["Close"].pct_change()

# 5. Create previous-day return feature
data["Previous_Day_Return"] = data["Daily_Return"].shift(1)

# 6. Create return from two trading observations ago
data["Return_2_Days_Ago"] = data["Daily_Return"].shift(2)

# 7. Display the results
print(data[[
    "Date",
    "Close",
    "Daily_Return",
    "Previous_Day_Return",
    "Return_2_Days_Ago"
]].head(10))




# 8. Calculate 5-observation moving average
data["MA_5"] = data["Close"].rolling(window=5).mean()

# 9. Display the new feature
print(data[["Date", "Close", "MA_5"]].head(10))



# 10. Calculate percentage change in trading volume
data["Volume_Change"] = data["Volume"].pct_change()

# 11. Display the new feature
print(data[["Date", "Volume", "Volume_Change"]].head(10))




# 12. Check missing values in our features
print("\nMissing values:")
print(data[[
    "Daily_Return",
    "Previous_Day_Return",
    "Return_2_Days_Ago",
    "MA_5",
    "Volume_Change"
]].isnull().sum())

# 13. Check infinite values in our features
import numpy as np

features = data[[
    "Daily_Return",
    "Previous_Day_Return",
    "Return_2_Days_Ago",
    "MA_5",
    "Volume_Change"
]]

print("\nInfinite values:")
print(np.isinf(features).sum())




# 14. Find rows with infinite volume changes
infinite_rows = data[np.isinf(data["Volume_Change"])]

print("\nRows with infinite Volume_Change:")
print(
    infinite_rows[
        ["Date", "Volume", "Volume_Change"]
    ]
)

# 15. Display the previous observation for investigation
for i in infinite_rows.index:
    if i > 0:
        print("\nPrevious observation:")
        print(data.loc[i - 1, ["Date", "Volume"]])




# Replace infinite volume changes with missing values
data["Volume_Change"] = data["Volume_Change"].replace(
    [np.inf, -np.inf],
    np.nan
)

print("\nInfinite values after cleaning:")
print(np.isinf(data[["Volume_Change"]]).sum())

print("\nMissing volume changes after cleaning:")
print(data["Volume_Change"].isnull().sum())




# Create the next trading observation's closing price
data["Next_Close"] = data["Close"].shift(-1)

# Create the target variable
data["Target"] = (
    data["Next_Close"] > data["Close"]
).astype("Int64")

# The final row has no next observation, so its target is unknown
data.loc[data["Next_Close"].isna(), "Target"] = pd.NA

# Display the result
print("\nTarget variable:")
print(data[["Date", "Close", "Next_Close", "Target"]].tail(10))


# Create a separate dataset for machine learning
model_data = data.copy()

# Remove rows with missing values
model_data = model_data.dropna()



print("\nModel-ready dataset:")
print("Original rows:", len(data))
print("Rows after cleaning:", len(model_data))
print(model_data[["Date", "Close", "MA_5", "Target"]].head())