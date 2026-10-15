
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD THE STOCK DATASET
# ==========================================

data = pd.read_csv("data/reliance_stock_data.csv")

# Convert Date column into proper datetime format
data["Date"] = pd.to_datetime(data["Date"])

# Sort data by date to ensure chronological order
data = data.sort_values("Date").reset_index(drop=True)


# ==========================================
# 2. CHECK DATA TYPES
# ==========================================

print("Data types:")
print(data.dtypes)


# ==========================================
# 3. CHECK MISSING VALUES
# ==========================================

print("\nMissing values:")
print(data.isnull().sum())


# ==========================================
# 4. STATISTICAL SUMMARY
# ==========================================

print("\nStatistical summary:")
print(data.describe())


# ==========================================
# 5. CHECK ZERO-VOLUME RECORDS
# ==========================================

print("\nZero-volume records:")
print((data["Volume"] == 0).sum())

print("\nRows with zero volume:")
print(data[data["Volume"] == 0])


# Display two rows before and after zero-volume records
print("\nRows around zero-volume records:")

zero_volume_rows = data.index[data["Volume"] == 0]

for i in zero_volume_rows:
    print(f"\nRows around index {i}:")
    print(data.iloc[max(0, i - 2):i + 3])


# ==========================================
# 6. CHECK DUPLICATE DATES
# ==========================================

print("\nDuplicate dates:")
print(data["Date"].duplicated().sum())

print("\nRows with duplicate dates:")
print(
    data[data["Date"].duplicated(keep=False)]
    .sort_values("Date")
)


# ==========================================
# 7. CHECK OHLC CONSISTENCY
# ==========================================

invalid_ohlc = data[
    (data["High"] < data["Low"]) |
    (data["High"] < data["Open"]) |
    (data["High"] < data["Close"]) |
    (data["Low"] > data["Open"]) |
    (data["Low"] > data["Close"])
]

print("\nInvalid OHLC records:")
print("Count:", len(invalid_ohlc))
print(invalid_ohlc)


# ==========================================
# 8. CALCULATE DAILY RETURNS
# ==========================================

# Calculate daily returns in decimal form
data["Daily_Return"] = data["Close"].pct_change()

# Convert decimal returns into percentages
data["Daily_Return_Pct"] = data["Daily_Return"] * 100

# Display the first 10 observations
print("\nDaily returns:")
print(
    data[
        ["Date", "Close", "Daily_Return", "Daily_Return_Pct"]
    ].head(10)
)


# ==========================================
# 9. PLOT CLOSING PRICE OVER TIME
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(data["Date"], data["Close"])

plt.title("Reliance Industries - Closing Price (5 Years)")
plt.xlabel("Date")
plt.ylabel("Adjusted Closing Price (₹)")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


# ==========================================
# 10. PLOT DAILY RETURNS
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(data["Date"], data["Daily_Return_Pct"])

plt.title("Reliance Industries - Daily Returns (5 Years)")
plt.xlabel("Date")
plt.ylabel("Daily Return (%)")

# Horizontal reference line at zero return
plt.axhline(y=0, linestyle="--")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()




# ==========================================
# 11. CALCULATE MOVING AVERAGES
# ==========================================

# Short-term trend: 20 trading observations
data["MA_20"] = data["Close"].rolling(window=20).mean()

# Longer-term trend: 50 trading observations
data["MA_50"] = data["Close"].rolling(window=50).mean()

# Display closing price and moving averages
print("\nMoving averages:")
print(data[["Date", "Close", "MA_20", "MA_50"]].tail(10))





# ==========================================
# 12. PLOT CLOSING PRICE AND MOVING AVERAGES
# ==========================================

plt.figure(figsize=(12, 6))

# Actual closing price
plt.plot(data["Date"], data["Close"], label="Closing Price")

# 20-observation moving average
plt.plot(data["Date"], data["MA_20"], label="MA 20")

# 50-observation moving average
plt.plot(data["Date"], data["MA_50"], label="MA 50")

plt.title("Reliance Industries - Closing Price and Moving Averages")
plt.xlabel("Date")
plt.ylabel("Adjusted Closing Price (₹)")

plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()



# ==========================================
# 13. CALCULATE ROLLING VOLATILITY
# ==========================================

# Calculate 20-observation rolling standard deviation of returns
data["Volatility_20"] = (
    data["Daily_Return"].rolling(window=20).std()
)

# Convert volatility from decimal returns into percentage units
data["Volatility_20_Pct"] = data["Volatility_20"] * 100

# Display recent values
print("\nRolling volatility:")
print(
    data[
        ["Date", "Daily_Return", "Volatility_20", "Volatility_20_Pct"]
    ].tail(10)
)



# ==========================================
# 14. PLOT ROLLING VOLATILITY
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    data["Date"],
    data["Volatility_20_Pct"],
    label="20-Observation Rolling Volatility"
)

plt.title("Reliance Industries - Rolling Volatility (20 Observations)")
plt.xlabel("Date")
plt.ylabel("Volatility (%)")

plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()










