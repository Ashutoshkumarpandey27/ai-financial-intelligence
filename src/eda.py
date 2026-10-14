
import pandas as pd

# Load the stock dataset
data = pd.read_csv("data/reliance_stock_data.csv")

# Convert Date column into proper datetime format
data["Date"] = pd.to_datetime(data["Date"])

# 1. Display data types
print("Data types:")
print(data.dtypes)

# 2. Check missing values
print("\nMissing values:")
print(data.isnull().sum())

# 3. Display statistical summary
print("\nStatistical summary:")
print(data.describe())
