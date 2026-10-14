
import pandas as pd

# Read the saved CSV file
data = pd.read_csv("data/reliance_stock_data.csv")

# Display first five rows
print(data.head())

# Display dataset dimensions
print("Dataset shape:", data.shape)

# Display column names
print("Columns:", data.columns.tolist())
