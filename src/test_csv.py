import pandas as pd

data = pd.read_csv("data/reliance_stock_data.csv", index_label="Date")

print(data.head())
print("Dataset shape:", data.shape)