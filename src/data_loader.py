
import yfinance as yf

# Download Reliance Industries stock data
data = yf.download(
    "RELIANCE.NS",
    period="5y",
    auto_adjust=True
)

# Check whether data was downloaded
if data.empty:
    print("No data downloaded. Check the ticker or internet connection.")

else:
    print("Data downloaded successfully!")

    # Display first five rows
    print(data.head())

    # Display dataset dimensions
    print("Dataset shape:", data.shape)

    # Inspect column structure
    print("Column structure:", data.columns)

    # Save data to CSV
    # Simplify column names for our single-stock dataset
    data.columns = data.columns.get_level_values("Price")

    # Save data to CSV
    data.to_csv("data/reliance_stock_data.csv", index_label="Date")

    print("Data saved successfully!")
