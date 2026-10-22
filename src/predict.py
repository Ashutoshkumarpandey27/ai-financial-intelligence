
import joblib
import pandas as pd

# Load the saved pipeline
model = joblib.load("models/logistic_regression.pkl")

# Load the latest row from the model-ready dataset
data = pd.read_csv("data/model_ready_dataset.csv")

feature_columns = [
    "Close",
    "Daily_Return",
    "Previous_Day_Return",
    "Return_2_Days_Ago",
    "MA_5",
    "Volume_Change",
]

latest_data = data[feature_columns].iloc[[-1]]

# Make prediction
prediction = model.predict(latest_data)[0]
probabilities = model.predict_proba(latest_data)[0]

print("Predicted next-day direction:",
      "UP" if prediction == 1 else "NOT UP")

print(f"Probability of NOT UP: {probabilities[0]:.2%}")
print(f"Probability of UP: {probabilities[1]:.2%}")
