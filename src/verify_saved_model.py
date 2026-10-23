
import joblib
import pandas as pd
import numpy as np

# 1. Load saved model and metadata
model = joblib.load("models/logistic_regression.pkl")
metadata = joblib.load("models/logistic_regression_metadata.pkl")

print("Model loaded successfully!")
print("Metadata loaded successfully!")

# 2. Read the model's expected features
features = metadata["feature_columns"]

# 3. Load the model-ready dataset
data = pd.read_csv("data/model_ready_dataset.csv")

# 4. Select one sample for prediction
X_sample = data[features].tail(1)

# 5. Generate prediction and probabilities
prediction = model.predict(X_sample)[0]
probabilities = model.predict_proba(X_sample)[0]

print("\n--- SAVED MODEL VERIFICATION ---")
print("Features:", features)
print("Prediction:", int(prediction))
print(f"Probability Not Up: {probabilities[0]:.2%}")
print(f"Probability Up: {probabilities[1]:.2%}")

print("\nTraining end date:", metadata["train_end"])
print("Test accuracy:", f"{metadata['test_accuracy']:.2%}")

# 6. Basic validation
assert prediction in [0, 1]
assert np.isclose(probabilities.sum(), 1.0)

print("\nSaved model verification completed successfully!")
