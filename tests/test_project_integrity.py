
import unittest
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "logistic_regression.pkl"
METADATA_PATH = ROOT / "models" / "logistic_regression_metadata.pkl"
DATA_PATH = ROOT / "data" / "model_ready_dataset.csv"

EXPECTED_FEATURES = [
    "Close",
    "Daily_Return",
    "Previous_Day_Return",
    "Return_2_Days_Ago",
    "MA_5",
    "Volume_Change",
]


class TestProjectIntegrity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        for path in [MODEL_PATH, METADATA_PATH, DATA_PATH]:
            if not path.exists():
                raise FileNotFoundError(f"Required project file missing: {path}")

        cls.model = joblib.load(MODEL_PATH)
        cls.metadata = joblib.load(METADATA_PATH)
        cls.data = pd.read_csv(DATA_PATH, parse_dates=["Date"])

    def test_dataset_is_not_empty(self):
        self.assertGreater(len(self.data), 0)

    def test_required_columns_exist(self):
        required = ["Date", *EXPECTED_FEATURES]
        missing = set(required) - set(self.data.columns)
        self.assertFalse(missing, f"Missing columns: {sorted(missing)}")

    def test_dates_are_valid_and_sorted(self):
        self.assertFalse(self.data["Date"].isna().any())
        dates = self.data["Date"]
        self.assertTrue(dates.is_monotonic_increasing)

    def test_model_feature_count(self):
        self.assertEqual(
            self.model.n_features_in_,
            len(EXPECTED_FEATURES),
        )

    def test_metadata_feature_order(self):
        saved_features = self.metadata.get("feature_columns")
        self.assertEqual(saved_features, EXPECTED_FEATURES)

    def test_latest_prediction_is_valid(self):
        latest = self.data.iloc[[-1]][EXPECTED_FEATURES]
        latest = latest.apply(pd.to_numeric, errors="coerce")

        self.assertFalse(latest.isna().any().any())

        probabilities = self.model.predict_proba(latest)[0]
        prediction = self.model.predict(latest)[0]

        self.assertIn(int(prediction), [0, 1])
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)
        self.assertTrue(all(0 <= p <= 1 for p in probabilities))


if __name__ == "__main__":
    unittest.main()

