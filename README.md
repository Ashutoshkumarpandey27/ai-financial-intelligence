# AI Financial Intelligence Engine

A machine learning research project that analyzes historical Reliance Industries stock data (`RELIANCE.NS`), predicts the probability of an upward intraday price movement, and evaluates a rule-based trading strategy using chronological testing, walk-forward validation, baseline comparisons, and transaction-cost sensitivity analysis.

> **Disclaimer:** This project is for educational and research purposes. Current experiments do not establish a consistently profitable trading strategy.

## Project Objectives

- Collect and analyze historical stock market data.
- Engineer features from historical price and volume information.
- Train a Logistic Regression classification model.
- Estimate the probability of an upward intraday movement.
- Backtest a rule-based trading strategy.
- Evaluate performance using chronological and walk-forward validation.
- Compare the model against simple baseline strategies.
- Test different probability thresholds and transaction-cost assumptions.
- Visualize returns, equity curves, drawdowns, and robustness results.

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Plotly
- Streamlit
- yfinance
- Joblib
- Git and GitHub

## Dataset

Historical data for Reliance Industries (`RELIANCE.NS`) is collected using Yahoo Finance.

The raw dataset contains OHLCV data:

- Open
- High
- Low
- Close
- Volume

### Features Used

The Logistic Regression model uses six features:

- `Close`
- `Daily_Return`
- `Previous_Day_Return`
- `Return_2_Days_Ago`
- `MA_5`
- `Volume_Change`

Future execution prices and target-alignment columns are not used as model features.

### Target Definition

The target represents the next trading session's intraday direction:

- `1`: Next session's Close is greater than its Open.
- `0`: Otherwise.

The model estimates the probability of an upward intraday movement using historical features.

## Machine Learning Pipeline

1. Download historical stock data.
2. Perform exploratory data analysis.
3. Generate features and the target variable.
4. Validate the dataset and target alignment.
5. Split the observations chronologically.
6. Scale features using `StandardScaler`.
7. Train a Logistic Regression classifier.
8. Generate predictions and probabilities.
9. Evaluate classification performance against a baseline.
10. Backtest the trading strategy.
11. Run walk-forward validation and robustness tests.
12. Display results through an interactive dashboard.

## Trading Strategy

The basic strategy trades when the model predicts class `1`.

- **Entry:** Next session's Open.
- **Exit:** Same session's Close.
- **No-trade condition:** Prediction of class `0`.
- **Initial capital:** INR 10,000.
- **Transaction costs:** Illustrative assumptions defined in the evaluation scripts.

The benchmark invests during each evaluated session from Open to Close. It is not continuous buy-and-hold investing.

Returns are compounded within individual evaluation windows. Average returns across separate walk-forward rounds are not equivalent to a single compounded portfolio return.

## Validation Methodology

### Chronological Holdout Testing

The model uses earlier observations for training and later observations for testing, preserving the time order of the data.

### Walk-Forward Validation

The walk-forward evaluation uses expanding training windows and multiple subsequent test windows. Each round evaluates the model on a different historical period.

### Baseline Comparison

The project compares Logistic Regression with:

- Majority-class baseline
- Always-UP baseline
- Matched-period intraday benchmark

Classification accuracy and trading performance are evaluated separately because higher accuracy does not necessarily imply profitable trading.

### Robustness Testing

The project investigates performance under different probability thresholds and transaction-cost assumptions.

These tests help evaluate sensitivity to historical periods, trade frequency, and estimated execution costs.

## Key Findings

The saved model metadata recorded a chronological holdout accuracy of approximately **58.94%**, compared with a recorded majority-class baseline accuracy of approximately **52.85%** for that evaluation.

The four-round walk-forward evaluation produced the following average results for Logistic Regression:

| Metric | Result |
|---|---:|
| Average classification accuracy | 53.89% |
| Average UP precision | 54.08% |
| Average UP recall | 35.36% |
| Average strategy return per round | -2.27% |
| Average benchmark return per round | -5.95% |
| Average excess return per round | +3.68 percentage points |
| Average maximum drawdown per round | -7.86% |

Results varied across rounds. Some periods outperformed the benchmark, while others produced negative strategy returns and underperformed.

These results describe historical experiments under stated assumptions. They do not demonstrate consistent profitability or guarantee future performance.

## Interactive Dashboard

The Streamlit dashboard presents:

- Strategy versus benchmark returns
- Final equity comparison
- Trade results and win rate
- Robustness testing results
- Performance tables and visualizations

Launch the dashboard from the project root:

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal in your browser.

## Project Structure

```text
ai-financial-intelligence/
├── app.py
├── data/
│   ├── reliance_stock_data.csv
│   └── model_ready_dataset.csv
├── models/
│   ├── logistic_regression.pkl
│   └── logistic_regression_metadata.pkl
├── reports/
│   ├── backtest_summary.csv
│   ├── baseline_comparison_summary.csv
│   ├── robustness_summary.csv
│   └── walk_forward_results.csv
├── src/
│   ├── data_loader.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── predict.py
│   ├── backtest.py
│   ├── validate_backtest.py
│   ├── baseline_comparison.py
│   ├── walk_forward_validation.py
│   ├── robustness_testing.py
│   ├── threshold_tuning.py
│   └── visualize_robustness.py
├── requirements.txt
├── README.md
└── .gitignore
```

This is a representative structure. Additional scripts and generated reports may exist in the repository.

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Ashutoshkumarpandey27/ai-financial-intelligence.git
cd ai-financial-intelligence
```

### 2. Create a Virtual Environment

On Windows:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Dashboard

```bash
python -m streamlit run app.py
```

## Useful Commands

Run commands from the project root with the virtual environment activated.

**Exploratory data analysis**

```bash
python src/eda.py
```

**Train and evaluate Logistic Regression**

```bash
python src/train_model.py
```

This script trains a model using its configured chronological split, evaluates it on the held-out test period, and saves the model and metadata. It overwrites the configured model files.

**Run the backtest**

```bash
python src/backtest.py
```

**Validate backtest reports**

```bash
python src/validate_backtest.py
```

**Compare baseline strategies**

```bash
python src/baseline_comparison.py
```

**Run walk-forward validation**

```bash
python src/walk_forward_validation.py
```

**Run robustness testing**

```bash
python src/robustness_testing.py
```

**Generate robustness visualizations**

```bash
python src/visualize_robustness.py
```

**Launch the dashboard**

```bash
python -m streamlit run app.py
```

Training and evaluation scripts can overwrite saved model files or reports. Run them intentionally and inspect their output.

## Reports and Artifacts

Reports are generated under the `reports/` directory, including backtest summaries, baseline comparisons, robustness summaries, and walk-forward results.

Saved model artifacts are stored under `models/`. The dashboard requires the relevant model, metadata, data, and report files to exist at the paths configured in the code.

## Limitations

- The model uses a limited set of historical price and volume features.
- Performance varies across evaluation periods.
- Some test windows contain relatively few trades, making performance estimates uncertain.
- Transaction costs are illustrative and are not broker-specific estimates.
- Brokerage, taxes, slippage, liquidity, and market impact can affect actual returns.
- The benchmark represents intraday Open-to-Close exposure, not continuous buy-and-hold investing.
- Historical backtests do not guarantee future performance.
- Further testing is required before any real-money deployment.

## Future Improvements

- Compare Logistic Regression with tree-based models.
- Evaluate probability calibration and threshold selection.
- Expand feature-alignment and leakage checks.
- Test additional chronological validation windows.
- Improve transaction-cost and slippage modeling.
- Add risk management and position-sizing rules.
- Compare with additional clearly defined benchmarks.
- Separate final model training from out-of-sample evaluation.
- Add downloadable reports and further dashboard explanations.

## Disclaimer

This project is for educational and research purposes only. It is not financial advice. Predictions and historical backtest results should not be used as the sole basis for investment decisions.
