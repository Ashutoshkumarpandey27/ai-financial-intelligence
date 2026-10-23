# AI Financial Intelligence Engine

A machine learning project that analyzes historical Reliance Industries stock data, predicts the probability of an upward intraday movement, and evaluates a rule-based trading strategy using chronological validation and transaction-cost sensitivity analysis.

## Project Objectives

- Collect and analyze historical stock market data.
- Engineer features from historical price and volume information.
- Train a Logistic Regression classification model.
- Backtest a probability-based trading strategy.
- Evaluate performance using walk-forward validation.
- Test different probability thresholds and transaction costs.
- Visualize strategy performance, drawdowns, and benchmark comparisons.

## Tech Stack

- Python
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- yfinance
- Git and GitHub

## Dataset

Historical data for Reliance Industries (`RELIANCE.NS`) was collected using Yahoo Finance.

The dataset contains historical OHLCV data:

- Open
- High
- Low
- Close
- Volume

Additional features include daily returns, lagged returns, a five-day moving average, and volume change.

### Target Definition

The target is:

- `1`: Next trading session's Close is greater than its Open.
- `0`: Otherwise.

The model uses historical features to estimate the probability of an upward intraday movement.

## Machine Learning Pipeline

1. Download historical stock data.
2. Perform exploratory data analysis.
3. Generate features and the target variable.
4. Split the data chronologically.
5. Scale features using `StandardScaler`.
6. Train a Logistic Regression classifier.
7. Generate predictions and probabilities.
8. Evaluate the model using classification metrics.
9. Backtest the trading rule.
10. Run walk-forward and robustness tests.

The model uses six features:

- Close
- Daily_Return
- Previous_Day_Return
- Return_2_Days_Ago
- MA_5
- Volume_Change

Future execution prices and the target-alignment columns are not used as model features.

## Trading Strategy

The strategy enters a trade when the model's predicted probability meets the selected threshold.

- Entry: Next session's Open.
- Exit: Same session's Close.
- Otherwise: Remain in cash.
- Initial capital: ₹10,000.
- Transaction costs: Tested at 0.1%, 0.2%, and 0.3% per round trip.

The strategy is evaluated against a benchmark that compounds the session's Open-to-Close returns. This benchmark is not equivalent to continuous buy-and-hold investing.

## Validation Methodology

### Chronological Testing

The model is trained on earlier observations and evaluated on later observations. The walk-forward evaluation uses expanding training windows and a one-row gap between training and test data.

### Threshold Tuning

Probability thresholds are selected using earlier chronological validation data. The selected threshold is then evaluated on a separate outer test window.

### Robustness Testing

The project compares fixed and tuned probability thresholds under multiple transaction-cost assumptions.

These procedures help investigate whether results depend on a particular time period, threshold, or cost assumption.

## Key Findings

The initial chronological holdout evaluation achieved approximately 58.94% classification accuracy, compared with a majority-class baseline accuracy of approximately 52.85%.

However, classification accuracy alone does not establish trading profitability.

In the four-round walk-forward robustness evaluation:

- The fixed 0.50 threshold produced an average strategy return of approximately -2.27% at the 0.1% transaction-cost assumption.
- The tuned-threshold strategy produced an average return of approximately -3.24% at the same cost assumption.
- At higher transaction costs, both strategies performed worse.
- The tuned strategy's lower trading frequency and drawdown did not consistently translate into higher returns.

The results suggest that transaction costs, time-period variation, and threshold selection materially affect strategy performance.

**Important:** The current experiments do not establish a consistently profitable trading strategy.

## Visualizations

The robustness visualization script generates the following charts:

- `reports/figures/walk_forward_returns.png`
- `reports/figures/transaction_cost_sensitivity.png`
- `reports/figures/drawdown_sensitivity.png`
- `reports/figures/excess_return_sensitivity.png`

Run the visualization script with:

```bash
python src/visualize_robustness.py
```

## How to Run

Clone the repository:

```bash
git clone https://github.com/Ashutoshkumarpandey27/ai-financial-intelligence.git
cd ai-financial-intelligence
```

Create and activate a virtual environment on Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run individual scripts from the project root, for example:

```bash
python src/eda.py
python src/robustness_testing.py
python src/visualize_robustness.py
```

Run other scripts only if they exist in the repository.

## Limitations

- The model uses a limited set of technical features.
- Results depend on the historical evaluation period.
- A small number of trades in some test windows makes performance estimates uncertain.
- Transaction costs and execution assumptions can materially change results.
- Historical backtests do not guarantee future performance.
- The current strategy requires further testing before any real-money use.

## Future Improvements

- Compare Logistic Regression with tree-based models.
- Add stronger leakage and feature-alignment tests.
- Evaluate calibration of predicted probabilities.
- Test additional chronological validation windows.
- Include more realistic slippage and execution assumptions.
- Add risk management and position-sizing rules.
- Compare against additional clearly defined benchmarks.
- Build a dashboard to explore predictions and backtest results.

## Disclaimer

This project is for educational and research purposes only. It is not financial advice and should not be used as the sole basis for investment decisions.
