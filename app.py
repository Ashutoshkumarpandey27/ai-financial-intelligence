from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "logistic_regression_current.pkl"
METADATA_PATH = (
    BASE_DIR / "models" / "logistic_regression_current_metadata.pkl"
)

DATA_PATH = BASE_DIR / "data" / "model_ready_dataset.csv"
RAW_DATA_PATH = BASE_DIR / "data" / "reliance_stock_data.csv"
REPORTS_DIR = BASE_DIR / "reports"

st.set_page_config(
    page_title="AI Financial Intelligence Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 2. FEATURE CONFIGURATION
# ============================================================

FEATURES = [
    "Close",
    "Daily_Return",
    "Previous_Day_Return",
    "Return_2_Days_Ago",
    "MA_5",
    "Volume_Change",
]

TARGET_DESCRIPTION = "Next session Close > next session Open"

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================


def find_column(df, possible_names):
    """Find a column using case-insensitive name matching."""
    lookup = {str(col).strip().lower(): col for col in df.columns}

    for name in possible_names:
        if name.lower() in lookup:
            return lookup[name.lower()]

    return None


def normalize_ohlcv_columns(df):
    """Normalize raw stock-data columns to Date, Open, High, Low, Close, Volume."""

    df = df.copy()

    # yfinance can sometimes produce a MultiIndex column structure.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            col[0] if isinstance(col, tuple) else col
            for col in df.columns
        ]

    date_col = find_column(df, ["Date", "Datetime", "Timestamp"])

    if date_col is None:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            date_col = find_column(
                df,
                ["Date", "Datetime", "Timestamp"],
            )
        else:
            raise ValueError(
                "Could not identify the Date column in the raw dataset."
            )

    if date_col != "Date":
        df = df.rename(columns={date_col: "Date"})

    required = ["Open", "High", "Low", "Close", "Volume"]

    rename_map = {}

    for required_col in required:
        matched_col = find_column(df, [required_col])

        if matched_col is None:
            raise ValueError(
                f"Required column '{required_col}' was not found "
                "in the raw stock dataset."
            )

        if matched_col != required_col:
            rename_map[matched_col] = required_col

    df = df.rename(columns=rename_map)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Remove timezone information if present.
    try:
        if df["Date"].dt.tz is not None:
            df["Date"] = df["Date"].dt.tz_localize(None)
    except (AttributeError, TypeError):
        pass

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date", "Open", "Close"])
    df = df.sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="last")
    df = df.reset_index(drop=True)

    return df


@st.cache_data(show_spinner=False)
def load_raw_data(file_path, modified_time):
    """Load the raw OHLCV dataset."""

    del modified_time  # Used to invalidate the cache when the file changes.

    if not Path(file_path).exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)
    df = normalize_ohlcv_columns(df)

    return df


def create_features(raw_df):
    """
    Generate the six features used by the current model.

    Features are calculated using information available by the
    end of each signal date. The target is not generated here.
    """

    df = raw_df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    # Daily return based on consecutive closing prices.
    df["Daily_Return"] = df["Close"].pct_change()

    # Previous trading session's return.
    df["Previous_Day_Return"] = df["Daily_Return"].shift(1)

    # Return from two trading sessions earlier.
    df["Return_2_Days_Ago"] = df["Daily_Return"].shift(2)

    # Five-session moving average of closing prices.
    df["MA_5"] = df["Close"].rolling(window=5).mean()

    # Change in volume compared with the previous session.
    df["Volume_Change"] = df["Volume"].pct_change()

    df = df.replace([np.inf, -np.inf], np.nan)

    return df


@st.cache_resource(show_spinner=False)
def load_model(model_path, metadata_path, model_modified, metadata_modified):
    """Load the trained model and its metadata."""

    del model_modified, metadata_modified

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    if not Path(metadata_path).exists():
        raise FileNotFoundError(
            f"Model metadata file not found: {metadata_path}"
        )

    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)

    return model, metadata


def load_csv_report(filename):
    """Load a saved report CSV if it exists."""

    path = REPORTS_DIR / filename

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


def metadata_value(metadata, *keys, default=None):
    """Return the first available metadata value among candidate keys."""

    if not isinstance(metadata, dict):
        return default

    for key in keys:
        if key in metadata and metadata[key] is not None:
            return metadata[key]

    return default


def format_percentage(value, decimals=2):
    """Format a numeric value as a percentage."""

    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "N/A"


def format_date(value):
    """Format a date-like value for display."""

    if value is None:
        return "N/A"

    try:
        return pd.to_datetime(value).strftime("%d %b %Y")
    except (TypeError, ValueError):
        return str(value)


def format_number(value, decimals=4):
    """Safely format a numeric value."""

    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def get_latest_prediction(raw_df, model, metadata):
    """
    Generate a prediction using the latest valid raw-data feature row.

    Leakage guard:
    The model's latest available training label must be strictly
    earlier than the signal date.
    """

    feature_df = create_features(raw_df)

    valid_rows = feature_df.dropna(subset=FEATURES).copy()

    if valid_rows.empty:
        raise ValueError(
            "There are not enough raw-data rows to calculate all "
            "the required model features."
        )

    latest_row = valid_rows.iloc[-1]
    signal_date = pd.to_datetime(latest_row["Date"])

    labels_available_through = metadata_value(
        metadata,
        "training_labels_available_through",
        "labels_available_through",
    )

    if labels_available_through is None:
        raise ValueError(
            "The model metadata does not contain "
            "'training_labels_available_through'. "
            "Retrain the current model with the leakage-safe "
            "training script before using this dashboard."
        )

    labels_date = pd.to_datetime(
        labels_available_through,
        errors="coerce",
    )

    if pd.isna(labels_date):
        raise ValueError(
            "The training-label availability date in model metadata "
            "could not be parsed."
        )

    # Prevent using a model whose training labels include information
    # from the signal date or a later date.
    if labels_date.normalize() >= signal_date.normalize():
        raise ValueError(
            "Leakage-safety check failed. The model's training labels "
            f"are available through {format_date(labels_date)}, but "
            f"the signal date is {format_date(signal_date)}. "
            "Retrain the model using only labels available before "
            "the signal date."
        )

    X_latest = pd.DataFrame(
        [latest_row[FEATURES].to_dict()],
        columns=FEATURES,
    )

    prediction = int(model.predict(X_latest)[0])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_latest)[0]

        classes = list(model.classes_)

        if 1 in classes:
            up_index = classes.index(1)
            probability_up = float(probabilities[up_index])
        else:
            probability_up = 0.0

        if 0 in classes:
            not_up_index = classes.index(0)
            probability_not_up = float(probabilities[not_up_index])
        else:
            probability_not_up = 0.0
    else:
        probability_up = float(prediction == 1)
        probability_not_up = float(prediction == 0)

    result = {
        "signal_date": signal_date,
        "prediction": prediction,
        "probability_up": probability_up,
        "probability_not_up": probability_not_up,
        "features": X_latest.iloc[0],
        "labels_available_through": labels_date,
    }

    return result


def safe_numeric_series(df, column):
    """Return a numeric series if the requested column exists."""

    if df is None or column not in df.columns:
        return None

    return pd.to_numeric(df[column], errors="coerce")


# ============================================================
# 4. PAGE HEADER
# ============================================================

st.title("📈 AI Financial Intelligence Engine")

st.markdown(
    """
    An interactive research dashboard for stock-market data,
    machine-learning predictions, backtesting, and model evaluation.

    **Dataset:** Reliance Industries (RELIANCE.NS)
    """
)

st.caption(
    "Research and educational tool only. Predictions are uncertain "
    "and should not be treated as financial advice."
)

st.divider()

# ============================================================
# 5. LOAD MODEL, METADATA, AND DATA
# ============================================================

try:
    raw_modified_time = RAW_DATA_PATH.stat().st_mtime
    model_modified_time = MODEL_PATH.stat().st_mtime
    metadata_modified_time = METADATA_PATH.stat().st_mtime

    raw_df = load_raw_data(
        str(RAW_DATA_PATH),
        raw_modified_time,
    )

    model, metadata = load_model(
        str(MODEL_PATH),
        str(METADATA_PATH),
        model_modified_time,
        metadata_modified_time,
    )

except FileNotFoundError as exc:
    st.error(str(exc))

    st.info(
        "Check that the raw data, current model, and current model "
        "metadata files exist in the expected project folders."
    )

    st.stop()

except Exception as exc:
    st.error(f"Could not load the project files: {exc}")
    st.stop()


# ============================================================
# 6. SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Dashboard Controls")

st.sidebar.markdown("### Project Information")

st.sidebar.write("**Stock:** Reliance Industries")
st.sidebar.write("**Ticker:** RELIANCE.NS")
st.sidebar.write("**Model:** Logistic Regression")
st.sidebar.write("**Target:** Next-session direction")

raw_start_date = raw_df["Date"].min()
raw_end_date = raw_df["Date"].max()

st.sidebar.markdown("### Dataset Information")

st.sidebar.write(
    f"**First raw-data date:** {format_date(raw_start_date)}"
)

st.sidebar.write(
    f"**Latest raw-data date:** {format_date(raw_end_date)}"
)

st.sidebar.write(
    f"**Raw-data rows:** {len(raw_df):,}"
)

st.sidebar.caption(
    "The prediction uses the latest valid feature row in the "
    "available raw dataset. It does not automatically fetch "
    "live market data."
)

if st.sidebar.button("🔄 Refresh dashboard data", use_container_width=True):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()


# ============================================================
# 7. GENERATE LATEST PREDICTION
# ============================================================

try:
    latest_result = get_latest_prediction(
        raw_df,
        model,
        metadata,
    )

except Exception as exc:
    st.error(f"Prediction could not be generated: {exc}")
    st.stop()


# ============================================================
# 8. MODEL OVERVIEW
# ============================================================

st.header("1. Model Overview")

col1, col2, col3, col4 = st.columns(4)

training_rows = metadata_value(
    metadata,
    "training_rows",
    "n_training_rows",
)

holdout_accuracy = metadata_value(
    metadata,
    "holdout_accuracy",
    "test_accuracy",
    "accuracy",
)

baseline_accuracy = metadata_value(
    metadata,
    "baseline_accuracy",
    "majority_class_accuracy",
)

precision_up = metadata_value(
    metadata,
    "precision_up",
)

recall_up = metadata_value(
    metadata,
    "recall_up",
)

f1_up = metadata_value(
    metadata,
    "f1_up",
)

with col1:
    st.metric(
        "Holdout Accuracy",
        format_percentage(holdout_accuracy),
    )

with col2:
    st.metric(
        "Baseline Accuracy",
        format_percentage(baseline_accuracy),
    )

with col3:
    st.metric(
        "Training Rows",
        f"{int(training_rows):,}"
        if training_rows is not None
        else "N/A",
    )

with col4:
    st.metric(
        "Latest Signal Date",
        format_date(latest_result["signal_date"]),
    )


with st.expander("View model training details"):
    training_start = metadata_value(
        metadata,
        "training_start",
        "train_start",
    )

    training_end = metadata_value(
        metadata,
        "training_end",
        "train_end",
    )

    holdout_start = metadata_value(
        metadata,
        "holdout_start",
        "test_start",
    )

    holdout_end = metadata_value(
        metadata,
        "holdout_end",
        "test_end",
    )

    labels_through = metadata_value(
        metadata,
        "training_labels_available_through",
        "labels_available_through",
    )

    details = {
        "Target": TARGET_DESCRIPTION,
        "Features": ", ".join(FEATURES),
        "Training start": format_date(training_start),
        "Training end": format_date(training_end),
        "Holdout start": format_date(holdout_start),
        "Holdout end": format_date(holdout_end),
        "Training labels available through": format_date(
            labels_through
        ),
        "Raw dataset latest date": format_date(raw_end_date),
    }

    st.table(
        pd.DataFrame(
            list(details.items()),
            columns=["Property", "Value"],
        )
    )

    if holdout_accuracy is not None and baseline_accuracy is not None:
        difference = (
            float(holdout_accuracy) - float(baseline_accuracy)
        ) * 100

        st.write(
            f"Holdout accuracy minus baseline accuracy: "
            f"**{difference:+.2f} percentage points**"
        )

st.caption(
    "Holdout performance is historical. It does not guarantee "
    "future predictive performance."
)

st.divider()


# ============================================================
# 9. LATEST PREDICTION
# ============================================================

st.header("2. Latest Model Prediction")

prediction_col1, prediction_col2 = st.columns([1, 1])

with prediction_col1:
    if latest_result["prediction"] == 1:
        st.success("Prediction: UP")
    else:
        st.warning("Prediction: NOT UP")

    st.metric(
        "Probability of UP",
        format_percentage(latest_result["probability_up"]),
    )

    st.metric(
        "Probability of NOT UP",
        format_percentage(latest_result["probability_not_up"]),
    )

    st.write(
        f"**Signal date:** "
        f"{format_date(latest_result['signal_date'])}"
    )

    st.write(
        "**Training labels available through:** "
        f"{format_date(latest_result['labels_available_through'])}"
    )

with prediction_col2:
    probability_df = pd.DataFrame(
        {
            "Outcome": ["UP", "NOT UP"],
            "Probability": [
                latest_result["probability_up"],
                latest_result["probability_not_up"],
            ],
        }
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(
        probability_df["Outcome"],
        probability_df["Probability"],
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Latest Prediction Probabilities")

    for index, value in enumerate(probability_df["Probability"]):
        ax.text(
            index,
            value + 0.02,
            f"{value:.1%}",
            ha="center",
        )

    st.pyplot(fig)
    plt.close(fig)

probability_difference = abs(
    latest_result["probability_up"]
    - latest_result["probability_not_up"]
)

if probability_difference < 0.10:
    st.warning(
        "The model's probabilities are close together. "
        "This is a low-confidence signal, not a strong directional "
        "indication."
    )
else:
    st.info(
        "The model favors one outcome, but this probability is not "
        "a guarantee that the market will move in that direction."
    )

st.caption(
    "UP means the model predicts the next session's closing price "
    "will be greater than its opening price. NOT UP means the model "
    "predicts that this condition will not hold."
)

st.divider()


# ============================================================
# 10. LATEST MODEL INPUTS
# ============================================================

st.header("3. Latest Model Inputs")

latest_features = latest_result["features"]

feature_display = pd.DataFrame(
    {
        "Feature": FEATURES,
        "Value": [
            latest_features[feature]
            for feature in FEATURES
        ],
    }
)

feature_display["Value"] = feature_display["Value"].map(
    lambda value: (
        f"{value:,.6f}"
        if pd.notna(value)
        else "N/A"
    )
)

st.dataframe(
    feature_display,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "These values are calculated from the latest valid raw-data "
    "row available to the application."
)

st.divider()


# ============================================================
# 11. HISTORICAL STOCK DATA
# ============================================================

st.header("4. Historical Stock Data")

chart_df = raw_df.copy()

chart_df["Date"] = pd.to_datetime(chart_df["Date"])

if len(chart_df) > 0:
    min_date = chart_df["Date"].min().date()
    max_date = chart_df["Date"].max().date()

    date_col1, date_col2 = st.columns(2)

    with date_col1:
        start_date = st.date_input(
            "Start date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            key="historical_start_date",
        )

    with date_col2:
        end_date = st.date_input(
            "End date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="historical_end_date",
        )

    if start_date > end_date:
        st.error("Start date must be on or before the end date.")
    else:
        filtered_chart_df = chart_df[
            (chart_df["Date"].dt.date >= start_date)
            & (chart_df["Date"].dt.date <= end_date)
        ].copy()

        if filtered_chart_df.empty:
            st.warning("No stock data exists for the selected date range.")
        else:
            fig, ax = plt.subplots(figsize=(12, 5))

            ax.plot(
                filtered_chart_df["Date"],
                filtered_chart_df["Close"],
                label="Closing Price",
            )

            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.set_title("Reliance Closing Price")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.xticks(rotation=30)
            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            with st.expander("View historical stock data"):
                st.dataframe(
                    filtered_chart_df,
                    use_container_width=True,
                    hide_index=True,
                )

st.divider()


# ============================================================
# 12. CLASSIFICATION METRICS
# ============================================================

st.header("5. Classification Metrics")

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric(
        "Precision — UP",
        format_percentage(precision_up),
    )

with metric_col2:
    st.metric(
        "Recall — UP",
        format_percentage(recall_up),
    )

with metric_col3:
    st.metric(
        "F1 Score — UP",
        format_number(f1_up, 4),
    )

st.markdown(
    """
    **How to interpret these metrics**

    - **Precision:** Of the sessions predicted as UP, how many
      actually satisfied the UP condition?
    - **Recall:** Of all sessions that actually satisfied the UP
      condition, how many did the model identify?
    - **F1 score:** The harmonic mean of precision and recall.

    These metrics describe historical holdout performance.
    They are not estimates of guaranteed future returns.
    """
)

st.divider()


# ============================================================
# 13. HISTORICAL BACKTEST RESULTS
# ============================================================

st.header("6. Historical Backtest Results")

backtest_summary = load_csv_report("backtest_final_summary.csv")
backtest_trades = load_csv_report("backtest_trades.csv")

if backtest_summary is None:
    st.info(
        "Backtest summary was not found in the reports folder. "
        "Run the backtest script to generate it."
    )
else:
    st.subheader("Backtest Summary")

    st.dataframe(
        backtest_summary,
        use_container_width=True,
        hide_index=True,
    )

    # Display common summary metrics if the expected fields exist.
    summary_row = (
        backtest_summary.iloc[0]
        if not backtest_summary.empty
        else None
    )

    if summary_row is not None:
        summary_lookup = {
            str(column).strip().lower(): column
            for column in backtest_summary.columns
        }

        def summary_field(*names):
            for name in names:
                col = summary_lookup.get(name.lower())
                if col is not None:
                    return summary_row[col]
            return None

        sessions = summary_field(
            "sessions",
            "total_sessions",
        )

        trades = summary_field(
            "trades",
            "total_trades",
        )

        winning_trades = summary_field(
            "winning_trades",
            "winning",
            "wins",
        )

        losing_trades = summary_field(
            "losing_trades",
            "losing",
            "losses",
        )

        win_rate = summary_field(
            "win_rate",
            "win rate",
        )

        strategy_return = summary_field(
            "strategy_return",
            "total_strategy_return",
        )

        max_drawdown = summary_field(
            "max_drawdown",
            "maximum_drawdown",
        )

        sharpe = summary_field(
            "annualized_sharpe",
            "sharpe_ratio",
            "sharpe",
        )

        equity_col1, equity_col2, equity_col3 = st.columns(3)

        with equity_col1:
            st.metric(
                "Sessions",
                f"{int(sessions):,}"
                if pd.notna(sessions)
                else "N/A",
            )

        with equity_col2:
            st.metric(
                "Trades",
                f"{int(trades):,}"
                if pd.notna(trades)
                else "N/A",
            )

        with equity_col3:
            st.metric(
                "Win Rate",
                format_percentage(win_rate),
            )

        return_col1, return_col2, return_col3 = st.columns(3)

        with return_col1:
            st.metric(
                "Strategy Return",
                format_percentage(strategy_return),
            )

        with return_col2:
            st.metric(
                "Maximum Drawdown",
                format_percentage(max_drawdown),
            )

        with return_col3:
            st.metric(
                "Annualized Sharpe",
                format_number(sharpe),
            )

        if (
            winning_trades is not None
            or losing_trades is not None
        ):
            st.write(
                f"**Winning trades:** "
                f"{winning_trades if winning_trades is not None else 'N/A'}"
            )

            st.write(
                f"**Losing trades:** "
                f"{losing_trades if losing_trades is not None else 'N/A'}"
            )

if backtest_trades is not None:
    with st.expander("View individual backtest trades"):
        st.dataframe(
            backtest_trades,
            use_container_width=True,
            hide_index=True,
        )

st.caption(
    "The backtest results are loaded from saved reports. "
    "They are not automatically regenerated when the model changes."
)

st.divider()


# ============================================================
# 14. ROBUSTNESS TESTING
# ============================================================

st.header("7. Robustness Testing")

robustness_summary = load_csv_report("robustness_summary.csv")
robustness_results = load_csv_report("robustness_results.csv")

if robustness_summary is not None:
    st.subheader("Robustness Summary")

    st.dataframe(
        robustness_summary,
        use_container_width=True,
        hide_index=True,
    )

elif robustness_results is not None:
    st.subheader("Robustness Results")

    st.dataframe(
        robustness_results,
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info(
        "No saved robustness report was found. Run the robustness "
        "testing script to create the report."
    )

st.markdown(
    """
    Robustness testing checks whether the historical results remain
    similar under different assumptions or market conditions.

    A strategy that performs well in one backtest may still fail
    when market conditions change.
    """
)

st.divider()


# ============================================================
# 15. BASELINE COMPARISON
# ============================================================

st.header("8. Baseline Comparison")

baseline_comparison = load_csv_report(
    "baseline_comparison_summary.csv"
)

if baseline_comparison is not None:
    st.dataframe(
        baseline_comparison,
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info(
        "The baseline comparison report was not found. "
        "Run the model comparison script to generate it."
    )

st.markdown(
    """
    A model should be evaluated against simple alternatives.
    Accuracy above a baseline does not automatically mean that
    a strategy is profitable after transaction costs and slippage.
    """
)

st.divider()


# ============================================================
# 16. DATA QUALITY INFORMATION
# ============================================================

st.header("9. Data Quality")

quality_col1, quality_col2, quality_col3 = st.columns(3)

with quality_col1:
    st.metric(
        "Raw Dataset Rows",
        f"{len(raw_df):,}",
    )

with quality_col2:
    st.metric(
        "Missing Close Values",
        f"{raw_df['Close'].isna().sum():,}",
    )

with quality_col3:
    st.metric(
        "Zero-Volume Rows",
        f"{(raw_df['Volume'] == 0).sum():,}",
    )

with st.expander("View data-quality checks"):
    st.write("**Dataset date range**")

    st.write(
        f"{format_date(raw_df['Date'].min())} to "
        f"{format_date(raw_df['Date'].max())}"
    )

    st.write("**Missing values by column**")

    missing_df = (
        raw_df.isna()
        .sum()
        .rename("Missing Values")
        .reset_index()
    )

    missing_df.columns = ["Column", "Missing Values"]

    st.dataframe(
        missing_df,
        use_container_width=True,
        hide_index=True,
    )

    st.write("**Duplicate dates**")

    duplicate_dates = int(
        raw_df["Date"].duplicated().sum()
    )

    st.write(duplicate_dates)

st.caption(
    "Zero trading volume and other unusual observations should be "
    "investigated before drawing conclusions from historical results."
)

st.divider()


# ============================================================
# 17. DISCLAIMER
# ============================================================

st.header("10. Disclaimer")

st.warning(
    """
    This application is for educational and research purposes only.

    - Predictions are based on historical data and a trained model.
    - Historical accuracy does not guarantee future performance.
    - A predicted UP direction does not guarantee a profitable trade.
    - Transaction costs, slippage, liquidity, and market conditions
      can materially affect actual trading results.
    - The application does not provide personalized financial advice.
    - The available dataset may be delayed or stale and is not
      necessarily live market data.

    Do not make investment decisions solely on the basis of this
    dashboard or its predictions.
    """
)

st.caption(
    "AI Financial Intelligence Engine | "
    "Built for machine-learning experimentation and financial research"
)