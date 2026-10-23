
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AI Financial Intelligence Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "logistic_regression.pkl"
METADATA_PATH = BASE_DIR / "models" / "logistic_regression_metadata.pkl"
DATA_PATH = BASE_DIR / "data" / "model_ready_dataset.csv"
REPORTS_DIR = BASE_DIR / "reports"


# ==========================================================
# 2. LOAD MODEL, METADATA AND DATA
# ==========================================================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    return model, metadata


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["Date"])


def load_report(filename):
    """Load a report CSV if it exists."""
    path = REPORTS_DIR / filename

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


try:
    model, metadata = load_model()
    data = load_data()

except Exception as error:
    st.error(f"Unable to load the project files: {error}")
    st.info(
        "Check that the model, metadata and model-ready dataset "
        "exist in the expected project folders."
    )
    st.stop()


# ==========================================================
# 3. VALIDATE DATA AND MODEL FEATURES
# ==========================================================

if data.empty:
    st.error("The model-ready dataset is empty.")
    st.stop()

if "Date" not in data.columns:
    st.error("The dataset does not contain a Date column.")
    st.stop()

data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
data = data.dropna(subset=["Date"])
data = data.sort_values("Date").reset_index(drop=True)

if data.empty:
    st.error("The dataset does not contain valid dates.")
    st.stop()

features = metadata.get("feature_columns")

if not features:
    st.error("The model metadata does not contain feature_columns.")
    st.stop()

missing_features = [
    feature for feature in features
    if feature not in data.columns
]

if missing_features:
    st.error(f"Missing required model features: {missing_features}")
    st.stop()

# Prevent future information from being used as model input.
forbidden_features = {"Target", "Next_Open", "Next_Close"}

if forbidden_features.intersection(features):
    st.error(
        "The saved model metadata contains future information "
        "as model input features. Retrain the model before using "
        "this dashboard."
    )
    st.stop()

latest_row = data.iloc[[-1]]

X_latest = latest_row[features].apply(
    pd.to_numeric,
    errors="coerce",
)

if X_latest.isna().any().any():
    st.error(
        "The latest row contains missing or non-numeric model features."
    )
    st.stop()

if not np.isfinite(X_latest.to_numpy(dtype=float)).all():
    st.error("The latest model inputs contain infinite values.")
    st.stop()

if not hasattr(model, "predict_proba"):
    st.error("The saved model does not support probability predictions.")
    st.stop()

if not hasattr(model, "classes_"):
    st.error("The saved model does not expose its trained classes.")
    st.stop()

classes = list(model.classes_)

if 0 not in classes or 1 not in classes:
    st.error(
        "The saved model does not contain both expected classes: 0 and 1."
    )
    st.stop()


# ==========================================================
# 4. SIDEBAR — PROJECT INFORMATION
# ==========================================================

st.sidebar.title("📊 Project Information")

st.sidebar.write("**Stock:** Reliance Industries")
st.sidebar.write("**Ticker:** RELIANCE.NS")
st.sidebar.write("**Model:** Logistic Regression")
st.sidebar.write("**Prediction:** Next-session intraday direction")

st.sidebar.divider()
st.sidebar.subheader("Available Dataset")

st.sidebar.write(f"**First date:** {data['Date'].min().date()}")
st.sidebar.write(f"**Last date:** {data['Date'].max().date()}")
st.sidebar.write(f"**Records:** {len(data):,}")

st.sidebar.caption(
    "The dataset is historical and may not include today's market data. "
    "This dashboard does not fetch live stock prices."
)

if st.sidebar.button("🔄 Refresh dashboard data"):
    load_model.clear()
    load_data.clear()
    st.rerun()


# ==========================================================
# 5. MAIN HEADER
# ==========================================================

st.title("📈 AI Financial Intelligence Engine")

st.caption(
    "Machine learning-based stock direction analysis "
    "for Reliance Industries (RELIANCE.NS)"
)

st.info(
    "Test accuracy measures historical classification performance. "
    "It does not prove that a trading strategy is profitable."
)


# ==========================================================
# 6. MODEL OVERVIEW
# ==========================================================

st.header("Model Overview")

test_accuracy = metadata.get("test_accuracy")
baseline_accuracy = metadata.get("baseline_accuracy")
train_rows = metadata.get("train_rows")

col1, col2, col3, col4 = st.columns(4)

col1.metric("ML Model", "Logistic Regression")

if test_accuracy is not None:
    col2.metric("Test Accuracy", f"{float(test_accuracy):.2%}")
else:
    col2.metric("Test Accuracy", "Not available")

if baseline_accuracy is not None:
    col3.metric(
        "Baseline Accuracy",
        f"{float(baseline_accuracy):.2%}",
    )
else:
    col3.metric("Baseline Accuracy", "Not available")

if train_rows is not None:
    col4.metric("Training Samples", f"{int(train_rows):,}")
else:
    col4.metric("Training Samples", "Not available")

train_start = metadata.get("train_start", "Unknown")
train_end = metadata.get("train_end", "Unknown")
test_start = metadata.get("test_start", "Unknown")
test_end = metadata.get("test_end", "Unknown")

st.caption(
    f"Training period: {train_start} to {train_end} "
    f"| Test period: {test_start} to {test_end}"
)


# ==========================================================
# 7. LATEST MODEL PREDICTION
# ==========================================================

st.divider()
st.header("Latest Available Model Prediction")

prediction = int(model.predict(X_latest)[0])
probabilities = model.predict_proba(X_latest)[0]

probability_up = float(probabilities[classes.index(1)])
probability_not_up = float(probabilities[classes.index(0)])

col1, col2, col3 = st.columns(3)

col1.metric(
    "Predicted Direction",
    "UP" if prediction == 1 else "NOT UP",
)

col2.metric(
    "Probability UP",
    f"{probability_up:.2%}",
)

col3.metric(
    "Probability NOT UP",
    f"{probability_not_up:.2%}",
)

latest_date = latest_row["Date"].iloc[0]

st.write(f"**Latest feature date:** {latest_date.date()}")

st.caption(
    "UP = next session's Close is greater than its Open. "
    "NOT UP = next session's Close is less than or equal to its Open. "
    "Probabilities are model estimates, not guarantees."
)


# ==========================================================
# 8. INTERACTIVE PREDICTION PROBABILITY CHART
# ==========================================================

probability_chart = pd.DataFrame({
    "Direction": ["NOT UP", "UP"],
    "Probability": [probability_not_up, probability_up],
})

probability_chart["Percentage"] = (
    probability_chart["Probability"] * 100
)

st.subheader("Prediction Probabilities")

probability_fig = px.bar(
    probability_chart,
    x="Direction",
    y="Percentage",
    text=probability_chart["Percentage"].map(
        lambda value: f"{value:.2f}%"
    ),
    hover_data={
        "Direction": True,
        "Percentage": ":.2f",
        "Probability": False,
    },
    labels={
        "Direction": "Predicted Direction",
        "Percentage": "Probability (%)",
    },
    title="Model's Estimated Next-Session Direction",
)

probability_fig.update_traces(
    textposition="outside",
    cliponaxis=False,
)

probability_fig.update_layout(
    yaxis=dict(
        range=[0, 100],
        ticksuffix="%",
        dtick=20,
        title="Probability",
    ),
    xaxis_title="Predicted Direction",
    showlegend=False,
    height=420,
    margin=dict(t=80, b=50, l=50, r=30),
)

st.plotly_chart(
    probability_fig,
    use_container_width=True,
)


# ==========================================================
# 9. LATEST MODEL INPUTS
# ==========================================================

st.divider()
st.header("Latest Model Inputs")

input_table = pd.DataFrame({
    "Feature": features,
    "Latest Value": [
        float(X_latest.iloc[0][feature])
        for feature in features
    ],
})

st.dataframe(
    input_table,
    use_container_width=True,
    hide_index=True,
)




# ==========================================================
# 10. INTERACTIVE HISTORICAL MARKET DATA
# ==========================================================

st.divider()
st.header("Historical Market Data")

# Prepare historical stock data
chart_data = data[["Date", "Close"]].copy()

chart_data["Date"] = pd.to_datetime(
    chart_data["Date"],
    errors="coerce",
)

chart_data["Close"] = pd.to_numeric(
    chart_data["Close"],
    errors="coerce",
)

# Remove invalid rows and sort by date
chart_data = chart_data.dropna(
    subset=["Date", "Close"]
).sort_values("Date").reset_index(drop=True)

if chart_data.empty:
    st.warning(
        "No valid closing-price data is available to chart."
    )

else:
    min_date = chart_data["Date"].min().date()
    max_date = chart_data["Date"].max().date()

    st.subheader("Filter Historical Data")

    # Date selection controls
    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            key="historical_start_date",
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="historical_end_date",
        )

    if start_date > end_date:
        st.error(
            "Start Date cannot be after End Date."
        )

    else:
        # Filter data according to the selected dates
        filtered_data = chart_data[
            (chart_data["Date"].dt.date >= start_date)
            & (chart_data["Date"].dt.date <= end_date)
        ].copy()

        if filtered_data.empty:
            st.warning(
                "No stock-price data is available "
                "for the selected date range."
            )

        else:
            st.caption(
                f"Showing {len(filtered_data):,} records "
                f"from {start_date} to {end_date}."
            )

            # Interactive historical closing-price chart
            historical_fig = px.line(
                filtered_data,
                x="Date",
                y="Close",
                title="RELIANCE.NS - Historical Closing Price",
                labels={
                    "Date": "Date",
                    "Close": "Closing Price (INR)",
                },
                hover_data={
                    "Date": "|%d %b %Y",
                    "Close": ":,.2f",
                },
            )

            historical_fig.update_traces(
                line=dict(width=2),
                hovertemplate=(
                    "Date: %{x|%d %b %Y}<br>"
                    "Closing Price: ₹%{y:,.2f}"
                    "<extra></extra>"
                ),
            )

            historical_fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Closing Price (INR)",
                hovermode="x unified",
                height=480,
                margin=dict(
                    t=75,
                    b=50,
                    l=60,
                    r=30,
                ),
            )

            # Interactive date-range slider
            historical_fig.update_xaxes(
                rangeslider_visible=True,
            )

            st.plotly_chart(
                historical_fig,
                use_container_width=True,
                key="historical_closing_price_chart",
            )

    st.caption(
        f"Available dataset dates: "
        f"{min_date} to {max_date}"
    )

    # Show the latest ten records
    with st.expander("View latest dataset records"):
        st.dataframe(
            data.tail(10),
            use_container_width=True,
            hide_index=True,
        )


# ==========================================================
# 11. CLASSIFICATION PERFORMANCE
# ==========================================================


# ==========================================================
# 11. CLASSIFICATION PERFORMANCE
# ==========================================================

st.divider()
st.header("Classification Performance")

precision = metadata.get("precision_up")
recall = metadata.get("recall_up")
f1_score = metadata.get("f1_up")

col1, col2, col3 = st.columns(3)

if precision is not None:
    col1.metric("UP Precision", f"{float(precision):.2%}")
else:
    col1.metric("UP Precision", "Not available")

if recall is not None:
    col2.metric("UP Recall", f"{float(recall):.2%}")
else:
    col2.metric("UP Recall", "Not available")

if f1_score is not None:
    col3.metric("UP F1 Score", f"{float(f1_score):.4f}")
else:
    col3.metric("UP F1 Score", "Not available")

st.markdown(
    """
    - **Precision:** Of sessions predicted UP, how many were actually UP?
    - **Recall:** Of all actual UP sessions, how many did the model identify?
    - **F1 score:** A combined measure of precision and recall.
    """
)


# ==========================================================
# 12. HISTORICAL BACKTEST RESULTS
# ==========================================================

st.divider()
st.header("Historical Backtest Results")

backtest_summary = load_report("backtest_summary.csv")

if backtest_summary is not None and not backtest_summary.empty:

    if not {"Metric", "Value"}.issubset(backtest_summary.columns):
        st.warning(
            "The backtest report must contain 'Metric' and 'Value' columns."
        )

    else:
        metrics = dict(
            zip(
                backtest_summary["Metric"],
                backtest_summary["Value"],
            )
        )

        def get_numeric_metric(name):
            value = metrics.get(name)

            if value is None or pd.isna(value):
                return None

            try:
                return float(value)
            except (ValueError, TypeError):
                return None

        trading_sessions = get_numeric_metric("Trading Sessions")
        completed_trades = get_numeric_metric(
            "UP Signals / Completed Trades"
        )
        win_rate = get_numeric_metric("Win Rate")
        strategy_return = get_numeric_metric("ML Strategy Return")
        benchmark_return = get_numeric_metric(
            "Matched-Period Benchmark Return"
        )
        max_drawdown = get_numeric_metric("Maximum Drawdown")
        sharpe_ratio = get_numeric_metric("Annualized Sharpe Ratio")
        initial_capital = get_numeric_metric("Initial Capital")
        final_strategy_equity = get_numeric_metric(
            "Final Strategy Equity"
        )
        final_benchmark_equity = get_numeric_metric(
            "Final Benchmark Equity"
        )

        # A. KEY PERFORMANCE METRICS

        st.subheader("Performance Summary")

        col1, col2, col3, col4 = st.columns(4)

        if win_rate is not None:
            col1.metric("Win Rate", f"{win_rate:.2%}")

        if strategy_return is not None:
            col2.metric(
                "ML Strategy Return",
                f"{strategy_return:+.2%}",
            )

        if benchmark_return is not None:
            col3.metric(
                "Benchmark Return",
                f"{benchmark_return:+.2%}",
            )

        if max_drawdown is not None:
            col4.metric(
                "Maximum Drawdown",
                f"{max_drawdown:.2%}",
            )

        col5, col6, col7 = st.columns(3)

        if trading_sessions is not None:
            col5.metric(
                "Trading Sessions",
                f"{int(trading_sessions):,}",
            )

        if completed_trades is not None:
            col6.metric(
                "Completed Trades",
                f"{int(completed_trades):,}",
            )

        if sharpe_ratio is not None:
            col7.metric(
                "Annualized Sharpe Ratio",
                f"{sharpe_ratio:.4f}",
            )
        
        # ------------------------------------------------------
        # B. STRATEGY VS BENCHMARK RETURNS
        # ------------------------------------------------------

        st.subheader("Strategy vs Benchmark Returns")

        if (
            strategy_return is not None
            and benchmark_return is not None
        ):
            return_chart = pd.DataFrame({
                "Strategy": [
                    "ML Strategy",
                    "Matched-Period Benchmark",
                ],
                "Return (%)": [
                    strategy_return * 100,
                    benchmark_return * 100,
                ],
            })

            return_fig = px.bar(
                return_chart,
                x="Strategy",
                y="Return (%)",
                color="Strategy",
                text=return_chart["Return (%)"].map(
                    lambda value: f"{value:+.2f}%"
                ),
                hover_data={
                    "Strategy": True,
                    "Return (%)": ":.2f",
                },
                title="ML Strategy vs Market Benchmark",
                labels={
                    "Strategy": "Investment Strategy",
                    "Return (%)": "Total Return (%)",
                },
            )

            return_fig.update_traces(
                textposition="outside",
                cliponaxis=False,
            )

            # Zero line makes positive and negative returns clear.
            return_fig.add_hline(
                y=0,
                line_width=1,
                line_dash="dash",
                line_color="gray",
            )

            return_fig.update_layout(
                xaxis_title="",
                yaxis_title="Total Return (%)",
                showlegend=False,
                height=450,
                hovermode="x unified",
                margin=dict(
                    t=80,
                    b=70,
                    l=60,
                    r=40,
                ),
            )

            st.plotly_chart(
                return_fig,
                use_container_width=True,
                key="strategy_benchmark_returns_chart",
            )

            st.caption(
                "Returns are from the saved historical backtest. "
                "Past results do not guarantee future performance."
            )

        else:
            st.info(
                "Strategy or benchmark return data is unavailable "
                "in the backtest report."
            )

        
        # ------------------------------------------------------
        # C. FINAL EQUITY COMPARISON
        # ------------------------------------------------------

        st.subheader("Final Equity Comparison")

        if (
            initial_capital is not None
            and final_strategy_equity is not None
            and final_benchmark_equity is not None
        ):
            equity_chart = pd.DataFrame({
                "Portfolio": [
                    "Initial Capital",
                    "ML Strategy",
                    "Benchmark",
                ],
                "Portfolio Value (INR)": [
                    initial_capital,
                    final_strategy_equity,
                    final_benchmark_equity,
                ],
            })

            equity_chart["Display Value"] = (
                equity_chart["Portfolio Value (INR)"].map(
                    lambda value: f"INR {value:,.2f}"
                )
            )

            equity_fig = px.bar(
                equity_chart,
                x="Portfolio",
                y="Portfolio Value (INR)",
                color="Portfolio",
                text="Display Value",
                custom_data=["Display Value"],
                title="Initial Capital vs Final Portfolio Values",
                labels={
                    "Portfolio": "Portfolio",
                    "Portfolio Value (INR)": "Portfolio Value (INR)",
                },
            )

            equity_fig.update_traces(
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Portfolio Value: %{customdata[0]}"
                    "<extra></extra>"
                ),
            )

            equity_fig.update_layout(
                xaxis_title="",
                yaxis_title="Portfolio Value (INR)",
                showlegend=False,
                height=450,
                hovermode="x",
                margin=dict(
                    t=80,
                    b=65,
                    l=65,
                    r=45,
                ),
            )

            equity_fig.update_yaxes(
                tickformat=",.0f",
                rangemode="tozero",
            )

            st.plotly_chart(
                equity_fig,
                use_container_width=True,
                key="final_equity_comparison_chart",
            )

            st.caption(
                "These values are from the historical backtest. "
                "They do not guarantee future investment returns."
            )

        else:
            st.info(
                "Initial capital or final portfolio values are "
                "unavailable in the backtest report."
            )

      


        # ------------------------------------------------------
        # D. TRADE RESULTS
        # ------------------------------------------------------

        st.subheader("Trade Results")

        winning_trades = get_numeric_metric("Winning Trades")
        losing_trades = get_numeric_metric("Losing Trades")

        if (
            winning_trades is not None
            and losing_trades is not None
        ):
            winning_trades = int(winning_trades)
            losing_trades = int(losing_trades)

            total_trades = winning_trades + losing_trades

            if total_trades > 0:
                win_rate = (
                    winning_trades / total_trades
                ) * 100

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Winning Trades",
                        f"{winning_trades:,}",
                    )

                with col2:
                    st.metric(
                        "Losing Trades",
                        f"{losing_trades:,}",
                    )

                with col3:
                    st.metric(
                        "Win Rate",
                        f"{win_rate:.2f}%",
                    )

                trade_chart = pd.DataFrame({
                    "Result": [
                        "Winning Trades",
                        "Losing Trades",
                    ],
                    "Number of Trades": [
                        winning_trades,
                        losing_trades,
                    ],
                })

                trade_chart["Percentage"] = (
                    trade_chart["Number of Trades"]
                    / total_trades
                ) * 100

                trade_fig = px.bar(
                    trade_chart,
                    x="Result",
                    y="Number of Trades",
                    color="Result",
                    text=trade_chart[
                        "Number of Trades"
                    ].map(lambda value: f"{value:,}"),
                    custom_data=["Percentage"],
                    title="Winning vs Losing Trades",
                    labels={
                        "Result": "Trade Outcome",
                        "Number of Trades": "Number of Trades",
                    },
                )

                trade_fig.update_traces(
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Number of Trades: %{y:,}<br>"
                        "Share of Trades: %{customdata[0]:.2f}%"
                        "<extra></extra>"
                    ),
                )

                trade_fig.update_layout(
                    showlegend=False,
                    height=420,
                    hovermode="x",
                    xaxis_title="",
                    yaxis_title="Number of Trades",
                    margin=dict(
                        t=75,
                        b=55,
                        l=55,
                        r=35,
                    ),
                )

                trade_fig.update_yaxes(
                    rangemode="tozero",
                )

                st.plotly_chart(
                    trade_fig,
                    use_container_width=True,
                    key="trade_results_chart",
                )

            else:
                st.info(
                    "No completed winning or losing trades "
                    "are available in the backtest summary."
                )

        else:
            st.info(
                "Winning or losing trade counts are unavailable "
                "in the backtest report."
            )

        with st.expander("View original backtest report"):
            st.dataframe(
                backtest_summary,
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            "These results use the saved historical backtest summary. "
            "Win rate is the percentage of winning trades among the "
            "reported winning and losing trades. It does not measure "
            "profitability by itself or guarantee future returns."
        )
       

else:
    st.warning(
        "Backtest summary not found. Run your backtest script "
        "to generate reports/backtest_summary.csv."
    )


# ==========================================================

# ==========================================================
# 13. ROBUSTNESS TESTING
# ==========================================================

st.divider()
st.header("Robustness Testing")

robustness_summary = load_report("robustness_summary.csv")

if (
    robustness_summary is not None
    and not robustness_summary.empty
):
    # Make a separate display copy so original data is unchanged.
    robustness_display = robustness_summary.copy()

    # Convert fractional values into percentages for display.
    percentage_columns = [
        "Average_Win_Rate",
        "Average_Strategy_Return",
        "Average_Benchmark_Return",
        "Average_Excess_Return",
        "Average_Max_Drawdown",
    ]

    for column in percentage_columns:
        if column in robustness_display.columns:
            robustness_display[column] = (
                robustness_display[column] * 100
            ).round(2)

    if "Average_Sharpe" in robustness_display.columns:
        robustness_display["Average_Sharpe"] = (
            robustness_display["Average_Sharpe"].round(3)
        )

    if "Average_Threshold" in robustness_display.columns:
        robustness_display["Average_Threshold"] = (
            robustness_display["Average_Threshold"].round(3)
        )

    if "Average_Trades" in robustness_display.columns:
        robustness_display["Average_Trades"] = (
            robustness_display["Average_Trades"].round(2)
        )

    st.caption(
        "Results averaged across validation rounds. "
        "Percentage columns are displayed in percent."
    )

    st.dataframe(
        robustness_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Average_Win_Rate": st.column_config.NumberColumn(
                "Avg Win Rate (%)",
                format="%.2f%%",
            ),
            "Average_Strategy_Return": st.column_config.NumberColumn(
                "Avg Strategy Return (%)",
                format="%.2f%%",
            ),
            "Average_Benchmark_Return": st.column_config.NumberColumn(
                "Avg Benchmark Return (%)",
                format="%.2f%%",
            ),
            "Average_Excess_Return": st.column_config.NumberColumn(
                "Avg Excess Return (%)",
                format="%.2f%%",
            ),
            "Average_Max_Drawdown": st.column_config.NumberColumn(
                "Avg Max Drawdown (%)",
                format="%.2f%%",
            ),
            "Average_Sharpe": st.column_config.NumberColumn(
                "Avg Sharpe",
                format="%.3f",
            ),
        },
    )

    # ------------------------------------------------------
    # STRATEGY RETURN VS BENCHMARK RETURN
    # ------------------------------------------------------

    required_columns = [
        "Strategy",
        "Cost",
        "Average_Strategy_Return",
        "Average_Benchmark_Return",
    ]

    if all(
        column in robustness_summary.columns
        for column in required_columns
    ):
        robustness_chart_data = robustness_summary.copy()

        robustness_chart_data["Strategy and Cost"] = (
            robustness_chart_data["Strategy"].astype(str)
            + " | Cost "
            + (
                robustness_chart_data["Cost"] * 100
            ).map(lambda value: f"{value:.1f}%")
        )

        robustness_chart_data["Strategy Return (%)"] = (
            robustness_chart_data["Average_Strategy_Return"] * 100
        )

        robustness_chart_data["Benchmark Return (%)"] = (
            robustness_chart_data["Average_Benchmark_Return"] * 100
        )

        robustness_chart_data = robustness_chart_data.melt(
            id_vars=["Strategy and Cost"],
            value_vars=[
                "Strategy Return (%)",
                "Benchmark Return (%)",
            ],
            var_name="Return Type",
            value_name="Return (%)",
        )

        import plotly.express as px

        robustness_fig = px.bar(
            robustness_chart_data,
            x="Strategy and Cost",
            y="Return (%)",
            color="Return Type",
            barmode="group",
            title="Average Strategy Return vs Benchmark",
            labels={
                "Strategy and Cost": "Strategy | Transaction Cost",
                "Return (%)": "Average Return (%)",
            },
        )

        robustness_fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
        )

        robustness_fig.update_layout(
            height=480,
            xaxis_title="",
            yaxis_title="Average Return (%)",
            margin=dict(t=75, b=100, l=55, r=35),
            legend_title="",
        )

        st.plotly_chart(
            robustness_fig,
            use_container_width=True,
            key="robustness_returns_chart",
        )

    st.caption(
        "Negative returns indicate losses over the measured period. "
        "Excess return compares strategy return with benchmark return; "
        "it does not by itself establish profitability. "
        "Past results do not guarantee future performance."
    )

else:
    st.warning(
        "Robustness summary not found. Run your robustness testing "
        "script to generate reports/robustness_summary.csv."
    )


# ==========================================================
# 14. BASELINE COMPARISON
# ==========================================================

st.divider()
st.header("Strategy Baseline Comparison")

baseline_summary = load_report("baseline_comparison_summary.csv")

if (
    baseline_summary is not None
    and not baseline_summary.empty
):
    st.dataframe(
        baseline_summary,
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info(
        "Baseline comparison report is not available yet."
    )


# ==========================================================
# 15. DISCLAIMER
# ==========================================================

st.divider()

st.warning(
    "**RESEARCH PROJECT — NOT FINANCIAL ADVICE.** "
    "This experimental model uses historical data and can make incorrect "
    "predictions. Historical accuracy does not guarantee future performance. "
    "Previous robustness testing did not establish consistent profitability. "
    "Real trading may involve transaction costs, slippage, liquidity limits, "
    "taxes and market risk. Do not use this dashboard alone to make "
    "investment decisions."
)

st.caption(
    "AI Financial Intelligence Engine | Experimental ML dashboard"
)
