"""
Smart Energy Analytics System
==============================
A production-quality Streamlit dashboard for AI-powered energy consumption
forecasting, analytics, alerting and reporting, built on top of a pre-trained
LSTM model.

Run with:
    streamlit run app.py
"""

import os
import io
import datetime
import warnings

import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# TensorFlow is imported lazily inside the model loader so the rest of the
# app can still run (with a clear warning) even if TF is not installed.
try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False


# =====================================================================
# CONSTANTS / CONFIG
# =====================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# The user's actual project folder — checked first, before any other location.
USER_PROJECT_DIR = r"C:\Users\abdul\Desktop\smart-energy-analytics"


def resolve_path(filename: str, subfolder: str) -> str:
    """
    Find `filename` by checking, in order:
      1. The known project folder on the user's machine (flat, no subfolders)
      2. Next to this script, inside `subfolder/` (e.g. models/final_energy_model.keras)
      3. Directly next to this script (e.g. final_energy_model.keras)
      4. Relative to the current working directory, inside `subfolder/`
      5. Relative to the current working directory, directly
    Falls back to option 1's path if nothing is found (used for error messages).
    """
    candidates = [
        os.path.join(USER_PROJECT_DIR, filename),
        os.path.join(BASE_DIR, subfolder, filename),
        os.path.join(BASE_DIR, filename),
        os.path.join(subfolder, filename),
        filename,
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


MODEL_PATH = resolve_path("final_energy_model.keras", "models")
SCALER_PATH = resolve_path("energy_scaler.pkl", "models")
FEATURES_PATH = resolve_path("selected_features.pkl", "models")
DATA_PATH = resolve_path("daily_energy_data.csv", "data")

SEQUENCE_LENGTH = 30  # LSTM look-back window (days)

CANDIDATE_DATE_COLUMNS = ["date", "Date", "DATE", "timestamp", "Timestamp"]
CANDIDATE_TARGET_COLUMNS = [
    "consumption", "Consumption", "energy_consumption", "Energy_Consumption",
    "energy", "Energy", "kwh", "kWh", "KWH", "value", "Value",
]

PRIMARY_COLOR = "#3B82F6"      # blue
PRIMARY_DARK = "#1D4ED8"
BG_DARK = "#0E1117"
CARD_BG = "#161B25"
CARD_BORDER = "#232A3B"
GREEN = "#22C55E"
ORANGE = "#F59E0B"
RED = "#EF4444"
TEXT_MUTED = "#9CA3AF"


# =====================================================================
# PAGE CONFIG + GLOBAL STYLE
# =====================================================================

st.set_page_config(
    page_title="Smart Energy Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_css() -> None:
    """Inject the dark + blue, Power BI-inspired theme."""
    st.markdown(
        f"""
        <style>
            html, body, [class*="css"] {{
                font-family: 'Segoe UI', 'Inter', 'Helvetica Neue', sans-serif;
            }}

            .stApp {{
                background-color: {BG_DARK};
                color: #E5E7EB;
            }}

            section[data-testid="stSidebar"] {{
                background-color: #0B0F17;
                border-right: 1px solid {CARD_BORDER};
            }}

            .sidebar-title {{
                font-size: 22px;
                font-weight: 800;
                color: white;
                padding: 6px 0 2px 0;
                letter-spacing: 0.3px;
            }}

            .sidebar-sub {{
                font-size: 12px;
                color: {TEXT_MUTED};
                padding-bottom: 18px;
                border-bottom: 1px solid {CARD_BORDER};
                margin-bottom: 14px;
            }}

            .kpi-card {{
                background: linear-gradient(145deg, {CARD_BG}, #10141d);
                border: 1px solid {CARD_BORDER};
                border-radius: 16px;
                padding: 18px 20px;
                box-shadow: 0 4px 14px rgba(0,0,0,0.25);
                height: 100%;
            }}

            .kpi-icon {{
                font-size: 26px;
                margin-bottom: 6px;
            }}

            .kpi-label {{
                font-size: 13px;
                color: {TEXT_MUTED};
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.4px;
            }}

            .kpi-value {{
                font-size: 26px;
                font-weight: 800;
                color: #FFFFFF;
                margin-top: 4px;
            }}

            .kpi-delta-up {{
                color: {GREEN};
                font-size: 13px;
                font-weight: 600;
            }}

            .kpi-delta-down {{
                color: {RED};
                font-size: 13px;
                font-weight: 600;
            }}

            .section-title {{
                font-size: 20px;
                font-weight: 800;
                color: white;
                margin: 6px 0 14px 0;
                border-left: 4px solid {PRIMARY_COLOR};
                padding-left: 10px;
            }}

            .info-card {{
                background-color: {CARD_BG};
                border: 1px solid {CARD_BORDER};
                border-radius: 14px;
                padding: 20px;
            }}

            .alert-card {{
                border-radius: 14px;
                padding: 16px 18px;
                margin-bottom: 12px;
                border-left: 6px solid;
                font-size: 14.5px;
            }}

            .alert-green {{
                background-color: rgba(34, 197, 94, 0.08);
                border-color: {GREEN};
                color: #D1FAE5;
            }}

            .alert-orange {{
                background-color: rgba(245, 158, 11, 0.10);
                border-color: {ORANGE};
                color: #FEF3C7;
            }}

            .alert-red {{
                background-color: rgba(239, 68, 68, 0.10);
                border-color: {RED};
                color: #FEE2E2;
            }}

            .stButton>button {{
                background-color: {PRIMARY_COLOR};
                color: white;
                border-radius: 10px;
                border: none;
                font-weight: 600;
                padding: 8px 18px;
            }}

            .stButton>button:hover {{
                background-color: {PRIMARY_DARK};
                color: white;
            }}

            div[data-testid="stMetricValue"] {{
                color: white;
            }}

            hr {{
                border-color: {CARD_BORDER};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# DATA / MODEL LOADING
# =====================================================================

@st.cache_data(show_spinner=False)
def load_energy_data() -> pd.DataFrame:
    """Load and lightly clean the historical daily energy dataset."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at '{DATA_PATH}'.")

    df = pd.read_csv(DATA_PATH)

    # Detect and standardize the date column
    date_col = next((c for c in CANDIDATE_DATE_COLUMNS if c in df.columns), None)
    if date_col is None:
        date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    return df


def detect_target_column(df: pd.DataFrame, selected_features) -> str:
    """Identify the energy consumption (target) column in the dataframe."""
    for c in CANDIDATE_TARGET_COLUMNS:
        if c in df.columns:
            return c

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns]

    if selected_features:
        remaining = [c for c in numeric_cols if c not in selected_features]
        if remaining:
            return remaining[0]

    if numeric_cols:
        return numeric_cols[-1]

    raise ValueError("Could not detect a numeric target consumption column.")


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the trained LSTM model, scaler and selected feature list."""
    errors = []
    model, scaler, selected_features = None, None, None

    if not TENSORFLOW_AVAILABLE:
        errors.append("TensorFlow is not installed — forecasting is disabled.")

    if TENSORFLOW_AVAILABLE:
        if os.path.exists(MODEL_PATH):
            try:
                model = load_model(MODEL_PATH)
            except Exception as e:
                errors.append(f"Failed to load model: {e}")
        else:
            errors.append(f"Model file not found at '{MODEL_PATH}'.")

    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            errors.append(f"Failed to load scaler: {e}")
    else:
        errors.append(f"Scaler file not found at '{SCALER_PATH}'.")

    if os.path.exists(FEATURES_PATH):
        try:
            selected_features = joblib.load(FEATURES_PATH)
            if isinstance(selected_features, (pd.Index, np.ndarray)):
                selected_features = list(selected_features)
        except Exception as e:
            errors.append(f"Failed to load selected features: {e}")
    else:
        errors.append(f"Selected features file not found at '{FEATURES_PATH}'.")

    return model, scaler, selected_features, errors


# =====================================================================
# FORECASTING LOGIC
# =====================================================================

def build_feature_frame(df: pd.DataFrame, target_col: str, selected_features):
    """Return the dataframe restricted to the model's expected feature columns."""
    if selected_features:
        missing = [f for f in selected_features if f not in df.columns]
        if missing:
            raise ValueError(
                f"The following expected features are missing from the data: {missing}"
            )
        feature_cols = list(selected_features)
    else:
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns]

    return df[feature_cols], feature_cols


def recursive_forecast(model, scaler, df, target_col, feature_cols, horizon: int):
    """
    Recursively forecast `horizon` future days using the trained LSTM model.

    The model is assumed to have been trained on sequences of length
    SEQUENCE_LENGTH over `feature_cols`, predicting the next-day target.
    """
    working_df = df.copy().reset_index(drop=True)
    feature_data = working_df[feature_cols].values.astype(float)

    if len(feature_data) < SEQUENCE_LENGTH:
        raise ValueError(
            f"Need at least {SEQUENCE_LENGTH} historical rows to forecast, "
            f"but only {len(feature_data)} are available."
        )

    scaled = scaler.transform(feature_data)
    target_idx = feature_cols.index(target_col) if target_col in feature_cols else 0

    window = scaled[-SEQUENCE_LENGTH:].copy()
    predictions = []

    for _ in range(horizon):
        model_input = window.reshape(1, SEQUENCE_LENGTH, len(feature_cols))
        pred_scaled = model.predict(model_input, verbose=0)
        pred_scaled_value = float(np.ravel(pred_scaled)[0])

        # Build the next feature row: carry forward the last known feature
        # values and overwrite the target column with the new prediction.
        next_row_scaled = window[-1].copy()
        next_row_scaled[target_idx] = pred_scaled_value
        window = np.vstack([window[1:], next_row_scaled])

        # Inverse-transform to get the prediction in the original scale.
        dummy_row = np.zeros((1, len(feature_cols)))
        dummy_row[0, target_idx] = pred_scaled_value
        inv_row = scaler.inverse_transform(dummy_row)
        predictions.append(float(inv_row[0, target_idx]))

    last_date = working_df["date"].max()
    future_dates = [last_date + datetime.timedelta(days=i + 1) for i in range(horizon)]

    result = pd.DataFrame({"date": future_dates, "predicted_consumption": predictions})
    return result


def estimate_confidence(historical_series: pd.Series) -> float:
    """A simple heuristic confidence score based on historical volatility."""
    if historical_series.empty or historical_series.mean() == 0:
        return 0.0
    cv = historical_series.std() / (abs(historical_series.mean()) + 1e-9)
    confidence = max(0.0, min(1.0, 1.0 - cv))
    return round(confidence * 100, 1)


# =====================================================================
# UI HELPERS
# =====================================================================

def kpi_card(col, icon: str, label: str, value: str, delta: str = None, positive: bool = True):
    delta_html = ""
    if delta is not None:
        css_class = "kpi-delta-up" if positive else "kpi-delta-down"
        arrow = "▲" if positive else "▼"
        delta_html = f'<div class="{css_class}">{arrow} {delta}</div>'

    col.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(text: str):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def alert_card(container, level: str, title: str, message: str):
    css_class = {"green": "alert-green", "orange": "alert-orange", "red": "alert-red"}[level]
    icon = {"green": "✅", "orange": "⚠", "red": "🚨"}[level]
    container.markdown(
        f"""
        <div class="alert-card {css_class}">
            <b>{icon} {title}</b><br>{message}
        </div>
        """,
        unsafe_allow_html=True,
    )


def styled_line_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    fig = px.line(df, x=x, y=y, title=title)
    fig.update_traces(line=dict(color=PRIMARY_COLOR, width=3))
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor=CARD_BG,
        paper_bgcolor=CARD_BG,
        font=dict(color="#E5E7EB"),
        title_font=dict(size=17, color="white"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# =====================================================================
# PAGES
# =====================================================================

def page_dashboard(df: pd.DataFrame, target_col: str, model, scaler, selected_features, load_errors):
    section_title("🏠 Dashboard Overview")

    series = df[target_col]
    current = series.iloc[-1]
    weekly_avg = series.tail(7).mean()
    monthly_avg = series.tail(30).mean()
    peak = series.max()
    lowest = series.min()

    predicted_tomorrow = None
    if model is not None and scaler is not None:
        try:
            feature_df, feature_cols = build_feature_frame(df, target_col, selected_features)
            combined = pd.concat([df[["date"]], feature_df], axis=1)
            forecast_df = recursive_forecast(model, scaler, combined, target_col, feature_cols, 1)
            predicted_tomorrow = forecast_df["predicted_consumption"].iloc[0]
        except Exception:
            predicted_tomorrow = None

    row1 = st.columns(3)
    kpi_card(row1[0], "⚡", "Current Consumption", f"{current:,.2f} kWh")
    kpi_card(
        row1[1], "🔮", "Predicted Tomorrow",
        f"{predicted_tomorrow:,.2f} kWh" if predicted_tomorrow is not None else "N/A",
    )
    kpi_card(row1[2], "📅", "Weekly Average", f"{weekly_avg:,.2f} kWh")

    row2 = st.columns(3)
    kpi_card(row2[0], "🗓", "Monthly Average", f"{monthly_avg:,.2f} kWh")
    kpi_card(row2[1], "🔺", "Peak Consumption", f"{peak:,.2f} kWh")
    kpi_card(row2[2], "🔻", "Lowest Consumption", f"{lowest:,.2f} kWh")

    st.write("")
    section_title("📈 Historical Consumption Trend")
    fig = styled_line_chart(df, "date", target_col, "Daily Energy Consumption Over Time")
    st.plotly_chart(fig, use_container_width=True)

    if predicted_tomorrow is not None:
        st.info(f"🔮 **Latest Prediction:** Tomorrow's estimated consumption is "
                f"**{predicted_tomorrow:,.2f} kWh**.")
    else:
        st.warning("Forecast unavailable — check the model/scaler files or the Forecast page for details.")

    if load_errors:
        with st.expander("⚠ Model loading notices"):
            for err in load_errors:
                st.write(f"- {err}")


def page_forecast(df: pd.DataFrame, target_col: str, model, scaler, selected_features, load_errors):
    section_title("📈 Energy Consumption Forecast")

    if model is None or scaler is None or not selected_features:
        st.error(
            "The forecasting model, scaler, or feature list could not be loaded. "
            "Please verify that all required files exist."
        )
        if load_errors:
            st.markdown("**Details:**")
            for err in load_errors:
                st.write(f"- {err}")
        else:
            st.write(f"- Model expected at: `{MODEL_PATH}`")
            st.write(f"- Scaler expected at: `{SCALER_PATH}`")
            st.write(f"- Features expected at: `{FEATURES_PATH}`")
        return

    horizon_map = {"Next 1 Day": 1, "Next 7 Days": 7, "Next 30 Days": 30}
    choice = st.radio("Select forecast horizon:", list(horizon_map.keys()), horizontal=True)
    horizon = horizon_map[choice]

    if st.button("🚀 Run Forecast"):
        with st.spinner("Running LSTM forecast..."):
            try:
                feature_df, feature_cols = build_feature_frame(df, target_col, selected_features)
                combined = pd.concat([df[["date"]], feature_df], axis=1)
                forecast_df = recursive_forecast(model, scaler, combined, target_col, feature_cols, horizon)
            except Exception as e:
                st.error(f"Forecast failed: {e}")
                return

        confidence = estimate_confidence(df[target_col].tail(60))

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("#### Prediction Table")
            display_df = forecast_df.copy()
            display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
            display_df["predicted_consumption"] = display_df["predicted_consumption"].round(2)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("#### Prediction Confidence")
            st.metric("Estimated Confidence", f"{confidence}%")
            st.progress(int(confidence))
            st.caption("Confidence is derived from recent historical volatility "
                       "and is indicative, not statistically guaranteed.")

        st.markdown("#### Forecast Chart")
        history_tail = df.tail(60)[["date", target_col]].rename(columns={target_col: "value"})
        history_tail["type"] = "Historical"
        future_plot = forecast_df.rename(columns={"predicted_consumption": "value"})
        future_plot["type"] = "Forecast"
        chart_df = pd.concat([history_tail, future_plot], ignore_index=True)

        fig = px.line(chart_df, x="date", y="value", color="type",
                      color_discrete_map={"Historical": "#60A5FA", "Forecast": "#F59E0B"},
                      title="Historical vs Forecasted Consumption")
        fig.update_traces(line=dict(width=3))
        fig.update_layout(
            template="plotly_dark", plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
            font=dict(color="#E5E7EB"), margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.session_state["last_forecast"] = forecast_df
    else:
        st.info("Choose a horizon and click **Run Forecast** to generate predictions.")


def page_analytics(df: pd.DataFrame, target_col: str):
    section_title("📊 Consumption Analytics")

    series = df[target_col]

    row1 = st.columns(4)
    kpi_card(row1[0], "📐", "Average Consumption", f"{series.mean():,.2f} kWh")
    kpi_card(row1[1], "🔺", "Maximum Consumption", f"{series.max():,.2f} kWh")
    kpi_card(row1[2], "🔻", "Minimum Consumption", f"{series.min():,.2f} kWh")
    kpi_card(row1[3], "📏", "Standard Deviation", f"{series.std():,.2f} kWh")

    st.write("")

    monthly = df.set_index("date")[target_col].resample("ME").mean().reset_index()
    weekly = df.set_index("date")[target_col].resample("W").mean().reset_index()

    c1, c2 = st.columns(2)
    with c1:
        fig_m = styled_line_chart(monthly, "date", target_col, "Monthly Trend")
        st.plotly_chart(fig_m, use_container_width=True)
    with c2:
        fig_w = styled_line_chart(weekly, "date", target_col, "Weekly Trend")
        st.plotly_chart(fig_w, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig_hist = px.histogram(df, x=target_col, nbins=30, title="Consumption Distribution")
        fig_hist.update_traces(marker_color=PRIMARY_COLOR)
        fig_hist.update_layout(template="plotly_dark", plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
                                font=dict(color="#E5E7EB"), margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)

    with c4:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] >= 2:
            corr = numeric_df.corr()
            fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="Blues",
                                  title="Correlation Heatmap")
            fig_corr.update_layout(template="plotly_dark", plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
                                    font=dict(color="#E5E7EB"), margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Not enough numeric columns to compute a correlation heatmap.")

    st.write("")
    section_title("Rolling Statistics")
    window = st.slider("Rolling window (days)", min_value=3, max_value=30, value=7)
    roll_df = df[["date", target_col]].copy()
    roll_df["rolling_avg"] = roll_df[target_col].rolling(window).mean()
    roll_df["rolling_std"] = roll_df[target_col].rolling(window).std()

    c5, c6 = st.columns(2)
    with c5:
        fig_ra = styled_line_chart(roll_df, "date", "rolling_avg", f"{window}-Day Rolling Average")
        st.plotly_chart(fig_ra, use_container_width=True)
    with c6:
        fig_rs = styled_line_chart(roll_df, "date", "rolling_std", f"{window}-Day Rolling Std Dev")
        fig_rs.update_traces(line=dict(color=ORANGE, width=3))
        st.plotly_chart(fig_rs, use_container_width=True)


def page_alerts(df: pd.DataFrame, target_col: str):
    section_title("⚠ Automated Alerts")

    series = df[target_col]
    monthly_avg = series.tail(30).mean()
    overall_std = series.std()
    latest = series.iloc[-1]
    previous = series.iloc[-2] if len(series) > 1 else latest

    high_threshold = monthly_avg + overall_std
    low_threshold = monthly_avg - overall_std
    spike_threshold = overall_std * 1.5

    alerts_generated = []

    if latest > high_threshold:
        alerts_generated.append(("red", "High Consumption Alert",
                                  f"Latest consumption ({latest:,.2f} kWh) is significantly above "
                                  f"the high threshold of {high_threshold:,.2f} kWh."))

    if latest < low_threshold:
        alerts_generated.append(("orange", "Low Consumption Alert",
                                  f"Latest consumption ({latest:,.2f} kWh) is below the low threshold "
                                  f"of {low_threshold:,.2f} kWh."))

    change = latest - previous
    if change > spike_threshold:
        alerts_generated.append(("red", "Sudden Spike Detected",
                                  f"Consumption jumped by {change:,.2f} kWh compared to the previous day."))

    if change < -spike_threshold:
        alerts_generated.append(("orange", "Sudden Drop Detected",
                                  f"Consumption dropped by {abs(change):,.2f} kWh compared to the previous day."))

    if latest > monthly_avg:
        alerts_generated.append(("orange", "Above Monthly Average",
                                  f"Latest consumption is {latest - monthly_avg:,.2f} kWh above "
                                  f"the 30-day average of {monthly_avg:,.2f} kWh."))
    else:
        alerts_generated.append(("green", "Below Monthly Average",
                                  f"Latest consumption is {monthly_avg - latest:,.2f} kWh below "
                                  f"the 30-day average of {monthly_avg:,.2f} kWh."))

    if not alerts_generated:
        alert_card(st, "green", "All Systems Normal", "No unusual consumption patterns detected.")
    else:
        for level, title, msg in alerts_generated:
            alert_card(st, level, title, msg)

    st.write("")
    section_title("Alert Thresholds")
    t1, t2, t3 = st.columns(3)
    kpi_card(t1, "🟢", "Low Threshold", f"{low_threshold:,.2f} kWh")
    kpi_card(t2, "🟠", "Monthly Average", f"{monthly_avg:,.2f} kWh")
    kpi_card(t3, "🔴", "High Threshold", f"{high_threshold:,.2f} kWh")


def page_reports(df: pd.DataFrame, target_col: str):
    section_title("📄 Reports & Downloads")

    st.markdown("#### Summary Statistics")
    summary = df[target_col].describe().to_frame(name=target_col)
    st.dataframe(summary, use_container_width=True)

    csv_summary = summary.to_csv().encode("utf-8")
    csv_full = df.to_csv(index=False).encode("utf-8")

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Raw Data", index=False)
        summary.to_excel(writer, sheet_name="Summary Statistics")
        if "last_forecast" in st.session_state:
            st.session_state["last_forecast"].to_excel(writer, sheet_name="Predictions", index=False)
    excel_buffer.seek(0)

    st.write("")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.download_button("⬇ Full Data (CSV)", data=csv_full,
                            file_name="energy_data.csv", mime="text/csv")
    with col2:
        st.download_button("⬇ Excel Report", data=excel_buffer,
                            file_name="energy_report.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with col3:
        st.download_button("⬇ Summary (CSV)", data=csv_summary,
                            file_name="summary_statistics.csv", mime="text/csv")
    with col4:
        if "last_forecast" in st.session_state:
            pred_csv = st.session_state["last_forecast"].to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Predictions (CSV)", data=pred_csv,
                                file_name="predictions.csv", mime="text/csv")
        else:
            st.button("⬇ Predictions (CSV)", disabled=True, help="Run a forecast first on the Forecast page.")


def page_about():
    section_title("ℹ About This Project")

    st.markdown(
        f"""
        <div class="info-card">
            <h3 style="color:white;">⚡ Smart Energy Analytics</h3>
            <p style="color:{TEXT_MUTED};">
            Smart Energy Analytics is an AI-powered dashboard that monitors, analyzes,
            and forecasts organizational energy consumption. It combines a trained
            LSTM deep-learning model with interactive visual analytics to help
            facilities and operations teams anticipate demand, detect anomalies,
            and make data-driven energy decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    section_title("Technology Stack")
    tech = [
        ("🧠", "TensorFlow"), ("🔁", "LSTM"), ("🐍", "Python"),
        ("🎈", "Streamlit"), ("📊", "Plotly"), ("🤖", "Scikit-learn"),
    ]
    cols = st.columns(3)
    for i, (icon, name) in enumerate(tech):
        kpi_card(cols[i % 3], icon, "Technology", name)

    st.write("")
    st.caption("© Smart Energy Analytics — AI-Powered Energy Forecasting Platform")


# =====================================================================
# MAIN APP
# =====================================================================

def main():
    inject_custom_css()

    with st.sidebar:
        st.markdown('<div class="sidebar-title">⚡ Smart Energy Analytics</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-sub">AI-Powered Energy Forecasting Platform</div>', unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "📈 Forecast", "📊 Analytics", "⚠ Alerts", "📄 Reports", "ℹ About"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.caption(f"Data source: `{DATA_PATH}`")
        st.caption(f"Model: `{MODEL_PATH}`")

    # Load data
    try:
        df = load_energy_data()
    except Exception as e:
        st.error(f"❌ Could not load energy data: {e}")
        st.info("Please make sure `data/daily_energy_data.csv` exists relative to this app.")
        st.stop()

    # Load model artifacts
    model, scaler, selected_features, load_errors = load_artifacts()

    try:
        target_col = detect_target_column(df, selected_features)
    except Exception as e:
        st.error(f"❌ Could not detect the energy consumption column: {e}")
        st.stop()

    if page == "🏠 Dashboard":
        page_dashboard(df, target_col, model, scaler, selected_features, load_errors)
    elif page == "📈 Forecast":
        page_forecast(df, target_col, model, scaler, selected_features, load_errors)
    elif page == "📊 Analytics":
        page_analytics(df, target_col)
    elif page == "⚠ Alerts":
        page_alerts(df, target_col)
    elif page == "📄 Reports":
        page_reports(df, target_col)
    elif page == "ℹ About":
        page_about()


if __name__ == "__main__":
    main()