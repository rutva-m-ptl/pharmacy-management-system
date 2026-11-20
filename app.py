import os
import json
import random
from datetime import datetime
from collections import deque

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix

st.set_page_config(
    page_title="Clinical Telemetry & Interactive Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for responsive alert banners
st.markdown("""
<style>
    .alert-banner-high {
        background-color: #fdecea;
        border-left: 6px solid #d93025;
        padding: 14px 18px;
        border-radius: 6px;
        color: #5f2120;
        margin-bottom: 16px;
        font-size: 15px;
    }
    .alert-banner-normal {
        background-color: #e6f4ea;
        border-left: 6px solid #1e8e3e;
        padding: 14px 18px;
        border-radius: 6px;
        color: #137333;
        margin-bottom: 16px;
        font-size: 15px;
    }
</style>
""", unsafe_allow_html=True)

NUM_FEATURES = ['age_norm', 'troponin', 'creatinine', 'systolic_bp', 'glucose']
CAT_FEATURES = ['admission_type']
FEATURE_COLS = NUM_FEATURES + CAT_FEATURES

# --- 1. Load or Auto-Train Model & Holdout Test Set ---
@st.cache_resource
def load_model_and_test_data():
    os.makedirs("ui_artifacts", exist_ok=True)
    model_path = "ui_artifacts/clinical_model.joblib"
    test_data_path = "ui_artifacts/test_cohort.csv"

    if not os.path.exists(model_path) or not os.path.exists(test_data_path):
        np.random.seed(42)
        X_num, y = make_classification(
            n_samples=2000, n_features=5, n_informative=4, n_redundant=1, random_state=42
        )
        df = pd.DataFrame(X_num, columns=NUM_FEATURES)
        df['admission_type'] = np.random.choice(['Emergency', 'Elective', 'Urgent'], size=2000, p=[0.5, 0.3, 0.2])
        df['patient_id'] = [f"TEST-{2000 + i}" for i in range(2000)]
        df['actual_outcome'] = y

        train_df, test_df = train_test_split(df, test_size=0.25, random_state=42, stratify=y)

        preprocessor = ColumnTransformer([
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), NUM_FEATURES),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CAT_FEATURES)
        ])

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', HistGradientBoostingClassifier(max_iter=120, learning_rate=0.05, random_state=42))
        ])
        pipeline.fit(train_df[FEATURE_COLS], train_df['actual_outcome'])

        # Precompute test set probabilities for preview & audit
        test_df = test_df.copy()
        test_df['predicted_risk'] = np.round(pipeline.predict_proba(test_df[FEATURE_COLS])[:, 1], 4)

        joblib.dump(pipeline, model_path)
        test_df.to_csv(test_data_path, index=False)

    return joblib.load(model_path), pd.read_csv(test_data_path)

model, df_test_cohort = load_model_and_test_data()

# --- 2. Stream Generators ---
def simulated_telemetry_stream():
    patient_counter = 1000
    while True:
        patient_counter += 1
        is_acute = random.random() < 0.28
        yield {
            "event_time": datetime.now().strftime("%H:%M:%S"),
            "patient_id": f"PT-{patient_counter}",
            "age_norm": round(random.gauss(0.2, 1.0), 3),
            "troponin": round(random.gauss(1.6 if is_acute else -0.3, 0.7), 3),
            "creatinine": round(random.gauss(0.8 if is_acute else -0.2, 0.8), 3),
            "systolic_bp": round(random.gauss(0.5, 1.0), 3),
            "glucose": round(random.gauss(0.4, 0.9), 3),
            "admission_type": random.choice(["Emergency", "Urgent", "Elective"])
        }

def kafka_telemetry_stream(bootstrap_servers="localhost:9092", topic="clinical-vitals"):
    from kafka import KafkaConsumer
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=True,
        consumer_timeout_ms=1000
    )
    for message in consumer:
        event = message.value
        event.setdefault("event_time", datetime.now().strftime("%H:%M:%S"))
        yield event

def websocket_telemetry_stream(ws_url="ws://localhost:8765/vitals"):
    from websockets.sync.client import connect
    with connect(ws_url) as websocket:
        while True:
            raw_msg = websocket.recv()
            event = json.loads(raw_msg)
            event.setdefault("event_time", datetime.now().strftime("%H:%M:%S"))
            yield event

# --- 3. Session State & Sidebar Controls ---
if "event_buffer" not in st.session_state:
    st.session_state.event_buffer = deque(maxlen=120)
if "streaming" not in st.session_state:
    st.session_state.streaming = False
if "stream_gen" not in st.session_state:
    st.session_state.stream_gen = None

st.sidebar.header("Telemetry Stream Controls")
source_type = st.sidebar.selectbox(
    "Ingestion Source",
    ["Simulated Bedside Monitor", "Apache Kafka", "WebSocket Feed"]
)

if source_type == "Apache Kafka":
    kafka_broker = st.sidebar.text_input("Bootstrap Servers", "localhost:9092")
    kafka_topic = st.sidebar.text_input("Kafka Topic", "clinical-vitals")
elif source_type == "WebSocket Feed":
    ws_endpoint = st.sidebar.text_input("WebSocket URI", "ws://localhost:8765/vitals")

risk_threshold = st.sidebar.slider("Clinical Escalation Threshold", 0.10, 0.90, 0.50, 0.05)
refresh_interval = st.sidebar.slider("Polling Interval (sec)", 0.5, 3.0, 1.0, 0.5)

st.sidebar.divider()
st.sidebar.subheader("Interactive Cohort Filters")
selected_admissions = st.sidebar.multiselect(
    "Filter Admission Types",
    options=["Emergency", "Urgent", "Elective"],
    default=["Emergency", "Urgent", "Elective"]
)

col_btn1, col_btn2 = st.sidebar.columns(2)
if col_btn1.button("Start Stream", type="primary", use_container_width=True):
    st.session_state.streaming = True
    if source_type == "Apache Kafka":
        st.session_state.stream_gen = kafka_telemetry_stream(kafka_broker, kafka_topic)
    elif source_type == "WebSocket Feed":
        st.session_state.stream_gen = websocket_telemetry_stream(ws_endpoint)
    else:
        st.session_state.stream_gen = simulated_telemetry_stream()

if col_btn2.button("Pause Stream", use_container_width=True):
    st.session_state.streaming = False

if st.sidebar.button("Clear Window Buffer", use_container_width=True):
    st.session_state.event_buffer.clear()

# --- 4. Main Interactive Dashboard ---
st.title("Clinical Risk, Telemetry & Validation Analytics Dashboard")
st.caption("Interactive Plotly & Altair/Vega-Lite visualizations over live patient streams and holdout validation data.")

# Top-level Navigation Tabs (Testing Preview works even before starting the live stream)
tab_live, tab_explore, tab_table, tab_test = st.tabs([
    "1. Live Telemetry & Multi-Chart View (Plotly)",
    "2. Biomarker Cross-Filtering & Correlations (Altair)",
    "3. Live Scored Event Log",
    "4. Testing Dataset Preview & Model Audit"
])

@st.fragment(run_every=refresh_interval if st.session_state.streaming else None)
def render_live_tabs():
    if st.session_state.streaming and st.session_state.stream_gen is not None:
        try:
            event = next(st.session_state.stream_gen)
            input_df = pd.DataFrame([{col: event[col] for col in FEATURE_COLS}])
            prob = float(model.predict_proba(input_df)[0, 1])

            event["risk_score"] = round(prob, 4)
            event["status"] = "High Risk" if prob >= risk_threshold else "Normal"
            st.session_state.event_buffer.append(event)
        except StopIteration:
            st.warning("Stream ended.")
            st.session_state.streaming = False
        except Exception as e:
            st.error(f"Stream connection interrupted: {e}")
            st.session_state.streaming = False

    if not st.session_state.event_buffer:
        with tab_live:
            st.info("Click **Start Stream** in the sidebar to begin ingesting live patient events (or explore **Tab 4: Testing Dataset Preview** immediately).")
        with tab_explore:
            st.info("Start the live stream in the sidebar to populate interactive Altair cross-filtering charts.")
        with tab_table:
            st.info("No live events buffered yet. Click **Start Stream** in the sidebar.")
        return

    df_raw = pd.DataFrame(list(st.session_state.event_buffer))
    df_live = df_raw[df_raw["admission_type"].isin(selected_admissions)].copy()

    if df_live.empty:
        with tab_live:
            st.warning("No events match the selected Admission Type filters.")
        return

    latest = df_live.iloc[-1]
    is_latest_high = latest["risk_score"] >= risk_threshold

    # --- TAB 1: PLOTLY MULTI-CHART TELEMETRY ---
    with tab_live:
        if is_latest_high:
            st.markdown(
                f"<div class='alert-banner-high'><b>IMMEDIATE TRIAGE ESCALATION:</b> Patient <b>{latest['patient_id']}</b> "
                f"({latest['admission_type']}) scored <b>{latest['risk_score']*100:.1f}%</b> risk at {latest['event_time']} "
                f"(Troponin: {latest['troponin']}, Creatinine: {latest['creatinine']}).</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='alert-banner-normal'><b>LATEST ADMISSION STABLE:</b> Patient <b>{latest['patient_id']}</b> "
                f"({latest['admission_type']}) scored <b>{latest['risk_score']*100:.1f}%</b> risk at {latest['event_time']}.</div>",
                unsafe_allow_html=True
            )

        high_risk_df = df_live[df_live["risk_score"] >= risk_threshold]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Filtered Window Events", len(df_live))
        k2.metric("High-Risk Escalations", len(high_risk_df), delta=f"{len(high_risk_df)/len(df_live):.1%}", delta_color="inverse")
        k3.metric("Mean Cohort Risk", f"{df_live['risk_score'].mean():.1%}")
        k4.metric("Peak Troponin (Window)", f"{df_live['troponin'].max():.2f}")

        # Row 1: Gauge + Real-Time Trajectory
        p_col1, p_col2 = st.columns([1, 2])
        with p_col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=latest["risk_score"] * 100,
                delta={"reference": risk_threshold * 100, "increasing": {"color": "#d93025"}, "decreasing": {"color": "#1e8e3e"}},
                number={"suffix": "%", "font": {"size": 32}},
                title={"text": f"Latest Risk: {latest['patient_id']}", "font": {"size": 15}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#d93025" if is_latest_high else "#1e8e3e"},
                    "steps": [
                        {"range": [0, risk_threshold * 100], "color": "#eef7f0"},
                        {"range": [risk_threshold * 100, 100], "color": "#fce8e6"}
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.8, "value": risk_threshold * 100}
                }
            ))
            fig_gauge.update_layout(height=290, margin=dict(l=20, r=20, t=45, b=15))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with p_col2:
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=df_live["event_time"],
                y=df_live["risk_score"] * 100,
                mode="lines+markers",
                name="Risk Score (%)",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=9, color=np.where(df_live["risk_score"] >= risk_threshold, "#d93025", "#1e8e3e")),
                customdata=df_live[["patient_id", "admission_type", "troponin", "creatinine"]],
                hovertemplate=(
                    "<b>Patient:</b> %{customdata[0]} (%{customdata[1]})<br>"
                    "<b>Time:</b> %{x}<br>"
                    "<b>Risk Score:</b> %{y:.1f}%<br>"
                    "<b>Troponin:</b> %{customdata[2]} | <b>Creatinine:</b> %{customdata[3]}<extra></extra>"
                )
            ))
            fig_ts.add_hline(y=risk_threshold * 100, line_dash="dash", line_color="#d93025", annotation_text=f"Threshold ({risk_threshold*100:.0f}%)")
            fig_ts.update_layout(
                title="Real-Time Risk Trajectory (Hover points for vitals)",
                yaxis_title="Risk Probability (%)",
                xaxis_title="Event Timestamp",
                yaxis=dict(range=[0, 105]),
                height=290,
                margin=dict(l=20, r=20, t=45, b=15)
            )
            st.plotly_chart(fig_ts, use_container_width=True)

        # Row 2: Radar Chart (Patient vs Cohort Profile) + Violin Risk Distribution
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            latest_vals = [float(latest[f]) for f in NUM_FEATURES]
            cohort_means = [float(df_live[f].mean()) for f in NUM_FEATURES]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=latest_vals + [latest_vals[0]],
                theta=NUM_FEATURES + [NUM_FEATURES[0]],
                fill='toself',
                name=f"Latest ({latest['patient_id']})",
                line_color="#d93025" if is_latest_high else "#1f77b4"
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=cohort_means + [cohort_means[0]],
                theta=NUM_FEATURES + [NUM_FEATURES[0]],
                fill='toself',
                name="Window Cohort Mean",
                line_color="#7f7f7f",
                opacity=0.6
            ))
            fig_radar.update_layout(
                title="Biomarker Radar: Latest Patient vs. Cohort Mean",
                polar=dict(radialaxis=dict(visible=True, range=[-2.5, 3.5])),
                height=320,
                margin=dict(l=40, r=40, t=50, b=25)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with r_col2:
            fig_violin = px.violin(
                df_live,
                x="admission_type",
                y="risk_score",
                color="admission_type",
                box=True,
                points="all",
                title="Risk Score Distribution by Admission Type",
                labels={"admission_type": "Admission Type", "risk_score": "Predicted Risk"}
            )
            fig_violin.add_hline(y=risk_threshold, line_dash="dash", line_color="black")
            fig_violin.update_layout(height=320, showlegend=False, margin=dict(l=20, r=20, t=50, b=25))
            st.plotly_chart(fig_violin, use_container_width=True)

    # --- TAB 2: ALTAIR LINKED BRUSHING & CORRELATION HEATMAP ---
    with tab_explore:
        st.caption("Drag a selection box on the **Scatter Plot** (left) to filter the **Bar Chart** (right). Below is the live **Biomarker Correlation Matrix**.")
        brush = alt.selection_interval()

        scatter = alt.Chart(df_live).mark_circle(size=85, opacity=0.85).encode(
            x=alt.X("troponin:Q", title="Troponin (Z-Score)"),
            y=alt.Y("creatinine:Q", title="Creatinine (Z-Score)"),
            color=alt.condition(
                brush,
                alt.Color("status:N", scale=alt.Scale(domain=["High Risk", "Normal"], range=["#d93025", "#1e8e3e"]), title="Triage Status"),
                alt.value("lightgray")
            ),
            tooltip=["patient_id", "admission_type", "troponin", "creatinine", "risk_score", "status"]
        ).add_params(brush).properties(width=420, height=300, title="Troponin vs. Creatinine (Drag to Brush-Select)")

        bars = alt.Chart(df_live).mark_bar().encode(
            x=alt.X("count():Q", title="Selected Patient Count"),
            y=alt.Y("admission_type:N", title="Admission Type"),
            color=alt.Color("status:N", scale=alt.Scale(domain=["High Risk", "Normal"], range=["#d93025", "#1e8e3e"]))
        ).transform_filter(brush).properties(width=350, height=300, title="Cohort Breakdown for Brushed Selection")

        st.altair_chart(scatter | bars, use_container_width=True)

        # Correlation Heatmap across Numeric Features + Risk Score
        if len(df_live) >= 3:
            corr_cols = NUM_FEATURES + ["risk_score"]
            corr_df = df_live[corr_cols].corr().reset_index().melt(id_vars="index")
            corr_df.columns = ["Feature 1", "Feature 2", "Correlation"]

            heatmap = alt.Chart(corr_df).mark_rect().encode(
                x=alt.X("Feature 1:N", title=None),
                y=alt.Y("Feature 2:N", title=None),
                color=alt.Color("Correlation:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])),
                tooltip=["Feature 1", "Feature 2", alt.Tooltip("Correlation:Q", format=".2f")]
            ).properties(width=550, height=280, title="Live Rolling Feature & Risk Correlation Heatmap")

            text = heatmap.mark_text(baseline="middle").encode(
                text=alt.Text("Correlation:Q", format=".2f"),
                color=alt.condition(alt.datum.Correlation > 0.5, alt.value("white"), alt.value("black"))
            )
            st.altair_chart(heatmap + text, use_container_width=True)

    # --- TAB 3: LIVE SCORED AUDIT LOG ---
    with tab_table:
        display_df = df_live.iloc[::-1][
            ["event_time", "patient_id", "admission_type", "troponin", "creatinine", "systolic_bp", "risk_score", "status"]
        ]
        st.dataframe(
            display_df,
            use_container_width=True,
            height=380,
            column_config={
                "risk_score": st.column_config.ProgressColumn(
                    "Risk Probability",
                    help="Model predicted probability of adverse event",
                    format="%.2f",
                    min_value=0.0,
                    max_value=1.0,
                ),
                "troponin": st.column_config.NumberColumn("Troponin", format="%.2f"),
                "creatinine": st.column_config.NumberColumn("Creatinine", format="%.2f"),
            }
        )

render_live_tabs()

# --- TAB 4: TESTING DATASET PREVIEW & HOLDOUT EVALUATION ---
with tab_test:
    st.subheader("Holdout Testing Dataset Preview & Model Performance Audit")
    st.caption(f"Inspecting {len(df_test_cohort)} unseen validation patients (`ui_artifacts/test_cohort.csv`) evaluated against active threshold **{risk_threshold:.2f}**.")

    # Filter controls for Testing Dataset
    t_col1, t_col2, t_col3 = st.columns([1, 1, 2])
    with t_col1:
        test_adm_filter = st.multiselect(
            "Test Admission Type",
            options=["Emergency", "Urgent", "Elective"],
            default=["Emergency", "Urgent", "Elective"],
            key="test_adm"
        )
    with t_col2:
        outcome_filter = st.selectbox(
            "Ground Truth Outcome",
            options=["All Patients", "Positive Cases (1)", "Negative Cases (0)"]
        )
    with t_col3:
        preview_rows = st.slider("Preview Row Count", 10, 200, 25, step=5)

    df_test_filtered = df_test_cohort[df_test_cohort["admission_type"].isin(test_adm_filter)].copy()
    if outcome_filter == "Positive Cases (1)":
        df_test_filtered = df_test_filtered[df_test_filtered["actual_outcome"] == 1]
    elif outcome_filter == "Negative Cases (0)":
        df_test_filtered = df_test_filtered[df_test_filtered["actual_outcome"] == 0]

    df_test_filtered["predicted_class"] = (df_test_filtered["predicted_risk"] >= risk_threshold).astype(int)
    df_test_filtered["audit_result"] = np.where(
        df_test_filtered["predicted_class"] == df_test_filtered["actual_outcome"],
        "Correct",
        "Misclassified"
    )

    # Dynamic Holdout Metrics at current threshold
    if len(df_test_filtered) > 0 and df_test_filtered["actual_outcome"].nunique() > 1:
        y_true_t = df_test_filtered["actual_outcome"]
        y_prob_t = df_test_filtered["predicted_risk"]
        y_pred_t = df_test_filtered["predicted_class"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test ROC-AUC", f"{roc_auc_score(y_true_t, y_prob_t):.4f}")
        m2.metric("Test Accuracy", f"{accuracy_score(y_true_t, y_pred_t):.1%}")
        m3.metric("Test Precision", f"{precision_score(y_true_t, y_pred_t, zero_division=0):.1%}")
        m4.metric("Test Recall", f"{recall_score(y_true_t, y_pred_t, zero_division=0):.1%}")

    # Side-by-side Test Distribution Histogram & Confusion Matrix
    tc1, tc2 = st.columns(2)
    with tc1:
        fig_hist = px.histogram(
            df_test_filtered,
            x="predicted_risk",
            color=df_test_filtered["actual_outcome"].map({1: "Actual Positive (1)", 0: "Actual Negative (0)"}),
            nbins=25,
            barmode="overlay",
            opacity=0.75,
            title="Test Set Predicted Risk Separation by Actual Outcome",
            labels={"predicted_risk": "Predicted Risk Score", "color": "Ground Truth"}
        )
        fig_hist.add_vline(x=risk_threshold, line_dash="dash", line_color="black")
        fig_hist.update_layout(height=300, margin=dict(l=20, r=20, t=45, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)

    with tc2:
        cm = confusion_matrix(df_test_filtered["actual_outcome"], df_test_filtered["predicted_class"], labels=[0, 1])
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=["Predicted Low (0)", "Predicted High (1)"],
            y=["Actual Low (0)", "Actual High (1)"],
            color_continuous_scale="Blues",
            title=f"Holdout Confusion Matrix (Threshold = {risk_threshold:.2f})"
        )
        fig_cm.update_layout(height=300, margin=dict(l=20, r=20, t=45, b=20))
        st.plotly_chart(fig_cm, use_container_width=True)

    # Interactive Test Dataset Preview Table + CSV Download
    st.markdown(f"**Showing top {min(preview_rows, len(df_test_filtered))} of {len(df_test_filtered)} filtered test records:**")
    ordered_cols = [
        "patient_id", "admission_type", "age_norm", "troponin", "creatinine",
        "systolic_bp", "glucose", "actual_outcome", "predicted_risk", "predicted_class", "audit_result"
    ]
    st.dataframe(
        df_test_filtered[ordered_cols].head(preview_rows),
        use_container_width=True,
        height=320,
        column_config={
            "predicted_risk": st.column_config.ProgressColumn(
                "Predicted Risk", format="%.2f", min_value=0.0, max_value=1.0
            )
        }
    )

    st.download_button(
        label="Download Full Scored Test Dataset (CSV)",
        data=df_test_filtered[ordered_cols].to_csv(index=False).encode("utf-8"),
        file_name="scored_holdout_test_cohort.csv",
        mime="text/csv"
    )