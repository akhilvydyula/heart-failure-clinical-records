from __future__ import annotations

import json

import joblib
import pandas as pd
import streamlit as st

from ingest import RAW_CSV_PATH, run_pipeline
from train_models import (
    BEST_MODEL_PATH,
    CLUSTER_REPORT_PATH,
    FEATURE_COLUMNS,
    FEATURE_IMPORTANCE_PATH,
    MODEL_REPORT_PATH,
    main as train_models_main,
)


st.set_page_config(
    page_title="Heart Failure Risk Dashboard",
    layout="wide",
)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    if not RAW_CSV_PATH.exists():
        run_pipeline()
    return pd.read_csv(RAW_CSV_PATH)


@st.cache_resource
def load_model():
    if not BEST_MODEL_PATH.exists():
        train_models_main()
    return joblib.load(BEST_MODEL_PATH)


@st.cache_data
def load_json_report(path: str) -> dict:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def binary_select(label: str, help_text: str | None = None) -> int:
    return int(
        st.selectbox(
            label,
            options=[0, 1],
            format_func=lambda value: "Yes" if value else "No",
            help=help_text,
        )
    )


def patient_input_form(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Patient Inputs")
    st.sidebar.caption("This dashboard is for teaching only, not clinical decision-making.")

    values = {
        "age": st.sidebar.slider("Age", 0.0, 120.0, float(df["age"].median()), 1.0),
        "anaemia": binary_select("Anaemia"),
        "creatinine_phosphokinase": st.sidebar.number_input(
            "Creatinine phosphokinase (mcg/L)",
            min_value=0,
            value=int(df["creatinine_phosphokinase"].median()),
            step=10,
        ),
        "diabetes": binary_select("Diabetes"),
        "ejection_fraction": st.sidebar.slider(
            "Ejection fraction (%)",
            0,
            100,
            int(df["ejection_fraction"].median()),
        ),
        "high_blood_pressure": binary_select("High blood pressure"),
        "platelets": st.sidebar.number_input(
            "Platelets (kiloplatelets/mL)",
            min_value=1.0,
            value=float(df["platelets"].median()),
            step=1000.0,
        ),
        "serum_creatinine": st.sidebar.number_input(
            "Serum creatinine (mg/dL)",
            min_value=0.1,
            value=float(df["serum_creatinine"].median()),
            step=0.1,
        ),
        "serum_sodium": st.sidebar.number_input(
            "Serum sodium (mEq/L)",
            min_value=1,
            value=int(df["serum_sodium"].median()),
            step=1,
        ),
        "sex": int(
            st.sidebar.selectbox(
                "Sex",
                options=[0, 1],
                format_func=lambda value: "Woman" if value == 0 else "Man",
            )
        ),
        "smoking": binary_select("Smoking"),
        "time": st.sidebar.slider("Follow-up time (days)", 0, int(df["time"].max()), int(df["time"].median())),
    }
    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


def risk_band(probability: float) -> str:
    if probability >= 0.6:
        return "High"
    if probability >= 0.3:
        return "Moderate"
    return "Low"


def show_prediction(model, patient_df: pd.DataFrame) -> None:
    probability = float(model.predict_proba(patient_df)[0, 1])
    prediction = int(probability >= 0.5)

    metric_col, band_col, class_col = st.columns(3)
    metric_col.metric("Predicted Death Event Probability", f"{probability:.1%}")
    band_col.metric("Risk Band", risk_band(probability))
    class_col.metric("Predicted Class", "Death event" if prediction else "No death event")

    st.progress(min(max(probability, 0.0), 1.0))
    st.caption("Prediction threshold shown here is 0.50. See model metrics for tuned threshold details.")


def show_project_reports() -> None:
    st.subheader("Model and Dataset Reports")
    metrics_col, importance_col = st.columns(2)

    if MODEL_REPORT_PATH.exists():
        metrics = load_json_report(str(MODEL_REPORT_PATH))
        selected_model = metrics["selected_model"]
        selected = metrics[selected_model]
        metrics_col.write(f"Selected model: `{selected_model}`")
        metrics_col.metric("ROC AUC", f"{selected['roc_auc']:.3f}")
        metrics_col.metric("F1 Score", f"{selected['f1']:.3f}")
        metrics_col.metric("Best F1 Threshold", f"{selected['best_threshold_for_f1']:.3f}")
    else:
        metrics_col.info("Run `invoke train` to generate model metrics.")

    if FEATURE_IMPORTANCE_PATH.exists():
        importance = pd.read_csv(FEATURE_IMPORTANCE_PATH).head(10)
        importance_col.write("Top feature importance")
        importance_col.bar_chart(importance.set_index("feature")["importance_mean"])
    else:
        importance_col.info("Run `invoke train` to generate feature importance.")

    if CLUSTER_REPORT_PATH.exists():
        st.write("Patient cluster summary")
        st.dataframe(pd.read_csv(CLUSTER_REPORT_PATH), use_container_width=True)


def main() -> None:
    st.title("Heart Failure Clinical Records Dashboard")
    st.write(
        "Interactive Streamlit app for the UCI Heart Failure Clinical Records teaching project."
    )

    df = load_dataset()
    model = load_model()
    patient_df = patient_input_form(df)

    show_prediction(model, patient_df)

    with st.expander("Current Patient Input", expanded=False):
        st.dataframe(patient_df, use_container_width=True)

    with st.expander("Dataset Preview", expanded=False):
        st.dataframe(df.head(25), use_container_width=True)

    show_project_reports()


if __name__ == "__main__":
    main()
