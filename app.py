"""
Mobile Price Classification — professional Streamlit dashboard (Assignment 2).
Paths resolve from this file's folder (C:\\AIML-ML-Assignment-2 when run locally).
"""

from __future__ import annotations

import json
from datetime import datetime
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
DEFAULT_TEST = ROOT / "test_data.csv"

MODELS: dict[str, str] = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbor": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

METRIC_LABELS = (
    "Accuracy",
    "AUC Score",
    "Precision",
    "Recall",
    "F1 Score",
    "MCC Score",
)

APP_THEME_STYLE = """
<style>
    .block-container { padding-top: 0.75rem !important; padding-bottom: 1.5rem !important; }
    header[data-testid="stHeader"] { background: rgba(0, 0, 0, 0); }
    div[data-testid="stToolbar"] { visibility: hidden; height: 0; position: fixed; }
    .stApp { background: linear-gradient(165deg, #eef2ff 0%, #f0f9ff 45%, #ecfeff 100%); }
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #dbeafe 0%, #e0e7ff 55%, #cffafe 100%);
        border-right: 3px solid #2563eb;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(145deg, #ffffff 0%, #f0f9ff 100%) !important;
        border: 2px solid #60a5fa !important;
        border-left: 6px solid #2563eb !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.12) !important;
    }
    div[data-testid="stMetric"] {
        border: 2px solid #93c5fd !important;
        border-top: 4px solid #2563eb !important;
        border-radius: 10px !important;
        padding: 0.55rem 0.8rem !important;
        background: linear-gradient(180deg, #ffffff 0%, #eff6ff 100%) !important;
    }
    .field-label {
        font-size: 1.05rem; font-weight: 700; color: #0f172a; margin: 0 0 0.35rem 0;
    }
    div[data-testid="stSelectbox"] label p,
    div[data-testid="stSelectbox"] label span {
        font-weight: 700 !important; color: #0f172a !important; font-size: 1.02rem !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        border: 2px solid #2563eb !important; border-radius: 8px !important;
    }
    .app-hero {
        background: linear-gradient(120deg, #1d4ed8 0%, #2563eb 40%, #0284c7 75%, #0891b2 100%);
        color: #f8fafc; padding: 1rem 1.35rem; border-radius: 12px;
        margin-bottom: 0.65rem !important; border: 2px solid #1e40af;
        box-shadow: 0 6px 18px rgba(29, 78, 216, 0.2);
    }
    .app-hero h1 {
        color: #ffffff !important; font-size: 1.55rem !important; font-weight: 700 !important;
        margin: 0 0 0.3rem 0 !important; padding: 0 !important;
    }
    .app-hero p { color: #e0f2fe; margin: 0; font-size: 0.94rem; line-height: 1.45; }
    .status-chip {
        display: inline-block; background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.35); border-radius: 999px;
        padding: 0.15rem 0.65rem; font-size: 0.78rem; margin-top: 0.45rem; margin-right: 0.35rem;
    }
    .classification-report-wrap {
        background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px;
        padding: 0.65rem 0.85rem; overflow-x: auto;
    }
    table.classification-report-table {
        width: 100%; border-collapse: collapse; font-family: "Segoe UI", sans-serif; font-size: 0.92rem;
    }
    table.classification-report-table th {
        background: #f8fafc; color: #334155; font-weight: 600; text-align: center;
        padding: 0.5rem 0.6rem; border-bottom: 2px solid #e2e8f0;
    }
    table.classification-report-table td {
        padding: 0.45rem 0.6rem; border-bottom: 1px solid #f1f5f9; text-align: center; color: #475569;
    }
    table.classification-report-table td.class-label {
        text-align: left; font-weight: 500; color: #0f172a;
    }
    table.classification-report-table td.report-number {
        font-weight: 600; font-size: 0.94rem; color: #0f172a; font-variant-numeric: tabular-nums;
    }
</style>
"""


@st.cache_resource
def load_metadata() -> dict:
    with (MODEL_DIR / "metadata.json").open(encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_resource
def load_model(file_name: str):
    return joblib.load(MODEL_DIR / file_name)


def validate_frame(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    target_column: str,
    class_labels: tuple[int, ...],
) -> tuple[bool, list[str], pd.DataFrame | None, pd.Series | None]:
    required = list(feature_columns) + [target_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return False, missing, None, None
    target = frame[target_column].astype(int)
    if not set(target.unique()).issubset(set(class_labels)):
        return False, [f"Target must use labels {list(class_labels)}."], None, None
    return True, [], frame[list(feature_columns)], target


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC Score": float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")),
        "Precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "F1 Score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "MCC Score": float(matthews_corrcoef(y_true, y_pred)),
    }


@st.cache_data(show_spinner="Evaluating All Models…")
def evaluate_all_models(
    feature_csv: bytes,
    target_values: tuple[int, ...],
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    features = pd.read_csv(BytesIO(feature_csv))
    y_true = np.array(target_values, dtype=int)
    rows: list[dict] = []
    for model_name, model_file in MODELS.items():
        model = load_model(model_file)
        preds = model.predict(features[list(feature_columns)])
        proba = model.predict_proba(features[list(feature_columns)])
        rows.append({"ML Model Name": model_name, **compute_metrics(y_true, preds, proba)})
    return pd.DataFrame(rows)


def build_classification_report_table(
    y_true: pd.Series, y_pred: np.ndarray, class_labels: list[int]
) -> pd.DataFrame:
    report = classification_report(
        y_true, y_pred, labels=class_labels, output_dict=True, zero_division=0
    )
    rows: list[dict] = []
    for label in class_labels:
        block = report[str(label)]
        rows.append(
            {
                "Class": f"Class {label}",
                "Precision": float(block["precision"]),
                "Recall": float(block["recall"]),
                "F1 Score": float(block["f1-score"]),
                "Support": float(block["support"]),
            }
        )
    if "accuracy" in report:
        rows.append(
            {
                "Class": "Accuracy",
                "Precision": np.nan,
                "Recall": np.nan,
                "F1 Score": float(report["accuracy"]),
                "Support": float(len(y_true)),
            }
        )
    for key, label in (("macro avg", "Macro Avg"), ("weighted avg", "Weighted Avg")):
        if key not in report:
            continue
        block = report[key]
        rows.append(
            {
                "Class": label,
                "Precision": float(block["precision"]),
                "Recall": float(block["recall"]),
                "F1 Score": float(block["f1-score"]),
                "Support": float(block["support"]),
            }
        )
    return pd.DataFrame(rows)


def format_report_cell(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{value:.3f}"


def render_classification_report_html(report_df: pd.DataFrame) -> str:
    header = "".join(f"<th>{col}</th>" for col in report_df.columns)
    body_rows = []
    for _, row in report_df.iterrows():
        cells = [f'<td class="class-label">{row["Class"]}</td>']
        for column in ("Precision", "Recall", "F1 Score", "Support"):
            cells.append(f'<td class="report-number">{format_report_cell(row[column])}</td>')
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"""
    <div class="classification-report-wrap">
        <table class="classification-report-table">
            <thead><tr>{header}</tr></thead>
            <tbody>{''.join(body_rows)}</tbody>
        </table>
    </div>
    """


def build_confusion_matrix_figure(matrix: np.ndarray, class_labels: list[int]) -> go.Figure:
    tick_labels = [f"Class {label}" for label in class_labels]
    axis_font = dict(size=16, color="#0f172a", family="Segoe UI, sans-serif")
    tick_font = dict(size=13, color="#0f172a", family="Segoe UI, sans-serif")
    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=tick_labels,
            y=tick_labels,
            text=[[str(int(v)) for v in row] for row in matrix],
            texttemplate="%{text}",
            textfont=dict(size=14, color="#ffffff", family="Segoe UI, sans-serif"),
            colorscale=[
                [0.0, "#7c3aed"],
                [0.45, "#6d28d9"],
                [0.75, "#5b21b6"],
                [1.0, "#4c1d95"],
            ],
            colorbar=dict(title=dict(text="Count", font=axis_font), tickfont=tick_font),
        )
    )
    figure.update_layout(
        template="plotly_white",
        title=dict(text="Confusion Matrix", font=dict(size=18, color="#0f172a")),
        height=440,
        margin=dict(l=78, r=28, t=64, b=76),
        xaxis=dict(
            title=dict(text="<b>Predicted Class</b>", font=axis_font, standoff=16),
            tickfont=tick_font,
        ),
        yaxis=dict(
            title=dict(text="<b>Actual Class</b>", font=axis_font, standoff=16),
            tickfont=tick_font,
            autorange="reversed",
        ),
    )
    return figure


def build_comparison_chart(comparison: pd.DataFrame) -> go.Figure:
    melted = comparison.melt(id_vars="ML Model Name", var_name="Metric", value_name="Score")
    figure = px.bar(
        melted,
        x="Metric",
        y="Score",
        color="ML Model Name",
        barmode="group",
        category_orders={"Metric": list(METRIC_LABELS)},
        title="All Models Comparison Chart",
        color_discrete_sequence=["#2563eb", "#0891b2", "#7c3aed", "#059669", "#ea580c"],
    )
    figure.update_layout(
        template="plotly_white",
        legend=dict(title="ML Model Name", orientation="h", y=-0.25),
        margin=dict(l=40, r=20, t=60, b=80),
    )
    figure.update_yaxes(range=[0, 1.05])
    return figure


def render_header() -> None:
    app_mtime = datetime.fromtimestamp((ROOT / "app.py").stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    st.markdown(
        f"""
        <div class="app-hero">
            <h1>Mobile Price Classification System</h1>
            <span class="status-chip">Five ML Models</span>
            <span class="status-chip">Inference Only</span>
            <span class="status-chip">Build {app_mtime}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Running from: `{ROOT}`")


def main() -> None:
    st.set_page_config(page_title="Mobile Price Classification System", layout="wide")
    st.markdown(APP_THEME_STYLE, unsafe_allow_html=True)

    meta = load_metadata()
    feature_columns = tuple(meta["feature_columns"])
    target_column = meta["target_column"]
    class_labels = tuple(meta["class_labels"])

    render_header()

    with st.sidebar:
        st.markdown("### Application Guide")
        st.markdown(
            "1. Upload **Test Dataset (CSV)**  \n"
            "2. Choose **Model Selection**  \n"
            "3. Review **Evaluation Metrics**  \n"
            "4. Inspect **Confusion Matrix** and **Classification Report**  \n"
            "5. Compare **All Models Performance**"
        )
        if DEFAULT_TEST.exists():
            st.download_button(
                "Download Sample Test Data",
                data=DEFAULT_TEST.read_bytes(),
                file_name="test_data.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with st.container(border=True):
        st.markdown("### Data Input And Model Selection")
        col_upload, col_model, col_status = st.columns([1.4, 1.1, 1])
        with col_upload:
            st.markdown('<p class="field-label">Upload Test Dataset (CSV)</p>', unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "Upload Test Dataset (CSV)",
                type=["csv"],
                label_visibility="collapsed",
            )
        with col_model:
            st.markdown('<p class="field-label">Model Selection</p>', unsafe_allow_html=True)
            selected_model = st.selectbox(
                "Model Selection",
                list(MODELS.keys()),
                label_visibility="visible",
            )
        with col_status:
            st.markdown('<p class="field-label">Inference Mode</p>', unsafe_allow_html=True)
            st.success("Pre-Trained Models Ready")

    if uploaded is not None:
        frame = pd.read_csv(uploaded)
        file_label = uploaded.name
    elif DEFAULT_TEST.exists():
        frame = pd.read_csv(DEFAULT_TEST)
        file_label = "test_data.csv (bundled)"
    else:
        st.warning("Add test_data.csv to the project folder or upload a labeled CSV.")
        return

    st.caption(f"Active file: **{file_label}** · Records: **{len(frame):,}**")

    ok, issues, features, target = validate_frame(frame, feature_columns, target_column, class_labels)
    if not ok:
        st.error("Uploaded Test Dataset Does Not Match The Required Schema.")
        st.write(", ".join(issues))
        return

    model = load_model(MODELS[selected_model])
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    selected_metrics = compute_metrics(target.to_numpy(), predictions, probabilities)
    matrix = confusion_matrix(target, predictions, labels=list(class_labels))
    all_models = evaluate_all_models(
        features.to_csv(index=False).encode("utf-8"),
        tuple(target.tolist()),
        feature_columns,
    )

    with st.container(border=True):
        st.markdown(f"### Evaluation Metrics — {selected_model}")
        cols = st.columns(6)
        for col, name in zip(cols, METRIC_LABELS):
            value = selected_metrics[name]
            col.metric(name, f"{value:.4f}" if name == "MCC Score" else f"{value:.2%}")

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("### Confusion Matrix")
            st.plotly_chart(build_confusion_matrix_figure(matrix, list(class_labels)), use_container_width=True)
    with right:
        with st.container(border=True):
            st.markdown("### Classification Report")
            report_df = build_classification_report_table(target, predictions, list(class_labels))
            st.markdown(render_classification_report_html(report_df), unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("### All Models Performance On Uploaded Test Data")
        st.dataframe(all_models.round(4), use_container_width=True, hide_index=True)
        st.plotly_chart(build_comparison_chart(all_models), use_container_width=True)


if __name__ == "__main__":
    main()
