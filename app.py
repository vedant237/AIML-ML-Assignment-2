"""
Assignment 2 — Streamlit evaluation app.
Upload labeled test CSV; pre-trained models predict and report metrics (no retraining).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

MODEL_CHOICES: dict[str, str] = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbor": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


@st.cache_resource
def load_metadata() -> dict:
    path = MODEL_DIR / "metadata.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_resource
def load_classifier(joblib_name: str):
    return joblib.load(MODEL_DIR / joblib_name)


def validate_upload(df: pd.DataFrame, meta: dict) -> tuple[pd.DataFrame, pd.Series] | None:
    target = meta["target_column"]
    features = meta["feature_columns"]
    required = features + [target]
    missing = [col for col in required if col not in df.columns]
    if missing:
        st.error("Missing columns in uploaded CSV: " + ", ".join(missing))
        return None

    extra = [c for c in df.columns if c not in required]
    if extra:
        st.warning("Ignoring extra columns: " + ", ".join(extra))

    x = df[features]
    y = df[target].astype(int)
    expected = set(meta["class_labels"])
    if not set(y.unique()).issubset(expected):
        st.error(f"Target must use class labels {sorted(expected)} only.")
        return None
    return x, y


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": float(
            roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="weighted",
            )
        ),
        "Precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "Recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def draw_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    figure, axis = plt.subplots(figsize=(6, 4.5))
    heatmap = sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=labels,
        yticklabels=labels,
        ax=axis,
        annot_kws={"size": 12, "weight": "bold", "color": "white"},
        linewidths=0.6,
        linecolor="#ffffff",
    )
    threshold = matrix.max() / 2.0 if matrix.max() > 0 else 0.5
    for text in heatmap.texts:
        value = int(text.get_text())
        text.set_color("white" if value >= threshold else "#1e1b4b")
    axis.set_xlabel("Predicted Class", fontsize=13, fontweight="bold", color="#0f172a")
    axis.set_ylabel("Actual Class", fontsize=13, fontweight="bold", color="#0f172a")
    axis.set_title("Confusion Matrix", fontsize=14, fontweight="bold", color="#0f172a")
    axis.tick_params(labelsize=11, colors="#0f172a")
    st.pyplot(figure)
    plt.close(figure)


def evaluate_model(name: str, model, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    preds = model.predict(x)
    proba = model.predict_proba(x)
    return score_predictions(y.to_numpy(), preds, proba)


def main() -> None:
    st.set_page_config(
        page_title="Mobile Price Classification | ML Assignment 2",
        page_icon="📱",
        layout="wide",
    )

    meta = load_metadata()

    st.title("Mobile Price Range Classification")
    st.caption(
        "BITS AIML — Assignment 2 | Pre-trained scikit-learn models | "
        "Upload **test_data.csv** (features + `price_range`). Models are **not** retrained here."
    )

    st.markdown(
        """
        <style>
            div[data-testid="stSidebar"] .model-selection-label {
                font-size: 1.05rem;
                font-weight: 700;
                color: #0f172a;
                margin: 0.5rem 0 0.35rem 0;
            }
            div[data-testid="stSidebar"] div[data-testid="stSelectbox"] label p {
                font-weight: 700 !important;
                color: #0f172a !important;
                font-size: 1rem !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Controls")
        st.markdown('<p class="model-selection-label">Model Selection</p>', unsafe_allow_html=True)
        selected = st.selectbox(
            "Model Selection",
            list(MODEL_CHOICES.keys()),
            label_visibility="collapsed",
        )
        uploaded = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])
        if DEFAULT_TEST.exists():
            st.download_button(
                label="Download Sample Test Data",
                data=DEFAULT_TEST.read_bytes(),
                file_name="test_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
        st.markdown(
            "**Required:** all 20 feature columns and `price_range` target.  \n"
            "Use the repository file `test_data.csv` for evaluation."
        )

    if uploaded is not None:
        data = pd.read_csv(uploaded)
        st.caption(f"Active file: **{uploaded.name}**")
    elif DEFAULT_TEST.exists():
        data = pd.read_csv(DEFAULT_TEST)
        st.caption("Active file: **test_data.csv** (bundled hold-out). Upload a CSV to replace.")
    else:
        st.info("Upload `test_data.csv` or add it to the repository to view metrics.")
        return
    st.subheader("Uploaded data preview")
    st.dataframe(data.head(10), use_container_width=True)

    parsed = validate_upload(data, meta)
    if parsed is None:
        return
    x, y = parsed

    model = load_classifier(MODEL_CHOICES[selected])
    y_pred = model.predict(x)
    y_proba = model.predict_proba(x)
    metrics = score_predictions(y.to_numpy(), y_pred, y_proba)

    st.subheader(f"Evaluation Metrics — {selected}")
    display_metrics = {
        "Accuracy": metrics["Accuracy"],
        "AUC Score": metrics["AUC"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1 Score": metrics["F1"],
        "MCC Score": metrics["MCC"],
    }
    metric_cols = st.columns(6)
    for column, (label, value) in zip(metric_cols, display_metrics.items()):
        column.metric(label, f"{value:.4f}")

    labels = meta["class_labels"]
    st.subheader("Confusion Matrix")
    draw_confusion_matrix(y.to_numpy(), y_pred, labels)

    st.subheader("Classification Report")
    st.code(
        classification_report(y, y_pred, labels=labels, zero_division=0),
        language="text",
    )

    st.subheader("All Models Performance On Uploaded Test Data")
    comparison = []
    for model_name, file_name in MODEL_CHOICES.items():
        clf = load_classifier(file_name)
        row = evaluate_model(model_name, clf, x, y)
        comparison.append(
            {
                "ML Model Name": model_name,
                "Accuracy": row["Accuracy"],
                "AUC Score": row["AUC"],
                "Precision": row["Precision"],
                "Recall": row["Recall"],
                "F1 Score": row["F1"],
                "MCC Score": row["MCC"],
            }
        )
    st.dataframe(pd.DataFrame(comparison).round(4), use_container_width=True)


if __name__ == "__main__":
    main()
