"""
Train five classification models on the Mobile Price dataset and export
artifacts required for GitHub + Streamlit deployment (Assignment 2).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "mobile_price_dataset.csv"
MODEL_DIR = ROOT / "model"
METRICS_PATH = ROOT / "training_metrics.json"
TEST_EXPORT_PATH = ROOT / "test_data.csv"

TARGET_COLUMN = "price_range"
RANDOM_STATE = 42
TEST_FRACTION = 0.25

MODEL_SPECS: list[tuple[str, str, object]] = [
    (
        "logistic_regression",
        "Logistic Regression",
        LogisticRegression(max_iter=2500, random_state=RANDOM_STATE),
    ),
    (
        "decision_tree",
        "Decision Tree",
        DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=12),
    ),
    (
        "knn",
        "K-Nearest Neighbor",
        KNeighborsClassifier(n_neighbors=5),
    ),
    (
        "naive_bayes",
        "Naive Bayes",
        GaussianNB(),
    ),
    (
        "svm",
        "Support Vector Machine",
        SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    ),
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in {DATA_PATH.name}.")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != TARGET_COLUMN]


def build_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", MinMaxScaler()),
            ("classifier", estimator),
        ]
    )


def multiclass_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return float(
        roc_auc_score(
            y_true,
            y_proba,
            multi_class="ovr",
            average="weighted",
        )
    )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": multiclass_auc(y_true, y_proba),
        "precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    features = feature_columns(df)
    if len(features) < 12:
        raise ValueError("Assignment requires at least 12 input features.")
    if len(df) < 500:
        raise ValueError("Assignment requires at least 500 instances.")

    x = df[features]
    y = df[TARGET_COLUMN].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_FRACTION,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    test_frame = x_test.copy()
    test_frame[TARGET_COLUMN] = y_test.values
    test_frame.to_csv(TEST_EXPORT_PATH, index=False)

    metadata = {
        "dataset_file": DATA_PATH.name,
        "dataset_source": "Kaggle — Mobile Price Classification (train split)",
        "target_column": TARGET_COLUMN,
        "feature_columns": features,
        "class_labels": sorted(y.unique().tolist()),
        "n_total": int(len(df)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "models": {},
    }

    metrics_by_model: dict[str, dict] = {}

    for slug, display_name, estimator in MODEL_SPECS:
        pipeline = build_pipeline(clone(estimator))
        pipeline.fit(x_train, y_train)

        y_pred = pipeline.predict(x_test)
        y_proba = pipeline.predict_proba(x_test)

        result = evaluate(y_test.to_numpy(), y_pred, y_proba)
        result["display_name"] = display_name
        metrics_by_model[slug] = result

        joblib.dump(pipeline, MODEL_DIR / f"{slug}.joblib")
        metadata["models"][slug] = {
            "file": f"{slug}.joblib",
            "display_name": display_name,
        }

    with METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metrics_by_model, handle, indent=2)

    with (MODEL_DIR / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved labeled test set: {TEST_EXPORT_PATH.name} ({len(x_test)} rows)")
    print(f"Trained models written to: {MODEL_DIR}")
    print("\nHeld-out test metrics:")
    for slug, m in metrics_by_model.items():
        print(
            f"  {m['display_name']:26} "
            f"Acc={m['accuracy']:.4f}  AUC={m['auc']:.4f}  "
            f"F1={m['f1']:.4f}  MCC={m['mcc']:.4f}"
        )


if __name__ == "__main__":
    main()
