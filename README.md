# Machine Learning — Assignment 2 (AIML)

End-to-end **multi-class classification** project with five scikit-learn models, evaluation metrics, GitHub artifacts, and a Streamlit deployment app.

---

## a. Problem statement

Classify mobile phones into one of **four price ranges** (0 = low, 1 = medium, 2 = high, 3 = very high) using hardware and connectivity specifications. The workflow covers training on a labeled dataset, saving all classifiers, evaluating on a held-out **test_data.csv**, and serving predictions through Streamlit **without retraining** on user uploads.

---

## b. Dataset description

| Field | Value |
|--------|--------|
| **Name** | Mobile Price Classification |
| **Source** | [Kaggle — Mobile Price Classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification) (public repository) |
| **File in repo** | `mobile_price_dataset.csv` (renamed from the original `train.csv`) |
| **Instances** | 2,000 |
| **Features** | 20 numeric inputs (`battery_power`, `ram`, `px_height`, etc.) |
| **Target** | `price_range` — integer classes **0, 1, 2, 3** (multi-class classification) |
| **Test export** | `test_data.csv` — 500-row stratified hold-out (25%) with **features + target** for Streamlit upload |

**Note:** The separate Kaggle `test.csv` (with `id` only, no labels) is **not** used for metric evaluation because the assignment requires uploaded CSVs to include the target column.

**Requirement check:** ≥ 12 features ✓ | ≥ 500 instances ✓ | classification ✓

---

## c. Github Repository Link

`https://github.com/vedant237/AIML-ML-Assignment-2`

---

## d. Models used

All models trained on **1,500** rows; metrics below on **500** held-out rows (`test_data.csv`).  
Regenerate anytime: `python train_models.py` (run this on **BITS Virtual Lab** for your screenshot).

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.9640 | 0.9984 | 0.9643 | 0.9640 | 0.9640 | 0.9521 |
| Decision Tree | 0.8260 | 0.8840 | 0.8262 | 0.8260 | 0.8256 | 0.7683 |
| K-Nearest Neighbor | 0.5320 | 0.7741 | 0.5474 | 0.5320 | 0.5382 | 0.3767 |
| Naive Bayes | 0.8060 | 0.9490 | 0.8062 | 0.8060 | 0.8059 | 0.7415 |
| Support Vector Machine | 0.8660 | 0.9795 | 0.8675 | 0.8660 | 0.8666 | 0.8214 |

**Streamlit app:** `app.py` — professional evaluation dashboard (Plotly charts, bundled `test_data.csv`, CSV upload, five models, full metrics). Re-run `python train_models.py` and refresh `training_metrics.json` / this table if you retrain.

### Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Best overall on this split: highest accuracy, AUC, F1, and MCC; linear boundaries work well after scaling on separable price tiers driven by RAM and camera features. |
| Decision Tree | Solid mid-tier performance but more confusion between adjacent price bands (1 vs 2); non-linear splits help but limited depth controls overfitting. |
| K-Nearest Neighbor | Weakest accuracy; distance-based voting struggles when many scaled features contribute noise; adjacent classes overlap in feature space. |
| Naive Bayes | Gaussian NB assumes feature independence (not strictly true) yet achieves ~81% accuracy with strong performance on extreme classes 0 and 3. |
| Support Vector Machine | Strong second place with RBF kernel capturing non-linear structure; slightly more errors on middle classes than logistic regression. |
| **Overall, Winner for your dataset?** | **Logistic Regression** |

---

## Repository structure (submission checklist)

```
AIML-ML-Assignment-2/
├── app.py                      # Streamlit app (deploy this file)
├── train_models.py             # Training + export script
├── requirements.txt
├── README.md
├── mobile_price_dataset.csv    # Full labeled training data
├── test_data.csv               # Labeled hold-out for app upload
├── training_metrics.json       # Metrics for README table
└── model/
    ├── metadata.json
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

---

## Local / BITS Virtual Lab setup

```powershell
cd C:\AIML-ML-Assignment-2
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

In the browser: upload `test_data.csv` → pick a model → verify metrics, confusion matrix, and classification report.

---

## Streamlit Community Cloud deployment

1. Push the full repo (include `model/*.joblib` and CSV files).
2. Open [https://streamlit.io/cloud](https://streamlit.io/cloud) → sign in with GitHub.
3. **New app** → select this repository → branch `main` → main file **`app.py`** → **Deploy**.
4. Add the live URL to your submission PDF.

---

## PDF submission (order)

1. GitHub repository link  
2. Live Streamlit app link  
3. Screenshot of execution on **BITS Virtual Lab**  
4. This README (sections **a** through **d**)

**Deadline:** 8 November 2026, 23:59 — no draft submissions.
