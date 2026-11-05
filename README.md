# Machine Learning — Assignment 2 (AIML)

Mobile Price Classification · Multi-Class Models · Streamlit Deployment

---

## a. Problem statement

The objective is to predict the price range of a mobile phone using hardware and connectivity attributes. This is a multi-class classification problem with four target classes (0–3).

The project implements five classification algorithms on one dataset, evaluates them with standard metrics, stores trained models in the repository, and exposes an interactive Streamlit application. The deployed app loads saved models and evaluates uploaded labeled test CSV files only; no retraining is performed in the app.

---

## b. Dataset description

| Item | Description |
|------|-------------|
| Dataset name | Mobile Price Classification |
| Repository | [Kaggle — Mobile Price Classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification) |
| File in GitHub repo | `mobile_price_dataset.csv` (training data; renamed from Kaggle `train.csv`) |
| Number of instances | 2,000 |
| Number of features | 20 (numeric): e.g. `battery_power`, `ram`, `px_height`, `px_width`, `mobile_wt` |
| Target variable | `price_range` — classes 0, 1, 2, 3 |
| Problem type | Multi-class classification |
| Test file for app | `test_data.csv` — 500 stratified hold-out rows (25%) with all features and target |

Assignment criteria: minimum 12 features and 500 instances — satisfied (20 features, 2,000 instances).

---

## c. Github Repository Link

| Resource | Link |
|----------|------|
| GitHub repository | [https://github.com/vedant237/AIML-ML-Assignment-2](https://github.com/vedant237/AIML-ML-Assignment-2) |
| Live Streamlit application | [https://aiml-ml-assignment-2-7zp9qjtttffmzswfxtrdzb.streamlit.app/](https://aiml-ml-assignment-2-7zp9qjtttffmzswfxtrdzb.streamlit.app/) |

---

## d. Models used

All models are trained on the same dataset (`mobile_price_dataset.csv`, 1,500 training rows after split). Evaluation metrics below are computed on the held-out test set (`test_data.csv`, 500 rows) using `train_models.py`.

Models implemented:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor Classifier  
4. Naive Bayes Classifier (Gaussian)  
5. Support Vector Machine  

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.9640 | 0.9984 | 0.9643 | 0.9640 | 0.9640 | 0.9521 |
| Decision Tree | 0.8260 | 0.8840 | 0.8262 | 0.8260 | 0.8256 | 0.7683 |
| K-Nearest Neighbor | 0.5320 | 0.7741 | 0.5474 | 0.5320 | 0.5382 | 0.3767 |
| Naive Bayes | 0.8060 | 0.9490 | 0.8062 | 0.8060 | 0.8059 | 0.7415 |
| Support Vector Machine | 0.8660 | 0.9795 | 0.8675 | 0.8660 | 0.8666 | 0.8214 |

Metrics: weighted precision, recall, and F1 where applicable; AUC computed with one-vs-rest (weighted). Source: `training_metrics.json`.

### Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Highest accuracy, AUC, F1, and MCC on the test split; strong separation of price tiers after feature scaling, especially driven by RAM and camera-related attributes. |
| Decision Tree | Moderate performance with more errors between adjacent classes 1 and 2; non-linear splits help but depth limits reduce overfitting. |
| K-Nearest Neighbor | Lowest accuracy among the five models; distance-based decisions degrade when feature scales and class boundaries overlap. |
| Naive Bayes | Reasonable performance (~81% accuracy) despite independence assumptions; stronger on extreme classes 0 and 3 than on middle tiers. |
| Support Vector Machine | Second-best overall; RBF kernel captures non-linear patterns with slightly more confusion on middle classes than logistic regression. |
| Overall, Winner for your dataset? | Logistic Regression |

---

## Appendix — Repository contents

```
AIML-ML-Assignment-2/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── mobile_price_dataset.csv
├── test_data.csv
├── training_metrics.json
├── .streamlit/config.toml
└── model/
    ├── metadata.json
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

## Appendix — Local execution (BITS Virtual Lab)

```powershell
cd C:\AIML-ML-Assignment-2
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```
