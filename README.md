# Credit Card Fraud Detection

Data science portfolio project: a binary classification pipeline for detecting fraudulent credit card transactions on an extremely imbalanced dataset (~0.17% fraud).

**Author:** Sebastian Tapia

*(This README is also available in Spanish: [README.es.md](README.es.md))*

---

> **Note:** my professional background is in banks and insurance companies, environments that handle highly confidential information that cannot be shared or used outside those contexts. For that reason, this portfolio project is built entirely on a public dataset (Kaggle), with no data or information coming from my professional activity.

---

## 🗣️ Executive summary (in plain terms)

Imagine working at a bank where thousands of credit card purchases go through the system every day. The vast majority are normal purchases, but every so often one is fraud — someone using stolen card details. The catch is that fraud is extremely rare: fewer than 2 in every 1,000 purchases are fraudulent, which makes it like finding a needle in a haystack.

This project builds a system that looks at a transaction's data and decides: "this looks suspicious, flag it for review" or "this looks normal, approve it." The process followed these steps:

1. **Look at the data first** — compare what normal vs. fraudulent purchases look like (amounts, time of day) before building anything.
2. **Teach the system to recognize something unusual** — since there are almost no fraud examples to learn from, techniques were used so the system pays extra attention to those rare cases instead of overlooking them.
3. **Compare several methods and keep the one that actually performs best**, based on evidence rather than preference.
4. **Fine-tune the winning method's details** to squeeze out a bit more performance.
5. **Decide the cutoff point based on real cost** — the cost of reviewing a normal purchase by mistake isn't the same as the cost of letting a real fraud slip through, so the point at which the system decides to "flag for review" is chosen to minimize that cost for the business.
6. **Explain the decisions** — the system doesn't just say "fraud"; it also shows which factors influenced each prediction the most, so it's trustworthy rather than a black box.

The result is a system that catches **9 out of 10 real frauds**, while flagging less than 0.31% of normal purchases for unnecessary manual review — a reasonable trade-off, given that the cost of missing a real fraud is far higher than the cost of reviewing one extra legitimate purchase.

---

## 📌 Project overview (technical)

The goal is to build a model that identifies fraudulent transactions as accurately as possible, prioritizing **recall** (catching real fraud) without generating an unmanageable volume of false positives (legitimate transactions flagged for manual review).

The project covers the full cycle: data exploration, class imbalance handling, rigorous model comparison via cross-validation, cost-based threshold selection, interpretability with SHAP, and a production-scoring example.

## 📊 Dataset

- **Source:** [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) (Kaggle)
- **Rows:** 284,807 transactions
- **Features:** 30 numerical features (`V1`–`V28` are PCA components already applied by the original authors for confidentiality, plus `Time` and `Amount`)
- **Target:** `Class` (1 = fraud, 0 = legitimate)
- **Imbalance:** only ~0.17% of transactions are fraud — the core challenge of the project

## 🛠️ Methodology

1. **Exploratory data analysis**: class distribution, descriptive statistics, `Amount` distribution by class (log scale), and transaction distribution by hour of day per class.
2. **Preprocessing**: stratified 80/20 train/test split to preserve the fraud ratio in both sets.
3. **Handling class imbalance**: systematic comparison of strategies — `SMOTE` (synthetic oversampling), `class_weight`/`scale_pos_weight` (loss-function weighting), and combinations of both.
4. **Model selection via cross-validation (5-fold, stratified)**: Logistic Regression, Random Forest, and XGBoost were compared, each under different balancing strategies, using **PR-AUC** (more informative than ROC-AUC on highly imbalanced classes) as the selection metric. The final model is whichever has the best average CV PR-AUC, with no algorithm favored in advance.
5. **Hyperparameter tuning**: `RandomizedSearchCV` (20 iterations, same 5-fold CV) on the ablation's winning configuration, tuning `n_estimators`, `max_depth`, and other algorithm-specific parameters.
6. **Overfitting diagnostics**: comparing train vs. test metrics to detect generalization gaps.
7. **Cost-based threshold selection**: instead of using the default 0.5 threshold, the cutoff point is optimized to minimize an estimated cost (cost of reviewing a false positive vs. cost of a missed fraud).
8. **Interpretability**: feature importance via `feature_importances_` and SHAP values to explain the model's predictions.
9. **Scoring simulation**: a `score_transaction()` function that runs the full pipeline on a new transaction and returns a decision (`APPROVE` / `REVIEW`), also available as a standalone script (`predict_en.py`).

## 📈 Results

| Metric | Value |
|---|---|
| Final model | XGBoost (with SMOTE) |
| PR-AUC (CV, before tuning) | 0.8520 |
| PR-AUC (CV, after tuning) | 0.8566 |
| Best hyperparameters | `n_estimators=400`, `max_depth=5`, `learning_rate=0.2`, `subsample=0.6`, `colsample_bytree=0.8` |
| ROC-AUC (test) | 0.9834 |
| PR-AUC (test) | 0.8721 |
| Optimal threshold | 0.05 (based on false-positive cost = 5, false-negative cost = 100) |
| Precision / Recall at optimal threshold (fraud class) | 0.50 / 0.90 |

**Most influential features** (per SHAP): `V14`, `V4`, `V1`, `V3`, `V8`, and `Time` account for the largest share of the model's predictive impact.

**Business interpretation**: at the optimal threshold (0.05), out of 56,962 test transactions, **175 are flagged for manual review** (~0.31% of the total), of which 88 are real fraud and 87 are false alarms. This means **~90% of real fraud is caught** (88 of 98 cases), missing only 10 fraud cases, in exchange for reviewing a very small volume of legitimate transactions.

> **Note on overfitting**: the final model reaches a perfect PR-AUC of 1.0000 on the training set vs. 0.8721 on test — a gap indicating the model is memorizing some training noise. The test result is still solid, but a natural next step would be to reduce this gap (lower `max_depth`, stronger regularization) and check whether test PR-AUC improves.

## 🧰 Tech stack

- **Python** — pandas, NumPy
- **Visualization** — Matplotlib, Seaborn
- **Modeling** — scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **Interpretability** — SHAP
- **Persistence** — joblib

## 📁 Project structure

This repo ships both a Spanish and an English version of the notebook and scoring script, sharing the same methodology and results:

```
├── fraud_detection.ipynb     # Main notebook (English): EDA, model comparison, tuning, interpretability
├── predict_en.py             # Standalone scoring script (English)
├── Deteccion_fraude.ipynb     # Main notebook (Spanish version)
├── predict.py                 # Standalone scoring script (Spanish version)
├── creditcard.csv            # Dataset (not included in the repo due to size — download from Kaggle)
├── fraud_model.pkl           # Final trained, serialized model (generated when running fraud_detection.ipynb)
├── .gitignore                 # Excludes dataset, serialized models, checkpoints, and virtual envs
├── README.md                  # This file (English)
└── README.es.md               # Spanish version
```

## ▶️ How to run it

1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the project root.
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost shap joblib
   ```
3. Open and run `fraud_detection.ipynb` from top to bottom. This generates `fraud_model.pkl`.
4. To score new transactions without opening the notebook:
   ```bash
   python predict_en.py new_transactions.csv
   ```
   This generates `scoring_results.csv` with the fraud probability and decision (`APPROVE`/`REVIEW`) for each row. The script can also be imported as a module (`from predict_en import score_transaction, load_model`) to integrate into another system, such as an API.

## 🔍 Design decisions and limitations

- **PR-AUC** was used instead of accuracy as the main metric, because with ~0.17% fraud a model that always predicts "no fraud" would score 99.8% accuracy while being useless.
- The decision threshold is set using an illustrative business cost (`costo_fp` / `costo_fn` in the notebook); in a real-world case, these values should be calibrated with actual operational data (manual review cost, average loss per undetected fraud, etc.).
- The dataset already ships with `V1`–`V28` transformed via PCA, which limits direct business analysis of those variables (we don't know what "V14" represents in real terms) — only `Time` and `Amount` are directly interpretable.
- Hyperparameter tuning uses `RandomizedSearchCV` (20 iterations) rather than an exhaustive search, for computational cost reasons — a finer search (Optuna, Bayesian optimization) is a natural extension.
- The final model shows a perfect train fit (PR-AUC = 1.0000) vs. 0.8721 on test — an overfitting signal that, while not invalidating the test result, leaves room for improvement via stronger regularization (lower `max_depth`, higher `reg_lambda`/`reg_alpha`, or higher `min_child_weight`).

## 🚀 Possible extensions

- Finer hyperparameter search with Optuna (Bayesian optimization instead of random sampling).
- Deploying `predict_en.py` as an API (FastAPI/Flask) for real-time scoring.
- Data drift monitoring if the model were used in production.
