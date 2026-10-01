# AML Risk Detection

A two-layer machine learning pipeline for Anti-Money Laundering (AML) monitoring. Layer 1 flags suspicious transactions, and Layer 2 assigns each client a risk tier (Low / Medium / High). An interactive HTML dashboard shows the results.

## Dataset
[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1): a synthetic mobile-money transaction dataset. The file is about 470 MB, so it is not included here. To run the notebook, download it from Kaggle and place it in the same folder as the notebook. The notebook reads it as `paysim.csv.csv`, so rename the file or edit that line.

This project uses a random sample of 100,000 transactions (`random_state=42`), of which 141 are fraud (0.14%).

## Approach

**Preprocessing and features**
- Encoded transaction type and removed duplicates
- Engineered features: balance-mismatch errors for sender and receiver, "account drained to zero" flag, high-risk transaction type flag (TRANSFER / CASH_OUT), and a large-transaction flag (top 5% by amount)

**Layer 1: Transaction monitoring**
- Target: `isFraud`
- Stratified 80/20 train/test split (28 fraud cases in the test set)
- 5 models compared: Logistic Regression, Decision Tree, Random Forest, KNN, XGBoost
- Class imbalance handled with `class_weight='balanced'` and XGBoost `scale_pos_weight`
- Features scaled for Logistic Regression and KNN

**Layer 2: Client risk classification**
- Transactions aggregated per client (count, amounts, behaviour ratios)
- Risk tiers: High if the client has any fraud, Medium if the behaviour score is in the top 20% of the rest, otherwise Low
- Stratified 75/25 split; 4 models compared (Logistic Regression, Decision Tree, Random Forest, KNN)
- Feature importance chart for explainability

## Results

**Layer 1 (transaction level)**

| Model | Precision | Recall | F1 | AUROC |
|---|---|---|---|---|
| Random Forest | 1.00 | 1.00 | 1.00 | 1.000 |
| XGBoost | 1.00 | 1.00 | 1.00 | 1.000 |
| Decision Tree | 0.97 | 1.00 | 0.98 | 1.000 |
| Logistic Regression | 0.06 | 1.00 | 0.11 | 0.999 |
| KNN | 0.92 | 0.79 | 0.85 | 0.893 |

**Layer 2 (client level, macro F1):** Decision Tree 1.00, Random Forest 1.00, KNN 0.92, Logistic Regression 0.69.

## Limitations (please read)
- **Likely feature leakage in Layer 1.** The near-perfect scores probably come from balance-based features, which strongly reveal fraud in PaySim. These results would not carry over to real bank data.
- **Layer 2 labels are rule-based.** The risk tiers are built from the same behaviour features the models are trained on, so the high scores show that the models learned the rule, not that they found new patterns.
- **Almost every client has one transaction** in the 100k sample (99,999 unique clients), which limits what client-level aggregation can show.
- **Very few fraud cases** (141 total, 28 in the test set), so metrics are unstable.
- PaySim is synthetic, so this project demonstrates a workflow, not production performance.

## Files
| File | Description |
|---|---|
| `AML_Client_Risk_and_Transaction_Monitoring.ipynb` | Main notebook (full pipeline) |
| `AML_Risk_Dashboard.html` | Interactive dashboard (download and open in a browser) |
| `build_dashboard.py`, `dashboard_template.html` | Code that generates the dashboard |
| `txn_model_comparison.csv`, `client_model_comparison.csv` | Model results |
| `transaction_sample.csv`, `client_risk_table.csv` | Sample outputs used by the dashboard |

## Tools
Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn, Jupyter


