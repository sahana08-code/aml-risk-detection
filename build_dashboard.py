"""
Run this AFTER running the notebook (AML_Client_Risk_and_Transaction_Monitoring.ipynb)
on your real paysim.csv.csv file. The notebook's last cells save:
  - client_risk_table.csv
  - transaction_sample.csv
  - txn_model_comparison.csv
  - client_model_comparison.csv
  - paysim.csv.csv (your real file, already in the folder)

This script reads those and rebuilds AML_Risk_Dashboard.html with YOUR real numbers.

Usage:
    python3 build_dashboard.py

Requires: dashboard_template.html in the same folder (the HTML/CSS/JS shell —
don't edit it, this script only injects fresh data into it).
"""

import pandas as pd
import json
import os

REQUIRED_FILES = [
    "client_risk_table.csv",
    "transaction_sample.csv",
    "txn_model_comparison.csv",
    "client_model_comparison.csv",
    "paysim.csv.csv",
    "dashboard_template.html",
]

missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
if missing:
    raise FileNotFoundError(
        f"Missing required file(s): {missing}\n"
        f"Run the notebook fully first (it saves the CSVs), and make sure "
        f"dashboard_template.html and your real paysim.csv.csv are in this same folder."
    )

client_df = pd.read_csv("client_risk_table.csv")
txn_comp = pd.read_csv("txn_model_comparison.csv")
client_comp = pd.read_csv("client_model_comparison.csv")

# Recompute transaction-type fraud rates from the same 100k sample the notebook used
# (random_state=42 matches the notebook, so this is consistent with your trained models)
df = pd.read_csv("paysim.csv.csv")
if len(df) > 100_000:
    df = df.sample(n=100_000, random_state=42).reset_index(drop=True)

fraud_by_type = df.groupby('type')['isFraud'].mean().round(5).to_dict()
type_counts = df['type'].value_counts().to_dict()
tier_counts = client_df['risk_tier'].value_counts().to_dict()

# Demo client table: all High risk + a sample of Medium/Low (cap ~300 rows so the
# dashboard stays fast to load)
high = client_df[client_df['risk_tier'] == 'High']
n_medium = min(120, (client_df['risk_tier'] == 'Medium').sum())
n_low = min(150, (client_df['risk_tier'] == 'Low').sum())
medium = client_df[client_df['risk_tier'] == 'Medium'].sample(n=n_medium, random_state=1) if n_medium > 0 else client_df.iloc[0:0]
low = client_df[client_df['risk_tier'] == 'Low'].sample(n=n_low, random_state=1) if n_low > 0 else client_df.iloc[0:0]
demo_clients = pd.concat([high, medium, low]).reset_index(drop=True)

demo_cols = ['nameOrig', 'txn_count', 'total_amount', 'avg_amount', 'max_amount',
             'high_risk_type_ratio', 'large_txn_ratio', 'became_zero_ratio',
             'risk_tier', 'predicted_tier']
demo_clients = demo_clients[[c for c in demo_cols if c in demo_clients.columns]].round(2)

dashboard_data = {
    "tier_counts": tier_counts,
    "fraud_by_type": fraud_by_type,
    "type_counts": type_counts,
    "txn_model_comparison": txn_comp.round(4).to_dict(orient='records'),
    "client_model_comparison": client_comp.round(4).to_dict(orient='records'),
    "clients": demo_clients.to_dict(orient='records'),
    "total_clients": int(len(client_df)),
    "total_transactions": int(len(df)),
    "overall_fraud_rate": round(float(df['isFraud'].mean()) * 100, 4)
}

data_json = json.dumps(dashboard_data, separators=(',', ':'))

with open("dashboard_template.html") as f:
    html = f.read()

if "__DATA_JSON__" not in html:
    raise ValueError("dashboard_template.html doesn't contain the __DATA_JSON__ placeholder -- "
                      "make sure you're using the original template, not an already-built dashboard.")

html = html.replace("__DATA_JSON__", data_json)

with open("AML_Risk_Dashboard.html", "w") as f:
    f.write(html)

print("Dashboard rebuilt: AML_Risk_Dashboard.html")
print("Tier counts:", tier_counts)
print("Total clients:", len(client_df), " | Total transactions:", len(df))
print("Overall fraud rate: {:.4f}%".format(dashboard_data["overall_fraud_rate"]))
