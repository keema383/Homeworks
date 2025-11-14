# CLV_Calculation.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import final model and processed data from your previous script
from Parametric_Models import (
    final_model_sig,
    test_sig,
    duration_col,
    event_col
)

# ==========================================================
# 0. PARAMETERS FOR CLV (ADJUST TO MATCH YOUR SLIDES)
# ==========================================================
# ARPU: average revenue per user per period (e.g. per month)
ARPU = 50.0          # change if your slides use a different value
discount_rate = 0.01 # per month discount rate
time_horizon = 60    # in months (e.g. 5 years)

# Time steps in months for CLV formula
timeline_months = np.arange(1, time_horizon + 1)

# IMPORTANT: model was trained on scaled tenure: tenure_scaled = tenure / 10
# So 1 unit of model time = 10 months.
model_times = timeline_months / 10.0


# ==========================================================
# 1. LOAD ORIGINAL DATA FOR SEGMENT INFORMATION
# ==========================================================

# Use the same CSV as in Parametric_Models.py
df_raw = pd.read_csv("telco.csv")

# test_sig has the same index as the original df rows used in test set
# Align original information for those customers:
test_raw = df_raw.loc[test_sig.index].copy()

# We'll attach CLV here:
# test_raw will have columns like 'gender', 'internet', 'region', 'marital', 'custcat', etc.


# ==========================================================
# 2. FUNCTION TO COMPUTE CLV FOR ONE CUSTOMER
# ==========================================================

# covariate columns used by the final model (no duration/event)
covariate_cols = test_sig.columns.difference([duration_col, event_col])

def compute_clv_for_row(row_covariates, model):
    """
    row_covariates: a Series with the covariate values (one row from test_sig[covariate_cols])
    model: fitted lifelines AFT model (final_model_sig)

    CLV(t) = sum_{t=1..T} S(t) * ARPU / (1 + r)^t
    where S(t) is survival probability at month t.
    """
    # reshape to DataFrame expected by lifelines
    row_df = row_covariates.to_frame().T

    # predict survival at model_times (scaled time)
    sf = model.predict_survival_function(row_df, times=model_times)
    survival_probs = sf.iloc[:, 0].values  # survival probability at each time point

    # discount factors for each month
    discount_factors = 1.0 / (1.0 + discount_rate) ** timeline_months

    # CLV = sum S(t) * ARPU * discounted
    clv = np.sum(survival_probs * ARPU * discount_factors)
    return clv


# ==========================================================
# 3. COMPUTE CLV FOR ALL CUSTOMERS IN TEST SET
# ==========================================================

clv_values = []

for i in range(len(test_sig)):
    row_cov = test_sig[covariate_cols].iloc[i]
    clv_i = compute_clv_for_row(row_cov, final_model_sig)
    clv_values.append(clv_i)

# Attach CLV to both test_sig and test_raw (for convenience)
test_sig = test_sig.copy()
test_sig["CLV"] = clv_values

test_raw["CLV"] = clv_values

print("\nSample of CLV per customer (first 10 rows):")
print(test_raw[["CLV"]].head(10))


# ==========================================================
# 4. EXPLORE CLV BY SEGMENTS (FROM ORIGINAL DATA)
# ==========================================================

def clv_by_segment(df, segment_col):
    if segment_col not in df.columns:
        print(f"\n[WARNING] Column '{segment_col}' not found in test_raw, skipping.")
        return
    print(f"\n=== CLV by {segment_col} ===")
    print(df.groupby(segment_col)["CLV"].mean())

# Example segments – adapt to your dataset columns
segment_columns = ["gender", "internet", "voice", "marital", "custcat", "region", "retire"]

for col in segment_columns:
    clv_by_segment(test_raw, col)


# ==========================================================
# 5. OPTIONAL: CLV DISTRIBUTION PLOT
# ==========================================================

plt.figure(figsize=(7, 4))
plt.hist(test_raw["CLV"], bins=30, edgecolor="black")
plt.title("CLV Distribution (Test Set)")
plt.xlabel("CLV")
plt.ylabel("Number of Customers")
plt.grid(True)
plt.tight_layout()
plt.show()
