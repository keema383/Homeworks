# Parametric_Models.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from lifelines import (
    WeibullAFTFitter,
    LogLogisticAFTFitter,
    LogNormalAFTFitter,
    GeneralizedGammaRegressionFitter,
    PiecewiseExponentialRegressionFitter,
)


# =========================
# 1. LOAD & PREPARE DATA
# =========================

# change this to your actual file name
df = pd.read_csv("telco.csv")

# original duration and event columns
duration_original = "tenure"
event_col = "churn_event"

# churn → event 0/1
if df["churn"].dtype == "O":
    df[event_col] = df["churn"].map({"Yes": 1, "No": 0, "yes": 1, "no": 0})
else:
    df[event_col] = df["churn"]

# scale tenure to help convergence
df["tenure_scaled"] = df[duration_original] / 10.0
duration_col = "tenure_scaled"

# define columns
cat_cols = ["region", "marital", "ed", "retire", "gender",
            "voice", "internet", "forward", "custcat"]
num_cols = ["age", "address", "income"]

# one-hot encode categoricals
df_model = pd.get_dummies(
    df[[duration_col, event_col] + cat_cols + num_cols],
    columns=cat_cols,
    drop_first=True
)

# drop rare dummy columns (very few 1s)
X_cols = df_model.columns.difference([duration_col, event_col])
rare_cols = [c for c in X_cols if df_model[c].sum() < 10]  # threshold can be tuned
df_model = df_model.drop(columns=rare_cols)

print("Dropped rare dummy columns:", rare_cols)

# train / test split
train_df, test_df = train_test_split(df_model, test_size=0.3, random_state=42)


# =========================
# 2. FIT AFT MODELS
# =========================

models = {
    "WeibullAFT": WeibullAFTFitter(penalizer=0.01),
    "LogLogisticAFT": LogLogisticAFTFitter(penalizer=0.01),
    "LogNormalAFT": LogNormalAFTFitter(penalizer=0.01),
    "GenGammaAFT": GeneralizedGammaRegressionFitter(penalizer=0.01),
    "PiecewiseExpAFT": PiecewiseExponentialRegressionFitter(
        breakpoints=[6, 12, 24, 36], penalizer=0.1
    ),
}

results = []
fitted_models = {}

for name, fitter in models.items():
    print(f"\n=== Fitting {name} ===")
    fitter._scipy_fit_method = "SLSQP"  # more stable sometimes

    try:
        fitter.fit(
            train_df,
            duration_col=duration_col,
            event_col=event_col,
        )

        cindex = fitter.score(train_df, scoring_method="concordance_index")

        results.append({
            "model": name,
            "AIC": fitter.AIC_,
            "loglik": fitter.log_likelihood_,
            "cindex": cindex,
        })

        fitted_models[name] = fitter
        print(f"{name} fitted successfully.")

    except Exception as e:
        print(f"{name} FAILED to converge / fit. Error:")
        print(e)

# show comparison table
results_df = pd.DataFrame(results).sort_values("AIC")
print("\nModel comparison:")
print(results_df)


# =========================
# 3. PLOT SURVIVAL CURVES
# =========================

if len(fitted_models) > 0:
    covariate_cols = train_df.columns.difference([duration_col, event_col])
    x_ref = train_df[covariate_cols].mean().to_frame().T

    times = np.linspace(0, train_df[duration_col].max(), 100)

    plt.figure(figsize=(8, 5))

    for name, fitter in fitted_models.items():
        sf = fitter.predict_survival_function(x_ref, times=times)
        plt.plot(sf.index, sf.iloc[:, 0], label=name)

    plt.xlabel("Scaled tenure")
    plt.ylabel("Survival probability (P[not churned])")
    plt.title("Survival curves for a typical subscriber")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =========================
# 4. CHOOSE FINAL MODEL
# =========================

if results_df.empty:
    print("No model converged, cannot choose final model.")
    raise SystemExit

best_model_name = results_df.iloc[0]["model"]
final_fitter = fitted_models[best_model_name]

print(f"\nChosen final model: {best_model_name}")
print(final_fitter.summary)


# =========================
# 5. KEEP ONLY SIGNIFICANT FEATURES
# =========================

summary = final_fitter.summary.copy()
idx = summary.index

# We want covariate names (not 'Intercept'), with p < 0.05
if isinstance(idx, pd.MultiIndex):
    # level 0 = parameter name (lambda_, rho_, etc), level 1 = covariate
    param_level = idx.get_level_values(0)
    cov_level = idx.get_level_values(1)

    # exclude intercept rows
    mask_cov = cov_level != "Intercept"

    sig_mask = (summary["p"] < 0.05) & mask_cov
    sig_covariates = cov_level[sig_mask]

    # unique covariate names
    sig_vars = sorted(set(sig_covariates.tolist()))
else:
    # simple Index (not typical for AFT, but just in case)
    sig_rows = summary[summary["p"] < 0.05]
    sig_vars = sig_rows.index.tolist()

print("\nSignificant variables (p < 0.05):")
print(sig_vars)

# if no significant vars, just keep everything (except duration and event)
if len(sig_vars) == 0:
    print("No significant variables at p < 0.05. Using all covariates.")
    sig_vars = train_df.columns.difference([duration_col, event_col]).tolist()

cols_to_keep = [duration_col, event_col] + sig_vars
train_sig = train_df[cols_to_keep].copy()
test_sig = test_df[cols_to_keep].copy()

# refit final model with only significant vars
FinalModelClass = type(final_fitter)
final_model_sig = FinalModelClass(penalizer=0.01)
final_model_sig._scipy_fit_method = "SLSQP"

final_model_sig.fit(train_sig, duration_col=duration_col, event_col=event_col)

print("\nFinal model with significant features only:")
print(final_model_sig.summary)

final_cindex = final_model_sig.score(test_sig, scoring_method="concordance_index")
print("\nFinal model test concordance index:", final_cindex)
