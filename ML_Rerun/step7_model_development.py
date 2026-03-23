"""
Model Development - Hospital Length of Stay Prediction
=======================================================
Single-dataset evaluation using Nested Cross-Validation (gold standard),
Rubin's Rules pooling, and Bootstrap internal validation.

Architecture:
  Outer loop  (5-fold) - unbiased performance estimate per fold
  Inner loop  (5-fold) - hyperparameter tuning via RandomizedSearchCV
  Rubin's Rules        - pools fold-level metrics + importances with CIs
  Bootstrap            - optimism correction on final model

Prerequisites: X_transformed, y in scope (NO separate test set needed).
Run in notebook with:  %run -i step7_model_development.py
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from copy import deepcopy

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from scipy import stats
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR   = Path(".")
OUTER_FOLDS  = 5      # outer CV folds  - performance estimation
INNER_FOLDS  = 5      # inner CV folds  - hyperparameter tuning
N_ITER       = 40     # RandomizedSearchCV iterations per fold
M_BOOTSTRAP  = 50     # bootstrap samples for Rubin's Rules
N_VALIDATION = 200    # bootstrap iterations for optimism correction
RANDOM_SEED  = 42
np.random.seed(RANDOM_SEED)

print("=" * 70)
print("MODEL DEVELOPMENT -- LENGTH OF STAY (NESTED CV, SINGLE DATASET)")
print("=" * 70)
print(f"\n  Samples          : {len(y)}")
print(f"  Features         : {X_transformed.shape[1]}")
print(f"  Outer folds      : {OUTER_FOLDS}  (performance estimation)")
print(f"  Inner folds      : {INNER_FOLDS}  (hyperparameter tuning)")
print(f"  Tuning iters     : {N_ITER} per fold per model")
print(f"  Rubin's M        : {M_BOOTSTRAP} bootstrap datasets")
print(f"  Bootstrap loops  : {N_VALIDATION} (optimism correction)")


# =============================================================================
# HELPERS
# =============================================================================

def safe_log(arr): return np.log1p(np.asarray(arr, dtype=float))
def safe_exp(arr): return np.expm1(np.asarray(arr, dtype=float))

def rmse_orig(y_true_log, y_pred_log):
    return np.sqrt(mean_squared_error(safe_exp(y_true_log), safe_exp(y_pred_log)))

scorer = make_scorer(rmse_orig, greater_is_better=False)

def metrics(y_true, y_pred):
    return (
        np.sqrt(mean_squared_error(y_true, y_pred)),
        mean_absolute_error(y_true, y_pred),
        r2_score(y_true, y_pred),
    )

def rubins_rules(estimates, variances):
    m     = len(estimates)
    Q_bar = np.mean(estimates)
    U_bar = np.mean(variances)
    B     = np.var(estimates, ddof=1)
    T     = U_bar + (1 + 1/m) * B
    SE    = np.sqrt(T)
    if U_bar > 0 and B > 0:
        r      = (1 + 1/m) * B / U_bar
        nu_old = (m - 1) * (1 + 1/r) ** 2
        nu     = (nu_old * 500) / (nu_old + 500)
    else:
        nu = np.inf
    t = stats.t.ppf(0.975, df=nu) if np.isfinite(nu) else 1.96
    return Q_bar, SE, nu, Q_bar - t*SE, Q_bar + t*SE

def pool_metric(values):
    var = np.full(len(values), np.var(values, ddof=1))
    q, _, _, lo, hi = rubins_rules(values, var)
    return q, lo, hi


# =============================================================================
# STEP 1 -- PREPARE DATA
# =============================================================================

y_log     = safe_log(y)
n_samples = len(y_log)
X_np      = X_transformed.values if hasattr(X_transformed, 'values') else np.array(X_transformed)
y_np      = np.asarray(y_log,  dtype=float)
y_raw     = np.asarray(y,      dtype=float)
cols      = list(X_transformed.columns)

baseline_rmse = np.sqrt(mean_squared_error(
    y_raw, np.full(n_samples, y_raw.mean())
))
print(f"\n  Naive baseline RMSE (predict mean): {baseline_rmse:.3f} days")

outer_cv = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_SEED)
inner_cv = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_SEED)

NEEDS_SCALE = {"ElasticNet"}


# =============================================================================
# STEP 2 -- MODEL CONFIGS
# =============================================================================

model_configs = {

    "GradientBoosting": {
        "model" : GradientBoostingRegressor(random_state=RANDOM_SEED),
        "params": {
            "n_estimators"    : [200, 400, 600, 800],
            "learning_rate"   : [0.005, 0.01, 0.05, 0.1],
            "max_depth"       : [2, 3, 4, 5],
            "subsample"       : [0.6, 0.7, 0.8, 1.0],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features"    : ["sqrt", "log2", None],
        },
    },

    "XGBoost": {
        "model" : XGBRegressor(
            random_state=RANDOM_SEED, eval_metric="rmse",
            enable_categorical=True, verbosity=0),
        "params": {
            "n_estimators"    : [200, 400, 600],
            "learning_rate"   : [0.005, 0.01, 0.05, 0.1],
            "max_depth"       : [2, 3, 4, 5, 6],
            "subsample"       : [0.6, 0.7, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
            "reg_alpha"       : [0, 0.1, 0.5, 1.0],
            "reg_lambda"      : [0.5, 1.0, 2.0, 5.0],
        },
    },

    "LightGBM": {
        "model" : LGBMRegressor(random_state=RANDOM_SEED, verbose=-1),
        "params": {
            "n_estimators"     : [200, 400, 600, 800],
            "learning_rate"    : [0.005, 0.01, 0.05, 0.1],
            "max_depth"        : [-1, 4, 6, 8, 10],
            "num_leaves"       : [15, 31, 50, 80, 120],
            "subsample"        : [0.6, 0.7, 0.8, 1.0],
            "colsample_bytree" : [0.6, 0.7, 0.8, 1.0],
            "reg_alpha"        : [0, 0.1, 0.5, 1.0],
            "reg_lambda"       : [0, 0.1, 0.5, 1.0],
            "min_child_samples": [5, 10, 20, 30],
        },
    },

    "RandomForest": {
        "model" : RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
        "params": {
            "n_estimators"     : [200, 400, 600],
            "max_depth"        : [None, 5, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf" : [1, 2, 4, 8],
            "max_features"     : ["sqrt", "log2", 0.5],
        },
    },

    "ElasticNet": {
        "model" : ElasticNet(random_state=RANDOM_SEED, max_iter=5000),
        "params": {
            "alpha"   : [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        },
    },
}


# =============================================================================
# STEP 3 -- NESTED CROSS-VALIDATION
# =============================================================================

print(f"\n{'=' * 70}")
print(f"STEP 3 -- NESTED CV  (outer={OUTER_FOLDS}-fold, inner={INNER_FOLDS}-fold)")
print(f"{'=' * 70}")

nested_results = {name: [] for name in model_configs}

for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X_np)):
    print(f"\n  -- Outer fold {fold_idx+1}/{OUTER_FOLDS} "
          f"(train={len(train_idx)}, val={len(val_idx)}) --")

    X_tr, X_val = X_np[train_idx], X_np[val_idx]
    y_tr, y_val = y_np[train_idx], y_np[val_idx]
    y_val_orig  = y_raw[val_idx]

    # Scale for ElasticNet (fit ONLY on train)
    sc         = StandardScaler()
    X_tr_sc    = sc.fit_transform(X_tr)
    X_val_sc   = sc.transform(X_val)

    X_tr_df     = pd.DataFrame(X_tr,    columns=cols)
    X_val_df    = pd.DataFrame(X_val,   columns=cols)
    X_tr_sc_df  = pd.DataFrame(X_tr_sc, columns=cols)
    X_val_sc_df = pd.DataFrame(X_val_sc, columns=cols)

    for name, cfg in model_configs.items():
        use_scaled = name in NEEDS_SCALE
        X_fit  = X_tr_sc_df  if use_scaled else X_tr_df
        X_eval = X_val_sc_df if use_scaled else X_val_df

        search = RandomizedSearchCV(
            estimator           = deepcopy(cfg["model"]),
            param_distributions = cfg["params"],
            n_iter              = N_ITER,
            cv                  = inner_cv,
            scoring             = scorer,
            n_jobs              = -1,
            random_state        = RANDOM_SEED,
            verbose             = 0,
        )
        search.fit(X_fit, y_tr)
        best_fold = search.best_estimator_

        y_pred_orig       = safe_exp(best_fold.predict(X_eval))
        rmse, mae, r2     = metrics(y_val_orig, y_pred_orig)

        coefs = (best_fold.feature_importances_
                 if hasattr(best_fold, 'feature_importances_')
                 else np.abs(best_fold.coef_))

        resid_var = np.var(y_val_orig - y_pred_orig)

        nested_results[name].append({
            "best_params": search.best_params_,
            "inner_rmse" : -search.best_score_,
            "rmse"       : rmse,
            "mae"        : mae,
            "r2"         : r2,
            "coefs"      : coefs,
            "coef_var"   : np.full(len(coefs), resid_var / max(len(y_val_orig), 1)),
            "y_pred"     : y_pred_orig,
            "y_true"     : y_val_orig,
            "val_idx"    : val_idx,
        })
        print(f"    {name:<18}: inner={search.best_score_*-1:.3f}  "
              f"val_RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.4f}")


# =============================================================================
# STEP 4 -- RUBIN'S RULES POOLING ACROSS FOLDS
# =============================================================================

print(f"\n{'=' * 70}")
print("STEP 4 -- RUBIN'S RULES POOLING ACROSS FOLDS")
print(f"{'=' * 70}")

pooled_results = {}

for name, folds in nested_results.items():
    rmse_p, rmse_lo, rmse_hi = pool_metric([f["rmse"] for f in folds])
    mae_p,  mae_lo,  mae_hi  = pool_metric([f["mae"]  for f in folds])
    r2_p,   r2_lo,   r2_hi   = pool_metric([f["r2"]   for f in folds])

    coef_mat = np.array([f["coefs"]    for f in folds])
    var_mat  = np.array([f["coef_var"] for f in folds])

    Q_list, SE_list, lo_list, hi_list = [], [], [], []
    for j in range(coef_mat.shape[1]):
        q, se, _, lo, hi = rubins_rules(coef_mat[:, j], var_mat[:, j])
        Q_list.append(q); SE_list.append(se)
        lo_list.append(lo); hi_list.append(hi)

    feat_df = pd.DataFrame({
        "Feature"   : cols,
        "Importance": Q_list,
        "SE"        : SE_list,
        "CI_Lower"  : lo_list,
        "CI_Upper"  : hi_list,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    pooled_results[name] = {
        "rmse"     : rmse_p, "rmse_ci"  : (rmse_lo, rmse_hi),
        "mae"      : mae_p,  "mae_ci"   : (mae_lo,  mae_hi),
        "r2"       : r2_p,   "r2_ci"    : (r2_lo,   r2_hi),
        "feat_df"  : feat_df,
        "fold_rmse": [f["rmse"] for f in folds],
    }

    print(f"\n  {name}:")
    print(f"    RMSE : {rmse_p:.3f}  (95% CI: {rmse_lo:.3f}-{rmse_hi:.3f})")
    print(f"    MAE  : {mae_p:.3f}  (95% CI: {mae_lo:.3f}-{mae_hi:.3f})")
    print(f"    R2   : {r2_p:.4f}  (95% CI: {r2_lo:.4f}-{r2_hi:.4f})")

summary_rows = [
    [name,
     pooled_results[name]["rmse"],
     pooled_results[name]["rmse_ci"][0],
     pooled_results[name]["rmse_ci"][1],
     pooled_results[name]["mae"],
     pooled_results[name]["r2"]]
    for name in pooled_results
]
summary_df = pd.DataFrame(
    summary_rows,
    columns=["Model", "RMSE", "RMSE_CI_lo", "RMSE_CI_hi", "MAE", "R2"]
).sort_values("RMSE").reset_index(drop=True)

print(f"\n{'=' * 70}")
print("NESTED CV SUMMARY (Rubin's Rules pooled, original scale)")
print(f"{'=' * 70}")
print(summary_df.round(4).to_string(index=False))
print(f"\n  Naive baseline RMSE: {baseline_rmse:.3f}")

best_name = summary_df.iloc[0]["Model"]
print(f"\n  Best model: {best_name}  "
      f"(RMSE={pooled_results[best_name]['rmse']:.3f}, "
      f"R2={pooled_results[best_name]['r2']:.4f})")


# =============================================================================
# STEP 5 -- FINAL MODEL (retrain on full dataset)
# =============================================================================

print(f"\n{'=' * 70}")
print(f"STEP 5 -- FINAL MODEL ({best_name}) TRAINED ON FULL DATASET")
print(f"{'=' * 70}")

best_folds  = nested_results[best_name]
best_params = best_folds[
    int(np.argmin([f["rmse"] for f in best_folds]))
]["best_params"]
print(f"\n  Using best-fold params: {best_params}")

use_scaled_final = best_name in NEEDS_SCALE
if use_scaled_final:
    sc_final = StandardScaler()
    X_final  = pd.DataFrame(sc_final.fit_transform(X_np), columns=cols)
else:
    X_final  = pd.DataFrame(X_np, columns=cols)

final_model = type(model_configs[best_name]["model"])(**best_params)
final_model.fit(X_final, y_log)

y_apparent_pred              = safe_exp(final_model.predict(X_final))
apparent_rmse, apparent_mae, apparent_r2 = metrics(y_raw, y_apparent_pred)

print(f"\n  Apparent (full data) : RMSE={apparent_rmse:.3f}  "
      f"MAE={apparent_mae:.3f}  R2={apparent_r2:.4f}")
print(f"  Nested CV RMSE (unbiased): {pooled_results[best_name]['rmse']:.3f}")
print(f"  Beats naive baseline     : "
      f"{pooled_results[best_name]['rmse'] < baseline_rmse}")


# =============================================================================
# STEP 6 -- RUBIN'S RULES BOOTSTRAP (feature importance CIs)
# =============================================================================

print(f"\n{'=' * 70}")
print(f"STEP 6 -- RUBIN'S RULES BOOTSTRAP ({M_BOOTSTRAP} samples)")
print(f"{'=' * 70}")

all_coefs     = []
all_variances = []
all_rmse_b    = []
all_mae_b     = []
all_r2_b      = []
has_imp       = hasattr(final_model, 'feature_importances_')

for m_idx in range(M_BOOTSTRAP):
    idx = np.random.choice(n_samples, size=n_samples, replace=True)
    X_b = X_final.iloc[idx].reset_index(drop=True)
    y_b = pd.Series(y_log).iloc[idx].reset_index(drop=True)

    m_b = type(final_model)(**best_params)
    m_b.fit(X_b, y_b)

    coefs = m_b.feature_importances_ if has_imp else np.abs(m_b.coef_)
    all_coefs.append(coefs)

    oob_idx = np.setdiff1d(np.arange(n_samples), np.unique(idx))
    if len(oob_idx) > 5:
        X_oob  = X_final.iloc[oob_idx]
        y_oob  = pd.Series(y_log).iloc[oob_idx]
        p_oob  = m_b.predict(X_oob)
        var    = np.full(len(coefs),
                         np.var(y_oob.values - p_oob) / max(len(oob_idx), 1))
        p_orig = safe_exp(p_oob)
        y_orig = safe_exp(y_oob.values)
        all_rmse_b.append(np.sqrt(mean_squared_error(y_orig, p_orig)))
        all_mae_b.append(mean_absolute_error(y_orig, p_orig))
        all_r2_b.append(r2_score(y_orig, p_orig))
    else:
        var = np.ones(len(coefs)) * 1e-4
    all_variances.append(var)

    if (m_idx + 1) % 10 == 0:
        print(f"  Bootstrap {m_idx+1:>3}/{M_BOOTSTRAP} done")

coef_mat = np.array(all_coefs)
var_mat  = np.array(all_variances)
Q_list, SE_list, lo_list, hi_list = [], [], [], []
for j in range(coef_mat.shape[1]):
    q, se, _, lo, hi = rubins_rules(coef_mat[:, j], var_mat[:, j])
    Q_list.append(q); SE_list.append(se)
    lo_list.append(lo); hi_list.append(hi)

boot_coef_df = pd.DataFrame({
    "Feature"   : cols,
    "Importance": Q_list,
    "SE"        : SE_list,
    "CI_Lower"  : lo_list,
    "CI_Upper"  : hi_list,
}).sort_values("Importance", ascending=False).reset_index(drop=True)

rmse_boot_p, rmse_boot_lo, rmse_boot_hi = pool_metric(all_rmse_b)
r2_boot_p,   r2_boot_lo,   r2_boot_hi   = pool_metric(all_r2_b)

print(f"\n  Bootstrap OOB RMSE: {rmse_boot_p:.3f}  "
      f"(95% CI: {rmse_boot_lo:.3f}-{rmse_boot_hi:.3f})")
print(f"  Bootstrap OOB R2  : {r2_boot_p:.4f}  "
      f"(95% CI: {r2_boot_lo:.4f}-{r2_boot_hi:.4f})")


# =============================================================================
# STEP 7 -- BOOTSTRAP OPTIMISM CORRECTION
# Reference: Harrell (2015), Steyerberg (2019)
# =============================================================================

print(f"\n{'=' * 70}")
print(f"STEP 7 -- BOOTSTRAP OPTIMISM CORRECTION ({N_VALIDATION} iterations)")
print(f"{'=' * 70}")

opt_rmse = []
opt_r2   = []

for b in range(N_VALIDATION):
    idx = np.random.choice(n_samples, size=n_samples, replace=True)
    X_b = X_final.iloc[idx].reset_index(drop=True)
    y_b = pd.Series(y_log).iloc[idx].reset_index(drop=True)

    m_b = type(final_model)(**best_params)
    m_b.fit(X_b, y_b)

    rmse_train, _, r2_train = metrics(safe_exp(y_b), safe_exp(m_b.predict(X_b)))
    rmse_orig,  _, r2_orig  = metrics(y_raw,         safe_exp(m_b.predict(X_final)))

    opt_rmse.append(rmse_train - rmse_orig)
    opt_r2.append(r2_train - r2_orig)

    if (b + 1) % 50 == 0:
        print(f"  Completed {b+1}/{N_VALIDATION} iterations")

mean_opt_rmse  = np.mean(opt_rmse)
mean_opt_r2    = np.mean(opt_r2)
corrected_rmse = apparent_rmse + mean_opt_rmse
corrected_r2   = apparent_r2   - mean_opt_r2

print(f"\n  Optimism RMSE    : {mean_opt_rmse:.4f}")
print(f"  Optimism R2      : {mean_opt_r2:.4f}")
print(f"  Corrected RMSE   : {corrected_rmse:.3f}  (apparent: {apparent_rmse:.3f})")
print(f"  Corrected R2     : {corrected_r2:.4f}  (apparent: {apparent_r2:.4f})")
print(f"  Nested CV RMSE   : {pooled_results[best_name]['rmse']:.3f}  (unbiased)")


# =============================================================================
# STEP 8 -- VISUALISATIONS
# =============================================================================

print(f"\n{'=' * 70}")
print("STEP 8 -- GENERATING VISUALISATIONS")
print(f"{'=' * 70}")

# Figure 1: Model Comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(summary_df))]

for ax, metric, title in [
    (axes[0], "RMSE", "Nested CV RMSE (lower=better)"),
    (axes[1], "MAE",  "Nested CV MAE  (lower=better)"),
    (axes[2], "R2",   "Nested CV R2   (higher=better)"),
]:
    bars = ax.barh(summary_df["Model"], summary_df[metric],
                   color=colors, edgecolor="white", alpha=0.9)
    if metric == "RMSE":
        ax.barh(summary_df["Model"], summary_df["RMSE"],
                xerr=[summary_df["RMSE"] - summary_df["RMSE_CI_lo"],
                      summary_df["RMSE_CI_hi"] - summary_df["RMSE"]],
                color=colors, edgecolor="white", alpha=0.0,
                capsize=4, error_kw={"elinewidth": 1.5, "ecolor": "black"})
    ax.set_xlabel(metric); ax.set_title(title)
    ax.invert_yaxis(); ax.grid(True, axis="x", alpha=0.3)
    for bar, val in zip(bars, summary_df[metric]):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)

axes[0].axvline(baseline_rmse, color="red", lw=1.5, linestyle="--",
                label=f"Baseline {baseline_rmse:.2f}")
axes[0].legend(fontsize=8)
plt.suptitle("Nested CV Model Comparison (Rubin's Rules pooled)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved: model_comparison.png")

# Figure 2: Per-fold RMSE
fig, ax = plt.subplots(figsize=(10, 4))
fold_labels = [f"Fold {i+1}" for i in range(OUTER_FOLDS)]
x       = np.arange(OUTER_FOLDS)
width   = 0.15
palette = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]

for i, (name, color) in enumerate(zip(nested_results.keys(), palette)):
    fold_rmse = [nested_results[name][f]["rmse"] for f in range(OUTER_FOLDS)]
    ax.bar(x + i*width, fold_rmse, width, label=name, color=color, alpha=0.85)

ax.axhline(baseline_rmse, color="black", lw=1.5, linestyle="--",
           label=f"Baseline {baseline_rmse:.2f}")
ax.set_xticks(x + width*2); ax.set_xticklabels(fold_labels)
ax.set_ylabel("RMSE (days)"); ax.set_title("Per-Fold RMSE -- Outer CV Folds")
ax.legend(fontsize=8, loc="upper right"); ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fold_rmse.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved: fold_rmse.png")

# Figure 3: Actual vs Predicted (OOF + Apparent)
oof_pred = np.zeros(n_samples)
oof_true = np.zeros(n_samples)
for fold_data in nested_results[best_name]:
    oof_pred[fold_data["val_idx"]] = fold_data["y_pred"]
    oof_true[fold_data["val_idx"]] = fold_data["y_true"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, y_t, y_p, label, color in [
    (axes[0], oof_true,  oof_pred,        "OOF Predictions", "#2196F3"),
    (axes[1], y_raw,     y_apparent_pred, "Apparent (Full)", "#FF5722"),
]:
    ax.scatter(y_t, y_p, alpha=0.4, s=20, color=color, edgecolors="none")
    lim = max(float(y_t.max()), float(y_p.max())) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1.5, label="Perfect prediction")
    cal   = LinearRegression().fit(y_p.reshape(-1,1), y_t)
    x_ln  = np.linspace(0, lim, 100)
    ax.plot(x_ln, cal.predict(x_ln.reshape(-1,1)),
            "r-", lw=1.5, label=f"Fit (slope={cal.coef_[0]:.2f})")
    r2_  = r2_score(y_t, y_p)
    mae_ = mean_absolute_error(y_t, y_p)
    ax.set_xlabel("Actual LOS (days)", fontsize=11)
    ax.set_ylabel("Predicted LOS (days)", fontsize=11)
    ax.set_title(f"{label} -- {best_name}\nR2={r2_:.3f}  MAE={mae_:.2f} days",
                 fontsize=12)
    ax.legend(fontsize=9); ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.grid(True, alpha=0.3)
plt.suptitle(f"Actual vs Predicted -- {best_name}", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved: actual_vs_predicted.png")

# Figure 4: Residual Diagnostics (OOF)
residuals = oof_true - oof_pred
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].scatter(oof_pred, residuals, alpha=0.4, s=20, color="#9C27B0")
axes[0].axhline(0, color="k", lw=1.5, linestyle="--")
axes[0].set_xlabel("Predicted LOS (days)")
axes[0].set_ylabel("Residual (Actual - Predicted)")
axes[0].set_title("Residuals vs Predicted (OOF)")
axes[0].grid(True, alpha=0.3)

axes[1].hist(residuals, bins=40, color="#9C27B0", alpha=0.7, edgecolor="white")
axes[1].axvline(0, color="k", lw=1.5, linestyle="--")
axes[1].set_xlabel("Residual (days)"); axes[1].set_ylabel("Count")
axes[1].set_title(f"Residual Distribution\n"
                  f"Mean={residuals.mean():.2f}  SD={residuals.std():.2f}")
axes[1].grid(True, alpha=0.3)

stats.probplot(residuals, dist="norm", plot=axes[2])
axes[2].set_title("Q-Q Plot of Residuals (OOF)")
axes[2].grid(True, alpha=0.3)
plt.suptitle(f"Residual Diagnostics -- {best_name} (Out-of-Fold)",
             fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "residual_diagnostics.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved: residual_diagnostics.png")

# Figure 5: Feature Importances
top_n = min(15, len(boot_coef_df))
top   = boot_coef_df.head(top_n)
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(top["Feature"], top["Importance"],
        xerr=[top["Importance"] - top["CI_Lower"],
              top["CI_Upper"]   - top["Importance"]],
        color="#2196F3", alpha=0.8, capsize=4,
        error_kw={"elinewidth": 1.2})
ax.set_xlabel("Pooled Feature Importance (Rubin's Rules)")
ax.set_title(f"Top {top_n} Features -- {best_name}")
ax.invert_yaxis(); ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved: feature_importance.png")

# Figure 6: Bootstrap Optimism
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(opt_rmse, bins=30, color="#FF9800", alpha=0.8, edgecolor="white")
axes[0].axvline(mean_opt_rmse, color="red", lw=2,
                label=f"Mean={mean_opt_rmse:.4f}")
axes[0].set_xlabel("RMSE Optimism"); axes[0].set_ylabel("Count")
axes[0].set_title("Bootstrap RMSE Optimism")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].hist(opt_r2, bins=30, color="#4CAF50", alpha=0.8, edgecolor="white")
axes[1].axvline(mean_opt_r2, color="red", lw=2,
                label=f"Mean={mean_opt_r2:.4f}")
axes[1].set_xlabel("R2 Optimism"); axes[1].set_ylabel("Count")
axes[1].set_title("Bootstrap R2 Optimism")
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.suptitle(f"Bootstrap Internal Validation (n={N_VALIDATION})", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "bootstrap_optimism.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved: bootstrap_optimism.png")


# =============================================================================
# STEP 9 -- SAVE OUTPUTS
# =============================================================================

print(f"\n{'=' * 70}")
print("STEP 9 -- SAVING OUTPUTS")
print(f"{'=' * 70}")

summary_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
print("  Saved: model_comparison.csv")

boot_coef_df.to_csv(OUTPUT_DIR / "pooled_feature_importances.csv", index=False)
print("  Saved: pooled_feature_importances.csv")

oof_df = pd.DataFrame({
    "actual_los"   : oof_true,
    "predicted_los": oof_pred,
    "residual"     : oof_true - oof_pred,
    "abs_error"    : np.abs(oof_true - oof_pred),
})
oof_df.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
print("  Saved: oof_predictions.csv")

performance = {
    "analysis_date": datetime.now().isoformat(),
    "best_model"   : best_name,
    "n_samples"    : int(n_samples),
    "n_features"   : int(X_transformed.shape[1]),
    "methods": {
        "evaluation"         : f"Nested CV (outer={OUTER_FOLDS}, inner={INNER_FOLDS})",
        "tuning"             : f"RandomizedSearchCV (n_iter={N_ITER})",
        "pooling"            : f"Rubin's Rules (folds + bootstrap M={M_BOOTSTRAP})",
        "df_adjustment"      : "Barnard-Rubin (1999)",
        "internal_validation": f"Bootstrap optimism correction (n={N_VALIDATION})",
        "target_transform"   : "log1p / expm1 (full dataset)",
    },
    "all_models": {
        name: {
            "nested_cv_rmse"   : float(pooled_results[name]["rmse"]),
            "nested_cv_rmse_ci": [float(pooled_results[name]["rmse_ci"][0]),
                                  float(pooled_results[name]["rmse_ci"][1])],
            "nested_cv_mae"    : float(pooled_results[name]["mae"]),
            "nested_cv_r2"     : float(pooled_results[name]["r2"]),
        }
        for name in pooled_results
    },
    "best_model_performance": {
        "nested_cv_rmse"    : float(pooled_results[best_name]["rmse"]),
        "nested_cv_rmse_ci" : [float(pooled_results[best_name]["rmse_ci"][0]),
                                float(pooled_results[best_name]["rmse_ci"][1])],
        "nested_cv_r2"      : float(pooled_results[best_name]["r2"]),
        "apparent_rmse"     : float(apparent_rmse),
        "apparent_r2"       : float(apparent_r2),
        "corrected_rmse"    : float(corrected_rmse),
        "corrected_r2"      : float(corrected_r2),
        "bootstrap_oob_rmse": float(rmse_boot_p),
        "bootstrap_oob_r2"  : float(r2_boot_p),
        "baseline_rmse"     : float(baseline_rmse),
        "beats_baseline"    : bool(pooled_results[best_name]["rmse"] < baseline_rmse),
    },
    "bootstrap_validation": {
        "n_iterations"  : N_VALIDATION,
        "optimism_rmse" : float(mean_opt_rmse),
        "optimism_r2"   : float(mean_opt_r2),
        "corrected_rmse": float(corrected_rmse),
        "corrected_r2"  : float(corrected_r2),
    },
    "top_features": boot_coef_df.head(10)["Feature"].tolist(),
}

with open(OUTPUT_DIR / "model_performance.json", "w") as f:
    json.dump(performance, f, indent=2)
print("  Saved: model_performance.json")

model_rows = "\n".join(
    f"| {row['Model']} | {row['RMSE']:.3f} "
    f"({row['RMSE_CI_lo']:.3f}-{row['RMSE_CI_hi']:.3f}) "
    f"| {row['MAE']:.3f} | {row['R2']:.4f} |"
    for _, row in summary_df.iterrows()
)
feat_rows = "\n".join(
    f"| {r['Feature']} | {r['Importance']:.4f} | "
    f"({r['CI_Lower']:.4f}-{r['CI_Upper']:.4f}) |"
    for _, r in boot_coef_df.head(10).iterrows()
)

report = f"""# Model Development Report -- Length of Hospital Stay
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Methods
- **Evaluation**: Nested CV (outer={OUTER_FOLDS}-fold, inner={INNER_FOLDS}-fold)
- **Tuning**: RandomizedSearchCV (n_iter={N_ITER} per fold)
- **Pooling**: Rubin's Rules across CV folds + bootstrap (M={M_BOOTSTRAP})
- **df adjustment**: Barnard-Rubin (1999)
- **Optimism correction**: Bootstrap ({N_VALIDATION} iterations, Harrell 2015)
- **Target transform**: log1p / expm1

## Model Comparison (Nested CV, Rubin's Rules)
| Model | RMSE (95% CI) | MAE | R2 |
|-------|--------------|-----|-----|
{model_rows}
| Naive baseline | {baseline_rmse:.3f} | - | - |

## Best Model: {best_name}
| Metric | Value |
|--------|-------|
| Nested CV RMSE (95% CI) | {pooled_results[best_name]['rmse']:.3f} ({pooled_results[best_name]['rmse_ci'][0]:.3f}-{pooled_results[best_name]['rmse_ci'][1]:.3f}) |
| Nested CV R2 (95% CI) | {pooled_results[best_name]['r2']:.4f} ({pooled_results[best_name]['r2_ci'][0]:.4f}-{pooled_results[best_name]['r2_ci'][1]:.4f}) |
| Apparent RMSE | {apparent_rmse:.3f} |
| Optimism-corrected RMSE | {corrected_rmse:.3f} |
| Bootstrap OOB RMSE | {rmse_boot_p:.3f} |
| Beats naive baseline | {pooled_results[best_name]['rmse'] < baseline_rmse} |

## Top 10 Features (Bootstrap Rubin's Rules)
| Feature | Importance | 95% CI |
|---------|------------|--------|
{feat_rows}

## References
1. Rubin DB (1987). Multiple Imputation for Nonresponse in Surveys.
2. Barnard J, Rubin DB (1999). Biometrika 86(4):948-955.
3. Harrell FE (2015). Regression Modeling Strategies. Springer.
4. Steyerberg EW (2019). Clinical Prediction Models. Springer.
5. Varma S, Simon R (2006). Bias in error estimation with nested CV. BMC Bioinformatics.
"""

with open(OUTPUT_DIR / "model_report.md", "w") as f:
    f.write(report)
print("  Saved: model_report.md")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(f"\n{'=' * 70}")
print("COMPLETE -- FINAL SUMMARY")
print(f"{'=' * 70}")
print(f"\n  Best model          : {best_name}")
print(f"  Nested CV RMSE      : {pooled_results[best_name]['rmse']:.3f}  "
      f"(95% CI: {pooled_results[best_name]['rmse_ci'][0]:.3f}-"
      f"{pooled_results[best_name]['rmse_ci'][1]:.3f})")
print(f"  Nested CV R2        : {pooled_results[best_name]['r2']:.4f}")
print(f"  Apparent RMSE       : {apparent_rmse:.3f}")
print(f"  Optimism-corr RMSE  : {corrected_rmse:.3f}")
print(f"  Bootstrap OOB RMSE  : {rmse_boot_p:.3f}")
print(f"  Naive baseline RMSE : {baseline_rmse:.3f}")
print(f"  Beats baseline      : "
      f"{pooled_results[best_name]['rmse'] < baseline_rmse}")
print(f"\n  Outputs saved to    : {OUTPUT_DIR.resolve()}")
print("=" * 70)