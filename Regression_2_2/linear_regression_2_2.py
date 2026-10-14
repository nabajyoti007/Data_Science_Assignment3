"""
Linear Regression 2.2 - Predict goals scored by a team in a FIFA World Cup 2026 match.
Run feature_engineering_2_2.py first.
Requires: pandas, numpy, scipy, scikit-learn, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

data_path = Path("data/worldcup2026_regression_2_2.csv")
out = Path("outputs_2_2")
out.mkdir(exist_ok=True)

df = pd.read_csv(data_path)

FEATURES = [
    "team_fifa_rank", "opp_fifa_rank",
    "team_avg_goals_scored", "opp_avg_goals_conceded",
    "team_top_confed", "opp_top_confed",
    "team_wc_appearances", "opp_wc_appearances",
]
TARGET = "goals_scored"

X, y = df[FEATURES], df[TARGET]
assert len(df) == 208 and X.shape[1] == 8
assert not X.isnull().any().any()

print("========== DATASET ==========")
print("Rows:", len(df), "| Matches:", df["match_id"].nunique(), "| Features:", len(FEATURES))
print(df[FEATURES + [TARGET]].describe().round(3).T.to_string())

# ==========================================
# STEP 1: EDA - CORRELATION MATRIX
# ==========================================
corr = df[FEATURES + [TARGET]].corr()
print("\n========== CORRELATION WITH GOALS SCORED ==========")
print(corr[TARGET].drop(TARGET).sort_values().round(3).to_string())

plt.figure(figsize=(11, 8))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
plt.title("Correlation Matrix - Linear Regression 2.2")
plt.tight_layout()
plt.savefig(out / "correlation_heatmap_2_2.png", dpi=300)
plt.close()

# Distribution of the target
plt.figure(figsize=(7, 4.5))
counts = y.value_counts().sort_index()
plt.bar(counts.index, counts.values)
plt.xlabel("Goals scored by a team in a match")
plt.ylabel("Number of team-matches")
plt.title("Distribution of Goals Scored (208 team-matches)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(out / "goals_distribution_2_2.png", dpi=300)
plt.close()

# ==========================================
# STEP 2: MULTICOLLINEARITY (VIF)
# ==========================================
def vif_table(frame):
    vals = []
    for col in frame.columns:
        others = frame.drop(columns=col)
        r2 = LinearRegression().fit(others, frame[col]).score(others, frame[col])
        vals.append(1 / (1 - r2))
    return pd.DataFrame({"Feature": frame.columns, "VIF": vals})

vif = vif_table(X).sort_values("VIF", ascending=False)
print("\n========== VIF ==========")
print(vif.round(3).to_string(index=False))
vif.to_csv(out / "vif_results_2_2.csv", index=False)

# ==========================================
# STEP 3: FULL-SAMPLE OLS INFERENCE
# ==========================================
def ols_summary(Xf, yv):
    A = np.column_stack([np.ones(len(Xf)), Xf.values])
    beta, *_ = np.linalg.lstsq(A, yv.values, rcond=None)
    resid = yv.values - A @ beta
    n, p = A.shape
    sigma2 = resid @ resid / (n - p)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(A.T @ A)))
    t = beta / se
    pvals = 2 * stats.t.sf(np.abs(t), n - p)
    r2 = 1 - (resid @ resid) / ((yv - yv.mean()) ** 2).sum()
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p)
    f_stat = (r2 / (p - 1)) / ((1 - r2) / (n - p))
    f_p = stats.f.sf(f_stat, p - 1, n - p)
    table = pd.DataFrame({
        "Variable": ["Intercept"] + list(Xf.columns),
        "Coefficient": beta, "Std_Error": se, "t_value": t, "p_value": pvals,
    })
    return table, r2, adj_r2, f_stat, f_p

ols_table, r2_full, adj_r2_full, f_stat, f_p = ols_summary(X, y)
print("\n========== OLS ON ALL 208 ROWS ==========")
print(ols_table.round(4).to_string(index=False))
print(f"R² = {r2_full:.4f} | Adjusted R² = {adj_r2_full:.4f} | F = {f_stat:.3f} (p = {f_p:.2e})")
ols_table.to_csv(out / "ols_full_sample_2_2.csv", index=False)

# ==========================================
# STEP 4: CHRONOLOGICAL TRAIN/TEST SPLIT (BY MATCH)
# ==========================================
# Same 80/20 split as 2.1: first 83 matches train, last 21 matches test.
# Splitting by match keeps both rows of a match on the same side.
split_match = df["match_id"].drop_duplicates().iloc[int(104 * 0.8) - 1]
train = df["match_id"] <= split_match
X_train, X_test, y_train, y_test = X[train], X[~train], y[train], y[~train]
print("\n========== SPLIT ==========")
print(f"Train rows: {len(X_train)} (matches 1-{split_match}) | Test rows: {len(X_test)}")

model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_clip = np.clip(y_pred, 0, None)  # goals cannot be negative

baseline_pred = np.full(len(y_test), y_train.mean())

def metrics(actual, pred):
    return (mean_absolute_error(actual, pred),
            np.sqrt(mean_squared_error(actual, pred)),
            r2_score(actual, pred))

lr_m, base_m = metrics(y_test, y_pred), metrics(y_test, baseline_pred)
comparison = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2"],
    "Linear Regression": lr_m,
    "Baseline (train mean)": base_m,
})
print("\n========== TEST SET PERFORMANCE ==========")
print(comparison.round(4).to_string(index=False))
print(f"Baseline prediction = {y_train.mean():.4f} goals")
print(f"Negative predictions in test set: {(y_pred < 0).sum()}")
comparison.to_csv(out / "model_comparison_2_2.csv", index=False)

coefs = pd.DataFrame({"Feature": FEATURES, "Coefficient": model.coef_})
coefs.loc[len(coefs)] = ["Intercept", model.intercept_]
print("\n========== TRAINED MODEL COEFFICIENTS ==========")
print(coefs.round(4).to_string(index=False))
coefs.to_csv(out / "regression_coefficients_2_2.csv", index=False)

# ==========================================
# STEP 5: 5-FOLD CROSS-VALIDATION (GROUPED BY MATCH)
# ==========================================
cv_rows = []
for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, df["match_id"]), 1):
    m = LinearRegression().fit(X.iloc[tr], y.iloc[tr])
    p = m.predict(X.iloc[te])
    b = np.full(len(te), y.iloc[tr].mean())
    cv_rows.append([fold, *metrics(y.iloc[te], p), *metrics(y.iloc[te], b)])
cv = pd.DataFrame(cv_rows, columns=["Fold", "LR_MAE", "LR_RMSE", "LR_R2",
                                    "Base_MAE", "Base_RMSE", "Base_R2"])
print("\n========== 5-FOLD GROUPED CROSS-VALIDATION ==========")
print(cv.round(4).to_string(index=False))
print("Mean:", cv.drop(columns="Fold").mean().round(4).to_dict())
cv.to_csv(out / "cross_validation_2_2.csv", index=False)

# ==========================================
# STEP 6: SAVE PREDICTIONS
# ==========================================
test_rows = df[~train]
results = pd.DataFrame({
    "row_id": test_rows["row_id"].values,
    "match_id": test_rows["match_id"].values,
    "team": test_rows["team"].values,
    "opponent": test_rows["opponent"].values,
    "actual_goals": y_test.values,
    "predicted_goals": y_pred.round(3),
    "baseline_prediction": baseline_pred.round(3),
    "residual": (y_test.values - y_pred).round(3),
})
results.to_csv(out / "test_predictions_2_2.csv", index=False)

# ==========================================
# STEP 7: PLOTS
# ==========================================
plt.figure(figsize=(7, 5))
jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(y_test))
plt.scatter(y_test + jitter, y_pred, alpha=0.75, label="Test team-matches")
lim = [min(0, y_pred.min()) - 0.2, max(y_test.max(), y_pred.max()) + 0.3]
plt.plot(lim, lim, "--", label="Perfect prediction")
plt.xlabel("Actual Goals Scored (jittered)")
plt.ylabel("Predicted Goals Scored")
plt.title("Linear Regression 2.2: Actual vs Predicted (test set)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(out / "actual_vs_predicted_2_2.png", dpi=300); plt.close()

plt.figure(figsize=(7, 5))
plt.scatter(y_pred, y_test.values - y_pred, alpha=0.75)
plt.axhline(0, linestyle="--", label="Zero residual")
plt.xlabel("Predicted Goals Scored")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Linear Regression 2.2: Residual Analysis (test set)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(out / "residual_plot_2_2.png", dpi=300); plt.close()

# Q-Q plot of full-sample residuals (normality check)
A = np.column_stack([np.ones(len(X)), X.values])
full_resid = y.values - A @ ols_table["Coefficient"].values
plt.figure(figsize=(6, 5))
stats.probplot(full_resid, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals (all 208 rows)")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(out / "qq_plot_2_2.png", dpi=300); plt.close()

x = np.arange(2); w = 0.35
plt.figure(figsize=(7, 5))
plt.bar(x - w / 2, lr_m[:2], w, label="Linear Regression")
plt.bar(x + w / 2, base_m[:2], w, label="Baseline (train mean)")
plt.xticks(x, ["MAE", "RMSE"]); plt.ylabel("Error in goals")
plt.title("Linear Regression 2.2 vs Baseline (test set)")
plt.legend(); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
plt.savefig(out / "model_comparison_2_2.png", dpi=300); plt.close()

plt.figure(figsize=(8, 5))
sig = ols_table.iloc[1:]
colors = ["tab:green" if p < 0.05 else "tab:gray" for p in sig["p_value"]]
plt.barh(sig["Variable"], sig["Coefficient"], color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Coefficient (goals per unit change)")
plt.title("OLS Coefficients, all 208 rows (green = p < 0.05)")
plt.tight_layout()
plt.savefig(out / "coefficients_2_2.png", dpi=300); plt.close()

print("\nAll outputs saved to", out)
