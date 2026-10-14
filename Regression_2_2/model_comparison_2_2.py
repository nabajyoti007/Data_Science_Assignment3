"""
Linear Regression 2.2 - Competing algorithms: systematic evaluation & recommendation
Run feature_engineering_2_2.py first.
Requires: pandas, numpy, scipy, scikit-learn, matplotlib

Protocol
  * 8 candidate models (2 baselines-of-sorts, 4 linear-family, 2 non-linear)
  * 10 x repeated 5-fold cross-validation, folds grouped by match_id
    (both rows of a match always land in the same fold -> no leakage)
  * Same folds for every model (paired comparison)
  * Pairwise test vs OLS with the Nadeau-Bengio corrected resampled t-test
    (ordinary paired t-tests are over-confident on overlapping CV folds)
  * Final check on the chronological hold-out (last 21 matches = knockouts)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, PoissonRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_poisson_deviance
from sklearn.inspection import permutation_importance

SEED = 42
out = Path("outputs_2_2")
out.mkdir(exist_ok=True)

df = pd.read_csv("data/worldcup2026_regression_2_2.csv")
FEATURES = [
    "team_fifa_rank", "opp_fifa_rank",
    "team_avg_goals_scored", "opp_avg_goals_conceded",
    "team_top_confed", "opp_top_confed",
    "team_wc_appearances", "opp_wc_appearances",
]
X, y, groups = df[FEATURES], df["goals_scored"], df["match_id"]

# ------------------------------------------------------------------
# Palette (validated categorical slots, light mode) & chart style
# ------------------------------------------------------------------
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlepad": 22, "axes.titlesize": 13, "axes.titlecolor": INK,
    "font.size": 10,
})
FAMILY_COLOR = {"Baseline": INK2, "Linear": C["blue"], "Count (GLM)": C["aqua"],
                "Non-linear": C["orange"]}

# ------------------------------------------------------------------
# Candidate models
# ------------------------------------------------------------------
def ols_pvalues(Xs, ys):
    A = np.column_stack([np.ones(len(Xs)), Xs.values])
    beta, *_ = np.linalg.lstsq(A, ys.values, rcond=None)
    r = ys.values - A @ beta
    dof = len(ys) - A.shape[1]
    se = np.sqrt(np.diag(r @ r / dof * np.linalg.inv(A.T @ A)))
    return pd.Series(2 * stats.t.sf(np.abs(beta / se), dof)[1:], index=Xs.columns)


class BackwardOLS(LinearRegression):
    """OLS with backward elimination done INSIDE each training set:
    repeatedly drop the least significant variable until all p < threshold.
    Selection never sees the test fold, so the CV estimate stays honest."""
    def __init__(self, threshold=0.10):
        super().__init__()
        self.threshold = threshold

    def fit(self, X, y):
        cols = list(X.columns)
        while len(cols) > 1:
            p = ols_pvalues(X[cols], y)
            if p.max() < self.threshold:
                break
            cols.remove(p.idxmax())
        self.cols_ = cols
        return super().fit(X[cols], y)

    def predict(self, X):
        return super().predict(X[self.cols_])


alphas = np.logspace(-3, 3, 50)
MODELS = {
    "Mean baseline":       ("Baseline", DummyRegressor(strategy="mean")),
    "OLS (8 vars)":        ("Linear", LinearRegression()),
    "OLS backward-selected": ("Linear", BackwardOLS(threshold=0.10)),
    "Ridge":               ("Linear", make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))),
    "Lasso":               ("Linear", make_pipeline(StandardScaler(), LassoCV(alphas=alphas, cv=5, random_state=SEED, max_iter=20000))),
    "Poisson regression":  ("Count (GLM)", make_pipeline(StandardScaler(), PoissonRegressor(alpha=0.01, max_iter=1000))),
    "Random forest":       ("Non-linear", RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1)),
    "Gradient boosting":   ("Non-linear", GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=SEED)),
    "k-NN (k=15)":         ("Non-linear", make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15))),
}

# ------------------------------------------------------------------
# Repeated grouped K-fold (shuffled groups)
# ------------------------------------------------------------------
K, REPEATS = 5, 10
match_ids = groups.unique()


def grouped_folds(rng):
    ids = rng.permutation(match_ids)
    for chunk in np.array_split(ids, K):
        te = groups.isin(chunk).values
        yield np.where(~te)[0], np.where(te)[0]


def score(y_true, pred):
    pred_pos = np.clip(pred, 1e-6, None)  # Poisson deviance needs > 0
    return {"MAE": mean_absolute_error(y_true, pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, pred)),
            "R2": r2_score(y_true, pred),
            "PoissonDev": mean_poisson_deviance(y_true, pred_pos)}


rows = []
rng = np.random.default_rng(SEED)
for rep in range(REPEATS):
    for fold, (tr, te) in enumerate(grouped_folds(rng)):
        for name, (fam, est) in MODELS.items():
            m = clone(est).fit(X.iloc[tr], y.iloc[tr])
            s = score(y.iloc[te], m.predict(X.iloc[te]))
            rows.append({"model": name, "family": fam, "repeat": rep, "fold": fold, **s})
cv = pd.DataFrame(rows)
cv.to_csv(out / "cv_all_models_2_2.csv", index=False)

summary = (cv.groupby(["model", "family"])
             .agg(MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"),
                  RMSE_mean=("RMSE", "mean"), R2_mean=("R2", "mean"),
                  PoissonDev_mean=("PoissonDev", "mean"))
             .reset_index().sort_values("MAE_mean"))

# ------------------------------------------------------------------
# Corrected resampled t-test (Nadeau & Bengio, 2003) vs OLS (8 vars)
# ------------------------------------------------------------------
n_test = len(df) / K
n_train = len(df) - n_test
J = K * REPEATS


def corrected_ttest(a, b):
    d = a - b
    mean, var = d.mean(), d.var(ddof=1)
    se = np.sqrt((1 / J + n_test / n_train) * var)
    t = mean / se
    p = 2 * stats.t.sf(abs(t), J - 1)
    half = stats.t.ppf(0.975, J - 1) * se
    return mean, mean - half, mean + half, t, p


ref = cv[cv.model == "OLS (8 vars)"].sort_values(["repeat", "fold"])["MAE"].values
tests = []
for name in MODELS:
    if name == "OLS (8 vars)":
        continue
    a = cv[cv.model == name].sort_values(["repeat", "fold"])["MAE"].values
    mean, lo, hi, t, p = corrected_ttest(a, ref)
    tests.append({"model": name, "MAE_diff_vs_OLS": mean, "CI_low": lo, "CI_high": hi,
                  "t": t, "p_value": p})
tests = pd.DataFrame(tests).sort_values("MAE_diff_vs_OLS")
tests.to_csv(out / "corrected_ttests_vs_ols_2_2.csv", index=False)

# ------------------------------------------------------------------
# Chronological hold-out (train matches 1-83, test 84-104)
# ------------------------------------------------------------------
train = df["match_id"] <= 83
hold = []
holdout_preds = {}
for name, (fam, est) in MODELS.items():
    m = clone(est).fit(X[train], y[train])
    p = m.predict(X[~train])
    holdout_preds[name] = p
    hold.append({"model": name, **{f"holdout_{k}": v for k, v in score(y[~train], p).items()},
                 "negative_preds": int((p < 0).sum())})
hold = pd.DataFrame(hold)

leaderboard = summary.merge(hold, on="model").merge(
    tests[["model", "MAE_diff_vs_OLS", "p_value"]], on="model", how="left")
leaderboard.to_csv(out / "model_leaderboard_2_2.csv", index=False)

pd.set_option("display.width", 200)
print("========== MODEL LEADERBOARD (10x5 grouped CV + hold-out) ==========")
print(leaderboard.round(4).to_string(index=False))
print("\n========== CORRECTED RESAMPLED t-TEST: MAE(model) - MAE(OLS) ==========")
print(tests.round(4).to_string(index=False))

# ------------------------------------------------------------------
# FIGURE 1 - Leaderboard: CV MAE distribution per model
# ------------------------------------------------------------------
order = summary["model"].tolist()[::-1]  # best at top
fig, ax = plt.subplots(figsize=(9, 5.6))
jit = np.random.default_rng(1)
for i, name in enumerate(order):
    vals = cv[cv.model == name]["MAE"].values
    fam = MODELS[name][0]
    col = FAMILY_COLOR[fam]
    ax.scatter(vals, i + jit.uniform(-0.18, 0.18, len(vals)), s=10, color=col, alpha=0.35, linewidths=0)
    mu = vals.mean()
    lo, hi = np.percentile(vals, [25, 75])
    ax.plot([lo, hi], [i, i], color=col, linewidth=3, solid_capstyle="round")
    ax.scatter([mu], [i], s=70, color=col, edgecolor=SURF, linewidth=2, zorder=3)
    ax.text(mu, i + 0.3, f"{mu:.3f}", va="bottom", ha="center", color=INK, fontsize=8.5, fontweight="bold")
base = summary.loc[summary.model == "Mean baseline", "MAE_mean"].item()
ax.axvline(base, color=INK2, linestyle=(0, (4, 3)), linewidth=1)
ax.text(base + 0.005, len(order) - 0.6, "predict-the-average", color=INK2, fontsize=8, va="bottom")
ax.set_yticks(range(len(order)), order)
ax.set_xlim(0.6, 1.45)
ax.set_xlabel("Mean absolute error (goals) - lower is better")
ax.set_title("Which algorithm predicts a team's goals best?", loc="left")
ax.text(0, 1.02, "50 cross-validation folds per model (dots); bar = interquartile range; big dot = mean",
        transform=ax.transAxes, color=INK2, fontsize=9)
handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, markersize=7, label=f)
           for f, c in FAMILY_COLOR.items()]
ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9)
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(out / "fig1_model_leaderboard_2_2.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------------------
# FIGURE 2 - Forest plot: is any model *significantly* better than OLS?
# ------------------------------------------------------------------
t2 = tests.sort_values("MAE_diff_vs_OLS", ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9, 5))
for i, r in t2.iterrows():
    sig = r.p_value < 0.05
    col = (C["green"] if r.MAE_diff_vs_OLS < 0 else C["red"]) if sig else INK2
    ax.plot([r.CI_low, r.CI_high], [i, i], color=col, linewidth=2, solid_capstyle="round")
    ax.scatter([r.MAE_diff_vs_OLS], [i], s=60, color=col, edgecolor=SURF, linewidth=2, zorder=3)
    ax.text(1.02, i, f"p = {r.p_value:.3f}", va="center", fontsize=9, color=INK, transform=ax.get_yaxis_transform())
ax.axvline(0, color=INK, linewidth=1)
ax.set_yticks(range(len(t2)), t2["model"])
ax.set_xlim(-0.12, 0.28)
ax.set_xlabel("Difference in MAE vs OLS (goals), 95% CI  -  left of 0 = better than OLS")
ax.set_title("Corrected resampled t-test: each model vs linear regression", loc="left")
ax.text(0, 1.02, "Green/red = significantly better/worse at 5%; grey = no significant difference",
        transform=ax.transAxes, color=INK2, fontsize=9)
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(out / "fig2_significance_forest_2_2.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------------------
# FIGURE 3 - Expected-goals map from the linear model:
#            team rank vs opponent rank
# ------------------------------------------------------------------
ols = LinearRegression().fit(X, y)
grid = np.arange(1, 86)
T, O = np.meshgrid(grid, grid)
base_row = X.median()
G = pd.DataFrame({c: np.full(T.size, base_row[c]) for c in FEATURES})
G["team_fifa_rank"], G["opp_fifa_rank"] = T.ravel(), O.ravel()
Z = ols.predict(G).reshape(T.shape)
fig, ax = plt.subplots(figsize=(8, 6.4))
im = ax.imshow(Z, origin="lower", extent=[1, 85, 1, 85], cmap="Blues", aspect="auto")
cs = ax.contour(T, O, Z, levels=[0.5, 1.0, 1.5, 2.0], colors=INK, linewidths=0.8)
ax.clabel(cs, fmt="%.1f goals", fontsize=8)
ax.grid(False)
ax.plot([1, 85], [1, 85], color=SURF, linestyle=(0, (3, 3)), linewidth=1)
for t, o, lab in [(10, 82, "Germany v Curaçao (scored 7)"), (2, 1, "Spain v France, SF (scored 2)"), (83, 6, "Haiti v Brazil (scored 0)")]:
    ax.scatter([t], [o], s=40, color=C["orange"], edgecolor=SURF, linewidth=1.5, zorder=3)
    ax.annotate(lab, (t, o), xytext=(-150 if t > 60 else 6, 8 if o < 70 else -14), textcoords="offset points", fontsize=8, color=INK)
cb = fig.colorbar(im, ax=ax, shrink=0.85)
cb.set_label("Predicted goals scored", color=INK2)
ax.set_xlabel("Team's FIFA rank (1 = best)")
ax.set_ylabel("Opponent's FIFA rank (1 = best)")
ax.set_title("The ranking gap drives the goals", loc="left")
ax.text(0, 1.02, "Linear-model prediction with all other variables held at their median",
        transform=ax.transAxes, color=INK2, fontsize=9)
fig.tight_layout()
fig.savefig(out / "fig3_expected_goals_map_2_2.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------------------
# FIGURE 4 - Knockout storyline: predicted vs actual, per team, hold-out
# ------------------------------------------------------------------
test_df = df[~train].copy()
test_df["pred"] = holdout_preds["OLS (8 vars)"]
test_df["label"] = test_df["team"] + " v " + test_df["opponent"] + "  (" + test_df["round"].str.replace("Quarter-finals", "QF").str.replace("Semi-finals", "SF").str.replace("Third-place match", "3rd").str.replace("Round of 16", "R16").str.replace("Round of 32", "R32") + ")"
test_df = test_df.iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9, 11))
for i, r in test_df.iterrows():
    ax.plot([r.pred, r.goals_scored], [i, i], color=GRID, linewidth=2, zorder=1)
ax.scatter(test_df["pred"], test_df.index, s=45, color=C["blue"], edgecolor=SURF, linewidth=1.5, zorder=3, label="Predicted (linear regression)")
ax.scatter(test_df["goals_scored"], test_df.index, s=45, color=C["orange"], edgecolor=SURF, linewidth=1.5, zorder=3, marker="D", label="Actual goals")
ax.set_yticks(test_df.index, test_df["label"], fontsize=8)
ax.set_xlabel("Goals scored")
ax.set_title("Knockout stage: where the model was right and wrong", loc="left")
ax.text(0, 1.008, "Hold-out set (matches 84-104), model trained only on earlier matches. Grey bar = error.",
        transform=ax.transAxes, color=INK2, fontsize=9)
ax.legend(loc="lower right", frameon=False)
ax.grid(axis="y", visible=False)
ax.set_ylim(-0.8, len(test_df) - 0.2)
fig.tight_layout()
fig.savefig(out / "fig4_knockout_storyline_2_2.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------------------
# FIGURE 5 - Do the models reproduce the *shape* of football scoring?
#            Observed goal-count frequencies vs Poisson-implied frequencies
# ------------------------------------------------------------------
pois = clone(MODELS["Poisson regression"][1]).fit(X, y)
mu = pois.predict(X)
ks = np.arange(0, 8)
expected_pois = np.array([stats.poisson.pmf(k, mu).sum() for k in ks])
lin_mu = np.clip(ols.predict(X), 0, None)
resid_sd = np.std(y - ols.predict(X), ddof=9)
expected_lin = np.array([(stats.norm.cdf(k + 0.5, lin_mu, resid_sd) - stats.norm.cdf(k - 0.5, lin_mu, resid_sd)).sum() for k in ks])
observed = np.array([(y == k).sum() if k < 7 else (y >= 7).sum() for k in ks])
fig, ax = plt.subplots(figsize=(9, 5))
w = 0.27
ax.bar(ks - w, observed, w, color=INK2, label="Observed", edgecolor=SURF, linewidth=1)
ax.bar(ks, expected_pois, w, color=C["aqua"], label="Poisson regression (implied)", edgecolor=SURF, linewidth=1)
ax.bar(ks + w, expected_lin, w, color=C["blue"], label="Linear regression (normal errors)", edgecolor=SURF, linewidth=1)
for k, o in zip(ks, observed):
    ax.text(k - w, o + 1, str(o), ha="center", fontsize=8, color=INK)
ax.set_xticks(ks, [str(k) for k in ks[:-1]] + ["7+"])
ax.set_xlabel("Goals scored by a team in a match")
ax.set_ylabel("Number of team-matches (of 208)")
ax.set_title("Goals are counts, and the Poisson model knows it", loc="left")
ax.text(0, 1.02, "Linear regression puts probability on negative and fractional goals; Poisson matches the zero-heavy shape",
        transform=ax.transAxes, color=INK2, fontsize=9)
ax.legend(frameon=False)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(out / "fig5_goal_distribution_fit_2_2.png", dpi=300)
plt.close(fig)
neg_mass = (stats.norm.cdf(-0.5, lin_mu, resid_sd)).sum()
print(f"\nLinear model: expected number of team-matches predicted below -0.5 goals = {neg_mass:.1f} of 208")
print("Observed counts:", observed.tolist())
print("Poisson expected:", expected_pois.round(1).tolist())
print("Linear expected :", expected_lin.round(1).tolist())

# ------------------------------------------------------------------
# FIGURE 6 - Which variables matter? Permutation importance (OLS vs RF)
# ------------------------------------------------------------------
imp_rows = []
for name in ["OLS (8 vars)", "Poisson regression", "Random forest"]:
    est = clone(MODELS[name][1]).fit(X[train], y[train])
    pi = permutation_importance(est, X[~train], y[~train], scoring="neg_mean_absolute_error",
                                n_repeats=200, random_state=SEED)
    for f, m_, s_ in zip(FEATURES, pi.importances_mean, pi.importances_std):
        imp_rows.append({"model": name, "feature": f, "importance": m_, "sd": s_})
imp = pd.DataFrame(imp_rows)
imp.to_csv(out / "permutation_importance_2_2.csv", index=False)
feat_order = imp[imp.model == "OLS (8 vars)"].sort_values("importance")["feature"].tolist()
fig, ax = plt.subplots(figsize=(9, 5.5))
h = 0.26
cols = {"OLS (8 vars)": C["blue"], "Poisson regression": C["aqua"], "Random forest": C["orange"]}
for j, (name, col) in enumerate(cols.items()):
    sub = imp[imp.model == name].set_index("feature").loc[feat_order]
    ax.barh(np.arange(len(feat_order)) + (j - 1) * h, sub["importance"], h, color=col,
            label=name, edgecolor=SURF, linewidth=1)
ax.axvline(0, color=INK, linewidth=1)
ax.set_yticks(range(len(feat_order)), feat_order)
ax.set_xlabel("Increase in hold-out MAE when the variable is shuffled (goals)")
ax.set_title("What actually drives the predictions?", loc="left")
ax.text(0, 1.02, "Permutation importance on the knockout hold-out; values near 0 = variable adds little",
        transform=ax.transAxes, color=INK2, fontsize=9)
ax.legend(frameon=False, loc="lower right")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(out / "fig6_permutation_importance_2_2.png", dpi=300)
plt.close(fig)

print("\nFigures saved to", out)

# Which variables does backward elimination keep? (selection frequency across all CV training sets)
sel = []
rng = np.random.default_rng(SEED)
for rep in range(REPEATS):
    for tr, te in grouped_folds(rng):
        sel += BackwardOLS(0.10).fit(X.iloc[tr], y.iloc[tr]).cols_
freq = pd.Series(sel).value_counts().reindex(FEATURES, fill_value=0) / J
print("\n========== BACKWARD ELIMINATION: % of 50 training sets keeping each variable ==========")
print((freq * 100).round(0).sort_values(ascending=False).to_string())
final = BackwardOLS(0.10).fit(X, y)
print("Variables kept on all 208 rows:", final.cols_)
print("Coefficients:", dict(zip(final.cols_, np.round(final.coef_, 4))), "Intercept:", round(final.intercept_, 4))
freq.to_csv(out / "backward_selection_frequency_2_2.csv", header=["selection_rate"])
