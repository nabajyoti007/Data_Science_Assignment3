# FIFA World Cup 2026 — Linear Regression 2.1

This project uses Python to predict the **goal difference between two teams** in the 2026 FIFA World Cup. It includes **104 matches** and **exactly eight pre-match predictors**. Goal difference is calculated as Team A's goals minus Team B's goals.

## Files and folders

| File | What it does |
|---|---|
| `data/worldcup2026_matches_clean.csv` | Match results collected from FBref, with goals and goal difference. |
| `data/worldcup2026_prepared.csv` | Match data after cleaning team names, checking dates and scores. |
| `data/fifa_rankings_april2026.csv` | FIFA men's ranking positions from 1 April 2026, used as pre-tournament information. |
| `data/worldcup2026_regression_2_1.csv` | Final dataset: 104 rows, eight predictors, one target and a match ID. |
| `check_dataset.py` | Checks record counts, missing values, and match IDs. |
| `02_prepare_matches.py` | Standardises team names and dates, and verifies goal differences. |
| `feature_engineering.py` | Builds eight predictors from rankings and previous tournament results, without using the current match's result. |
| `linear_regression.py` | Trains the model on 83 earlier matches, tests on 21 later matches, and compares results with a simple baseline. |
| `04_eda.py` | Creates the correlation matrix and heatmap. |
| `05_vif_analysis.py` | Checks overlap among predictors using VIF. |
| `outputs/` | Saves predictions, coefficients, model metrics, VIF results and charts (actual vs predicted, residuals, correlation and baseline comparison). |

## Eight predictors

Ranking difference, previous win-rate difference, average goals-scored difference, average goals-conceded difference, clean-sheet-rate difference, rest-days difference, host-nation advantage, and knockout-stage indicator.

The previous-performance features use up to five earlier **2026 World Cup** matches. For teams without previous tournament matches, the script uses zero as a starting value. This is a modelling assumption, not a real performance measurement.

## Run the project

From the project folder in VS Code, install the libraries:

```powershell
py -m pip install pandas numpy matplotlib scikit-learn statsmodels
```

Then run:

```powershell
py check_dataset.py
py 02_prepare_matches.py
py feature_engineering.py
py 04_eda.py
py 05_vif_analysis.py
py linear_regression.py
```

## Results

| Metric | Linear Regression | Baseline |
|---|---:|---:|
| MAE | 1.4016 | 1.4263 |
| RMSE | 1.8791 | 1.7470 |
| R² | -0.5059 | -0.3017 |

Linear regression had a slightly lower MAE, but the baseline performed better on RMSE and R². This shows that predicting football goal differences is difficult with a small dataset and limited pre-match history.

# Linear Regression 2.2 – Goals scored by a team in a WC 2026 match

## How to run
```
python feature_engineering_2_2.py   # builds data/worldcup2026_regression_2_2.csv (208 rows)
python linear_regression_2_2.py     # EDA, VIF, OLS, train/test, CV, plots -> outputs_2_2/
```
Needs pandas, numpy, scipy, scikit-learn, matplotlib (no statsmodels).

## Dataset
208 rows = 104 matches x 2 teams. Each row is one team's view of one match.
ID fields: row_id, match_id (plus date, round, team and opponent for readability, not used in the model).
Target: `goals_scored`, the goals the team scored in normal plus extra time. Penalty shoot-out kicks are not counted.

## The 8 explanatory variables (all known before kick-off)
| # | Variable | Meaning | Source | Shared with 2.1? |
|---|---|---|---|---|
| X1 | team_fifa_rank | Team's FIFA ranking (April 2026) | FIFA ranking file | Yes (ranking_diff) |
| X2 | opp_fifa_rank | Opponent's FIFA ranking | FIFA ranking file | Yes (ranking_diff) |
| X3 | team_avg_goals_scored | Team's mean goals scored in its earlier 2026 WC matches | Match results | Yes (avg_goals_scored_diff) |
| X4 | opp_avg_goals_conceded | Opponent's mean goals conceded in its earlier 2026 WC matches | Match results | Yes (avg_goals_conceded_diff) |
| X5 | team_top_confed | 1 if the team is from UEFA or CONMEBOL | Confederation membership | No |
| X6 | opp_top_confed | 1 if the opponent is from UEFA or CONMEBOL | Confederation membership | No |
| X7 | team_wc_appearances | Team's previous World Cup finals appearances (1930–2022) | FIFA historical records | No |
| X8 | opp_wc_appearances | Opponent's previous World Cup finals appearances | FIFA historical records | No |

Shared with 2.1: 4 of 8 (counted conservatively, since X1 and X2 both come from ranking_diff). That is within the limit of 4.

Leakage control: X3 and X4 only use matches played on an earlier date. In a team's first match there is no 2026 history, so both use 1.344, the 2022 World Cup average of goals per team per match (172 goals over 128 team-matches).

## Results
**Full-sample OLS (208 rows):** R² = 0.252, adjusted R² = 0.222, F = 8.40, p < 0.001.
- Significant at the 5% level:
  - team_fifa_rank: −0.014 goals per rank place, so a stronger team scores more.
  - opp_fifa_rank: +0.012, so a weaker opponent means more goals.
  - team_wc_appearances: +0.043 per previous appearance (p = 0.0499).
- Marginal: opp_avg_goals_conceded, +0.17 (p = 0.08).
- VIF: all below 2.5, so multicollinearity is not a concern.

**Chronological 80/20 split** (train on matches 1–83 = 166 rows, test on matches 84–104 = 42 rows):

| Metric | Linear Regression | Baseline (train mean 1.47) |
|---|---|---|
| MAE | 0.983 | 1.045 |
| RMSE | 1.287 | 1.315 |
| R² | 0.040 | −0.002 |

**5-fold cross-validation grouped by match:** mean MAE 0.958 vs 1.127 for the baseline, and mean R² 0.126 vs −0.059.

## Limitations (for the discussion section)
- Goals are small non-negative counts (mostly 0–3), so linear regression is an approximation. A Poisson model would suit the data better, and one test prediction comes out slightly negative.
- The model under-predicts big scorelines such as 4–6 goals, which are rare and hard to forecast.
- The two rows of each match are not independent: one team's goals are the other's goals conceded. Splitting by match keeps both rows on the same side of train/test.
- The rolling-form variables are noisy early in the tournament because they are based on only 0–2 matches.
- The test set covers knockout matches only, which are tighter and lower-scoring. That partly explains why test R² is lower than the CV R².

---

## Competing algorithms: systematic evaluation (`model_comparison_2_2.py`)

### Protocol
- **9 models:**
  - mean baseline
  - OLS (8 variables)
  - OLS with backward elimination (p < 0.10)
  - Ridge
  - Lasso
  - Poisson regression (GLM)
  - random forest
  - gradient boosting
  - k-NN (k = 15)
- **Cross-validation:** 10 x 5-fold, grouped by match, so both rows of a match always fall in the same fold. Every model gets identical folds, which makes the comparisons paired.
- **Significance test:** the Nadeau–Bengio *corrected* resampled t-test of each model's MAE against OLS. A plain paired t-test is over-confident here because CV training sets overlap. This is the two-sample t-test idea from Objective 1, adapted for CV.
- **Hold-out check:** a second test on the chronological hold-out (matches 84–104, the knockouts).
- **Feature selection without leakage:** backward elimination runs *inside* each training fold. A first version picked variables using p-values from all 208 rows. That looked significantly better than OLS (p = 0.04), but the advantage disappeared once selection was done properly (p = 0.66). It's a good example of selection leakage to mention in the report.

### Leaderboard (CV mean, lower MAE is better)
| Model | CV MAE | CV RMSE | CV R² | Hold-out MAE | ΔMAE vs OLS | p |
|---|---|---|---|---|---|---|
| k-NN (k=15) | 0.941 | 1.244 | 0.171 | 0.956 | −0.023 | 0.40 |
| Ridge | 0.953 | 1.252 | 0.157 | 0.968 | −0.012 | 0.44 |
| Poisson regression | 0.956 | 1.296 | 0.075 | 0.981 | −0.008 | 0.66 |
| Lasso | 0.960 | 1.264 | 0.139 | 0.962 | −0.004 | 0.75 |
| Random forest | 0.962 | 1.274 | 0.121 | 0.945 | −0.002 | 0.93 |
| OLS (8 vars) | 0.964 | 1.269 | 0.126 | 0.983 | – | – |
| OLS backward-selected | 0.971 | 1.278 | 0.116 | 0.976 | +0.007 | 0.66 |
| Gradient boosting | 0.985 | 1.296 | 0.089 | 0.963 | +0.021 | 0.62 |
| Mean baseline | 1.114 | 1.384 | −0.017 | 1.045 | +0.150 | **0.005** |

### Findings
1. Every model beats the predict-the-average baseline. For OLS the improvement is about 0.15 goals of MAE (p = 0.005).
2. **No algorithm is significantly better than plain linear regression** (all p > 0.40). With only 208 rows, the extra flexibility of random forests, boosting and k-NN can't beat a straight line. Where the true signal is mostly linear, simple models win.
3. Backward elimination keeps `opp_fifa_rank` in 100% of training sets and `team_fifa_rank` in 88%. It keeps `team_wc_appearances` in 50% and `opp_avg_goals_conceded` in 42%. The confederation variables and `opp_wc_appearances` are never kept. Team strength (ranking) is the dominant signal.
4. Linear regression gets the *shape* of scoring wrong (Fig 5). Goals are zero-heavy counts. Under normal errors, OLS implies about 17 of 208 team-matches would score fewer than −0.5 goals, and it under-predicts 0-goal games (34 expected vs 55 observed). Poisson matches the observed distribution closely (58 vs 55 zeros, 64 vs 71 ones).

### Recommendation
- **Use Ridge regression as the final 2.2 model.** It's still linear regression, so it meets the brief, and it has the best linear-family CV error. Shrinkage makes the coefficients more stable when several variables carry overlapping information about team strength. It made no negative predictions on the hold-out.
- **Keep OLS for interpretation.** Its coefficients and p-values explain the effects, for example each place a team drops in the FIFA ranking costs about 0.014 goals.
- **Suggest Poisson regression as the next step** when you need probabilities, such as P(team scores 0) or the most likely scoreline. Its accuracy matches OLS, but it respects the count nature of goals.
- **Don't use k-NN,** even though it has the lowest mean CV MAE. The gap isn't significant (p = 0.40), its hold-out error is worse than random forest's, and it gives no interpretable coefficients.
- **Biggest likely improvement:** better variables, not a more complex algorithm. For example, pre-match rolling xG from FBref.

### Figures (data story, in order)
1. `fig1_model_leaderboard_2_2.png`: all 50 CV scores per model. They overlap heavily; only the baseline stands apart.
2. `fig2_significance_forest_2_2.png`: corrected t-test 95% CIs. Every interval crosses zero except the baseline's.
3. `fig3_expected_goals_map_2_2.png`: predicted goals across every team-rank vs opponent-rank combination, with real fixtures marked.
4. `fig4_knockout_storyline_2_2.png`: predicted vs actual goals for every knockout team-match. Shows where the model misses, such as the 6–4 third-place match.
5. `fig5_goal_distribution_fit_2_2.png`: why a count model fits football scoring better.
6. `fig6_permutation_importance_2_2.png`: team rank and World Cup experience drive predictions in all three model families. This uses a small hold-out (42 rows), so treat it as indicative.
