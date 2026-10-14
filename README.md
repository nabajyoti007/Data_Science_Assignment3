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
