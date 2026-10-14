
import pandas as pd
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor
)
from statsmodels.tools.tools import add_constant

df = pd.read_csv(
    "data/worldcup2026_regression_2_1.csv"
)

features = [
    "ranking_diff",
    "win_rate_diff",
    "avg_goals_scored_diff",
    "avg_goals_conceded_diff",
    "clean_sheet_rate_diff",
    "rest_days_diff",
    "host_advantage",
    "knockout_stage"
]

X = add_constant(df[features])

vif = pd.DataFrame({
    "Feature": features,
    "VIF": [
        variance_inflation_factor(
            X.values, i
        )
        for i in range(1, len(features) + 1)
    ]
})

print("\n===== VIF RESULTS =====")
print(vif.sort_values("VIF", ascending=False))

vif.to_csv(
    "outputs/vif_results.csv",
    index=False
)

print("\nVIF results saved successfully!")
