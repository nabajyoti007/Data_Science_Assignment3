
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

df = pd.read_csv(
    "data/worldcup2026_regression_2_1.csv"
)

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

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

# Calculate correlations
corr = df[features + ["goal_difference"]].corr()

print("\n===== CORRELATION MATRIX =====")
print(corr.round(3).to_string())

# Create correlation heatmap
plt.figure(figsize=(11, 8))

plt.imshow(
    corr,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns
)

plt.title("Correlation Matrix - Linear Regression 2.1")
plt.tight_layout()

plt.savefig(
    output_dir / "correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nCorrelation heatmap saved successfully!")
