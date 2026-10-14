
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# LOAD REGRESSION DATA

data_path = Path("data/worldcup2026_regression_2_1.csv")
output_dir = Path("outputs")

output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_path)

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

X = df[features]
y = df["goal_difference"]

# Validate dataset
assert len(df) == 104, "Dataset must contain 104 matches"
assert len(features) == 8, "Exactly 8 features required"
assert X.shape[1] == 8
assert df["match_id"].is_unique
assert not X.isnull().any().any()
assert not y.isnull().any()

print("========== DATASET INFORMATION ==========")
print("Dataset shape:", df.shape)
print("Total matches:", len(df))
print("Explanatory variables:", len(features))

# TRAIN/TEST SPLIT

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\n========== DATA SPLITTING ==========")
print("Training matches:", len(X_train))
print("Testing matches:", len(X_test))


# TRAIN LINEAR REGRESSION MODEL

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nLinear Regression model trained successfully!")

# EVALUATE LINEAR REGRESSION

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("\n========== LINEAR REGRESSION ==========")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# BASELINE MODEL


baseline_value = y_train.mean()

baseline_pred = np.full(
    len(y_test),
    baseline_value
)

baseline_mae = mean_absolute_error(
    y_test, baseline_pred
)

baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_pred)
)

baseline_r2 = r2_score(
    y_test, baseline_pred
)

print("\n========== BASELINE MODEL ==========")
print(f"Baseline prediction: {baseline_value:.4f}")
print(f"Baseline MAE: {baseline_mae:.4f}")
print(f"Baseline RMSE: {baseline_rmse:.4f}")
print(f"Baseline R²: {baseline_r2:.4f}")


# COMPARE BOTH MODELS

comparison = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "R2"
    ],
    "Linear Regression": [
        mae,
        rmse,
        r2
    ],
    "Baseline": [
        baseline_mae,
        baseline_rmse,
        baseline_r2
    ]
})

print("\n========== MODEL COMPARISON ==========")
print(comparison.to_string(index=False))

comparison.to_csv(
    output_dir / "model_comparison.csv",
    index=False
)

print("\nModel comparison saved successfully!")


# MODEL COEFFICIENTS

coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})

coefficients = coefficients.sort_values(
    by="Coefficient",
    ascending=False
)

print("\n========== MODEL COEFFICIENTS ==========")
print(coefficients.to_string(index=False))

print(f"\nIntercept: {model.intercept_:.4f}")

coefficients.to_csv(
    output_dir / "regression_coefficients.csv",
    index=False
)

# SAVE TEST PREDICTIONS

results = pd.DataFrame({
    "match_id": df.iloc[split_index:]["match_id"].values,
    "actual_goal_difference": y_test.values,
    "predicted_goal_difference": y_pred,
    "baseline_prediction": baseline_pred,
    "residual": y_test.values - y_pred,
    "absolute_error": np.abs(y_test.values - y_pred)
})

results.to_csv(
    output_dir / "test_predictions.csv",
    index=False
)

print("\n========== SAMPLE TEST PREDICTIONS ==========")
print(results.head(10).to_string(index=False))

# ACTUAL VS PREDICTED GRAPH

plt.figure(figsize=(7, 5))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.8,
    label="Predicted matches"
)

limits = [
    min(y_test.min(), y_pred.min()),
    max(y_test.max(), y_pred.max())
]

plt.plot(
    limits,
    limits,
    linestyle="--",
    label="Perfect prediction"
)

plt.xlabel("Actual Goal Difference")
plt.ylabel("Predicted Goal Difference")

plt.title(
    "Linear Regression 2.1: Actual vs Predicted"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    output_dir / "actual_vs_predicted.png",
    dpi=300
)

plt.close()

# RESIDUAL ANALYSIS GRAPH

residuals = y_test.values - y_pred

plt.figure(figsize=(7, 5))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.8
)

plt.axhline(
    y=0,
    linestyle="--",
    label="Zero residual"
)

plt.xlabel("Predicted Goal Difference")
plt.ylabel("Residual (Actual - Predicted)")

plt.title(
    "Linear Regression 2.1: Residual Analysis"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    output_dir / "residual_plot.png",
    dpi=300
)

plt.close()

# MODEL COMPARISON GRAPH

metric_names = ["MAE", "RMSE"]

regression_values = [mae, rmse]

baseline_values = [
    baseline_mae,
    baseline_rmse
]

x = np.arange(len(metric_names))
width = 0.35

plt.figure(figsize=(7, 5))

plt.bar(
    x - width / 2,
    regression_values,
    width,
    label="Linear Regression"
)

plt.bar(
    x + width / 2,
    baseline_values,
    width,
    label="Baseline"
)

plt.xticks(x, metric_names)

plt.ylabel("Error in Goals")

plt.title(
    "Linear Regression vs Baseline Performance"
)

plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    output_dir / "model_comparison.png",
    dpi=300
)

plt.close()

# SUMMARY

print("\n========== FINAL SUMMARY ==========")

print("Total matches:", len(df))
print("Training matches:", len(X_train))
print("Testing matches:", len(X_test))
print("Number of explanatory variables:", len(features))

if mae < baseline_mae:
    print(
        "Linear Regression has lower MAE "
        "than the baseline."
    )
elif mae > baseline_mae:
    print(
        "The baseline has lower MAE "
        "than Linear Regression."
    )
else:
    print(
        "Both models have the same MAE."
    )

if rmse < baseline_rmse:
    print(
        "Linear Regression has lower RMSE "
        "than the baseline."
    )
elif rmse > baseline_rmse:
    print(
        "The baseline has lower RMSE "
        "than Linear Regression."
    )
else:
    print(
        "Both models have the same RMSE."
    )

print("\n========== FILES SAVED ==========")

saved_files = [
    "model_comparison.csv",
    "regression_coefficients.csv",
    "test_predictions.csv",
    "actual_vs_predicted.png",
    "residual_plot.png",
    "model_comparison.png"
]

for filename in saved_files:
    print(output_dir / filename)

print("\nLinear Regression 2.1 evaluation completed!")
