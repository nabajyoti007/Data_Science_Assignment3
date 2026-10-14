
import pandas as pd

df = pd.read_csv("data/worldcup2026_prepared.csv")

print("========== DATASET CHECK ==========")

print("Total matches:", len(df))
print("Total columns:", len(df.columns))

teams = set(df["team_a"]) | set(df["team_b"])

print("Unique teams:", len(teams))
print("Unique match IDs:", df["match_id"].nunique())

print("\nMissing required values:")
required = [
    "match_id", "Date", "team_a", "team_b",
    "team_a_goals", "team_b_goals", "goal_difference"
]
print(df[required].isnull().sum())

# Verify goal differences
expected = df["team_a_goals"] - df["team_b_goals"]

assert (expected == df["goal_difference"]).all()
assert len(df) == 104
assert len(teams) == 48
assert df["match_id"].is_unique

print("\nPASS: Dataset checks completed!")

print("\nFirst five matches:")
print(df[required].head().to_string(index=False))
