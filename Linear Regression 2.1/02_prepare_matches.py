
import pandas as pd

# LOAD THE DATASET

df = pd.read_csv("data/worldcup2026_matches_clean.csv")

print("===== ORIGINAL DATASET =====")
print("Total matches:", len(df))
print("Total columns:", len(df.columns))

# STANDARDISE MATCH DATES

df["Date"] = pd.to_datetime(
    df["Date"],
    format="%Y-%m-%d",
    errors="raise"
)

# CLEAN TEAM NAMES

df["team_a"] = (
    df["Home"]
    .str.replace(r"\s+[a-z]{2,3}$", "", regex=True)
    .str.strip()
)

df["team_b"] = (
    df["Away"]
    .str.replace(r"^[a-z]{2,3}\s+", "", regex=True)
    .str.strip()
)

# VERIFY GOAL DIFFERENCE

df["goal_difference_check"] = (
    df["team_a_goals"] - df["team_b_goals"]
)

assert (
    df["goal_difference_check"] == df["goal_difference"]
).all(), "Error: Goal difference mismatch!"

# Temporary verification column is no longer needed
df = df.drop(columns=["goal_difference_check"])

print("\nPASS: Goal differences verified")

# CHECK MISSING VALUES

required_columns = [
    "match_id",
    "Date",
    "team_a",
    "team_b",
    "team_a_goals",
    "team_b_goals",
    "goal_difference"
]

print("\n===== MISSING VALUES =====")
print(df[required_columns].isnull().sum())

assert not df[required_columns].isnull().any().any(), (
    "Error: Missing values in required columns!"
)

# CHECK DUPLICATE MATCH IDs

duplicate_count = df["match_id"].duplicated().sum()

print("\nDuplicate match IDs:", duplicate_count)

assert duplicate_count == 0, (
    "Error: Duplicate match IDs found!"
)

# VERIFY TEAM COUNT

clean_teams = sorted(
    set(df["team_a"]).union(set(df["team_b"]))
)

print("\n===== CLEANED NATIONAL TEAMS =====")
for index, team in enumerate(clean_teams, start=1):
    print(f"{index}. {team}")

print("\nUnique national teams:", len(clean_teams))


# VALIDATE DATASET

assert len(df) == 104, (
    f"Expected 104 matches, found {len(df)}"
)

assert len(clean_teams) == 48, (
    f"Expected 48 teams, found {len(clean_teams)}"
)

assert (df["team_a"] != df["team_b"]).all(), (
    "Error: A team cannot play against itself!"
)

print("\nPASS: 104 matches verified")
print("PASS: 48 national teams verified")
print("PASS: No duplicate match IDs")
print("PASS: No missing required values")

# PREVIEW THE DATASET

print("\n===== FIRST 10 MATCHES =====")

print(
    df[
        [
            "match_id",
            "Date",
            "team_a",
            "team_b",
            "team_a_goals",
            "team_b_goals",
            "goal_difference"
        ]
    ].head(10).to_string(index=False)
)


# SAVE PREPARED DATASET

output_path = "data/worldcup2026_prepared.csv"

df.to_csv(output_path, index=False)

print("\n===== COMPLETED =====")
print("Prepared dataset saved to:", output_path)
print("Final dataset shape:", df.shape)
