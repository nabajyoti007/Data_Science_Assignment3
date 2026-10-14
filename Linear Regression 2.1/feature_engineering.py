
import pandas as pd
from collections import defaultdict
from pathlib import Path

# LOAD THE DATASETS

data_dir = Path("data")

matches = pd.read_csv(
    data_dir / "worldcup2026_prepared.csv",
    parse_dates=["Date"]
)

rankings = pd.read_csv(
    data_dir / "fifa_rankings_april2026.csv"
)

print("World Cup matches:", len(matches))
print("FIFA ranking records:", len(rankings))

# VALIDATE THE DATA

required_match_columns = [
    "match_id", "Date", "Round",
    "team_a", "team_b",
    "team_a_goals", "team_b_goals",
    "goal_difference"
]

assert all(
    col in matches.columns
    for col in required_match_columns
), "Missing required match columns"

assert len(matches) == 104
assert matches["match_id"].is_unique

assert {"team", "ranking"}.issubset(rankings.columns)

assert rankings["team"].is_unique
assert rankings["ranking"].notna().all()

teams = set(matches["team_a"]) | set(matches["team_b"])
ranked_teams = set(rankings["team"])

missing_teams = teams - ranked_teams

if missing_teams:
    raise ValueError(
        f"Missing FIFA rankings: {sorted(missing_teams)}"
    )

ranking_map = dict(
    zip(rankings["team"], rankings["ranking"])
)

print("PASS: Datasets validated")

# SORT MATCHES BY DATE

matches = matches.sort_values(
    ["Date", "match_id"],
    kind="stable"
).reset_index(drop=True)

# PREPARE TEAM HISTORY

team_history = defaultdict(list)

host_countries = {"Mexico", "Canada", "USA"}


def get_previous_stats(team, match_date):
    """
    Calculate team statistics using only
    previously completed World Cup matches.
    """

    # Only matches before the current date
    previous = [
        game for game in team_history[team]
        if game["date"] < match_date
    ]

    # Use up to five previous matches
    previous = previous[-5:]

    n = len(previous)

    if n == 0:
        return {
            "win_rate": 0.0,
            "avg_scored": 0.0,
            "avg_conceded": 0.0,
            "clean_sheet_rate": 0.0,
            "rest_days": None
        }

    wins = sum(
        game["gf"] > game["ga"]
        for game in previous
    )

    goals_scored = sum(
        game["gf"] for game in previous
    )

    goals_conceded = sum(
        game["ga"] for game in previous
    )

    clean_sheets = sum(
        game["ga"] == 0 for game in previous
    )

    rest_days = (
        match_date - previous[-1]["date"]
    ).days

    return {
        "win_rate": wins / n,
        "avg_scored": goals_scored / n,
        "avg_conceded": goals_conceded / n,
        "clean_sheet_rate": clean_sheets / n,
        "rest_days": rest_days
    }

# CALCULATE EIGHT FEATURES

records = []

for _, match in matches.iterrows():

    team_a = match["team_a"]
    team_b = match["team_b"]
    match_date = match["Date"]

    stats_a = get_previous_stats(team_a, match_date)
    stats_b = get_previous_stats(team_b, match_date)

    # For first tournament appearances,
    # rest-day difference uses a zero baseline.
    if (
        stats_a["rest_days"] is None
        or stats_b["rest_days"] is None
    ):
        rest_diff = 0
    else:
        rest_diff = (
            stats_a["rest_days"]
            - stats_b["rest_days"]
        )

    row = {
        "match_id": match["match_id"],

        # X1: FIFA ranking difference
        "ranking_diff": (
            ranking_map[team_a]
            - ranking_map[team_b]
        ),

        # X2: Recent win-rate difference
        "win_rate_diff": (
            stats_a["win_rate"]
            - stats_b["win_rate"]
        ),

        # X3: Recent goals-scored difference
        "avg_goals_scored_diff": (
            stats_a["avg_scored"]
            - stats_b["avg_scored"]
        ),

        # X4: Recent goals-conceded difference
        "avg_goals_conceded_diff": (
            stats_a["avg_conceded"]
            - stats_b["avg_conceded"]
        ),

        # X5: Clean-sheet-rate difference
        "clean_sheet_rate_diff": (
            stats_a["clean_sheet_rate"]
            - stats_b["clean_sheet_rate"]
        ),

        # X6: Rest-days difference
        "rest_days_diff": rest_diff,

        # X7: Host-country advantage
        "host_advantage": (
            int(team_a in host_countries)
            - int(team_b in host_countries)
        ),

        # X8: Knockout-stage indicator
        "knockout_stage": int(
            match["Round"] != "Group stage"
        ),

        # Target variable
        "goal_difference": match["goal_difference"]
    }

    records.append(row)

    # UPDATE HISTORY AFTER FEATURE CREATION


    team_history[team_a].append({
        "date": match_date,
        "gf": match["team_a_goals"],
        "ga": match["team_b_goals"]
    })

    team_history[team_b].append({
        "date": match_date,
        "gf": match["team_b_goals"],
        "ga": match["team_a_goals"]
    })

# CREATE FINAL DATASET

final_df = pd.DataFrame(records)

feature_columns = [
    "ranking_diff",
    "win_rate_diff",
    "avg_goals_scored_diff",
    "avg_goals_conceded_diff",
    "clean_sheet_rate_diff",
    "rest_days_diff",
    "host_advantage",
    "knockout_stage"
]


# VALIDATE EIGHT FEATURES

assert len(final_df) == 104, "Incorrect match count"

assert len(feature_columns) == 8, (
    "Exactly eight explanatory variables required"
)

assert final_df["match_id"].is_unique

assert final_df[
    feature_columns + ["goal_difference"]
].notna().all().all(), "Missing model values found"

print("\nPASS: 104 matches")
print("PASS: Exactly 8 explanatory variables")
print("PASS: No missing predictor values")


# SAVE FINAL REGRESSION DATASET

output_file = (
    data_dir / "worldcup2026_regression_2_1.csv"
)

final_df.to_csv(output_file, index=False)

print("\n===== FEATURE ENGINEERING COMPLETED =====")
print("Dataset shape:", final_df.shape)
print("Saved file:", output_file)

print("\nFirst five records:")
print(final_df.head().to_string(index=False))
