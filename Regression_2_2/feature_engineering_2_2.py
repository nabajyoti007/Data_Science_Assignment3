"""
Linear Regression 2.2 - Feature engineering
Builds a 208-row dataset (one row per team per match) to predict
the number of goals a team scores in a FIFA World Cup 2026 match.

All eight explanatory variables use information available BEFORE kick-off.
"""

import pandas as pd
from collections import defaultdict
from pathlib import Path

data_dir = Path("data")

# ==========================================
# STEP 1: LOAD DATA
# ==========================================
matches = pd.read_csv(data_dir / "worldcup2026_prepared.csv", parse_dates=["Date"])
rankings = pd.read_csv(data_dir / "fifa_rankings_april2026.csv", encoding="utf-8-sig")
ranking_map = dict(zip(rankings["team"], rankings["ranking"]))

assert len(matches) == 104

# ==========================================
# STEP 2: STATIC PRE-TOURNAMENT TEAM INFORMATION
# ==========================================
# Confederation of each team. UEFA and CONMEBOL are treated as the
# "top" confederations: between them they have won every World Cup.
confederation = {
    "Algeria": "CAF", "Argentina": "CONMEBOL", "Australia": "AFC", "Austria": "UEFA",
    "Belgium": "UEFA", "Bosnia–Herz": "UEFA", "Brazil": "CONMEBOL", "Cabo Verde": "CAF",
    "Canada": "CONCACAF", "Colombia": "CONMEBOL", "Congo DR": "CAF", "Croatia": "UEFA",
    "Curaçao": "CONCACAF", "Czechia": "UEFA", "Côte d'Ivoire": "CAF", "Ecuador": "CONMEBOL",
    "Egypt": "CAF", "England": "UEFA", "France": "UEFA", "Germany": "UEFA",
    "Ghana": "CAF", "Haiti": "CONCACAF", "IR Iran": "AFC", "Iraq": "AFC",
    "Japan": "AFC", "Jordan": "AFC", "Korea Republic": "AFC", "Mexico": "CONCACAF",
    "Morocco": "CAF", "Netherlands": "UEFA", "New Zealand": "OFC", "Norway": "UEFA",
    "Panama": "CONCACAF", "Paraguay": "CONMEBOL", "Portugal": "UEFA", "Qatar": "AFC",
    "Saudi Arabia": "AFC", "Scotland": "UEFA", "Senegal": "CAF", "South Africa": "CAF",
    "Spain": "UEFA", "Sweden": "UEFA", "Switzerland": "UEFA", "Tunisia": "CAF",
    "Türkiye": "UEFA", "USA": "CONCACAF", "Uruguay": "CONMEBOL", "Uzbekistan": "AFC",
}
TOP_CONFEDS = {"UEFA", "CONMEBOL"}

# Number of previous FIFA World Cup finals appearances (1930-2022).
# FIFA's successor rules apply: Germany includes West Germany,
# Czechia includes Czechoslovakia, Congo DR includes Zaire.
wc_appearances = {
    "Algeria": 4, "Argentina": 18, "Australia": 6, "Austria": 7, "Belgium": 14,
    "Bosnia–Herz": 1, "Brazil": 22, "Cabo Verde": 0, "Canada": 2, "Colombia": 6,
    "Congo DR": 1, "Croatia": 6, "Curaçao": 0, "Czechia": 9, "Côte d'Ivoire": 3,
    "Ecuador": 4, "Egypt": 3, "England": 16, "France": 16, "Germany": 20,
    "Ghana": 4, "Haiti": 1, "IR Iran": 6, "Iraq": 1, "Japan": 7, "Jordan": 0,
    "Korea Republic": 11, "Mexico": 17, "Morocco": 6, "Netherlands": 11,
    "New Zealand": 2, "Norway": 3, "Panama": 1, "Paraguay": 8, "Portugal": 8,
    "Qatar": 1, "Saudi Arabia": 6, "Scotland": 8, "Senegal": 3, "South Africa": 3,
    "Spain": 16, "Sweden": 12, "Switzerland": 12, "Tunisia": 6, "Türkiye": 2,
    "USA": 11, "Uruguay": 14, "Uzbekistan": 0,
}

teams = set(matches["team_a"]) | set(matches["team_b"])
for name, mapping in [("ranking", ranking_map), ("confederation", confederation),
                      ("wc_appearances", wc_appearances)]:
    missing = teams - set(mapping)
    assert not missing, f"Missing {name} for {sorted(missing)}"
assert len(teams) == 48

# ==========================================
# STEP 3: ROLLING TOURNAMENT FORM (NO LEAKAGE)
# ==========================================
# Goals per team per match at the 2022 World Cup (172 goals / 128 team-matches).
# Used as the neutral starting value before a team has played in 2026.
PRIOR_GOALS = 172 / 128

matches = matches.sort_values(["Date", "match_id"], kind="stable").reset_index(drop=True)
history = defaultdict(list)  # team -> list of dicts {date, gf, ga}


def prior_average(team, date, key):
    """Average of `key` over the team's matches played strictly before `date`."""
    prev = [g[key] for g in history[team] if g["date"] < date]
    return sum(prev) / len(prev) if prev else PRIOR_GOALS


records = []
for _, m in matches.iterrows():
    sides = [
        (m["team_a"], m["team_b"], m["team_a_goals"]),
        (m["team_b"], m["team_a"], m["team_b_goals"]),
    ]
    for team, opp, goals in sides:
        records.append({
            "row_id": len(records) + 1,
            "match_id": m["match_id"],
            "date": m["Date"].date(),
            "round": m["Round"],
            "team": team,
            "opponent": opp,
            # X1: team's FIFA ranking (lower = stronger)
            "team_fifa_rank": ranking_map[team],
            # X2: opponent's FIFA ranking
            "opp_fifa_rank": ranking_map[opp],
            # X3: team's average goals scored in earlier 2026 WC matches
            "team_avg_goals_scored": round(prior_average(team, m["Date"], "gf"), 4),
            # X4: opponent's average goals conceded in earlier 2026 WC matches
            "opp_avg_goals_conceded": round(prior_average(opp, m["Date"], "ga"), 4),
            # X5: team is from UEFA or CONMEBOL
            "team_top_confed": int(confederation[team] in TOP_CONFEDS),
            # X6: opponent is from UEFA or CONMEBOL
            "opp_top_confed": int(confederation[opp] in TOP_CONFEDS),
            # X7: team's previous World Cup appearances
            "team_wc_appearances": wc_appearances[team],
            # X8: opponent's previous World Cup appearances
            "opp_wc_appearances": wc_appearances[opp],
            # Target
            "goals_scored": int(goals),
        })

    # Update history only AFTER both rows for this match are built
    history[m["team_a"]].append({"date": m["Date"], "gf": m["team_a_goals"], "ga": m["team_b_goals"]})
    history[m["team_b"]].append({"date": m["Date"], "gf": m["team_b_goals"], "ga": m["team_a_goals"]})

df = pd.DataFrame(records)

FEATURES = [
    "team_fifa_rank", "opp_fifa_rank",
    "team_avg_goals_scored", "opp_avg_goals_conceded",
    "team_top_confed", "opp_top_confed",
    "team_wc_appearances", "opp_wc_appearances",
]

# ==========================================
# STEP 4: VALIDATE
# ==========================================
assert len(df) == 208, "Expected 208 team-match rows"
assert len(FEATURES) == 8
assert df["row_id"].is_unique
assert (df.groupby("match_id").size() == 2).all()
assert not df[FEATURES + ["goals_scored"]].isnull().any().any()
assert df["goals_scored"].sum() == (matches["team_a_goals"] + matches["team_b_goals"]).sum()

out = data_dir / "worldcup2026_regression_2_2.csv"
df.to_csv(out, index=False)
print("PASS: 208 rows, 8 explanatory variables, no missing values")
print("Saved:", out)
print(df.head(6).to_string(index=False))
