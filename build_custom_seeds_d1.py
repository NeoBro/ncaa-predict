#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd

from ncaa_predict.data_loader import load_ncaa_schools
from ncaa_predict.tourney_pipeline import build_kaggle_to_ncaa_id_map


def parse_args():
    p = argparse.ArgumentParser(
        description="Build custom seeds from query-year regular season, restricted to mapped D-I teams."
    )
    p.add_argument("--year", "-y", type=int, required=True)
    p.add_argument("--kaggle-dir", required=True)
    p.add_argument("--games-csv", default=None, help="Defaults to csv/subsets/ncaa_games_<year>_regular_season.csv")
    p.add_argument("--out-csv", default=None, help="Defaults to brackets/custom_seeds_<year>.csv")
    p.add_argument("--seed-template-year", type=int, default=2023)
    p.add_argument("--min-games", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    year = int(args.year)
    games_csv = args.games_csv or f"csv/subsets/ncaa_games_{year}_regular_season.csv"
    out_csv = args.out_csv or f"brackets/custom_seeds_{year}.csv"

    games = pd.read_csv(games_csv)
    games = games[games["year"] == year].copy()
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    games = games[games["game_date"].notna()].copy()

    schools = load_ncaa_schools()
    kaggle_to_ncaa, unresolved, ambiguous = build_kaggle_to_ncaa_id_map(args.kaggle_dir, schools)
    d1_ids = set(int(v) for v in kaggle_to_ncaa.values())
    if not d1_ids:
        raise ValueError("No mapped D-I IDs found. Check --kaggle-dir and team mappings.")

    # Build team performance table from the regular season subset.
    t = games.assign(
        win=(games["score"] > games["opponent_score"]).astype(int),
        loss=(games["score"] < games["opponent_score"]).astype(int),
        margin=(games["score"] - games["opponent_score"]).astype(float),
    )
    team = (
        t.groupby("school_id", as_index=False)
        .agg(
            gp=("win", "size"),
            wins=("win", "sum"),
            losses=("loss", "sum"),
            mov=("margin", "mean"),
        )
    )
    team["win_pct"] = team["wins"] / team["gp"]

    # Strength proxy: average opponent win%.
    opp_stats = team[["school_id", "win_pct"]].rename(
        columns={"school_id": "opponent_id", "win_pct": "opp_win_pct"}
    )
    with_opp = games[["school_id", "opponent_id"]].merge(opp_stats, on="opponent_id", how="left")
    sos = with_opp.groupby("school_id", as_index=False)["opp_win_pct"].mean().rename(
        columns={"opp_win_pct": "sos"}
    )
    team = team.merge(sos, on="school_id", how="left")
    team["sos"] = team["sos"].fillna(team["sos"].median())

    # Restrict to mapped D-I teams and minimum game count.
    team = team[team["school_id"].isin(d1_ids)].copy()
    team = team[team["gp"] >= int(args.min_games)].copy()

    # Composite score tuned to produce realistic top seeds.
    team["rank_score"] = (
        (team["win_pct"] * 0.55)
        + (team["sos"] * 0.30)
        + ((team["mov"] / 25.0) * 0.15)
    )
    team = team.sort_values(["rank_score", "win_pct", "sos", "mov", "wins"], ascending=False)

    seed_template = pd.read_csv(Path(args.kaggle_dir) / "MNCAATourneySeeds.csv")
    seed_codes = seed_template[seed_template["Season"] == int(args.seed_template_year)]["Seed"].astype(str).tolist()
    if len(seed_codes) != 68:
        raise ValueError(
            "Template season %s has %s seeds; expected 68" % (args.seed_template_year, len(seed_codes))
        )

    schools_df = pd.read_csv("csv/ncaa_schools.csv")[["school_id", "school_name"]]
    ranked = team.merge(schools_df, on="school_id", how="left").dropna(subset=["school_name"]).copy()
    ranked = ranked.drop_duplicates(subset=["school_name"]).reset_index(drop=True)
    if len(ranked) < len(seed_codes):
        raise ValueError("Only %s eligible D-I teams found; need %s" % (len(ranked), len(seed_codes)))

    out = pd.DataFrame(
        {
            "season": year,
            "seed": seed_codes,
            "team_name": ranked.head(len(seed_codes))["school_name"].tolist(),
        }
    )
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    print("Wrote seeds:", out_csv, "rows:", len(out))
    print("Mapped D-I ids:", len(d1_ids), "unresolved_names:", len(unresolved), "ambiguous_names:", len(ambiguous))
    print("Top 16 seeded teams:")
    top = ranked.head(16)[["school_name", "gp", "wins", "losses", "win_pct", "sos", "mov"]]
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
