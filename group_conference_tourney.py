#!/usr/bin/env python3
import argparse
import os

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Group conference-tournament games by conference using a team->conference mapping."
    )
    p.add_argument("--year", "-y", type=int, required=True)
    p.add_argument(
        "--games-csv",
        default=None,
        help="Defaults to csv/subsets/ncaa_games_<year>_conference_tourney.csv",
    )
    p.add_argument(
        "--conference-map-csv",
        default="csv/team_conferences.csv",
        help="CSV with columns: school_id,conference (optional season)",
    )
    p.add_argument("--out-csv", default=None)
    p.add_argument("--summary-out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    year = int(args.year)
    games_csv = args.games_csv or f"csv/subsets/ncaa_games_{year}_conference_tourney.csv"
    out_csv = args.out_csv or f"csv/subsets/ncaa_games_{year}_conference_tourney_grouped.csv"
    summary_out = args.summary_out or f"reports/conference_tourney_{year}_summary.csv"

    if not os.path.exists(games_csv):
        raise FileNotFoundError(f"Missing games CSV: {games_csv}")
    if not os.path.exists(args.conference_map_csv):
        raise FileNotFoundError(
            f"Missing conference map CSV: {args.conference_map_csv}. "
            "Create it with columns school_id,conference (optional season)."
        )

    games = pd.read_csv(games_csv)
    conf = pd.read_csv(args.conference_map_csv)
    cols = {c.lower(): c for c in conf.columns}
    if "school_id" not in cols or "conference" not in cols:
        raise ValueError("conference map must include columns: school_id, conference")

    sid_col = cols["school_id"]
    conf_col = cols["conference"]
    season_col = cols.get("season")
    use = conf[[sid_col, conf_col] + ([season_col] if season_col else [])].copy()
    use = use.rename(columns={sid_col: "school_id", conf_col: "conference"})
    if season_col:
        use = use.rename(columns={season_col: "season"})
        use = use[(use["season"] == year) | (use["season"].isna())].copy()

    # Attach conference for both teams.
    a = use[["school_id", "conference"]].rename(columns={"conference": "team_conf"})
    b = use[["school_id", "conference"]].rename(
        columns={"school_id": "opponent_id", "conference": "opponent_conf"}
    )
    out = games.merge(a, on="school_id", how="left").merge(b, on="opponent_id", how="left")

    # Keep intra-conference games; mark unknown/mixed for auditing.
    out["conference_group"] = out["team_conf"]
    out.loc[out["team_conf"] != out["opponent_conf"], "conference_group"] = "MIXED_OR_UNKNOWN"
    out.to_csv(out_csv, index=False)

    summary = (
        out.groupby("conference_group", dropna=False)
        .size()
        .reset_index(name="games")
        .sort_values("games", ascending=False)
    )
    os.makedirs(os.path.dirname(summary_out), exist_ok=True)
    summary.to_csv(summary_out, index=False)

    print(f"Wrote grouped games: {out_csv} rows={len(out)}")
    print(f"Wrote summary: {summary_out}")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
