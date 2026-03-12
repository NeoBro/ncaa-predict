#!/usr/bin/env python3
import argparse
import json
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build query-year subsets for regular season and conference tournaments."
    )
    parser.add_argument("--year", type=int, required=True, help="Query season year, e.g. 2026")
    parser.add_argument(
        "--in-csv",
        default=None,
        help="Input games CSV. Defaults to csv/ncaa_games_<year>.csv if present, else csv/ncaa_games_all.csv.",
    )
    parser.add_argument("--out-dir", default="csv/subsets", help="Output directory")
    parser.add_argument(
        "--regular-end",
        default=None,
        help="Regular season end date (YYYY-MM-DD). Default: <year>-02-28",
    )
    parser.add_argument(
        "--conf-start",
        default=None,
        help="Conference tournament start date (YYYY-MM-DD). Default: <year>-03-01",
    )
    parser.add_argument(
        "--conf-end",
        default=None,
        help="Conference tournament end date (YYYY-MM-DD). Default: <year>-03-15",
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Optional summary JSON output path",
    )
    return parser.parse_args()


def resolve_input_csv(year, in_csv):
    if in_csv:
        return in_csv
    year_file = f"csv/ncaa_games_{year}.csv"
    if os.path.exists(year_file):
        return year_file
    return "csv/ncaa_games_all.csv"


def main():
    args = parse_args()
    in_csv = resolve_input_csv(args.year, args.in_csv)
    regular_end = pd.Timestamp(args.regular_end or f"{args.year}-02-28")
    conf_start = pd.Timestamp(args.conf_start or f"{args.year}-03-01")
    conf_end = pd.Timestamp(args.conf_end or f"{args.year}-03-15")
    if conf_end < conf_start:
        raise ValueError("--conf-end must be >= --conf-start")

    df = pd.read_csv(in_csv)
    if "year" not in df.columns or "game_date" not in df.columns:
        raise ValueError("Input CSV must contain 'year' and 'game_date' columns")

    df = df[df["year"] == args.year].copy()
    df["game_date_dt"] = pd.to_datetime(df["game_date"], errors="coerce")
    if df["game_date_dt"].isna().any():
        bad = int(df["game_date_dt"].isna().sum())
        raise ValueError(f"Found {bad} rows with invalid game_date values")

    regular = df[df["game_date_dt"] <= regular_end].copy()
    conf = df[(df["game_date_dt"] >= conf_start) & (df["game_date_dt"] <= conf_end)].copy()

    regular = regular.drop(columns=["game_date_dt"])
    conf = conf.drop(columns=["game_date_dt"])

    os.makedirs(args.out_dir, exist_ok=True)
    regular_path = os.path.join(args.out_dir, f"ncaa_games_{args.year}_regular_season.csv")
    conf_path = os.path.join(args.out_dir, f"ncaa_games_{args.year}_conference_tourney.csv")
    regular.to_csv(regular_path, index=False)
    conf.to_csv(conf_path, index=False)

    summary = {
        "year": args.year,
        "input_csv": in_csv,
        "regular_end": regular_end.strftime("%Y-%m-%d"),
        "conference_start": conf_start.strftime("%Y-%m-%d"),
        "conference_end": conf_end.strftime("%Y-%m-%d"),
        "rows_regular_season": int(len(regular)),
        "rows_conference_tourney": int(len(conf)),
        "regular_out": regular_path,
        "conference_out": conf_path,
    }

    if args.summary_out:
        out_dir = os.path.dirname(args.summary_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.summary_out, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
