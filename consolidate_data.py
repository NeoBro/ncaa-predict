#!/usr/bin/env python3
import argparse
import glob
import json
import os

import pandas as pd


def parse_year(path, prefix):
    name = os.path.basename(path)
    return int(name.replace(prefix, "").replace(".csv", ""))


def merge_yearlies(pattern, prefix, out_path):
    files = sorted(
        p for p in glob.glob(pattern)
        if os.path.basename(p).replace(prefix, "").replace(".csv", "").isdigit()
    )
    if not files:
        raise ValueError("No files matched: %s" % pattern)
    frames = []
    years = []
    for path in files:
        df = pd.read_csv(path)
        years.append(parse_year(path, prefix))
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True).drop_duplicates()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    merged.to_csv(out_path, index=False)
    return years, len(merged)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2002)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--csv-dir", default="csv")
    parser.add_argument("--games-out", default="csv/ncaa_games_all.csv")
    parser.add_argument("--players-out", default="csv/ncaa_players_all.csv")
    parser.add_argument("--inventory-out", default="reports/data_inventory.json")
    parser.add_argument(
        "--delete-shards",
        action="store_true",
        help="Delete per-school shard files in csv/games and csv/players after merge.")
    args = parser.parse_args()

    expected = set(range(args.start_year, args.end_year + 1))
    game_files = sorted(glob.glob(os.path.join(args.csv_dir, "ncaa_games_*.csv")))
    player_files = sorted(glob.glob(os.path.join(args.csv_dir, "ncaa_players_*.csv")))

    game_years = set(
        parse_year(p, "ncaa_games_")
        for p in game_files
        if os.path.basename(p).replace("ncaa_games_", "").replace(".csv", "").isdigit()
    )
    player_years = set(
        parse_year(p, "ncaa_players_")
        for p in player_files
        if os.path.basename(p).replace("ncaa_players_", "").replace(".csv", "").isdigit()
    )
    missing_games = sorted(expected - game_years)
    missing_players = sorted(expected - player_years)

    print("Missing game years:", missing_games if missing_games else "none")
    print("Missing player years:", missing_players if missing_players else "none")

    years_g, rows_g = merge_yearlies(
        os.path.join(args.csv_dir, "ncaa_games_*.csv"),
        "ncaa_games_",
        args.games_out,
    )
    years_p, rows_p = merge_yearlies(
        os.path.join(args.csv_dir, "ncaa_players_*.csv"),
        "ncaa_players_",
        args.players_out,
    )
    print("Merged games years: %s..%s (%s files) -> %s rows" % (
        min(years_g), max(years_g), len(years_g), rows_g))
    print("Merged players years: %s..%s (%s files) -> %s rows" % (
        min(years_p), max(years_p), len(years_p), rows_p))
    print("Wrote:", args.games_out)
    print("Wrote:", args.players_out)

    inventory = {
        "start_year": args.start_year,
        "end_year": args.end_year,
        "missing_games": missing_games,
        "missing_players": missing_players,
        "games_files": len(years_g),
        "players_files": len(years_p),
        "games_rows_merged": rows_g,
        "players_rows_merged": rows_p,
        "games_out": args.games_out,
        "players_out": args.players_out,
    }
    inv_dir = os.path.dirname(args.inventory_out)
    if inv_dir:
        os.makedirs(inv_dir, exist_ok=True)
    with open(args.inventory_out, "w") as f:
        json.dump(inventory, f, indent=2, sort_keys=True)
    print("Wrote:", args.inventory_out)

    if args.delete_shards:
        removed = 0
        for pattern in [
            os.path.join(args.csv_dir, "games", "ncaa_games_*_*.csv"),
            os.path.join(args.csv_dir, "players", "ncaa_players_*_*.csv"),
        ]:
            for path in glob.glob(pattern):
                os.remove(path)
                removed += 1
        print("Deleted shard files:", removed)
