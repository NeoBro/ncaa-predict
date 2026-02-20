#!/usr/bin/env python3
import argparse
import json
import os

from ncaa_predict.tourney_pipeline import (
    build_kaggle_to_ncaa_id_map,
    load_kaggle_tourney,
    validate_team_id_alignment,
)
from ncaa_predict.util import list_arg
from ncaa_predict.data_loader import load_ncaa_schools


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-dir", required=True)
    parser.add_argument("--all-games-csv", required=True)
    parser.add_argument(
        "--years", "-y", required=True,
        type=list_arg(type=int, container=list),
        help="Comma-separated tournament seasons to validate.")
    parser.add_argument("--stats-year-offset", type=int, default=-1)
    parser.add_argument("--min-id-coverage", type=float, default=0.98)
    parser.add_argument("--report-out", default=None)
    args = parser.parse_args()

    kaggle_games = load_kaggle_tourney(args.kaggle_dir)
    schools = load_ncaa_schools()
    mapping, unresolved, ambiguous = build_kaggle_to_ncaa_id_map(
        args.kaggle_dir, schools)
    print("Mapped Kaggle TeamIDs: %s" % len(mapping))
    print("Unresolved Kaggle TeamIDs: %s" % len(unresolved))
    print("Ambiguous Kaggle TeamIDs: %s" % len(ambiguous))
    report = validate_team_id_alignment(
        kaggle_games, args.all_games_csv, args.years,
        stats_year_offset=args.stats_year_offset,
        kaggle_to_ncaa_id_map=mapping)

    print("TeamID alignment report:")
    for row in report:
        print(
            "season=%s stats_year=%s coverage=%.4f missing=%s/%s" % (
                row["season"], row["stats_year"], row["coverage"],
                row["missing_team_count"], row["kaggle_team_count"]))

    bad = [r for r in report if r["coverage"] < args.min_id_coverage]
    if args.report_out:
        out_dir = os.path.dirname(args.report_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.report_out, "w") as f:
            json.dump({
                "alignment_report": report,
                "mapping_summary": {
                    "mapped_count": len(mapping),
                    "unresolved_count": len(unresolved),
                    "ambiguous_count": len(ambiguous),
                },
                "unresolved_examples": dict(list(unresolved.items())[:50]),
                "ambiguous_examples": dict(list(ambiguous.items())[:50]),
            }, f, indent=2, sort_keys=True)
        print("Wrote report to %s" % args.report_out)

    if bad:
        print("FAIL: coverage below threshold %.4f in seasons: %s" % (
            args.min_id_coverage, [r["season"] for r in bad]))
        raise SystemExit(2)
    print("PASS: all seasons meet minimum coverage %.4f" % args.min_id_coverage)
