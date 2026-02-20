#!/usr/bin/env python3
import argparse
import json


def fail(msg):
    print("FAIL:", msg)
    raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensitivity-report", required=True)
    parser.add_argument("--min-folds", type=int, default=3)
    parser.add_argument("--min-accuracy", type=float, default=0.63)
    parser.add_argument("--min-upset-recall", type=float, default=0.15)
    parser.add_argument("--max-missing-years", type=int, default=0)
    parser.add_argument("--data-inventory", default=None)
    args = parser.parse_args()

    with open(args.sensitivity_report) as f:
        rep = json.load(f)
    grid = rep.get("grid_summary", [])
    if not grid:
        fail("No grid_summary rows in sensitivity report")

    best = sorted(grid, key=lambda r: (r.get("accuracy_mean", 0.0), r.get("upset_recall_mean", 0.0)), reverse=True)[0]
    if int(best.get("folds", 0)) < int(args.min_folds):
        fail("Best config folds %s < min_folds %s" % (best.get("folds"), args.min_folds))
    if float(best.get("accuracy_mean", 0.0)) < float(args.min_accuracy):
        fail("Best accuracy %.4f < %.4f" % (best.get("accuracy_mean", 0.0), args.min_accuracy))
    if float(best.get("upset_recall_mean", 0.0) or 0.0) < float(args.min_upset_recall):
        fail("Best upset_recall %.4f < %.4f" % (best.get("upset_recall_mean", 0.0) or 0.0, args.min_upset_recall))

    if args.data_inventory:
        with open(args.data_inventory) as f:
            inv = json.load(f)
        missing = len(inv.get("missing_games", [])) + len(inv.get("missing_players", []))
        if missing > int(args.max_missing_years):
            fail("Missing year count %s > allowed %s" % (missing, args.max_missing_years))

    print("PASS: release gates satisfied")
    print("Selected best config:", best.get("config_key"))
