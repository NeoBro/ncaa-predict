#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np

from ncaa_predict.score_pipeline import (
    FEATURE_COLUMNS,
    choose_ensemble_weight,
    evaluate_models,
    fit_platt_scaler,
    fit_ridge_regression,
    save_pipeline,
)
from ncaa_predict.tourney_pipeline import (
    build_kaggle_to_ncaa_id_map,
    build_tourney_feature_dataset,
    load_kaggle_seeds,
    load_kaggle_tourney,
    validate_team_id_alignment,
)
from ncaa_predict.data_loader import load_ncaa_schools
from ncaa_predict.util import list_arg


def check_split_leakage(train_years, validation_years, test_years):
    train_set = set(train_years)
    val_set = set(validation_years)
    test_set = set(test_years)
    if train_set & val_set:
        raise ValueError("Leakage guard: overlap train/validation")
    if train_set & test_set:
        raise ValueError("Leakage guard: overlap train/test")
    if val_set & test_set:
        raise ValueError("Leakage guard: overlap validation/test")
    if train_years and validation_years and max(train_years) >= min(validation_years):
        raise ValueError("Leakage guard: train years must be before validation years")
    if validation_years and test_years and max(validation_years) >= min(test_years):
        raise ValueError("Leakage guard: validation years must be before test years")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-dir", required=True)
    parser.add_argument(
        "--all-games-csv", default="csv/ncaa_games_2018.csv",
        help="Path to consolidated season game CSVs. "
             "Use a concatenated file if you have one; default points at one file and "
             "is mainly for demonstration.")
    parser.add_argument(
        "--train-years", "-y", required=True,
        type=list_arg(type=int, container=list))
    parser.add_argument(
        "--validation-years", "-v", required=True,
        type=list_arg(type=int, container=list))
    parser.add_argument(
        "--test-years", "-t", default=[],
        type=list_arg(type=int, container=list))
    parser.add_argument("--stats-year-offset", type=int, default=-1)
    parser.add_argument("--ridge-l2", type=float, default=1.0)
    parser.add_argument(
        "--min-id-coverage", type=float, default=0.98,
        help="Minimum per-season TeamID coverage between Kaggle tournament teams "
             "and season stats IDs. (default: %(default)s)")
    parser.add_argument(
        "--alignment-report-out", default=None,
        help="Optional path to write TeamID alignment report JSON.")
    parser.add_argument("--model-out", "-o", required=True)
    args = parser.parse_args()

    check_split_leakage(args.train_years, args.validation_years, args.test_years)

    kaggle_games = load_kaggle_tourney(args.kaggle_dir)
    seeds = load_kaggle_seeds(args.kaggle_dir)
    if seeds is not None:
        print("Loaded tournament seeds: %s rows" % len(seeds))
    schools = load_ncaa_schools()
    kaggle_to_ncaa, unresolved, ambiguous = build_kaggle_to_ncaa_id_map(
        args.kaggle_dir, schools)
    print("Mapped Kaggle TeamIDs: %s" % len(kaggle_to_ncaa))
    print("Unresolved Kaggle TeamIDs: %s" % len(unresolved))
    print("Ambiguous Kaggle TeamIDs: %s" % len(ambiguous))

    used_years = sorted(set(args.train_years + args.validation_years + args.test_years))
    alignment_report = validate_team_id_alignment(
        kaggle_games, args.all_games_csv, used_years,
        stats_year_offset=args.stats_year_offset,
        kaggle_to_ncaa_id_map=kaggle_to_ncaa)
    min_coverage = min(r["coverage"] for r in alignment_report) if alignment_report else 1.0
    print("TeamID alignment minimum coverage: %.4f" % min_coverage)
    bad = [r for r in alignment_report if r["coverage"] < args.min_id_coverage]
    if bad:
        raise ValueError(
            "TeamID alignment below threshold %.4f for seasons: %s" % (
                args.min_id_coverage, [r["season"] for r in bad]))
    if args.alignment_report_out:
        out_dir = os.path.dirname(args.alignment_report_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.alignment_report_out, "w") as f:
            json.dump(alignment_report, f, indent=2, sort_keys=True)
        print("Wrote TeamID alignment report to %s" % args.alignment_report_out)

    x_train, y_a_train, y_b_train, _ = build_tourney_feature_dataset(
        kaggle_games, args.all_games_csv, args.train_years, args.stats_year_offset,
        kaggle_to_ncaa_id_map=kaggle_to_ncaa)
    x_val, y_a_val, y_b_val, _ = build_tourney_feature_dataset(
        kaggle_games, args.all_games_csv, args.validation_years, args.stats_year_offset,
        kaggle_to_ncaa_id_map=kaggle_to_ncaa)

    model_a = fit_ridge_regression(x_train, y_a_train, l2=args.ridge_l2)
    model_b = fit_ridge_regression(x_train, y_b_train, l2=args.ridge_l2)
    w = choose_ensemble_weight(x_val, y_a_val, y_b_val, model_a, model_b)

    val_base_a, val_base_b = ((0.5 * (x_val[:, 0] + x_val[:, 5])),
                              (0.5 * (x_val[:, 4] + x_val[:, 1])))
    val_ridge_a = model_a.predict(x_val)
    val_ridge_b = model_b.predict(x_val)
    val_ens_diff = (
        (w * val_ridge_a) + ((1 - w) * val_base_a)
    ) - (
        (w * val_ridge_b) + ((1 - w) * val_base_b)
    )
    y_val_win = (y_a_val > y_b_val).astype(np.float32)
    cal_a, cal_b = fit_platt_scaler(val_ens_diff, y_val_win)

    val_metrics, _, _ = evaluate_models(
        x_val, y_a_val, y_b_val, model_a, model_b, w,
        calibration={"a": cal_a, "b": cal_b})
    print("Tournament validation metrics:")
    print(val_metrics)

    test_metrics = {}
    if args.test_years:
        x_test, y_a_test, y_b_test, _ = build_tourney_feature_dataset(
            kaggle_games, args.all_games_csv, args.test_years, args.stats_year_offset,
            kaggle_to_ncaa_id_map=kaggle_to_ncaa)
        test_metrics, _, _ = evaluate_models(
            x_test, y_a_test, y_b_test, model_a, model_b, w,
            calibration={"a": cal_a, "b": cal_b})
        print("Tournament test metrics:")
        print(test_metrics)

    out_dir = os.path.dirname(args.model_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = {
        "config": {
            "kaggle_dir": args.kaggle_dir,
            "all_games_csv": args.all_games_csv,
            "train_years": args.train_years,
            "validation_years": args.validation_years,
            "test_years": args.test_years,
            "stats_year_offset": args.stats_year_offset,
            "ridge_l2": args.ridge_l2,
            "label_source": "MNCAATourneyCompactResults",
            "min_id_coverage": args.min_id_coverage,
        },
        "mapping_summary": {
            "mapped_count": len(kaggle_to_ncaa),
            "unresolved_count": len(unresolved),
            "ambiguous_count": len(ambiguous),
        },
        "alignment_report": alignment_report,
        "feature_columns": FEATURE_COLUMNS,
        "ensemble_weight": w,
        "ridge": {
            "a_coef": [float(v) for v in model_a.coef.tolist()],
            "a_intercept": float(model_a.intercept),
            "b_coef": [float(v) for v in model_b.coef.tolist()],
            "b_intercept": float(model_b.intercept),
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "calibration": {"method": "platt", "a": cal_a, "b": cal_b},
    }
    save_pipeline(args.model_out, payload)
    print("Saved tournament score pipeline to %s" % args.model_out)
