#!/usr/bin/env python3
import argparse
import json
import os
import statistics

import numpy as np

from ncaa_predict.data_loader import load_ncaa_schools
from ncaa_predict.score_pipeline import (
    choose_ensemble_weight,
    evaluate_models,
    fit_platt_scaler,
    fit_ridge_regression,
)
from ncaa_predict.tourney_pipeline import (
    build_kaggle_to_ncaa_id_map,
    build_tourney_feature_dataset,
    load_kaggle_tourney,
    validate_team_id_alignment,
)


def run_fold(
    kaggle_games,
    all_games_csv,
    train_years,
    validation_year,
    test_year,
    stats_year_offset,
    ridge_l2,
    kaggle_to_ncaa,
):
    x_train, y_a_train, y_b_train, _ = build_tourney_feature_dataset(
        kaggle_games, all_games_csv, train_years,
        stats_year_offset=stats_year_offset,
        kaggle_to_ncaa_id_map=kaggle_to_ncaa)
    x_val, y_a_val, y_b_val, _ = build_tourney_feature_dataset(
        kaggle_games, all_games_csv, [validation_year],
        stats_year_offset=stats_year_offset,
        kaggle_to_ncaa_id_map=kaggle_to_ncaa)
    x_test, y_a_test, y_b_test, _ = build_tourney_feature_dataset(
        kaggle_games, all_games_csv, [test_year],
        stats_year_offset=stats_year_offset,
        kaggle_to_ncaa_id_map=kaggle_to_ncaa)

    model_a = fit_ridge_regression(x_train, y_a_train, l2=ridge_l2)
    model_b = fit_ridge_regression(x_train, y_b_train, l2=ridge_l2)
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
    test_metrics, _, _ = evaluate_models(
        x_test, y_a_test, y_b_test, model_a, model_b, w,
        calibration={"a": cal_a, "b": cal_b})
    return w, val_metrics, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-dir", required=True)
    parser.add_argument("--all-games-csv", required=True)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--min-train-years", type=int, default=6)
    parser.add_argument("--stats-year-offset", type=int, default=-1)
    parser.add_argument("--ridge-l2", type=float, default=1.0)
    parser.add_argument("--min-id-coverage", type=float, default=0.98)
    parser.add_argument("--report-out", default=None)
    args = parser.parse_args()

    if args.end_year <= args.start_year:
        raise ValueError("end-year must be greater than start-year")

    kaggle_games = load_kaggle_tourney(args.kaggle_dir)
    schools = load_ncaa_schools()
    kaggle_to_ncaa, unresolved, ambiguous = build_kaggle_to_ncaa_id_map(
        args.kaggle_dir, schools)
    print("Mapped Kaggle TeamIDs: %s" % len(kaggle_to_ncaa))
    print("Unresolved Kaggle TeamIDs: %s" % len(unresolved))
    print("Ambiguous Kaggle TeamIDs: %s" % len(ambiguous))

    years = list(range(args.start_year, args.end_year + 1))
    alignment = validate_team_id_alignment(
        kaggle_games, args.all_games_csv, years,
        stats_year_offset=args.stats_year_offset,
        kaggle_to_ncaa_id_map=kaggle_to_ncaa)
    bad = [r for r in alignment if r["coverage"] < args.min_id_coverage]
    if bad:
        raise ValueError(
            "TeamID alignment below threshold %.4f for seasons: %s" % (
                args.min_id_coverage, [r["season"] for r in bad]))

    first_val = args.start_year + args.min_train_years
    if first_val + 1 > args.end_year:
        raise ValueError("Not enough years for rolling train/val/test")

    folds = []
    for val_year in range(first_val, args.end_year):
        test_year = val_year + 1
        train_years = list(range(args.start_year, val_year))
        w, val_m, test_m = run_fold(
            kaggle_games, args.all_games_csv, train_years, val_year, test_year,
            stats_year_offset=args.stats_year_offset,
            ridge_l2=args.ridge_l2,
            kaggle_to_ncaa=kaggle_to_ncaa)
        print(
            "Fold train=%s-%s val=%s test=%s w=%.2f test_acc=%.4f "
            "test_brier=%.4f test_logloss=%.4f test_mae=%.4f" % (
                min(train_years), max(train_years), val_year, test_year, w,
                test_m["ensemble"]["winner_accuracy"],
                test_m["ensemble"]["brier"],
                test_m["ensemble"]["log_loss"],
                test_m["ensemble"]["score_mae"]))
        folds.append({
            "train_years": train_years,
            "validation_year": val_year,
            "test_year": test_year,
            "ensemble_weight": w,
            "validation_metrics": val_m["ensemble"],
            "test_metrics": test_m["ensemble"],
        })

    accs = [f["test_metrics"]["winner_accuracy"] for f in folds]
    briers = [f["test_metrics"]["brier"] for f in folds]
    lls = [f["test_metrics"]["log_loss"] for f in folds]
    maes = [f["test_metrics"]["score_mae"] for f in folds]
    summary = {
        "folds": len(folds),
        "accuracy_mean": statistics.mean(accs),
        "accuracy_std": statistics.pstdev(accs),
        "brier_mean": statistics.mean(briers),
        "brier_std": statistics.pstdev(briers),
        "log_loss_mean": statistics.mean(lls),
        "log_loss_std": statistics.pstdev(lls),
        "score_mae_mean": statistics.mean(maes),
        "score_mae_std": statistics.pstdev(maes),
    }
    print("Backtest summary:")
    print(summary)

    if args.report_out:
        out_dir = os.path.dirname(args.report_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.report_out, "w") as f:
            json.dump({
                "config": {
                    "kaggle_dir": args.kaggle_dir,
                    "all_games_csv": args.all_games_csv,
                    "start_year": args.start_year,
                    "end_year": args.end_year,
                    "min_train_years": args.min_train_years,
                    "stats_year_offset": args.stats_year_offset,
                    "ridge_l2": args.ridge_l2,
                    "min_id_coverage": args.min_id_coverage,
                },
                "mapping_summary": {
                    "mapped_count": len(kaggle_to_ncaa),
                    "unresolved_count": len(unresolved),
                    "ambiguous_count": len(ambiguous),
                },
                "alignment_report": alignment,
                "fold_results": folds,
                "summary": summary,
            }, f, indent=2, sort_keys=True)
        print("Wrote backtest report to %s" % args.report_out)
