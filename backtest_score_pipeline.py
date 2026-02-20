#!/usr/bin/env python3
import argparse
import statistics

import numpy as np

from ncaa_predict.score_pipeline import (
    build_score_dataset,
    choose_ensemble_weight,
    evaluate_models,
    fit_platt_scaler,
    fit_ridge_regression,
)


def fold(train_years, validation_year, test_year, stats_year_offset, ridge_l2, max_bad_ratio):
    x_train, y_a_train, y_b_train, _ = build_score_dataset(
        train_years, stats_year_offset=stats_year_offset, max_bad_ratio=max_bad_ratio)
    x_val, y_a_val, y_b_val, _ = build_score_dataset(
        [validation_year], stats_year_offset=stats_year_offset, max_bad_ratio=max_bad_ratio)
    x_test, y_a_test, y_b_test, _ = build_score_dataset(
        [test_year], stats_year_offset=stats_year_offset, max_bad_ratio=max_bad_ratio)

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
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--min-train-years", type=int, default=6)
    parser.add_argument("--stats-year-offset", type=int, default=-1)
    parser.add_argument("--ridge-l2", type=float, default=1.0)
    parser.add_argument("--max-bad-ratio", type=float, default=0.06)
    args = parser.parse_args()

    first_val = args.start_year + args.min_train_years
    if first_val + 1 > args.end_year:
        raise ValueError("Not enough years for rolling train/val/test")

    fold_metrics = []
    for val_year in range(first_val, args.end_year):
        test_year = val_year + 1
        train_years = list(range(args.start_year, val_year))
        w, val_m, test_m = fold(
            train_years, val_year, test_year,
            stats_year_offset=args.stats_year_offset,
            ridge_l2=args.ridge_l2,
            max_bad_ratio=args.max_bad_ratio)
        print(
            "Fold train=%s-%s val=%s test=%s w=%.2f test_acc=%.4f test_brier=%.4f test_mae=%.4f" % (
                min(train_years), max(train_years), val_year, test_year, w,
                test_m["ensemble"]["winner_accuracy"],
                test_m["ensemble"]["brier"],
                test_m["ensemble"]["score_mae"]))
        fold_metrics.append(test_m["ensemble"])

    accs = [m["winner_accuracy"] for m in fold_metrics]
    briers = [m["brier"] for m in fold_metrics]
    maes = [m["score_mae"] for m in fold_metrics]
    lls = [m["log_loss"] for m in fold_metrics]
    print("Backtest summary:")
    print("folds=%s" % len(fold_metrics))
    print("accuracy mean=%.5f std=%.5f" % (statistics.mean(accs), statistics.pstdev(accs)))
    print("brier mean=%.5f std=%.5f" % (statistics.mean(briers), statistics.pstdev(briers)))
    print("log_loss mean=%.5f std=%.5f" % (statistics.mean(lls), statistics.pstdev(lls)))
    print("score_mae mean=%.5f std=%.5f" % (statistics.mean(maes), statistics.pstdev(maes)))
