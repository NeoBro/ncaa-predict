#!/usr/bin/env python3
import argparse
import os

import numpy as np

from ncaa_predict.score_pipeline import (
    FEATURE_COLUMNS,
    build_score_dataset,
    choose_ensemble_weight,
    evaluate_models,
    fit_ridge_regression,
    save_pipeline,
)
from ncaa_predict.util import list_arg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-years", "-y", required=True,
        type=list_arg(type=int, container=list),
        help="Comma-separated game years for training.")
    parser.add_argument(
        "--validation-years", "-v", required=True,
        type=list_arg(type=int, container=list),
        help="Comma-separated game years for validation.")
    parser.add_argument(
        "--test-years", "-t", default=[],
        type=list_arg(type=int, container=list),
        help="Comma-separated game years for test reporting.")
    parser.add_argument(
        "--stats-year-offset", default=-1, type=int,
        help="Offset for team-stat source year relative to game year. "
             "(default: %(default)s)")
    parser.add_argument(
        "--ridge-l2", default=1.0, type=float,
        help="L2 penalty for ridge score regressors. (default: %(default)s)")
    parser.add_argument(
        "--model-out", "-o", required=True,
        help="Output path for pipeline JSON.")
    parser.add_argument(
        "--allow-non-chronological-splits", action="store_true", default=False,
        help="By default, require train < validation < test years to reduce leakage.")
    args = parser.parse_args()

    train_set = set(args.train_years)
    val_set = set(args.validation_years)
    test_set = set(args.test_years)
    if train_set & val_set:
        raise ValueError(
            "Leakage guard: train/validation year overlap: %s" %
            sorted(train_set & val_set))
    if train_set & test_set:
        raise ValueError(
            "Leakage guard: train/test year overlap: %s" %
            sorted(train_set & test_set))
    if val_set & test_set:
        raise ValueError(
            "Leakage guard: validation/test year overlap: %s" %
            sorted(val_set & test_set))

    if not args.allow_non_chronological_splits:
        if args.train_years and args.validation_years:
            if max(args.train_years) >= min(args.validation_years):
                raise ValueError(
                    "Leakage guard: train years must be strictly before validation years. "
                    "Got max(train)=%s min(validation)=%s" %
                    (max(args.train_years), min(args.validation_years)))
        if args.validation_years and args.test_years:
            if max(args.validation_years) >= min(args.test_years):
                raise ValueError(
                    "Leakage guard: validation years must be strictly before test years. "
                    "Got max(validation)=%s min(test)=%s" %
                    (max(args.validation_years), min(args.test_years)))

    x_train, y_a_train, y_b_train, _ = build_score_dataset(
        args.train_years, stats_year_offset=args.stats_year_offset)
    x_val, y_a_val, y_b_val, _ = build_score_dataset(
        args.validation_years, stats_year_offset=args.stats_year_offset)

    model_a = fit_ridge_regression(x_train, y_a_train, l2=args.ridge_l2)
    model_b = fit_ridge_regression(x_train, y_b_train, l2=args.ridge_l2)
    ensemble_weight = choose_ensemble_weight(
        x_val, y_a_val, y_b_val, model_a, model_b)

    val_metrics, _, _ = evaluate_models(
        x_val, y_a_val, y_b_val, model_a, model_b, ensemble_weight)
    print("Validation metrics:")
    print(val_metrics)

    test_metrics = {}
    if args.test_years:
        x_test, y_a_test, y_b_test, _ = build_score_dataset(
            args.test_years, stats_year_offset=args.stats_year_offset)
        test_metrics, _, _ = evaluate_models(
            x_test, y_a_test, y_b_test, model_a, model_b, ensemble_weight)
        print("Test metrics:")
        print(test_metrics)

    out_dir = os.path.dirname(args.model_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = {
        "config": {
            "train_years": args.train_years,
            "validation_years": args.validation_years,
            "test_years": args.test_years,
            "stats_year_offset": args.stats_year_offset,
            "ridge_l2": args.ridge_l2,
        },
        "feature_columns": FEATURE_COLUMNS,
        "ensemble_weight": ensemble_weight,
        "ridge": {
            "a_coef": [float(v) for v in model_a.coef.tolist()],
            "a_intercept": float(model_a.intercept),
            "b_coef": [float(v) for v in model_b.coef.tolist()],
            "b_intercept": float(model_b.intercept),
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "residual_std": {
            "score_a": float(np.std(y_a_val - model_a.predict(x_val))),
            "score_b": float(np.std(y_b_val - model_b.predict(x_val))),
        },
    }
    save_pipeline(args.model_out, payload)
    print("Saved score pipeline to %s" % args.model_out)
