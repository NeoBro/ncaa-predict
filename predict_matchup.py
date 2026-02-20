#!/usr/bin/env python3
import argparse
import math

from ncaa_predict.score_pipeline import (
    baseline_predict_scores,
    load_pipeline,
    matchup_features,
)


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-in", "-m", required=True)
    parser.add_argument("--year", "-y", required=True, type=int)
    parser.add_argument("team_a")
    parser.add_argument("team_b")
    args = parser.parse_args()

    payload, model_a, model_b = load_pipeline(args.model_in)
    stats_offset = int(payload["config"]["stats_year_offset"])
    x, _, _, stats_year = matchup_features(
        args.team_a, args.team_b, args.year, stats_offset)

    ridge_a = float(model_a.predict(x)[0])
    ridge_b = float(model_b.predict(x)[0])
    base_a, base_b = baseline_predict_scores(x)
    base_a = float(base_a[0])
    base_b = float(base_b[0])
    w = float(payload["ensemble_weight"])
    ens_a = (w * ridge_a) + ((1 - w) * base_a)
    ens_b = (w * ridge_b) + ((1 - w) * base_b)

    diff_std = max(
        1e-6,
        math.sqrt(
            (float(payload["residual_std"]["score_a"]) ** 2) +
            (float(payload["residual_std"]["score_b"]) ** 2)))
    win_prob_a = normal_cdf((ens_a - ens_b) / diff_std)

    print("Using stats year: %s" % stats_year)
    print("Baseline score: %s %.1f - %s %.1f" %
          (args.team_a, base_a, args.team_b, base_b))
    print("Ridge score: %s %.1f - %s %.1f" %
          (args.team_a, ridge_a, args.team_b, ridge_b))
    print("Ensemble score: %s %.1f - %s %.1f" %
          (args.team_a, ens_a, args.team_b, ens_b))
    print("Ensemble win probability: %s %.3f | %s %.3f" % (
        args.team_a, win_prob_a, args.team_b, 1 - win_prob_a))
