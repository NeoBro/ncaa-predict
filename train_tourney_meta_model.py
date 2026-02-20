#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np

from ncaa_predict.data_loader import load_ncaa_schools
from ncaa_predict.score_pipeline import (
    _feature_row,
    _team_stats,
    baseline_predict_scores,
    load_pipeline,
    platt_predict,
)
from ncaa_predict.tourney_pipeline import (
    build_kaggle_to_ncaa_id_map,
    load_kaggle_tourney,
    load_seed_map,
    season_team_stats_from_csv,
    season_team_stats_from_csv_with_cutoff,
)
from ncaa_predict.util import list_arg


def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logreg(x, y, l2=1.0, lr=0.05, steps=3000):
    w = np.zeros(x.shape[1], dtype=np.float64)
    b = 0.0
    n = len(y)
    for _ in range(steps):
        p = sigmoid(x @ w + b)
        err = p - y
        grad_w = (x.T @ err) / n + l2 * w
        grad_b = float(np.mean(err))
        w -= lr * grad_w
        b -= lr * grad_b
    return w, float(b)


def metrics(y, p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    pred = p >= 0.5
    acc = float(np.mean(pred == y))
    brier = float(np.mean((p - y) ** 2))
    ll = float(-np.mean((y * np.log(p)) + ((1 - y) * np.log(1 - p))))
    return {"accuracy": acc, "brier": brier, "log_loss": ll}


def compute_seed_upset_priors(kaggle_games, years, kaggle_to_ncaa, seed_map):
    pair_counts = {}
    pair_upsets = {}
    total = 0
    upsets = 0
    season = kaggle_games[kaggle_games["Season"].isin(years)]
    for g in season.itertuples():
        wa = int(g.WTeamID)
        la = int(g.LTeamID)
        if wa not in kaggle_to_ncaa or la not in kaggle_to_ncaa:
            continue
        w_id = int(kaggle_to_ncaa[wa])
        l_id = int(kaggle_to_ncaa[la])
        sw = seed_map.get((int(g.Season), w_id))
        sl = seed_map.get((int(g.Season), l_id))
        if sw is None or sl is None or sw == sl:
            continue
        fav_seed = min(sw, sl)
        dog_seed = max(sw, sl)
        upset = (sw > sl)
        key = "%d-%d" % (fav_seed, dog_seed)
        pair_counts[key] = pair_counts.get(key, 0) + 1
        pair_upsets[key] = pair_upsets.get(key, 0) + (1 if upset else 0)
        total += 1
        upsets += (1 if upset else 0)
    global_upset = (upsets / total) if total else 0.35
    pair_rates = {}
    for key, n in pair_counts.items():
        # Laplace smoothing to keep rare pairs stable.
        pair_rates[key] = float((pair_upsets[key] + 1.0) / (n + 2.0))
    return pair_rates, float(global_upset)


def season_stats(all_games_csv, season):
    import pandas as pd
    games = pd.read_csv(
        all_games_csv,
        usecols=["year", "school_id", "opponent_id", "score", "opponent_score"])
    g = games[games["year"] == season]
    if len(g) == 0:
        return None
    return _team_stats(g)


def score_model_outputs(x, score_payload, score_model_a, score_model_b):
    ridge_a = score_model_a.predict(x)
    ridge_b = score_model_b.predict(x)
    base_a, base_b = baseline_predict_scores(x)
    w = float(score_payload["ensemble_weight"])
    ens_a = (w * ridge_a) + ((1 - w) * base_a)
    ens_b = (w * ridge_b) + ((1 - w) * base_b)
    diff = ens_a - ens_b
    cal = score_payload.get("calibration")
    if cal:
        p = platt_predict(diff, float(cal["a"]), float(cal["b"]))
    else:
        p = sigmoid(diff)
    return ens_a, ens_b, diff, p


def build_datasets(
    kaggle_games,
    all_games_csv,
    years,
    stats_year_offset,
    kaggle_to_ncaa,
    seed_map,
    score_payload,
    score_model_a,
    score_model_b,
    seed_pair_upset_priors,
    global_upset_prior,
    cutoff_month=None,
    cutoff_day=None,
):
    x_win, y_win = [], []
    x_upset, y_upset = [], []
    for year in years:
        stats_year = year + stats_year_offset
        if cutoff_month is None or cutoff_day is None:
            stats = season_team_stats_from_csv(all_games_csv, stats_year)
        else:
            stats = season_team_stats_from_csv_with_cutoff(
                all_games_csv, stats_year,
                cutoff_month=cutoff_month, cutoff_day=cutoff_day)
        if stats is None:
            continue
        if len(stats) == 0:
            continue
        season = kaggle_games[kaggle_games["Season"] == year]
        for g in season.itertuples():
            wa = int(g.WTeamID)
            la = int(g.LTeamID)
            if wa not in kaggle_to_ncaa or la not in kaggle_to_ncaa:
                continue
            w_id = int(kaggle_to_ncaa[wa])
            l_id = int(kaggle_to_ncaa[la])
            if w_id not in stats.index or l_id not in stats.index:
                continue

            # Two orientations for win model.
            for a_id, b_id, y in [(w_id, l_id, 1.0), (l_id, w_id, 0.0)]:
                row = np.array([_feature_row(stats.loc[a_id], stats.loc[b_id])], dtype=np.float32)
                _, _, diff, p = score_model_outputs(
                    row, score_payload, score_model_a, score_model_b)
                sa = seed_map.get((year, a_id), 20)
                sb = seed_map.get((year, b_id), 20)
                fav_seed = min(sa, sb)
                dog_seed = max(sa, sb)
                key = "%d-%d" % (fav_seed, dog_seed)
                upset_prior = float(seed_pair_upset_priors.get(key, global_upset_prior))
                if sa < sb:
                    p_a_seed = 1.0 - upset_prior
                elif sa > sb:
                    p_a_seed = upset_prior
                else:
                    p_a_seed = 0.5
                x_win.append([
                    float(p[0]),
                    float(diff[0]),
                    float(abs(diff[0])),
                    float(sa),
                    float(sb),
                    float(sb - sa),
                    float(p_a_seed),
                ])
                y_win.append(y)

            # Favorite-first row for upset model.
            sw = seed_map.get((year, w_id))
            sl = seed_map.get((year, l_id))
            if sw is None or sl is None or sw == sl:
                continue
            fav_id, dog_id = (w_id, l_id) if sw < sl else (l_id, w_id)
            upset = 1.0 if fav_id == l_id else 0.0
            fav_seed = min(sw, sl)
            dog_seed = max(sw, sl)
            key = "%d-%d" % (fav_seed, dog_seed)
            upset_prior = float(seed_pair_upset_priors.get(key, global_upset_prior))
            row = np.array([_feature_row(stats.loc[fav_id], stats.loc[dog_id])], dtype=np.float32)
            _, _, diff, p = score_model_outputs(
                row, score_payload, score_model_a, score_model_b)
            x_upset.append([
                float(p[0]),
                float(diff[0]),
                float(abs(diff[0])),
                float(fav_seed),
                float(dog_seed),
                float(dog_seed - fav_seed),
                float(upset_prior),
            ])
            y_upset.append(upset)
    return (
        np.array(x_win, dtype=np.float32),
        np.array(y_win, dtype=np.float32),
        np.array(x_upset, dtype=np.float32),
        np.array(y_upset, dtype=np.float32),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-dir", required=True)
    parser.add_argument("--all-games-csv", required=True)
    parser.add_argument("--score-model-in", required=True)
    parser.add_argument("--train-years", "-y", required=True, type=list_arg(type=int, container=list))
    parser.add_argument("--validation-years", "-v", required=True, type=list_arg(type=int, container=list))
    parser.add_argument("--test-years", "-t", default=[], type=list_arg(type=int, container=list))
    parser.add_argument("--stats-year-offset", type=int, default=0)
    parser.add_argument("--cutoff-month", type=int, default=3)
    parser.add_argument("--cutoff-day", type=int, default=15)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--model-out", "-o", required=True)
    args = parser.parse_args()

    schools = load_ncaa_schools()
    kaggle_to_ncaa, unresolved, ambiguous = build_kaggle_to_ncaa_id_map(args.kaggle_dir, schools)
    kaggle_games = load_kaggle_tourney(args.kaggle_dir)
    seed_map = load_seed_map(args.kaggle_dir, kaggle_to_ncaa_id_map=kaggle_to_ncaa)
    score_payload, score_model_a, score_model_b = load_pipeline(args.score_model_in)
    seed_pair_upset_priors, global_upset_prior = compute_seed_upset_priors(
        kaggle_games, args.train_years, kaggle_to_ncaa, seed_map)

    xw_tr, yw_tr, xu_tr, yu_tr = build_datasets(
        kaggle_games, args.all_games_csv, args.train_years, args.stats_year_offset,
        kaggle_to_ncaa, seed_map, score_payload, score_model_a, score_model_b,
        seed_pair_upset_priors, global_upset_prior,
        cutoff_month=args.cutoff_month, cutoff_day=args.cutoff_day)
    xw_va, yw_va, xu_va, yu_va = build_datasets(
        kaggle_games, args.all_games_csv, args.validation_years, args.stats_year_offset,
        kaggle_to_ncaa, seed_map, score_payload, score_model_a, score_model_b,
        seed_pair_upset_priors, global_upset_prior,
        cutoff_month=args.cutoff_month, cutoff_day=args.cutoff_day)

    ww, bw = fit_logreg(xw_tr, yw_tr, l2=args.l2)
    wu, bu = fit_logreg(xu_tr, yu_tr, l2=args.l2)

    win_val = metrics(yw_va, sigmoid(xw_va @ ww + bw))
    upset_val = metrics(yu_va, sigmoid(xu_va @ wu + bu))
    print("Validation win metrics:", win_val)
    print("Validation upset metrics:", upset_val)

    test_metrics = {}
    if args.test_years:
        xw_te, yw_te, xu_te, yu_te = build_datasets(
            kaggle_games, args.all_games_csv, args.test_years, args.stats_year_offset,
            kaggle_to_ncaa, seed_map, score_payload, score_model_a, score_model_b,
            seed_pair_upset_priors, global_upset_prior,
            cutoff_month=args.cutoff_month, cutoff_day=args.cutoff_day)
        test_metrics = {
            "win": metrics(yw_te, sigmoid(xw_te @ ww + bw)),
            "upset": metrics(yu_te, sigmoid(xu_te @ wu + bu)),
        }
        print("Test metrics:", test_metrics)

    out_dir = os.path.dirname(args.model_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.model_out, "w") as f:
        json.dump({
            "config": {
                "kaggle_dir": args.kaggle_dir,
                "all_games_csv": args.all_games_csv,
                "score_model_in": args.score_model_in,
                "train_years": args.train_years,
                "validation_years": args.validation_years,
                "test_years": args.test_years,
                "stats_year_offset": args.stats_year_offset,
                "cutoff_month": args.cutoff_month,
                "cutoff_day": args.cutoff_day,
                "l2": args.l2,
            },
            "mapping_summary": {
                "mapped_count": len(kaggle_to_ncaa),
                "unresolved_count": len(unresolved),
                "ambiguous_count": len(ambiguous),
            },
            "features": {
                "win": [
                    "score_win_prob_a", "score_margin", "score_margin_abs",
                    "seed_a", "seed_b", "seed_diff",
                    "seed_prior_win_a",
                ],
                "upset": [
                    "fav_score_win_prob", "fav_score_margin", "fav_margin_abs",
                    "fav_seed", "dog_seed", "seed_gap",
                    "seed_prior_upset",
                ],
            },
            "seed_upset_priors": {
                "global": global_upset_prior,
                "pair_upset_rate": seed_pair_upset_priors,
            },
            "win_model": {"coef": [float(v) for v in ww.tolist()], "intercept": float(bw)},
            "upset_model": {"coef": [float(v) for v in wu.tolist()], "intercept": float(bu)},
            "validation_metrics": {"win": win_val, "upset": upset_val},
            "test_metrics": test_metrics,
        }, f, indent=2, sort_keys=True)
    print("Saved meta model to %s" % args.model_out)
