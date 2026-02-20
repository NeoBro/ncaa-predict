#!/usr/bin/env python3
import argparse
import json
import os
import statistics

import numpy as np

from ncaa_predict.data_loader import load_ncaa_schools
from ncaa_predict.score_pipeline import _feature_row, load_pipeline
from ncaa_predict.tourney_pipeline import (
    build_kaggle_to_ncaa_id_map,
    load_kaggle_tourney,
    load_seed_map,
    season_team_stats_from_csv,
    season_team_stats_from_csv_with_cutoff,
)
from ncaa_predict.util import list_arg
from train_tourney_meta_model import (
    build_datasets,
    compute_seed_upset_priors,
    fit_logreg,
    score_model_outputs,
)


def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def meta_prob(features, model):
    coef = np.array(model["coef"], dtype=np.float32)
    b = float(model["intercept"])
    return float(sigmoid(np.dot(features, coef) + b))


def seed_upset_prior(priors, fav_seed, dog_seed):
    pair = priors.get("pair_upset_rate", {})
    key = "%d-%d" % (int(fav_seed), int(dog_seed))
    return float(pair.get(key, priors.get("global", 0.35)))


def make_win_features(score_p_a, diff, seed_a, seed_b, priors):
    fav_seed = min(seed_a, seed_b)
    dog_seed = max(seed_a, seed_b)
    upset_prior = seed_upset_prior(priors, fav_seed, dog_seed)
    if seed_a < seed_b:
        p_a_seed = 1.0 - upset_prior
    elif seed_a > seed_b:
        p_a_seed = upset_prior
    else:
        p_a_seed = 0.5
    return [
        float(score_p_a),
        float(diff),
        float(abs(diff)),
        float(seed_a),
        float(seed_b),
        float(seed_b - seed_a),
        float(p_a_seed),
    ]


def make_upset_features(fav_score_p, fav_margin, fav_seed, dog_seed, priors):
    upset_prior = seed_upset_prior(priors, fav_seed, dog_seed)
    gap = float(dog_seed - fav_seed)
    return [
        float(fav_score_p),
        float(fav_margin),
        float(abs(fav_margin)),
        float(fav_seed),
        float(dog_seed),
        gap,
        float(upset_prior),
    ]


def evaluate_year(
    year,
    kaggle_games,
    all_games_csv,
    kaggle_to_ncaa,
    seed_map,
    score_payload,
    score_model_a,
    score_model_b,
    win_model,
    upset_model,
    priors,
    stats_year_offset,
    upset_bias,
    upset_threshold,
    min_underdog_win_prob,
    cutoff_month,
    cutoff_day,
):
    stats_year = year + stats_year_offset
    if cutoff_month is None or cutoff_day is None:
        stats = season_team_stats_from_csv(all_games_csv, stats_year)
    else:
        stats = season_team_stats_from_csv_with_cutoff(
            all_games_csv, stats_year,
            cutoff_month=cutoff_month, cutoff_day=cutoff_day)
    if stats is None:
        return None
    season = kaggle_games[kaggle_games["Season"] == year]

    n_games = 0
    correct = 0
    upset_total = 0
    upset_hit = 0
    upset_picks = 0
    seed_pair_totals = {}
    seed_pair_hits = {}

    for g in season.itertuples():
        wk = int(g.WTeamID)
        lk = int(g.LTeamID)
        if wk not in kaggle_to_ncaa or lk not in kaggle_to_ncaa:
            continue
        w_id = int(kaggle_to_ncaa[wk])
        l_id = int(kaggle_to_ncaa[lk])
        if w_id not in stats.index or l_id not in stats.index:
            continue

        # Deterministic orientation independent from outcome.
        a_id, b_id = (w_id, l_id) if w_id < l_id else (l_id, w_id)
        actual_winner_is_a = 1.0 if a_id == w_id else 0.0

        feat = np.array([_feature_row(stats.loc[a_id], stats.loc[b_id])], dtype=np.float32)
        _, _, diff, p_score = score_model_outputs(feat, score_payload, score_model_a, score_model_b)

        sa = seed_map.get((year, a_id), 20)
        sb = seed_map.get((year, b_id), 20)
        p_a = meta_prob(
            make_win_features(float(p_score[0]), float(diff[0]), sa, sb, priors),
            win_model,
        )

        if sa < sb:
            fav_is_a = True
            fav_seed, dog_seed = sa, sb
            fav_margin = float(diff[0])
            fav_score_p = float(p_score[0])
        else:
            fav_is_a = False
            fav_seed, dog_seed = sb, sa
            fav_margin = -float(diff[0])
            fav_score_p = 1.0 - float(p_score[0])
        p_upset = meta_prob(
            make_upset_features(fav_score_p, fav_margin, fav_seed, dog_seed, priors),
            upset_model,
        )

        dog_is_a = not fav_is_a
        dog_win_prob = p_a if dog_is_a else (1.0 - p_a)
        pick_dog = (p_upset + float(upset_bias) >= float(upset_threshold)) and (
            dog_win_prob >= float(min_underdog_win_prob)
        )
        if pick_dog:
            pred_a = 1.0 if dog_is_a else 0.0
            upset_picks += 1
        else:
            pred_a = 1.0 if p_a >= 0.5 else 0.0

        is_correct = (pred_a == actual_winner_is_a)
        correct += 1 if is_correct else 0
        n_games += 1

        sw = seed_map.get((year, w_id))
        sl = seed_map.get((year, l_id))
        if sw is None or sl is None or sw == sl:
            continue
        actual_upset = sw > sl
        if actual_upset:
            upset_total += 1
            if is_correct:
                upset_hit += 1

        pair_key = "%d-%d" % (min(sw, sl), max(sw, sl))
        seed_pair_totals[pair_key] = seed_pair_totals.get(pair_key, 0) + 1
        if actual_upset and is_correct:
            seed_pair_hits[pair_key] = seed_pair_hits.get(pair_key, 0) + 1

    if n_games == 0:
        return None
    return {
        "year": int(year),
        "games": int(n_games),
        "accuracy": float(correct / n_games),
        "actual_upsets": int(upset_total),
        "upset_hits": int(upset_hit),
        "upset_recall": float(upset_hit / upset_total) if upset_total else None,
        "upset_pick_rate": float(upset_picks / n_games),
        "seed_pair_totals": seed_pair_totals,
        "seed_pair_upset_hits": seed_pair_hits,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-dir", required=True)
    parser.add_argument("--all-games-csv", required=True)
    parser.add_argument("--score-model-in", required=True)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--min-train-years", type=int, default=6)
    parser.add_argument("--stats-year-offset", type=int, default=0)
    parser.add_argument("--cutoff-month", type=int, default=3)
    parser.add_argument("--cutoff-day", type=int, default=15)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--upset-bias-grid", type=list_arg(type=float, container=list), default=[0.0, 0.05, 0.1])
    parser.add_argument("--upset-threshold-grid", type=list_arg(type=float, container=list), default=[1.0, 0.45, 0.4, 0.35])
    parser.add_argument("--min-dog-win-prob-grid", type=list_arg(type=float, container=list), default=[0.0, 0.3, 0.35, 0.4])
    parser.add_argument("--report-out", default="reports/tourney_sensitivity_backtest.json")
    args = parser.parse_args()

    kaggle_games = load_kaggle_tourney(args.kaggle_dir)
    schools = load_ncaa_schools()
    kaggle_to_ncaa, unresolved, ambiguous = build_kaggle_to_ncaa_id_map(args.kaggle_dir, schools)
    seed_map = load_seed_map(args.kaggle_dir, kaggle_to_ncaa_id_map=kaggle_to_ncaa)
    score_payload, score_model_a, score_model_b = load_pipeline(args.score_model_in)
    years = list(range(args.start_year, args.end_year + 1))
    first_test_idx = args.min_train_years + 1
    if first_test_idx >= len(years):
        raise ValueError("Not enough seasons for train/val/test rolling setup")

    configs = []
    for bias in args.upset_bias_grid:
        for thresh in args.upset_threshold_grid:
            for min_dog in args.min_dog_win_prob_grid:
                if thresh > 0.999 and (bias > 0.0 or min_dog > 0.0):
                    continue
                configs.append((float(bias), float(thresh), float(min_dog)))

    results = []
    for idx in range(first_test_idx, len(years)):
        test_year = years[idx]
        val_year = years[idx - 1]
        train_years = years[: idx - 1]

        priors_pair, priors_global = compute_seed_upset_priors(
            kaggle_games, train_years, kaggle_to_ncaa, seed_map)
        priors = {"pair_upset_rate": priors_pair, "global": priors_global}

        xw_tr, yw_tr, xu_tr, yu_tr = build_datasets(
            kaggle_games, args.all_games_csv, train_years, args.stats_year_offset,
            kaggle_to_ncaa, seed_map, score_payload, score_model_a, score_model_b,
            priors_pair, priors_global,
            cutoff_month=args.cutoff_month, cutoff_day=args.cutoff_day)
        xw_va, yw_va, xu_va, yu_va = build_datasets(
            kaggle_games, args.all_games_csv, [val_year], args.stats_year_offset,
            kaggle_to_ncaa, seed_map, score_payload, score_model_a, score_model_b,
            priors_pair, priors_global,
            cutoff_month=args.cutoff_month, cutoff_day=args.cutoff_day)

        if xw_tr.size == 0 or yw_tr.size == 0 or xu_tr.size == 0 or yu_tr.size == 0:
            print(
                "Skipping fold train=%s-%s val=%s test=%s: empty train dataset" % (
                    train_years[0], train_years[-1], val_year, test_year))
            continue
        if xw_va.size == 0 or yw_va.size == 0 or xu_va.size == 0 or yu_va.size == 0:
            print(
                "Skipping fold train=%s-%s val=%s test=%s: empty validation dataset" % (
                    train_years[0], train_years[-1], val_year, test_year))
            continue

        ww, bw = fit_logreg(np.vstack([xw_tr, xw_va]), np.concatenate([yw_tr, yw_va]), l2=args.l2)
        wu, bu = fit_logreg(np.vstack([xu_tr, xu_va]), np.concatenate([yu_tr, yu_va]), l2=args.l2)
        win_model = {"coef": ww.tolist(), "intercept": float(bw)}
        upset_model = {"coef": wu.tolist(), "intercept": float(bu)}

        fold_rows = []
        for bias, thresh, min_dog in configs:
            row = evaluate_year(
                test_year, kaggle_games, args.all_games_csv, kaggle_to_ncaa, seed_map,
                score_payload, score_model_a, score_model_b, win_model, upset_model, priors,
                args.stats_year_offset, bias, thresh, min_dog,
                args.cutoff_month, args.cutoff_day,
            )
            if row is None:
                continue
            row["config"] = {
                "upset_bias": bias,
                "upset_threshold": thresh,
                "min_dog_win_prob": min_dog,
            }
            fold_rows.append(row)
        if not fold_rows:
            print(
                "Skipping fold train=%s-%s val=%s test=%s: no evaluable games" % (
                    train_years[0], train_years[-1], val_year, test_year))
            continue
        results.append({
            "train_years": train_years,
            "validation_year": val_year,
            "test_year": test_year,
            "grid_results": fold_rows,
        })
        print(
            "Fold train=%s-%s val=%s test=%s evaluated_configs=%s" % (
                train_years[0], train_years[-1], val_year, test_year, len(fold_rows)))

    by_cfg = {}
    for fold in results:
        for r in fold["grid_results"]:
            c = r["config"]
            key = "%0.2f|%0.2f|%0.2f" % (
                c["upset_bias"], c["upset_threshold"], c["min_dog_win_prob"])
            by_cfg.setdefault(key, []).append(r)

    summary = []
    for key, rows in by_cfg.items():
        acc = [r["accuracy"] for r in rows]
        up = [r["upset_recall"] for r in rows if r["upset_recall"] is not None]
        upr = [r["upset_pick_rate"] for r in rows]
        summary.append({
            "config_key": key,
            "folds": len(rows),
            "accuracy_mean": float(statistics.mean(acc)),
            "accuracy_std": float(statistics.pstdev(acc)) if len(acc) > 1 else 0.0,
            "upset_recall_mean": float(statistics.mean(up)) if up else None,
            "upset_pick_rate_mean": float(statistics.mean(upr)),
        })
    summary = sorted(
        summary,
        key=lambda x: (
            -1.0 if x["upset_recall_mean"] is None else -x["upset_recall_mean"],
            -x["accuracy_mean"],
        ),
    )

    top_seed_pairs = {}
    for fold in results:
        for row in fold["grid_results"]:
            if row["config"]["upset_threshold"] != 1.0:
                continue
            for pair, n in row["seed_pair_totals"].items():
                top_seed_pairs.setdefault(pair, {"n": 0, "hits": 0})
                top_seed_pairs[pair]["n"] += n
                top_seed_pairs[pair]["hits"] += row["seed_pair_upset_hits"].get(pair, 0)

    seed_pair_summary = []
    for pair, d in top_seed_pairs.items():
        if d["n"] < 8:
            continue
        seed_pair_summary.append({
            "seed_pair": pair,
            "games": int(d["n"]),
            "upset_hit_rate": float(d["hits"] / d["n"]),
        })
    seed_pair_summary = sorted(seed_pair_summary, key=lambda x: -x["upset_hit_rate"])

    report = {
        "config": vars(args),
        "mapping_summary": {
            "mapped_count": len(kaggle_to_ncaa),
            "unresolved_count": len(unresolved),
            "ambiguous_count": len(ambiguous),
        },
        "fold_results": results,
        "grid_summary": summary,
        "seed_pair_summary": seed_pair_summary,
    }

    out_dir = os.path.dirname(args.report_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.report_out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print("Wrote sensitivity backtest: %s" % args.report_out)
    if summary:
        print("Top configs:")
        for row in summary[:5]:
            print(row)
