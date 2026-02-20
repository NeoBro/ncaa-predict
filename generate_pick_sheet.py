#!/usr/bin/env python3
import argparse
import csv
import json
import os

import numpy as np

from ncaa_predict.bracket import load_bracket
from ncaa_predict.data_loader import load_ncaa_games, load_ncaa_schools
from ncaa_predict.score_pipeline import (
    _feature_row,
    _team_stats,
    baseline_predict_scores,
    load_pipeline,
    platt_predict,
)
from ncaa_predict.tourney_pipeline import (
    build_kaggle_to_ncaa_id_map,
    load_custom_seed_map,
    normalize_team_name,
    load_seed_name_map,
    load_seed_map,
    season_team_stats_from_csv,
    season_team_stats_from_csv_with_cutoff,
)


def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def normalize_name(name):
    return normalize_team_name(name)


def resolve_id(name, schools_df):
    n = normalize_name(name)
    for row in schools_df.itertuples():
        if normalize_name(row.school_name) == n:
            return int(row.school_id)
    raise ValueError("Team not found: %s" % name)


def score_outputs(row, score_payload, score_model_a, score_model_b):
    x = np.array([row], dtype=np.float32)
    ridge_a = float(score_model_a.predict(x)[0])
    ridge_b = float(score_model_b.predict(x)[0])
    base_a, base_b = baseline_predict_scores(x)
    w = float(score_payload["ensemble_weight"])
    ens_a = (w * ridge_a) + ((1 - w) * float(base_a[0]))
    ens_b = (w * ridge_b) + ((1 - w) * float(base_b[0]))
    diff = ens_a - ens_b
    cal = score_payload.get("calibration")
    if cal:
        p = float(platt_predict(diff, float(cal["a"]), float(cal["b"])))
    else:
        p = float(sigmoid(diff))
    return ens_a, ens_b, diff, p


def meta_prob(features, model):
    coef = np.array(model["coef"], dtype=np.float32)
    b = float(model["intercept"])
    return float(sigmoid(np.dot(features, coef) + b))


def seed_upset_prior(meta_payload, fav_seed, dog_seed):
    priors = meta_payload.get("seed_upset_priors", {})
    pair = priors.get("pair_upset_rate", {})
    key = "%d-%d" % (int(fav_seed), int(dog_seed))
    return float(pair.get(key, priors.get("global", 0.35)))


def win_features(score_p_a, diff, seed_a, seed_b, meta_payload):
    fav_seed = min(seed_a, seed_b)
    dog_seed = max(seed_a, seed_b)
    upset_prior = seed_upset_prior(meta_payload, fav_seed, dog_seed)
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


def upset_features(fav_score_p, fav_margin, fav_seed, dog_seed, meta_payload):
    upset_prior = seed_upset_prior(meta_payload, fav_seed, dog_seed)
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


def simulate_pick(
    node, round_num, rows, stats, school_ids, seed_map, year, score_models, meta_payload,
    upset_bias=0.0, upset_threshold=0.0, min_dog_win_prob=0.0,
):
    left, right = node
    if isinstance(left, tuple):
        team_a = simulate_pick(
            left, round_num + 1, rows, stats, school_ids, seed_map, year, score_models, meta_payload,
            upset_bias=upset_bias, upset_threshold=upset_threshold, min_dog_win_prob=min_dog_win_prob)
    else:
        team_a = left
    if isinstance(right, tuple):
        team_b = simulate_pick(
            right, round_num + 1, rows, stats, school_ids, seed_map, year, score_models, meta_payload,
            upset_bias=upset_bias, upset_threshold=upset_threshold, min_dog_win_prob=min_dog_win_prob)
    else:
        team_b = right

    a_id = school_ids[team_a]
    b_id = school_ids[team_b]
    feat = _feature_row(stats.loc[a_id], stats.loc[b_id])
    score_payload, score_model_a, score_model_b = score_models
    ens_a, ens_b, diff, p_a_score = score_outputs(feat, score_payload, score_model_a, score_model_b)
    sa = seed_map.get((year, a_id), 20)
    sb = seed_map.get((year, b_id), 20)
    p_a_meta = meta_prob(
        win_features(p_a_score, diff, sa, sb, meta_payload),
        meta_payload["win_model"],
    )

    if sa < sb:
        favorite, underdog = team_a, team_b
        fav_seed, dog_seed = sa, sb
        fav_margin = diff
        fav_p_score = p_a_score
    else:
        favorite, underdog = team_b, team_a
        fav_seed, dog_seed = sb, sa
        fav_margin = -diff
        fav_p_score = 1 - p_a_score
    p_upset = meta_prob(
        upset_features(fav_p_score, fav_margin, fav_seed, dog_seed, meta_payload),
        meta_payload["upset_model"],
    )

    winner = team_a if p_a_meta >= 0.5 else team_b
    pick_rule = "win_prob"
    dog_team = underdog
    dog_win_prob = p_a_meta if dog_team == team_a else (1 - p_a_meta)
    if (p_upset + float(upset_bias) >= float(upset_threshold)) and (dog_win_prob >= float(min_dog_win_prob)):
        winner = dog_team
        pick_rule = "upset_rule"
    rows.append({
        "round": round_num,
        "team_a": team_a,
        "team_b": team_b,
        "seed_a": sa,
        "seed_b": sb,
        "score_pred_a": round(ens_a, 3),
        "score_pred_b": round(ens_b, 3),
        "win_prob_a": round(p_a_meta, 4),
        "win_prob_b": round(1 - p_a_meta, 4),
        "pick": winner,
        "favorite": favorite,
        "underdog": underdog,
        "upset_prob": round(p_upset, 4),
        "underdog_win_prob": round(dog_win_prob, 4),
        "pick_rule": pick_rule,
    })
    return winner


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-model-in", required=True)
    parser.add_argument("--meta-model-in", required=True)
    parser.add_argument("--kaggle-dir", required=True)
    parser.add_argument("--custom-seeds-csv", default=None, help="Optional custom seeds CSV (for current seasons not in Kaggle seeds).")
    parser.add_argument("--year", "-y", required=True, type=int)
    parser.add_argument("--all-games-csv", default=None, help="Optional consolidated games CSV for cutoff-aware stats.")
    parser.add_argument("--cutoff-month", type=int, default=None)
    parser.add_argument("--cutoff-day", type=int, default=None)
    parser.add_argument(
        "--bracket-file", default=None,
        help="Optional JSON file describing the bracket tree.")
    parser.add_argument(
        "--pick-style", choices=["safe", "balanced", "chaos", "custom"], default="safe",
        help="Preset upset policy. Use custom to fully control upset flags.")
    parser.add_argument("--upset-bias", type=float, default=0.0, help="Additive bias applied to upset_prob before threshold.")
    parser.add_argument("--upset-threshold", type=float, default=1.0, help="If upset_prob+bias >= threshold, pick underdog (subject to min underdog win prob).")
    parser.add_argument("--min-underdog-win-prob", type=float, default=0.0, help="Minimum underdog win probability required for upset override.")
    parser.add_argument("--out-csv", "-o", default="reports/pick_sheet.csv")
    args = parser.parse_args()
    bracket = load_bracket(args.bracket_file)

    score_payload, score_model_a, score_model_b = load_pipeline(args.score_model_in)
    with open(args.meta_model_in) as f:
        meta_payload = json.load(f)

    presets = {
        "safe": {"upset_bias": 0.0, "upset_threshold": 1.0, "min_underdog_win_prob": 0.0},
        "balanced": {"upset_bias": 0.05, "upset_threshold": 0.40, "min_underdog_win_prob": 0.35},
        "chaos": {"upset_bias": 0.10, "upset_threshold": 0.35, "min_underdog_win_prob": 0.30},
    }
    if args.pick_style != "custom":
        p = presets[args.pick_style]
        args.upset_bias = p["upset_bias"]
        args.upset_threshold = p["upset_threshold"]
        args.min_underdog_win_prob = p["min_underdog_win_prob"]

    schools = load_ncaa_schools()
    school_ids = {row.school_name: int(row.school_id) for row in schools.itertuples()}
    # Ensure every bracket team resolves.
    def flatten(node):
        a, b = node
        out = []
        out += flatten(a) if isinstance(a, tuple) else [a]
        out += flatten(b) if isinstance(b, tuple) else [b]
        return out
    for team in set(flatten(bracket)):
        if team not in school_ids:
            school_ids[team] = resolve_id(team, schools)

    kaggle_to_ncaa, _, _ = build_kaggle_to_ncaa_id_map(args.kaggle_dir, schools)
    seed_map = load_seed_map(args.kaggle_dir, kaggle_to_ncaa_id_map=kaggle_to_ncaa)
    seed_name_map = load_seed_name_map(args.kaggle_dir)
    if args.custom_seeds_csv:
        custom_id_seed, custom_name_seed = load_custom_seed_map(
            args.custom_seeds_csv, schools, year=args.year)
        seed_map.update(custom_id_seed)
        seed_name_map.update(custom_name_seed)

    stats_year = args.year + int(score_payload["config"]["stats_year_offset"])
    if args.all_games_csv:
        cutoff_month = args.cutoff_month
        cutoff_day = args.cutoff_day
        if cutoff_month is None:
            cutoff_month = int(score_payload["config"].get("cutoff_month", 3))
        if cutoff_day is None:
            cutoff_day = int(score_payload["config"].get("cutoff_day", 15))
        stats = season_team_stats_from_csv_with_cutoff(
            args.all_games_csv, stats_year, cutoff_month=cutoff_month, cutoff_day=cutoff_day)
        if stats is None:
            stats = season_team_stats_from_csv(args.all_games_csv, stats_year)
    else:
        games = load_ncaa_games(stats_year, max_bad_ratio=0.06)
        stats = _team_stats(games)

    rows = []
    # Fill missing ID-based seeds with name-based lookup to avoid losing upset features.
    def resolved_seed(team_name):
        team_id = school_ids[team_name]
        seed = seed_map.get((args.year, team_id))
        if seed is None:
            seed = seed_name_map.get((args.year, normalize_name(team_name)))
        return seed if seed is not None else 20

    bracket_teams = set(flatten(bracket))
    seed_map_resolved = {
        (args.year, school_ids[t]): resolved_seed(t) for t in bracket_teams
    }
    missing_seed_teams = sorted([
        t for t in bracket_teams
        if seed_map_resolved[(args.year, school_ids[t])] == 20
    ])

    champion = simulate_pick(
        bracket, 1, rows, stats, school_ids, seed_map_resolved, args.year,
        (score_payload, score_model_a, score_model_b), meta_payload,
        upset_bias=args.upset_bias,
        upset_threshold=args.upset_threshold,
        min_dog_win_prob=args.min_underdog_win_prob)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Champion pick: %s" % champion)
    print("Wrote pick sheet: %s" % args.out_csv)
    print(
        "Policy: style=%s bias=%.2f threshold=%.2f min_dog=%.2f" % (
            args.pick_style, args.upset_bias, args.upset_threshold, args.min_underdog_win_prob))
    if missing_seed_teams:
        print(
            "Seed fallback used for %s teams (default 20). Example: %s" % (
                len(missing_seed_teams), ", ".join(missing_seed_teams[:8])))
