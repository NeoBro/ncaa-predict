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
    load_seed_name_map,
    load_seed_map,
)


def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def normalize_name(name):
    text = name.lower()
    for c in ["'", ".", ",", "(", ")", "-", "&", "/"]:
        text = text.replace(c, " ")
    text = " ".join(text.split())
    repl = {
        "saint": "st",
        "mount": "mt",
        "state": "st",
        "university": "u",
    }
    toks = [repl.get(tok, tok) for tok in text.split()]
    out = " ".join(toks)
    aliases = {
        "liu brooklyn": "liu",
    }
    return aliases.get(out, out)


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


def simulate_pick(node, round_num, rows, stats, school_ids, seed_map, year, score_models, meta_payload):
    left, right = node
    if isinstance(left, tuple):
        team_a = simulate_pick(left, round_num + 1, rows, stats, school_ids, seed_map, year, score_models, meta_payload)
    else:
        team_a = left
    if isinstance(right, tuple):
        team_b = simulate_pick(right, round_num + 1, rows, stats, school_ids, seed_map, year, score_models, meta_payload)
    else:
        team_b = right

    a_id = school_ids[team_a]
    b_id = school_ids[team_b]
    feat = _feature_row(stats.loc[a_id], stats.loc[b_id])
    score_payload, score_model_a, score_model_b = score_models
    ens_a, ens_b, diff, p_a_score = score_outputs(feat, score_payload, score_model_a, score_model_b)
    sa = seed_map.get((year, a_id), 20)
    sb = seed_map.get((year, b_id), 20)
    win_features = [p_a_score, diff, abs(diff), float(sa), float(sb), float(sb - sa)]
    p_a_meta = meta_prob(win_features, meta_payload["win_model"])

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
    upset_features = [fav_p_score, fav_margin, abs(fav_margin), float(fav_seed), float(dog_seed), float(dog_seed - fav_seed)]
    p_upset = meta_prob(upset_features, meta_payload["upset_model"])

    winner = team_a if p_a_meta >= 0.5 else team_b
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
    })
    return winner


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-model-in", required=True)
    parser.add_argument("--meta-model-in", required=True)
    parser.add_argument("--kaggle-dir", required=True)
    parser.add_argument("--year", "-y", required=True, type=int)
    parser.add_argument(
        "--bracket-file", default=None,
        help="Optional JSON file describing the bracket tree.")
    parser.add_argument("--out-csv", "-o", default="reports/pick_sheet.csv")
    args = parser.parse_args()
    bracket = load_bracket(args.bracket_file)

    score_payload, score_model_a, score_model_b = load_pipeline(args.score_model_in)
    with open(args.meta_model_in) as f:
        meta_payload = json.load(f)

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

    stats_year = args.year + int(score_payload["config"]["stats_year_offset"])
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

    seed_map_resolved = {
        (args.year, school_ids[t]): resolved_seed(t) for t in set(flatten(bracket))
    }

    champion = simulate_pick(
        bracket, 1, rows, stats, school_ids, seed_map_resolved, args.year,
        (score_payload, score_model_a, score_model_b), meta_payload)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Champion pick: %s" % champion)
    print("Wrote pick sheet: %s" % args.out_csv)
