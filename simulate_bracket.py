#!/usr/bin/env python3
import argparse
import random
from collections import defaultdict

import pandas as pd

from ncaa_predict.bracket import load_bracket
from ncaa_predict.data_loader import load_ncaa_games, load_ncaa_schools
from ncaa_predict.score_pipeline import (
    baseline_predict_scores,
    load_pipeline,
    platt_predict,
)
from ncaa_predict.tourney_pipeline import normalize_team_name
from ncaa_predict.util import team_name_to_id


def team_stats_for_year(stats_year, max_bad_ratio):
    games = load_ncaa_games(stats_year, max_bad_ratio=max_bad_ratio)
    left = games[["school_id", "score", "opponent_score"]].rename(columns={
        "school_id": "team_id",
        "score": "team_score",
        "opponent_score": "opp_score",
    })
    right = games[["opponent_id", "opponent_score", "score"]].rename(columns={
        "opponent_id": "team_id",
        "opponent_score": "team_score",
        "score": "opp_score",
    })
    team_games = pd.concat([left, right], ignore_index=True)
    by_team = team_games.groupby("team_id")
    stats = pd.DataFrame({
        "offense": by_team["team_score"].mean(),
        "defense": by_team["opp_score"].mean(),
    })
    stats["total"] = stats["offense"] + stats["defense"]
    stats["margin"] = stats["offense"] - stats["defense"]
    return stats


def feature_row(team_a_stats, team_b_stats):
    return [
        team_a_stats["offense"],
        team_a_stats["defense"],
        team_a_stats["total"],
        team_a_stats["margin"],
        team_b_stats["offense"],
        team_b_stats["defense"],
        team_b_stats["total"],
        team_b_stats["margin"],
        team_a_stats["offense"] - team_b_stats["offense"],
        team_a_stats["defense"] - team_b_stats["defense"],
        team_a_stats["margin"] - team_b_stats["margin"],
    ]


def normalize_name(name):
    return normalize_team_name(name)


def resolve_team_id(team_name, schools):
    try:
        return int(team_name_to_id(team_name, schools))
    except Exception:
        wanted = normalize_name(team_name)
        matches = [
            row.school_id for row in schools.itertuples()
            if normalize_name(row.school_name) == wanted
        ]
        if len(matches) == 1:
            return int(matches[0])
        raise ValueError("Could not resolve team name: %s" % team_name)


def matchup_probability(team_a, team_b, model_a, model_b, payload, school_map, stats):
    team_a_id = school_map[team_a]
    team_b_id = school_map[team_b]
    row = feature_row(stats.loc[team_a_id], stats.loc[team_b_id])
    x = pd.DataFrame([row]).to_numpy(dtype="float32")
    ridge_a = float(model_a.predict(x)[0])
    ridge_b = float(model_b.predict(x)[0])
    base_a, base_b = baseline_predict_scores(x)
    w = float(payload["ensemble_weight"])
    ens_a = (w * ridge_a) + ((1 - w) * float(base_a[0]))
    ens_b = (w * ridge_b) + ((1 - w) * float(base_b[0]))
    if "calibration" in payload:
        cal = payload["calibration"]
        p_a = float(platt_predict(ens_a - ens_b, float(cal["a"]), float(cal["b"])))
    else:
        p_a = 1.0 if ens_a > ens_b else 0.0
    return max(0.0, min(1.0, p_a))


def flatten_teams(bracket):
    a, b = bracket
    teams = []
    if isinstance(a, tuple):
        teams.extend(flatten_teams(a))
    else:
        teams.append(a)
    if isinstance(b, tuple):
        teams.extend(flatten_teams(b))
    else:
        teams.append(b)
    return teams


def simulate_node(
    node, model_a, model_b, payload, school_map, stats, rng, round_counts, round_idx
):
    left, right = node
    if isinstance(left, tuple):
        team_a = simulate_node(
            left, model_a, model_b, payload, school_map, stats, rng, round_counts, round_idx + 1)
    else:
        team_a = left
    if isinstance(right, tuple):
        team_b = simulate_node(
            right, model_a, model_b, payload, school_map, stats, rng, round_counts, round_idx + 1)
    else:
        team_b = right

    p_a = matchup_probability(team_a, team_b, model_a, model_b, payload, school_map, stats)
    winner = team_a if rng.random() < p_a else team_b
    round_counts[winner][round_idx] += 1
    return winner


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-in", "-m", required=True)
    parser.add_argument("--year", "-y", required=True, type=int)
    parser.add_argument(
        "--bracket-file", default=None,
        help="Optional JSON file describing the bracket tree.")
    parser.add_argument("--n-sims", "-n", default=5000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max-bad-ratio", default=0.06, type=float)
    parser.add_argument("--top-k", default=20, type=int)
    args = parser.parse_args()
    bracket = load_bracket(args.bracket_file)

    payload, model_a, model_b = load_pipeline(args.model_in)
    stats_year = args.year + int(payload["config"]["stats_year_offset"])
    stats = team_stats_for_year(stats_year, max_bad_ratio=args.max_bad_ratio)
    schools = load_ncaa_schools()
    school_map = {
        team: resolve_team_id(team, schools)
        for team in set(flatten_teams(bracket))
    }

    bracket_teams = set(flatten_teams(bracket))
    missing_stats = sorted(
        [team for team in bracket_teams if school_map[team] not in stats.index])
    if missing_stats:
        raise ValueError("Teams missing stats for year %s: %s" % (stats_year, missing_stats))

    rng = random.Random(args.seed)
    champs = defaultdict(int)
    # round index 0=champion, 1=final, 2=final4, ...
    round_counts = defaultdict(lambda: defaultdict(int))
    for _ in range(args.n_sims):
        champ = simulate_node(
            bracket, model_a, model_b, payload, school_map, stats, rng, round_counts, 0)
        champs[champ] += 1

    print("Simulations: %s (seed=%s)" % (args.n_sims, args.seed))
    print("Stats year: %s" % stats_year)
    print("Top champion probabilities:")
    ranked = sorted(champs.items(), key=lambda kv: kv[1], reverse=True)
    for team, wins in ranked[:args.top_k]:
        print("%-25s %.4f" % (team, wins / args.n_sims))

    print("\nRound reach probabilities (top champions only):")
    for team, _ in ranked[:min(10, args.top_k)]:
        champ = round_counts[team][0] / args.n_sims
        final = round_counts[team][1] / args.n_sims
        final4 = round_counts[team][2] / args.n_sims
        elite8 = round_counts[team][3] / args.n_sims
        sweet16 = round_counts[team][4] / args.n_sims
        print(
            "%-25s champ=%.4f final=%.4f final4=%.4f elite8=%.4f sweet16=%.4f" % (
                team, champ, final, final4, elite8, sweet16))
