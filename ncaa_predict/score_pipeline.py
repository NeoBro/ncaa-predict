import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ncaa_predict.data_loader import load_ncaa_games, load_ncaa_schools
from ncaa_predict.util import team_name_to_id


FEATURE_COLUMNS = [
    "a_offense",
    "a_defense",
    "a_total",
    "a_margin",
    "b_offense",
    "b_defense",
    "b_total",
    "b_margin",
    "offense_diff",
    "defense_diff",
    "margin_diff",
]


@dataclass
class RegressionModel:
    coef: np.ndarray
    intercept: float

    def predict(self, x):
        return x @ self.coef + self.intercept


def _team_stats(games):
    by_team = games.groupby("school_id")
    stats = pd.DataFrame({
        "offense": by_team["score"].mean(),
        "defense": by_team["opponent_score"].mean(),
    })
    stats["total"] = stats["offense"] + stats["defense"]
    stats["margin"] = stats["offense"] - stats["defense"]
    return stats


def _feature_row(team_a_stats, team_b_stats):
    return np.array([
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
    ], dtype=np.float32)


def build_score_dataset(game_years, stats_year_offset=-1):
    feature_rows = []
    y_a = []
    y_b = []
    meta = []
    for game_year in game_years:
        games = load_ncaa_games(game_year)
        stats_year = game_year + stats_year_offset
        stats_games = load_ncaa_games(stats_year)
        stats = _team_stats(stats_games)
        usable = 0
        for game in games.itertuples():
            if game.school_id not in stats.index or game.opponent_id not in stats.index:
                continue
            row = _feature_row(stats.loc[game.school_id], stats.loc[game.opponent_id])
            feature_rows.append(row)
            y_a.append(float(game.score))
            y_b.append(float(game.opponent_score))
            meta.append({
                "game_year": int(game_year),
                "stats_year": int(stats_year),
                "school_id": int(game.school_id),
                "opponent_id": int(game.opponent_id),
            })
            usable += 1
        print("Score dataset year %s: usable games=%s" % (game_year, usable))
    if not feature_rows:
        raise ValueError("No usable games for score dataset")
    x = np.vstack(feature_rows)
    return x, np.array(y_a), np.array(y_b), meta


def fit_ridge_regression(x, y, l2=1.0):
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std == 0] = 1.0
    x_scaled = (x - x_mean) / x_std
    y_mean = y.mean()
    y_centered = y - y_mean
    xtx = x_scaled.T @ x_scaled
    ridge = xtx + (l2 * np.eye(xtx.shape[0]))
    coef_scaled = np.linalg.solve(ridge, x_scaled.T @ y_centered)
    coef = coef_scaled / x_std
    intercept = y_mean - np.dot(x_mean, coef)
    return RegressionModel(coef=coef, intercept=float(intercept))


def baseline_predict_scores(x):
    pred_a = 0.5 * (x[:, 0] + x[:, 5])
    pred_b = 0.5 * (x[:, 4] + x[:, 1])
    return pred_a, pred_b


def _mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate_models(x, y_a, y_b, model_a, model_b, ensemble_weight):
    ridge_a = model_a.predict(x)
    ridge_b = model_b.predict(x)
    base_a, base_b = baseline_predict_scores(x)
    ens_a = (ensemble_weight * ridge_a) + ((1 - ensemble_weight) * base_a)
    ens_b = (ensemble_weight * ridge_b) + ((1 - ensemble_weight) * base_b)

    results = {
        "baseline": {
            "score_mae": (_mae(y_a, base_a) + _mae(y_b, base_b)) / 2,
            "winner_accuracy": winner_accuracy(y_a, y_b, base_a, base_b),
        },
        "ridge": {
            "score_mae": (_mae(y_a, ridge_a) + _mae(y_b, ridge_b)) / 2,
            "winner_accuracy": winner_accuracy(y_a, y_b, ridge_a, ridge_b),
        },
        "ensemble": {
            "score_mae": (_mae(y_a, ens_a) + _mae(y_b, ens_b)) / 2,
            "winner_accuracy": winner_accuracy(y_a, y_b, ens_a, ens_b),
        },
    }
    return results, ens_a, ens_b


def winner_accuracy(y_a, y_b, pred_a, pred_b):
    true_win = y_a > y_b
    pred_win = pred_a > pred_b
    return float(np.mean(true_win == pred_win))


def choose_ensemble_weight(x_val, y_a_val, y_b_val, model_a, model_b):
    best_weight = 0.5
    best_mae = math.inf
    for weight in np.linspace(0.0, 1.0, 21):
        _, ens_a, ens_b = evaluate_models(
            x_val, y_a_val, y_b_val, model_a, model_b, weight)
        mae = (_mae(y_a_val, ens_a) + _mae(y_b_val, ens_b)) / 2
        if mae < best_mae:
            best_mae = mae
            best_weight = float(weight)
    return best_weight


def save_pipeline(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_pipeline(path):
    with open(path) as f:
        payload = json.load(f)
    model_a = RegressionModel(
        coef=np.array(payload["ridge"]["a_coef"], dtype=np.float32),
        intercept=float(payload["ridge"]["a_intercept"]))
    model_b = RegressionModel(
        coef=np.array(payload["ridge"]["b_coef"], dtype=np.float32),
        intercept=float(payload["ridge"]["b_intercept"]))
    return payload, model_a, model_b


def matchup_features(team_a, team_b, year, stats_year_offset):
    stats_year = year + stats_year_offset
    stats = _team_stats(load_ncaa_games(stats_year))
    schools = load_ncaa_schools()
    team_a_id = int(team_name_to_id(team_a, schools))
    team_b_id = int(team_name_to_id(team_b, schools))
    if team_a_id not in stats.index:
        raise ValueError("No stats for team %s in %s" % (team_a, stats_year))
    if team_b_id not in stats.index:
        raise ValueError("No stats for team %s in %s" % (team_b, stats_year))
    x = _feature_row(stats.loc[team_a_id], stats.loc[team_b_id]).reshape(1, -1)
    return x, team_a_id, team_b_id, stats_year
