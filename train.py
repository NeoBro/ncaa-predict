#!/usr/bin/env python3
import argparse
import functools
import multiprocessing
import os

import numpy as np
import pandas as pd
import tensorflow as tf


# All teams need to be the same size, so we pad them to this size
MAX_PLAYERS = 56

PLAYER_FEATURE_COLUMNS = [
    "height", "fg_percent", "3pt_percent", "freethrows_percent",
    "points_avg", "rebounds_avg", "assists_avg", "blocks_avg",
    "steals_avg"]


def load_csv(path, columns):
    this_dir = os.path.dirname(__file__)
    path = os.path.join(this_dir, path)
    df = pd.read_csv(path, usecols=list(columns))
    return df.apply(pd.to_numeric)


def load_ncaa_games(year):
    columns = ["year", "school_id", "opponent_id", "score", "opponent_score"]
    path = "csv/ncaa_games_%s.csv" % year
    return load_csv(path, columns)


def load_ncaa_players(year):
    columns = PLAYER_FEATURE_COLUMNS + ["school_id"]
    path = "csv/ncaa_players_%s.csv" % year
    players = load_csv(path, columns)
    players = players[~players["height"].isnull()]
    return players


def load_game(players, p):
    i, game = p
    this_team = players[players["school_id"] == game["school_id"]]
    this_team = this_team.as_matrix(columns=PLAYER_FEATURE_COLUMNS)
    other_team = players[players["school_id"] == game["opponent_id"]]
    other_team = other_team.as_matrix(columns=PLAYER_FEATURE_COLUMNS)

    if len(this_team) == 0 or len(other_team) == 0:
        return None, None
    while len(this_team) < MAX_PLAYERS:
        random_index = np.random.choice(this_team.shape[0])
        random_player = this_team[random_index]
        this_team = np.vstack([this_team, [random_player]])
    while len(other_team) < MAX_PLAYERS:
        random_index = np.random.choice(other_team.shape[0])
        random_player = other_team[random_index]
        other_team = np.vstack([other_team, [random_player]])
    teams = [this_team, other_team]
    if i % 1000 == 0:
        print("Handled row %s" % i)
    return np.stack(teams), [game["score"] > game["opponent_score"]]


def load_data(year):
    with multiprocessing.Pool(16) as pool:
        features_path = "data/features_%s.npy" % year
        labels_path = "data/labels_%s.npy" % year
        if not os.path.exists(features_path) \
                or not os.path.exists(labels_path):
            games = load_ncaa_games(year)
            players = load_ncaa_players(year).fillna(0)
            len_rows = games.shape[0]
            print("Iterating through %s games" % len_rows)
            f = functools.partial(load_game, players)
            res = pool.map(f, games.iterrows())
            features = [feature for feature, _ in res if feature is not None]
            labels = [label for _, label in res if label is not None]
            features = np.array(features, dtype=np.float32)
            labels = np.array(labels, dtype=np.float32)
            os.makedirs(os.path.dirname(features_path), exist_ok=True)
            os.makedirs(os.path.dirname(labels_path), exist_ok=True)
            np.save(features_path, features)
            np.save(labels_path, labels)
        return np.load(features_path), np.load(labels_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict-year", "-p", default=2016, type=int)
    parser.add_argument(
        "--train-years", "-y", default=list(range(2002, 2017)),
        type=lambda v: list(map(int, v.split(","))))
    parser.add_argument("--verbose", "-v", action="store_const", const=True)
    args = parser.parse_args()
    if args.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)

    data = [load_data(year) for year in args.train_years]
    features = np.vstack([features for features, _ in data])
    labels = np.vstack([labels for _, labels in data])
    assert len(features) == len(labels)

    feature_cols = \
        tf.contrib.learn.infer_real_valued_columns_from_input(features)

    estimator = tf.contrib.learn.LinearClassifier(
        feature_columns=feature_cols)
    estimator.fit(
        x=features, y=labels, steps=100000, batch_size=1000)

    test_features, test_labels = load_data(args.predict_year)
    print(estimator.evaluate(x=test_features, y=test_labels))
