#!/usr/bin/env python3
import argparse

import keras

from ncaa_predict.data_loader import load_data


def evaluate(model, year, player_year_offset):
    features, labels = load_data(year, player_year_offset=player_year_offset)
    print("\nEvaluating accuracy")
    loss, accuracy = model.evaluate(x=features, y=labels, verbose=1)
    print("\nLoss: %s, Accuracy: %s" % (loss, accuracy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-in", "-m", required=True)
    parser.add_argument("--year", "-y", default=2016, type=int)
    parser.add_argument(
        "--player-year-offset", default=-1, type=int,
        help="Offset applied to player stats year relative to game year. "
             "(default: %(default)s)")
    args = parser.parse_args()

    model = keras.models.load_model(args.model_in)
    evaluate(model, args.year, args.player_year_offset)
