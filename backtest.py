#!/usr/bin/env python3
import argparse
import statistics

from train import build_model
from ncaa_predict.data_loader import load_data, load_data_multiyear


def run_fold(train_years, validation_year, epochs, batch_size, player_year_offset):
    model = build_model()
    train_features, train_labels = load_data_multiyear(
        train_years, player_year_offset=player_year_offset)
    val_features, val_labels = load_data(
        validation_year, player_year_offset=player_year_offset)
    model.fit(
        x=train_features,
        y=train_labels,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        shuffle=True,
        validation_data=(val_features, val_labels))
    loss, accuracy = model.evaluate(x=val_features, y=val_labels, verbose=0)
    print(
        "Fold train=%s-%s validate=%s loss=%.5f accuracy=%.5f games=%s" % (
            min(train_years), max(train_years), validation_year,
            loss, accuracy, len(val_features)))
    return loss, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument(
        "--min-train-years",
        type=int,
        default=5,
        help="Minimum number of years required in the initial training window.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--player-year-offset", type=int, default=-1,
        help="Offset applied to player stats year relative to game year. "
             "(default: %(default)s)")
    args = parser.parse_args()

    if args.end_year <= args.start_year:
        raise ValueError("end-year must be greater than start-year")

    folds = []
    first_validation_year = args.start_year + args.min_train_years
    if first_validation_year > args.end_year:
        raise ValueError("Not enough years for requested min-train-years")

    for validation_year in range(first_validation_year, args.end_year + 1):
        train_years = list(range(args.start_year, validation_year))
        folds.append(run_fold(
            train_years=train_years,
            validation_year=validation_year,
            epochs=args.epochs,
            batch_size=args.batch_size,
            player_year_offset=args.player_year_offset))

    losses = [loss for loss, _ in folds]
    accuracies = [acc for _, acc in folds]
    print("Backtest summary:")
    print("folds=%s" % len(folds))
    print("loss mean=%.5f std=%.5f" % (statistics.mean(losses), statistics.pstdev(losses)))
    print("accuracy mean=%.5f std=%.5f" % (statistics.mean(accuracies), statistics.pstdev(accuracies)))
