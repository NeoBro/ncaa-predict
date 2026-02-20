#!/usr/bin/env python3
import argparse

import keras
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten
from keras.models import Sequential

from ncaa_predict.data_loader import N_FEATURES, N_PLAYERS, load_data, load_data_multiyear
from ncaa_predict.util import list_arg


DEFAULT_BATCH_SIZE = 512
DEFAULT_EPOCHS = 20


def configure_runtime(gpu_memory_growth, gpu_memory_limit_mb, mixed_precision):
    gpus = tf.config.list_physical_devices("GPU")
    if gpus and gpu_memory_limit_mb is not None:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(
                memory_limit=gpu_memory_limit_mb)])
        print("Configured GPU memory limit: %s MB on GPU 0" % gpu_memory_limit_mb)

    if gpus and gpu_memory_growth:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled GPU memory growth on %s GPU(s)" % len(gpus))

    if mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
        print("Enabled mixed precision policy: mixed_float16")


def build_model():
    model = Sequential([
        Conv2D(
            10, (2, N_PLAYERS), strides=(1, N_PLAYERS), activation="relu",
            input_shape=(2, N_PLAYERS, N_FEATURES)),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(16, activation="relu"),
        Dense(2, activation="softmax"),
    ])
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adagrad",
        metrics=["accuracy"])
    return model


def evaluate_year(model, year, label, player_year_offset):
    features, labels = load_data(year, player_year_offset=player_year_offset)
    loss, accuracy = model.evaluate(x=features, y=labels, verbose=0)
    print("%s %s: loss=%.5f accuracy=%.5f games=%s" %
          (label, year, loss, accuracy, len(features)))
    return loss, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size", "-b", default=DEFAULT_BATCH_SIZE, type=int,
        help="Training batch size. (default: %(default)s)")
    parser.add_argument(
        "--epochs", "-e", default=DEFAULT_EPOCHS, type=int,
        help="Number of training epochs. (default: %(default)s)")
    parser.add_argument(
        "--model-out", "-o", default=None,
        help="Output path for saved Keras model.")
    parser.add_argument(
        "--train-years", "-y", required=True,
        type=list_arg(type=int, container=list),
        help="Comma-separated years for training.")
    parser.add_argument(
        "--validation-years", "-v", required=True,
        type=list_arg(type=int, container=list),
        help="Comma-separated years for validation.")
    parser.add_argument(
        "--test-years", "-t", default=[],
        type=list_arg(type=int, container=list),
        help="Comma-separated years for final test reporting.")
    parser.add_argument(
        "--player-year-offset", default=-1, type=int,
        help="Offset applied to player stats year relative to game year. "
             "Use -1 to avoid same-season leakage. (default: %(default)s)")
    parser.add_argument(
        "--gpu-memory-growth", action="store_true", default=True,
        help="Enable TensorFlow GPU memory growth. (default: enabled)")
    parser.add_argument(
        "--no-gpu-memory-growth", action="store_false", dest="gpu_memory_growth",
        help="Disable TensorFlow GPU memory growth.")
    parser.add_argument(
        "--gpu-memory-limit-mb", default=None, type=int,
        help="Optional hard limit for GPU 0 memory in MB.")
    parser.add_argument(
        "--mixed-precision", action="store_true", default=False,
        help="Enable mixed_float16 policy.")
    args = parser.parse_args()

    overlap = set(args.train_years) & set(args.validation_years)
    if overlap:
        raise ValueError("Train/validation overlap is not allowed: %s" % sorted(overlap))

    configure_runtime(
        gpu_memory_growth=args.gpu_memory_growth,
        gpu_memory_limit_mb=args.gpu_memory_limit_mb,
        mixed_precision=args.mixed_precision)

    model = build_model()
    train_features, train_labels = load_data_multiyear(
        args.train_years, player_year_offset=args.player_year_offset)
    val_data = [
        load_data(year, player_year_offset=args.player_year_offset)
        for year in args.validation_years
    ]
    val_features = np.vstack([features for features, _ in val_data])
    val_labels = np.vstack([labels for _, labels in val_data])

    history = model.fit(
        x=train_features,
        y=train_labels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        shuffle=True,
        validation_data=(val_features, val_labels),
        verbose=1)

    best_epoch = int(np.argmax(history.history["val_accuracy"])) + 1
    best_val_acc = float(np.max(history.history["val_accuracy"]))
    print("Best validation accuracy: %.5f at epoch %s" % (best_val_acc, best_epoch))

    for year in args.validation_years:
        evaluate_year(
            model, year, "Validation",
            player_year_offset=args.player_year_offset)
    for year in args.test_years:
        evaluate_year(
            model, year, "Test",
            player_year_offset=args.player_year_offset)

    if args.model_out is not None:
        model.save(args.model_out)

    import gc
    gc.collect()
