"""Executes model training and evaluation."""

import argparse
import logging
import os
import subprocess
import sys

import hypertune
import tensorflow as tf
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

from trainer.model import Word2VecEstimator


MODEL_FILE_NAME = "model.joblib"


def dump_object(object_to_dump, output_path):
    """Pickle the object and save to the output_path.

    Args:
        object_to_dump: Python object to be pickled
        output_path: (string) output path which can be Google Cloud Storage

    Returns:
        None
    """
    if not tf.io.gfile.exists(output_path):
        tf.io.gfile.makedirs(os.path.dirname(output_path))
    with tf.io.gfile.GFile(output_path, "w") as wf:
        joblib.dump(object_to_dump, wf)


def _train_and_evaluate(estimator, dataset_path, output_dir):
    """Runs model training and evaluation.

    Args:
        estimator: (pipeline.Pipeline), Pipeline instance, assemble pre-processing
        steps and model training
        dataset_path: (string), Path containing training data
        output_dir: (string), directory that the trained model will be exported

    Returns:
        None
    """
    estimator.fit(dataset_path)

    loss = estimator.score(dataset_path)

    logging.info(loss)

    # Write model and eval metrics to `output_dir`
    model_output_path = os.path.join(output_dir, "model", MODEL_FILE_NAME)

    dump_object(estimator, model_output_path)

    # The default name of the metric is training/hptuning/metric.
    # We recommend that you assign a custom name
    # The only functional difference is that if you use a custom name,
    # you must set the hyperparameterMetricTag value in the
    # HyperparameterSpec object in your job request to match your chosen name.
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag="loss", metric_value=loss, global_step=1000,
    )


def run_experiment(flags):
    """Testbed for running model training and evaluation."""

    # Handle remote files
    if flags.input.startswith("gs://"):
        file_name = flags.input.split("/")[-1]
        file_path = f"data/{file_name}"
        subprocess.run(["gsutil", "cp", flags.input, file_path])
        flags.input = file_path

    # Get model
    estimator = Word2VecEstimator(
        flags.vector_size,
        flags.window,
        flags.min_count,
        flags.workers,
        flags.iterations,
    )

    # Create pipeline
    pipeline = Pipeline([("word2vec", estimator)])

    # Run training and evaluation
    _train_and_evaluate(pipeline, flags.input, flags.job_dir)


def parse_args(argv):
    """Parses command-line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input", help="Path to dataset.", required=True,
    )

    parser.add_argument(
        "--job-dir",
        help="Output directory for exporting model and other ",
        required=True,
    )

    parser.add_argument(
        "--log_level",
        help="Logging level.",
        choices=["DEBUG", "ERROR", "FATAL", "INFO", "WARN",],
        default="INFO",
    )

    parser.add_argument(
        "--vector_size",
        help="Dimensionality of the word vectors.",
        default=100,
        type=int,
    )

    parser.add_argument(
        "--window",
        help="Maximum distance between the current and predicted word within a sentence.",
        default=5,
        type=int,
    )

    parser.add_argument(
        "--min_count",
        help="The minimum number of samples required to be included in vocabulary.",
        default=5,
        type=int,
    )

    parser.add_argument(
        "--workers", help="Worker threads to train the model.", default=3, type=int,
    )

    parser.add_argument(
        "--iterations",
        help="Number of iterations (epochs) over the corpus.",
        default=5,
        type=int,
    )

    return parser.parse_args(argv)


def main():
    """Entry point."""
    flags = parse_args(sys.argv[1:])
    logging.basicConfig(level=flags.log_level.upper())
    run_experiment(flags)


if __name__ == "__main__":
    main()
