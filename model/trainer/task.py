"""Executes model training and evaluation."""

import argparse
import logging
import os
import sys

import hypertune
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

from trainer.model import Word2VecEstimator

METRIC_FILE_NAME = "eval_metrics.joblib"
MODEL_FILE_NAME = "model.joblib"

BASE_QUERY = """
select
  hashtags
from `{table}`
limit 1000000
"""


def read_df_from_bigquery(full_table_path, project_id=None, num_samples=None):
    """Read data from BigQuery and split into train and validation sets.

    Args:
        full_table_path: (string) full path of the table containing training data
        in the format of [project_id.dataset_name.table_name].
        project_id: (string, Optional) Google BigQuery Account project ID.
        num_samples: (int, Optional) Number of data samples to read.

    Returns:
        pandas.DataFrame
    """

    query = BASE_QUERY.format(table=full_table_path)
    limit = "limit {}".format(num_samples) if num_samples else ""
    query += limit

    data_df = pd.read_gbq(query, project_id=project_id, dialect="standard")

    return data_df


def read_df_from_gcs(file_pattern):
    """Read data from Google Cloud Storage, split into train and validation sets.

    Assume that the data on GCS is in csv format without header.
    The column names will be provided through metadata

    Args:
        file_pattern: (string) pattern of the files containing training data.
        For example: [gs://bucket/folder_name/prefix]

    Returns:
        pandas.DataFrame
    """
    pass


def upload_to_gcs(local_path, gcs_path):
    """Upload local file to Google Cloud Storage.

    Args:
        local_path: (string) Local file
        gcs_path: (string) Google Cloud Storage destination

    Returns:
        None
    """
    pass


def dump_object(object_to_dump, output_path):
    """Pickle the object and save to the output_path.

    Args:
        object_to_dump: Python object to be pickled
        output_path: (string) output path which can be Google Cloud Storage

    Returns:
        None
    """

    with open(output_path, "wb") as wf:
        joblib.dump(object_to_dump, wf)


def _train_and_evaluate(estimator, dataset, output_dir):
    """Runs model training and evaluation.

    Args:
        estimator: (pipeline.Pipeline), Pipeline instance, assemble pre-processing
        steps and model training
        dataset: (pandas.DataFrame), DataFrame containing training data
        output_dir: (string), directory that the trained model will be exported

    Returns:
        None
    """
    estimator.fit(dataset["hashtags"])

    scores = model_selection.cross_val_score(estimator, dataset["hashtags"], cv=3)

    logging.info(scores)

    # Write model and eval metrics to `output_dir`
    model_output_path = os.path.join(output_dir, "model", MODEL_FILE_NAME)

    metric_output_path = os.path.join(output_dir, "experiment", METRIC_FILE_NAME)

    dump_object(estimator, model_output_path)
    dump_object(scores, metric_output_path)

    # The default name of the metric is training/hptuning/metric.
    # We recommend that you assign a custom name
    # The only functional difference is that if you use a custom name,
    # you must set the hyperparameterMetricTag value in the
    # HyperparameterSpec object in your job request to match your chosen name.
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag="loss",
        metric_value=np.mean(scores),
        global_step=1000,
    )


def run_experiment(flags):
    """Testbed for running model training and evaluation."""
    # Get data for training and evaluation

    dataset = read_df_from_bigquery(flags.input, num_samples=flags.num_samples)

    # Get model
    estimator = Word2VecEstimator(flags.vector_size, flags.window, flags.min_count)

    # Create pipeline
    pipeline = Pipeline([("word2vec", estimator)])

    # Run training and evaluation
    _train_and_evaluate(pipeline, dataset, flags.job_dir)


def parse_args(argv):
    """Parses command-line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        help="""Dataset to use for training and evaluation.
              Can be BigQuery table or a file (CSV).
              If BigQuery table, specify as as PROJECT_ID.DATASET.TABLE_NAME.
            """,
        required=True,
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
        "--num_samples",
        help="Number of samples to read from `input`",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--vector_size",
        help="Dimensionality of the word vectors.",
        default=50,
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
        default=10,
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
