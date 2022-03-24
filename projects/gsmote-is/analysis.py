"""
Analyze the experiment.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from os.path import join, dirname, basename
from tempfile import mkdtemp

import mlflow
from rlearn.tools import summarize_datasets
from tools.datasets import load_datasets_from_db

PATH = join(dirname(__file__), 'artifacts')


def run_analysis(datasets):

    with mlflow.start_run():

        # Datasets summary
        datasets_summary_path = join(mkdtemp(), 'datasets_summary.csv')
        summarize_datasets(datasets).to_csv(datasets_summary_path)
        mlflow.log_artifact(datasets_summary_path)


if __name__ == '__main__':
    datasets = load_datasets_from_db(join(PATH, 'experiment', 'imbalanced.db'))
    run_analysis(datasets)
