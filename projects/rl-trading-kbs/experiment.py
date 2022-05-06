"""
Run the experiment.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import sys
from os.path import join, dirname

import mlflow
from rlearn.tools import ImbalancedExperiment
from tools.datasets import load_datasets_from_db

from artifacts.experiment.config import OVERSAMPLERS, CLASSIFIERS, SCORING

PATH = join(dirname(__file__), 'artifacts')


def run_experiment(datasets):

    with mlflow.start_run():
        experiment = ImbalancedExperiment(
            oversamplers=OVERSAMPLERS,
            classifiers=CLASSIFIERS,
            scoring=SCORING,
            n_splits=int(sys.argv[1]),
            n_runs=int(sys.argv[2]),
            random_state=int(sys.argv[3]),
        )
        experiment.fit(datasets)
        mlflow.sklearn.log_model(experiment, 'experiment_model')


if __name__ == '__main__':
    datasets = load_datasets_from_db(join(PATH, 'experiment', 'imbalanced.db'))
    run_experiment(datasets)
