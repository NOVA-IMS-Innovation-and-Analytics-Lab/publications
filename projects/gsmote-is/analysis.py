"""
Analyze the experiment.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from os.path import join, dirname
from tempfile import mkdtemp

import mlflow
from rlearn.tools import (
    summarize_datasets,
    calculate_wide_optimal,
    calculate_mean_sem_ranking,
)
from tools.datasets import load_datasets_from_db

PATH = join(dirname(__file__), 'artifacts')


def run_analysis(datasets):

    with mlflow.start_run():

        model_path = join(mlflow.get_artifact_uri(), 'experiment_model')
        experiment = mlflow.sklearn.load_model(model_path)

        # Datasets summary
        datasets_summary_path = join(mkdtemp(), 'datasets_summary.csv')
        summarize_datasets(datasets).to_csv(datasets_summary_path)
        mlflow.log_artifact(datasets_summary_path)

        # Wide optimal results
        wide_optimal = calculate_wide_optimal(experiment.results_)

        # Mean ranking
        from imblearn.metrics import geometric_mean_score
        from sklearn.metrics import SCORERS, make_scorer

        SCORERS['geometric_mean_score'] = make_scorer(geometric_mean_score)
        mean_ranking = calculate_mean_sem_ranking(experiment.results_)
        print(mean_ranking)


if __name__ == '__main__':
    datasets = load_datasets_from_db(join(PATH, 'experiment', 'imbalanced.db'))
    run_analysis(datasets)
