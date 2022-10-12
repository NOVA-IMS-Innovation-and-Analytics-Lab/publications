"""
Run the experiment.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from os.path import join, dirname, pardir

import mlflow
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import RandomOverSampler, SMOTE
from gsmote import GeometricSMOTE
from rlearn.experiment import ImbalancedClassificationExperiment
from tools.datasets import load_datasets_from_db


OVERSAMPLERS = [
    ('none', None, {}),
    ('ros', RandomOverSampler(), {}),
    ('smote', SMOTE(), {'k_neighbors': [3, 5]}),
    (
        'gsmote',
        GeometricSMOTE(),
        {
            'k_neighbors': [3, 5],
            'truncation_factor': [-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0],
            'deformation_factor': [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
        },
    ),
]
CLASSIFIERS = [
    ('lr', LogisticRegression(solver='saga', max_iter=1e4), {}),
    ('knn', KNeighborsClassifier(), {'n_neighbors': [3, 5]}),
    ('dt', DecisionTreeClassifier(), {'max_depth': [3, 6]}),
    (
        'gb',
        GradientBoostingClassifier(),
        {'max_depth': [3, 6], 'n_estimators': [50, 100]},
    ),
]
SCORING = {
    'roc_auc': make_scorer(roc_auc_score),
    'f1': make_scorer(f1_score),
    'geometric_mean_score': make_scorer(geometric_mean_score),
}
N_SPLITS = 5
N_RUNS = 3
RND_SEED = 10


def run_experiment():
    """Run the experiment and log the experiment model."""
    datasets_path = join(dirname(__file__), pardir, 'datasets', 'imbalanced_binary.db')
    datasets = load_datasets_from_db(datasets_path)
    with mlflow.start_run():
        experiment = ImbalancedClassificationExperiment(
            oversamplers=OVERSAMPLERS,
            classifiers=CLASSIFIERS,
            scoring=SCORING,
            n_splits=N_SPLITS,
            n_runs=3,
            random_state=RND_SEED,
            verbose=1,
        )
        experiment.fit(datasets)
        mlflow.sklearn.log_model(experiment, 'experiment_model')


if __name__ == '__main__':

    run_experiment()
