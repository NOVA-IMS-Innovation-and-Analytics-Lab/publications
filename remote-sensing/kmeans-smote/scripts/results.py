"""
Generate the main experimental results.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
#         Joao Fonseca <jpmrfonseca@gmail.com>
# License: MIT

import sys
from os.path import join, dirname

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import make_scorer, SCORERS
from imblearn.over_sampling import (
    SMOTE,
    RandomOverSampler,
    BorderlineSMOTE,
    ADASYN
)
from clover.over_sampling import (
    ClusterOverSampler,
    KMeansSMOTE
)
from rlearn.tools import ImbalancedExperiment

sys.path.append(join(dirname(__file__), '..', '..', '..'))
from utils import load_datasets, generate_paths


CONFIG = {
    'oversamplers': [
       ('NONE', None, {}),
       ('ROS', RandomOverSampler(), {}),
       ('SMOTE', ClusterOverSampler(SMOTE()), {'oversampler__k_neighbors': [3, 5]}),
       ('B-SMOTE', ClusterOverSampler(BorderlineSMOTE()), {'oversampler__k_neighbors': [3, 5]}),
       ('K-SMOTE', KMeansSMOTE(), [
           {'kmeans_estimator':[1], 'imbalance_ratio_threshold': [np.inf], 'k_neighbors':[3, 5]}, {
           'kmeans_estimator': [.05, 0.1, 0.3, 0.5, 0.7, 0.9],
           'distances_exponent': ['auto', 2, 5, 7],
           'imbalance_ratio_threshold': ['auto', 0.5, 0.75, 1.0],
           'k_neighbors': [3, 5]
           }]
       ),
    ],
    'classifiers': [
        ('CONSTANT CLASSIFIER', DummyClassifier(strategy='prior'), {}),
        ('LR', LogisticRegression(multi_class='multinomial', solver='sag', penalty='none', max_iter=1e4), {}),
        ('KNN', KNeighborsClassifier(), {'n_neighbors': [3, 5, 8]}),
        ('RF', RandomForestClassifier(), {'max_depth':
        [None, 3, 6], 'n_estimators': [50, 100, 200]})
    ],
    'scoring': ['accuracy', 'f1_macro', 'geometric_mean_score_macro'],
    'n_splits': 5,
    'n_runs': 3,
    'rnd_seed': 0,
    'n_jobs': -1
}


if __name__ == '__main__':

    # Extract paths
    data_dir, results_dir, _ = generate_paths()

    # Load datasets
    datasets = load_datasets(data_dir=data_dir)

    # Extract oversamplers
    oversamplers = CONFIG['oversamplers']

    # Generate oversamplers
    for oversampler in oversamplers:

        # Define and fit experiment
        experiment = ImbalancedExperiment(
            oversamplers=[oversampler],
            classifiers=CONFIG['classifiers'],
            scoring=CONFIG['scoring'],
            n_splits=CONFIG['n_splits'],
            n_runs=CONFIG['n_runs'],
            random_state=CONFIG['rnd_seed'],
            n_jobs=CONFIG['n_jobs']
        ).fit(datasets)

        # Save results
        file_name = f'{oversampler[0].replace("-", "").lower()}.pkl'
        experiment.results_.to_pickle(join(results_dir, file_name))
