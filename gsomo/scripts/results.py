"""
Generate the main experimental results.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import sys
from collections import Counter, OrderedDict
from os.path import join, dirname

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import make_scorer, SCORERS
from imblearn.metrics import geometric_mean_score
from clover.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from clover.distribution import DensityDistributor
from rlearn.tools import ImbalancedExperiment
from gsmote import GeometricSMOTE
from somlearn import SOM

sys.path.append('../..')
from utils import load_datasets


SCORERS['geometric_mean_score'] = make_scorer(geometric_mean_score)
CONFIG = {
    'classifiers': [
        ('LR', LogisticRegression(solver='liblinear', multi_class='auto'), {}),
        ('KNN', KNeighborsClassifier(), {'n_neighbors': [3, 5]}),
        ('DT', DecisionTreeClassifier(), {'max_depth': [3, 6]}),
        ('GBC', GradientBoostingClassifier(), {'max_depth': [3, 6], 'n_estimators': [50, 100]}),
    ],
    'oversamplers': [
        ('NO OVERSAMPLING', None, {}),
        ('RANDOM OVERSAMPLING', RandomOverSampler(), {}),
        ('SMOTE', SMOTE(), {'k_neighbors': [3, 5]}),
        ('BORDERLINE SMOTE', BorderlineSMOTE(), {'k_neighbors': [3, 5]}),
        ('ADASYN', ADASYN(), {'n_neighbors': [2, 3]}),
        ('G-SMOTE', GeometricSMOTE(), {
            'k_neighbors': [3, 5],
            'selection_strategy': ['combined', 'minority', 'majority'],
            'truncation_factor': [-1.0, -0.5, .0, 0.25, 0.5, 0.75, 1.0],
            'deformation_factor': [.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
        },
        ('SOMO', SMOTE(clusterer=SOM(), distributor=DensityDistributor()), {
            'k_neighbors': [3, 5],
            'clusterer__n_clusters': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'distributor__distances_exponent': [0, 1, 2, 5],
            'distributor__filtering_threshold': [0.0, 0.5, 1.0, 2.0],
            'distributor__distribution_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]
        },
        ('G-SOMO', GeometricSMOTE(clusterer=SOM(), distributor=DensityDistributor()), {
            'k_neighbors': [3, 5],
            'clusterer__n_clusters': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'distributor__distances_exponent': [0, 1, 2, 5],
            'distributor__filtering_threshold': [0.0, 0.5, 1.0, 2.0],
            'distributor__distribution_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]
        }
    )
    )

    )
    ],
    'scoring': ['f1', 'geometric_mean_score', 'roc_auc'],
    'n_splits': 5,
    'n_runs': 3,
    'rnd_seed': 0,
    'n_jobs': -1
}
DATA_PATH = join(dirname(__file__), '..', 'data', 'cgan.db')
RESULTS_PATH = join(dirname(__file__), '..', 'results', '{}.pkl')


if __name__ == '__main__':

    # TODO: Modify it to correspond to the paper

    # Load datasets
    datasets = load_datasets(DATA_PATH)

    # Run experiments and save results
    for oversampler in CONFIG['oversamplers']:

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
        name = oversampler[0].replace("-", "").replace(" ", "_").lower()
        experiment.results_.to_pickle(RESULTS_PATH.format(name))
