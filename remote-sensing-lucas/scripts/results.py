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
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import make_scorer, SCORERS
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from imblearn.metrics import geometric_mean_score
from gsmote import GeometricSMOTE
from rlearn.tools import ImbalancedExperiment

sys.path.append(join(dirname(__file__), '..', '..'))
from utils import load_datasets

SCORERS['geometric_mean_score'] = make_scorer(geometric_mean_score)
CONFIG = {
    'oversamplers': [
        ('NO OVERSAMPLING', None, {}),
        ('RANDOM OVERSAMPLING', RandomOverSampler(), {}),
        ('SMOTE', SMOTE(), {'k_neighbors': [3, 5]}),
        ('BORDERLINE SMOTE', BorderlineSMOTE(), {'k_neighbors': [3, 5]}),
        ('G-SMOTE', GeometricSMOTE(), {
            'k_neighbors': [3, 5],
            'selection_strategy': ['combined', 'minority', 'majority'],
            'truncation_factor': [-1.0, -0.5, .0, 0.25, 0.5, 0.75, 1.0],
            'deformation_factor': [.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
            }
        )],
    'classifiers': [
        ('CONSTANT CLASSIFIER', DummyClassifier(strategy='constant', constant=0), {}),
        ('LR', LogisticRegression(solver='liblinear', multi_class='auto'), {}),
        ('KNN', KNeighborsClassifier(), {'n_neighbors': [3, 5]}),
        ('DT', DecisionTreeClassifier(), {'max_depth': [3, 6]}),
        ('GBC', GradientBoostingClassifier(), {'max_depth': [3, 6], 'n_estimators': [50, 100]}),
    ],
    'scoring': ['accuracy', 'f1_macro', 'geometric_mean_score'],
    'n_splits': 5,
    'n_runs': 3,
    'rnd_seed': 0,
    'n_jobs': -1
}
DATA_PATH = join(dirname(__file__), '..', 'data')
FILE_PATH = join(dirname(__file__), '..', 'results', '{}.pkl')


if __name__ == '__main__':

    # Load lucas dataset
    datasets = load_datasets(data_path=DATA_PATH, data_type='csv')
    
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
        experiment.results_.to_pickle(
            FILE_PATH.format(oversampler[0].replace("-", "").replace(" ", "_").lower()
            )
        )
