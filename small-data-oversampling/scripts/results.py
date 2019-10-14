"""
Generate the main experimental results.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import sys
from collections import Counter, OrderedDict
from os.path import join, dirname

import pandas as pd
from sklearn.base import clone
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

sys.path.append('../..')
from utils import load_datasets


def generate_oversamplers(factor):
    """Generate a list of oversamplers that pre-apply undersampling."""
    if factor is None:
        return [('BENCHMARK METHOD', None, {})]
    return [
        ('NO OVERSAMPLING', UnderOverSampler(oversampler=None, factor=factor), {}),
        ('RANDOM OVERSAMPLING', UnderOverSampler(oversampler=RandomOverSampler(), factor=factor), {}),
        ('SMOTE', UnderOverSampler(oversampler=SMOTE(), factor=factor), {'oversampler__k_neighbors': [3, 5]}),
        ('BORDERLINE SMOTE', UnderOverSampler(oversampler=BorderlineSMOTE(), factor=factor), {'oversampler__k_neighbors': [3, 5]}),
        ('G-SMOTE', UnderOverSampler(oversampler=GeometricSMOTE(), factor=factor), {
            'oversampler__k_neighbors': [3, 5],
            'oversampler__selection_strategy': ['combined', 'minority', 'majority'],
            'oversampler__truncation_factor': [-1.0, -0.5, .0, 0.25, 0.5, 0.75, 1.0],
            'oversampler__deformation_factor': [.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
            }
        )]


class UnderOverSampler(BaseOverSampler):
    """A class that applies random undersampling and oversampling."""

    def __init__(self,
                 random_state=None,
                 oversampler=None,
                 factor=10):
        super(UnderOverSampler, self).__init__(sampling_strategy='auto')
        self.random_state = random_state
        self.oversampler = oversampler
        self.factor = factor

    def fit(self, X, y):
        self._deprecate_ratio()
        X, y, _ = self._check_X_y(X, y)
        self.sampling_strategy_ = OrderedDict([(list(Counter(y).keys())[0], 0)])
        return self

    def _fit_resample(self, X, y):
        counts = Counter(y)
        self.undersampler_ = RandomUnderSampler(random_state=self.random_state, sampling_strategy={k:int(v / self.factor) for k,v in counts.items()})
        X_resampled, y_resampled = self.undersampler_.fit_resample(X, y)
        if self.oversampler is not None:
            self.oversampler_ = clone(self.oversampler).set_params(random_state=self.random_state, sampling_strategy=dict(counts))
            X_resampled, y_resampled = self.oversampler_.fit_resample(X_resampled, y_resampled)
        return X_resampled, y_resampled


SCORERS['geometric_mean_score'] = make_scorer(geometric_mean_score)
CONFIG = {
    'classifiers': [
        ('CONSTANT CLASSIFIER', DummyClassifier(strategy='constant', constant=0), {}),
        ('LR', LogisticRegression(solver='liblinear', multi_class='auto'), {}),
        ('KNN', KNeighborsClassifier(), {'n_neighbors': [3, 5]}),
        ('DT', DecisionTreeClassifier(), {'max_depth': [3, 6]}),
        ('GBC', GradientBoostingClassifier(), {'max_depth': [3, 6], 'n_estimators': [50, 100]}),
    ],
    'scoring': ['accuracy', 'geometric_mean_score'],
    'n_splits': 5,
    'n_runs': 3,
    'rnd_seed': 0,
    'n_jobs': -1
}
FACTORS = [None, 2, 4, 10, 20]
DATA_PATH = join(dirname(__file__), '..', 'data', 'small_data_oversampling.db')
FILE_PATH = join(dirname(__file__), '..', 'results', '{}_{}.pkl')


if __name__ == '__main__':

    # Load datasets
    datasets = load_datasets(DATA_PATH)

    # Run experiments and save results
    for factor in FACTORS:
        
        # Define undersampling ratio
        if factor is not None:
            ratio = int(100 * (factor - 1) / factor)
        
        # Generate oversamplers
        oversamplers = generate_oversamplers(factor)
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
                FILE_PATH.format(
                    oversampler[0].replace("-", "").replace(" ", "_").lower(),
                    ratio
                )
            )

