"""
Analyze the experimental results.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from os.path import dirname, join
from os import pardir
from collections import Counter, OrderedDict


import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import make_scorer, SCORERS
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE
from imblearn.metrics import geometric_mean_score
from gsmote import GeometricSMOTE
from rlearn.tools import (
    select_results,
    combine_results,
    calculate_mean_sem_scores,
    calculate_mean_sem_perc_diff_scores,
    calculate_mean_sem_ranking,
    apply_friedman_test,
    apply_holms_test,
    ImbalancedExperiment,
)
from pubs.utils import (
    load_datasets,
    BinaryDatasets,
    generate_mean_std_tbl,
    generate_pvalues_tbl,
    sort_tbl,
)


OVERSAMPLERS_NAMES = [
    'NO OVERSAMPLING',
    'RANDOM OVERSAMPLING',
    'SMOTE',
    'BORDERLINE SMOTE',
    'G-SMOTE',
]
CLASSIFIERS_NAMES = ['LR', 'KNN', 'DT', 'GBC']
SCORERS['geometric_mean_score'] = make_scorer(geometric_mean_score)
CONFIG = {
    'classifiers': [
        ('CONSTANT CLASSIFIER', DummyClassifier(strategy='constant', constant=0), {}),
        ('LR', LogisticRegression(solver='liblinear', multi_class='auto'), {}),
        ('KNN', KNeighborsClassifier(), {'n_neighbors': [3, 5]}),
        ('DT', DecisionTreeClassifier(), {'max_depth': [3, 6]}),
        (
            'GBC',
            GradientBoostingClassifier(),
            {'max_depth': [3, 6], 'n_estimators': [50, 100]},
        ),
    ],
    'scoring': ['accuracy', 'geometric_mean_score'],
    'n_splits': 5,
    'n_runs': 3,
    'rnd_seed': 0,
    'n_jobs': -1,
}
UNDERSAMPLING_RATIOS = [50, 75, 90, 95]
FACTORS = [None, 2, 4, 10, 20]
RESULTS_NAMES = [
    'no_oversampling',
    'random_oversampling',
    'smote',
    'borderline_smote',
    'gsmote',
]


def generate_results(ratio):
    """Generate results including all oversamplers."""

    # Load results
    results = []
    for name in RESULTS_NAMES:
        file_path = join(RESULTS_PATH, f'{name}_{ratio}.pkl')
        results.append(pd.read_pickle(file_path))

    # Combine results
    results = combine_results(*results)

    # Select results
    results = select_results(results, classifiers_names=CLASSIFIERS_NAMES)

    return results


def generate_main_results():
    """Generate the main results of the experiment."""

    main_results = {}
    for ratio in UNDERSAMPLING_RATIOS:

        # Generate results
        results = generate_results(ratio)

        # Calculate results
        mean_sem_scores = sort_tbl(
            generate_mean_std_tbl(*calculate_mean_sem_scores(results)),
            ovrs_order=OVERSAMPLERS_NAMES,
            clfs_order=CLASSIFIERS_NAMES,
        )
        mean_sem_perc_diff_scores = sort_tbl(
            generate_mean_std_tbl(
                *calculate_mean_sem_perc_diff_scores(
                    results, ['NO OVERSAMPLING', 'G-SMOTE']
                )
            ),
            ovrs_order=OVERSAMPLERS_NAMES,
            clfs_order=CLASSIFIERS_NAMES,
        )
        mean_sem_ranking = sort_tbl(
            generate_mean_std_tbl(*calculate_mean_sem_ranking(results)),
            ovrs_order=OVERSAMPLERS_NAMES,
            clfs_order=CLASSIFIERS_NAMES,
        )

        # Populate main results
        main_results_names = (
            'mean_sem_scores',
            'mean_sem_perc_diff_scores',
            'mean_sem_ranking',
        )
        main_results[ratio] = zip(
            main_results_names,
            (mean_sem_scores, mean_sem_perc_diff_scores, mean_sem_ranking),
        )

    return main_results


def generate_statistical_results():
    """Generate the statistical results of the experiment."""

    # Combine experiments objects
    results = []
    for ratio in UNDERSAMPLING_RATIOS:

        # Generate results
        partial_results = generate_results(ratio)

        # Extract results
        cols = partial_results.columns
        partial_results = partial_results.reset_index()
        partial_results['Dataset'] = partial_results['Dataset'].apply(
            lambda name: f'{name}({ratio})'
        )
        partial_results.set_index(
            ['Dataset', 'Oversampler', 'Classifier', 'params'], inplace=True
        )
        partial_results.columns = cols
        results.append(partial_results)

    # Combine results
    results = combine_results(*results)

    # Calculate statistical results
    friedman_test = sort_tbl(
        generate_pvalues_tbl(apply_friedman_test(results)),
        ovrs_order=OVERSAMPLERS_NAMES,
        clfs_order=CLASSIFIERS_NAMES,
    )
    holms_test = sort_tbl(
        generate_pvalues_tbl(apply_holms_test(results, control_oversampler='G-SMOTE')),
        ovrs_order=OVERSAMPLERS_NAMES[:-1],
        clfs_order=CLASSIFIERS_NAMES,
    )
    statistical_results_names = ('friedman_test', 'holms_test')
    statistical_results = zip(statistical_results_names, (friedman_test, holms_test))

    return statistical_results


def generate_oversamplers(factor):
    """Generate a list of oversamplers that pre-apply undersampling."""
    if factor is None:
        return [('BENCHMARK METHOD', None, {})]
    return [
        ('NO OVERSAMPLING', UnderOverSampler(oversampler=None, factor=factor), {}),
        (
            'RANDOM OVERSAMPLING',
            UnderOverSampler(oversampler=RandomOverSampler(), factor=factor),
            {},
        ),
        (
            'SMOTE',
            UnderOverSampler(oversampler=SMOTE(), factor=factor),
            {'oversampler__k_neighbors': [3, 5]},
        ),
        (
            'BORDERLINE SMOTE',
            UnderOverSampler(oversampler=BorderlineSMOTE(), factor=factor),
            {'oversampler__k_neighbors': [3, 5]},
        ),
        (
            'G-SMOTE',
            UnderOverSampler(oversampler=GeometricSMOTE(), factor=factor),
            {
                'oversampler__k_neighbors': [3, 5],
                'oversampler__selection_strategy': ['combined', 'minority', 'majority'],
                'oversampler__truncation_factor': [
                    -1.0,
                    -0.5,
                    0.0,
                    0.25,
                    0.5,
                    0.75,
                    1.0,
                ],
                'oversampler__deformation_factor': [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
            },
        ),
    ]


class UnderOverSampler(BaseOverSampler):
    """A class that applies random undersampling and oversampling."""

    def __init__(self, random_state=None, oversampler=None, factor=10):
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
        self.undersampler_ = RandomUnderSampler(
            random_state=self.random_state,
            sampling_strategy={k: int(v / self.factor) for k, v in counts.items()},
        )
        X_resampled, y_resampled = self.undersampler_.fit_resample(X, y)
        if self.oversampler is not None:
            self.oversampler_ = clone(self.oversampler).set_params(
                random_state=self.random_state, sampling_strategy=dict(counts)
            )
            X_resampled, y_resampled = self.oversampler_.fit_resample(
                X_resampled, y_resampled
            )
        return X_resampled, y_resampled


if __name__ == '__main__':

    import os
    from random import random, randint
    from mlflow import log_metric, log_param, log_artifacts

    log_param("param1", randint(0, 100))

    # BinaryDatasets().download().save(DATA_PATH, 'small_data_oversampling')

    # # Load datasets
    # datasets = load_datasets(DATA_PATH)

    # # Run experiments and save results
    # for factor in FACTORS:

    #     # Define undersampling ratio
    #     if factor is not None:
    #         ratio = int(100 * (factor - 1) / factor)

    #     # Generate oversamplers
    #     oversamplers = generate_oversamplers(factor)
    #     for oversampler in oversamplers:

    #         # Define and fit experiment
    #         experiment = ImbalancedExperiment(
    #             oversamplers=[oversampler],
    #             classifiers=CONFIG['classifiers'],
    #             scoring=CONFIG['scoring'],
    #             n_splits=CONFIG['n_splits'],
    #             n_runs=CONFIG['n_runs'],
    #             random_state=CONFIG['rnd_seed'],
    #             n_jobs=CONFIG['n_jobs']
    #         ).fit(datasets)

    #         # Save results
    #         experiment.results_.to_pickle(
    #             FILE_PATH.format(
    #                 oversampler[0].replace("-", "").replace(" ", "_").lower(),
    #                 ratio
    #             )
    #         )

    # # Main results
    # main_results = generate_main_results()
    # for ratio, results in main_results.items():
    #     for name, result in results:
    #         result.to_csv(join(ANALYSIS_PATH, f'{name}_{ratio}.csv'), index=False)

    # # Statistical results
    # statistical_results = generate_statistical_results()
    # for name, result in statistical_results:
    #     result.to_csv(join(ANALYSIS_PATH, f'{name}.csv'), index=False)
