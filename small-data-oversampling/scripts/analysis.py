"""
Analyze the experimental results.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import sys
from os.path import dirname, join
from itertools import product

import pandas as pd
from sklearn.metrics import SCORERS, make_scorer
from imblearn.metrics import geometric_mean_score
from rlearn.tools import (
    select_results,
    combine_results,
    calculate_mean_sem_scores,
    calculate_mean_sem_perc_diff_scores,
    calculate_mean_sem_ranking,
    apply_friedman_test,
    apply_holms_test
)

sys.path.append('../..')
from utils import generate_mean_std_tbl, generate_pvalues_tbl, sort_tbl

UNDERSAMPLING_RATIOS = [50, 75, 90, 95]
RESULTS_NAMES = [
    'no_oversampling', 
    'random_oversampling',
    'smote',
    'borderline_smote',
    'gsmote'
]
OVERSAMPLERS_NAMES = ['NO OVERSAMPLING', 'RANDOM OVERSAMPLING', 'SMOTE', 'BORDERLINE SMOTE', 'G-SMOTE']
CLASSIFIERS_NAMES = ['LR', 'DT', 'KNN', 'GBC']
SCORERS['geometric_mean_score'] = make_scorer(geometric_mean_score)
RESULTS_PATH = join(dirname(__file__), '..', 'results')
ANALYSIS_PATH = join(dirname(__file__), '..', 'analysis')

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
        mean_sem_scores = sort_tbl(generate_mean_std_tbl(*calculate_mean_sem_scores(results)), ovrs_order=OVERSAMPLERS_NAMES, clfs_order=CLASSIFIERS_NAMES)
        mean_sem_perc_diff_scores = sort_tbl(generate_mean_std_tbl(*calculate_mean_sem_perc_diff_scores(results, ['NO OVERSAMPLING', 'G-SMOTE'])), ovrs_order=OVERSAMPLERS_NAMES, clfs_order=CLASSIFIERS_NAMES)
        mean_sem_ranking = sort_tbl(generate_mean_std_tbl(*calculate_mean_sem_ranking(results)), ovrs_order=OVERSAMPLERS_NAMES, clfs_order=CLASSIFIERS_NAMES)

        # Populate main results
        main_results_names = ('mean_sem_scores', 'mean_sem_perc_diff_scores', 'mean_sem_ranking')
        main_results[ratio] = zip(main_results_names, (mean_sem_scores, mean_sem_perc_diff_scores, mean_sem_ranking))
    
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
        partial_results['Dataset'] = partial_results['Dataset'].apply(lambda name: f'{name}({ratio})')
        partial_results.set_index(['Dataset', 'Oversampler', 'Classifier', 'params'], inplace=True)
        partial_results.columns = cols
        results.append(partial_results)

    # Combine results
    results = combine_results(*results)

    # Calculate statistical results
    friedman_test = sort_tbl(generate_pvalues_tbl(apply_friedman_test(results)), ovrs_order=OVERSAMPLERS_NAMES, clfs_order=CLASSIFIERS_NAMES)
    holms_test = sort_tbl(generate_pvalues_tbl(apply_holms_test(results, control_oversampler='G-SMOTE')), ovrs_order=OVERSAMPLERS_NAMES, clfs_order=CLASSIFIERS_NAMES)
    statistical_results_names = ('friedman_test', 'holms_test')
    statistical_results = zip(statistical_results_names, (friedman_test, holms_test))

    return statistical_results


if __name__ == '__main__':
    
    # Main results
    main_results = generate_main_results()
    for ratio, results in main_results.items():
        for name, result in results:
            result.to_csv(join(ANALYSIS_PATH, f'{name}_{ratio}.csv'), index=False)

    # Statistical results
    statistical_results = generate_statistical_results()
    for name, result in statistical_results:
        result.to_csv(join(ANALYSIS_PATH, f'{name}.csv'), index=False)
