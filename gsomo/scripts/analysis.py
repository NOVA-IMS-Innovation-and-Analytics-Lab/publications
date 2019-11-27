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
    combine_results,
    calculate_mean_sem_scores,
    calculate_mean_sem_perc_diff_scores,
    calculate_mean_sem_ranking,
    apply_friedman_test,
    apply_holms_test
)

sys.path.append(join(dirname(__file__), '..', '..'))
from utils import generate_mean_std_tbl, generate_pvalues_tbl, sort_tbl

RESULTS_NAMES = [
    'no_oversampling', 
    'random_oversampling',
    'smote',
    'kmeans_smote',
    'somo',
    'gsmote',
    'gsomo'
]
OVERSAMPLERS_NAMES = ['NO OVERSAMPLING', 'RANDOM OVERSAMPLING', 'SMOTE', 'K-MEANS SMOTE', 'SOMO', 'G-SMOTE', 'G-SOMO']
CLASSIFIERS_NAMES = ['LR', 'KNN', 'DT', 'GBC']
SCORERS['geometric_mean_score'] = make_scorer(geometric_mean_score)
RESULTS_PATH = join(dirname(__file__), '..', 'results')
ANALYSIS_PATH = join(dirname(__file__), '..', 'analysis')


def generate_results():
    """Generate results including all oversamplers."""

    # Load results
    results = []
    for name in RESULTS_NAMES:
        file_path = join(RESULTS_PATH, f'{name}.pkl')
        results.append(pd.read_pickle(file_path))
        
    # Combine results
    results = combine_results(*results)

    return results
    

def generate_main_results():
    """Generate the main results of the experiment."""

    # Generate results   
    results = generate_results()

    # Calculate results
    mean_sem_scores = sort_tbl(generate_mean_std_tbl(*calculate_mean_sem_scores(results)), ovrs_order=OVERSAMPLERS_NAMES, clfs_order=CLASSIFIERS_NAMES)
    keys = mean_sem_scores[['Classifier', 'Metric']]
    mean_sem_perc_diff_scores = []
    for oversampler in ('SMOTE', 'K-MEANS SMOTE', 'SOMO', 'G-SMOTE'):
        perc_diff_scores = sort_tbl(generate_mean_std_tbl(*calculate_mean_sem_perc_diff_scores(results, [oversampler, 'G-SOMO'])), ovrs_order=OVERSAMPLERS_NAMES, clfs_order=CLASSIFIERS_NAMES)
        perc_diff_scores = perc_diff_scores.rename(columns={'Difference': oversampler}).drop(columns=['Classifier', 'Metric'])
        mean_sem_perc_diff_scores.append(perc_diff_scores)
    mean_sem_perc_diff_scores = pd.concat([keys, pd.concat(mean_sem_perc_diff_scores, axis=1)], axis=1)
    mean_sem_ranking = sort_tbl(generate_mean_std_tbl(*calculate_mean_sem_ranking(results)), ovrs_order=OVERSAMPLERS_NAMES, clfs_order=CLASSIFIERS_NAMES)

    # Generate main results
    main_results_names = ('mean_sem_scores', 'mean_sem_perc_diff_scores', 'mean_sem_ranking')
    main_results = zip(main_results_names, (mean_sem_scores, mean_sem_perc_diff_scores, mean_sem_ranking))
    
    return main_results


def generate_statistical_results():
    """Generate the statistical results of the experiment."""
    
    # Generate results   
    results = generate_results()

    # Calculate statistical results
    friedman_test = sort_tbl(generate_pvalues_tbl(apply_friedman_test(results)), ovrs_order=OVERSAMPLERS_NAMES, clfs_order=CLASSIFIERS_NAMES)
    holms_test = sort_tbl(generate_pvalues_tbl(apply_holms_test(results, control_oversampler='G-SOMO')), ovrs_order=OVERSAMPLERS_NAMES[:-1], clfs_order=CLASSIFIERS_NAMES)
    
    # Generate statistical results
    statistical_results_names = ('friedman_test', 'holms_test')
    statistical_results = zip(statistical_results_names, (friedman_test, holms_test))

    return statistical_results


if __name__ == '__main__':
    
    # Main results
    main_results = generate_main_results()
    for name, result in main_results:
        result.to_csv(join(ANALYSIS_PATH, f'{name}.csv'), index=False)

    # Statistical results
    statistical_results = generate_statistical_results()
    for name, result in statistical_results:
        result.to_csv(join(ANALYSIS_PATH, f'{name}.csv'), index=False)
