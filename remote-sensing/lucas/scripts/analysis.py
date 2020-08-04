"""
Analyze the experimental results.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
#         João Fonseca <jpmrfonseca@gmail.com>
# License: MIT

import sys
from os.path import join, dirname
from collections import Counter, OrderedDict

from scipy.stats import wilcoxon
import numpy as np
import pandas as pd
from rlearn.tools import (
    combine_results,
    select_results,
    calculate_wide_optimal,
    calculate_ranking,
    calculate_mean_sem_perc_diff_scores
)

from utils import (
    sort_tbl, 
    generate_paths, 
    load_datasets, 
    make_bold, 
    generate_pvalues_tbl,
    SCORERS
)

LABELS_MAPPING = {'A':1, 'B':2, 'C':0, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7}
RESULTS_NAMES = ('none', 'ros', 'smote', 'bsmote', 'adasyn', 'gsmote')
OVRS_NAMES = ('NONE', 'ROS', 'SMOTE', 'B-SMOTE', 'ADASYN', 'G-SMOTE')
CLFS_NAMES = ('LR', 'KNN', 'DT', 'GBC', 'RF')
METRICS_MAPPING = OrderedDict([('accuracy', 'Accuracy'), ('f1_macro', 'F-score'), ('geometric_mean_score_macro', 'G-mean')])
BASELINE_OVRS = ('NONE', 'ROS', 'SMOTE')
MAIN_RESULTS_NAMES = ('dataset_description', 'wide_optimal', 'ranking', 'perc_diff_scores', 'wilcoxon_results')
ALPHA = 0.01


def describe_dataset(dataset):
    """Generates dataframe with dataset description."""
    name, (X, y) = dataset
    counts = Counter(y)
    description = [
        ['Dataset', name],
        ['Features', X.shape[-1] - 1],
        ['Instances', X.shape[0]],
        ['Instances of class C', counts[LABELS_MAPPING['C']]],
        ['Instances of class H', counts[LABELS_MAPPING['H']]],
        ['IR of class H', counts[LABELS_MAPPING['C']] / counts[LABELS_MAPPING['H']]]
    ]
    return pd.DataFrame(description)


def generate_main_results(data_path, results_path):
    """Generate the main results of the experiment."""
    
    # Load dataset
    dataset = load_datasets(data_dir=data_path)[0]

    # Load results
    results = []
    for name in RESULTS_NAMES:
        file_path = join(results_path, f'{name}.pkl')
        results.append(pd.read_pickle(file_path))
        
    # Combine and select results
    results = combine_results(*results)
    results = select_results(results, oversamplers_names=OVRS_NAMES, classifiers_names=CLFS_NAMES)

    # Extract metrics names
    metrics_names, *_ = zip(*METRICS_MAPPING.items())

    # Dataset description
    dataset_description = describe_dataset(dataset)
    
    # Scores
    wide_optimal = calculate_wide_optimal(results).drop(columns='Dataset')
    
    # Ranking
    ranking = calculate_ranking(results).drop(columns='Dataset')
    ranking.iloc[:, 2:] = ranking.iloc[:, 2:].astype(int)
    
    # Percentage difference
    perc_diff_scores = []
    for oversampler in BASELINE_OVRS:
        perc_diff_scores_ovs = calculate_mean_sem_perc_diff_scores(results, [oversampler, 'G-SMOTE'])[0]
        perc_diff_scores_ovs = perc_diff_scores_ovs[['Difference']].rename(columns={'Difference': oversampler})
        perc_diff_scores.append(perc_diff_scores_ovs)
    perc_diff_scores = sort_tbl(pd.concat([ranking[['Classifier', 'Metric']], pd.concat(perc_diff_scores, axis=1)], axis=1), clfs_order=CLFS_NAMES, ovrs_order=OVRS_NAMES, metrics_order=metrics_names)
    perc_diff_scores.iloc[:, 2:] = round(perc_diff_scores.iloc[:, 2:], 2)
    
    # Wilcoxon test
    pvalues = []
    for ovr in OVRS_NAMES[:-1]:
        mask = (wide_optimal['Metric'] != 'accuracy') if ovr == 'NONE' else np.repeat(True, len(wide_optimal))
        pvalues.append(wilcoxon(wide_optimal.loc[mask, ovr], wide_optimal.loc[mask, 'G-SMOTE']).pvalue)
    wilcoxon_results = pd.DataFrame({'Oversampler': OVRS_NAMES[:-1], 'p-value': pvalues, 'Significance': np.array(pvalues) < ALPHA})
        
    # Format results
    main_results = [(MAIN_RESULTS_NAMES[0], dataset_description)]
    for name, result in zip(MAIN_RESULTS_NAMES[1:], (wide_optimal, ranking, perc_diff_scores, wilcoxon_results)):
        if name != 'wilcoxon_results':
            result = sort_tbl(result, clfs_order=CLFS_NAMES, ovrs_order=OVRS_NAMES, metrics_order=metrics_names)
            result['Metric'] = result['Metric'].apply(lambda metric: METRICS_MAPPING[metric])
        if name == 'wide_optimal':
            result.iloc[:, 2:] = result.iloc[:, 2:].apply(lambda row: make_bold(row, True, 3), axis=1)
        elif name == 'ranking':
            result.iloc[:, 2:] = result.iloc[:, 2:].apply(lambda row: make_bold(row, False, 0), axis=1)
        elif name == 'wilcoxon_results':
            wilcoxon_results = generate_pvalues_tbl(wilcoxon_results)
        main_results.append((name, result))

    return main_results


if __name__ == '__main__':

    # Extract paths
    data_path, results_path, analysis_path = generate_paths()

    # Generate and save main results
    results = generate_main_results(data_path, results_path)
    for name, result in results:
        result.to_csv(join(analysis_path, f'{name}.csv'), index=False, header=(name != 'dataset_description'))
