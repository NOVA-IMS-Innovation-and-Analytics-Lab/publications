"""
Analyze the experimental results.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
#         João Fonseca <jpmrfonseca@gmail.com>
# License: MIT

import sys
from os.path import join, dirname
from collections import OrderedDict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rlearn.tools import (
    combine_results,
    select_results,
    summarize_datasets,
    calculate_mean_sem_scores,
    calculate_mean_sem_perc_diff_scores,
    calculate_mean_sem_ranking,
    apply_friedman_test,
    apply_holms_test
)

from utils import (
    load_datasets, 
    generate_paths,  
    generate_pvalues_tbl,
    generate_mean_std_tbl,
    sort_tbl,
    make_bold,
    RemoteSensingDatasets
)

RESULTS_NAMES = ('none', 'ros', 'smote', 'bsmote', 'ksmote')
OVRS_NAMES = ('NONE', 'ROS', 'SMOTE', 'B-SMOTE', 'K-SMOTE')
CLFS_NAMES = ('LR', 'KNN', 'DT', 'GBC', 'RF')
METRICS_MAPPING = dict([('accuracy', 'Accuracy'), ('f1_macro', 'F-score'), ('geometric_mean_score_macro', 'G-mean')])
DATASETS_NAMES = ('indian_pines', 'salinas', 'salinas_a', 'pavia_centre', 'pavia_university', 'kennedy_space_center', 'botswana')


def _make_bold(row, maximum=True, num_decimals=2):
    """Make bold the lowest or highest value(s)."""
    row = round(row, num_decimals)
    val = row.max() if maximum else row.min()
    mask = (row == val)
    formatter = '{0:.%sf}' % num_decimals
    row = row.apply(lambda el: formatter.format(el))
    row[mask] = '\\textbf{%s' % formatter.format(val)
    return row, mask

def generate_mean_std_tbl_bold(mean_vals, std_vals, maximum=True):
    """Generate table that combines mean and sem values."""
    index = mean_vals.iloc[:, :2]
    mean_bold = mean_vals.iloc[:, 2:].apply(lambda row: _make_bold(row, maximum)[0], axis=1)
    mask = mean_vals.iloc[:, 2:].apply(lambda row: _make_bold(row, maximum)[1], axis=1).values
    std_bold = np.round(std_vals.iloc[:, 2:], 2).astype(str)
    std_bold = np.where(mask, std_bold+'}', std_bold)
    scores = mean_bold + r" $\pm$ "  + std_bold

    tbl = pd.concat([index, scores], axis=1)
    return tbl

def generate_main_results():
    """Generate the main results of the experiment."""

    mean_sem_scores = sort_tbl(
        generate_mean_std_tbl_bold(*calculate_mean_sem_scores(results), maximum=True),
        ovrs_order=OVRS_NAMES, clfs_order=CLFS_NAMES
    )
    mean_sem_perc_diff_scores = sort_tbl(
        generate_mean_std_tbl(*calculate_mean_sem_perc_diff_scores(results, ['SMOTE', 'K-SMOTE'])), 
        ovrs_order=OVRS_NAMES, clfs_order=CLFS_NAMES
    )
    mean_sem_ranking = sort_tbl(
        generate_mean_std_tbl_bold(*calculate_mean_sem_ranking(results), maximum=False), 
        ovrs_order=OVRS_NAMES, clfs_order=CLFS_NAMES
    )
    main_results_names = ('mean_sem_scores', 'mean_sem_perc_diff_scores', 'mean_sem_ranking')

    return zip(main_results_names, (mean_sem_scores, mean_sem_perc_diff_scores, mean_sem_ranking))

def generate_statistical_results():
    """Generate the statistical results of the experiment."""

    friedman_test = sort_tbl(generate_pvalues_tbl(apply_friedman_test(results)), ovrs_order=OVRS_NAMES, clfs_order=CLFS_NAMES)
    holms_test = sort_tbl(generate_pvalues_tbl(apply_holms_test(results, control_oversampler='K-SMOTE')), ovrs_order=OVRS_NAMES[:-1], clfs_order=CLFS_NAMES)
    statistical_results_names = ('friedman_test', 'holms_test')
    statistical_results = zip(statistical_results_names, (friedman_test, holms_test))

    return statistical_results

def summarize_multiclass_datasets(datasets):
    summarized = summarize_datasets(datasets)\
        .rename(columns={'Dataset name':'Dataset', 'Imbalance Ratio': 'IR'})\
        .set_index('Dataset')\
        .join(pd.Series(dict([(name, dat[-1].unique().size) for name, dat in datasets]), name='Classes'))\
        .reset_index()
    summarized.loc[:,'Dataset'] = summarized.loc[:,'Dataset']\
        .apply(lambda x: x.title())
    return summarized


def plot_lulc_images():
    arrays_x = []
    arrays_y = []
    for dat_name in DATASETS_NAMES:
        X, y = RemoteSensingDatasets()._load_gic_dataset(dat_name)
        arrays_x.append(X[:,:,100])
        arrays_y.append(np.squeeze(y))

    for X, y, figname in zip(arrays_x, arrays_y, DATASETS_NAMES):
        plt.figure(
            figsize=(20,10),
            dpi=320
        )
        if figname == 'kennedy_space_center':
            X = np.clip(X, 0, 350)
        for i, (a, cmap) in enumerate(zip([X, y],['gist_gray','terrain'])):
            plt.subplot(2, 1, i+1)
            plt.imshow(
                a, cmap=plt.get_cmap(cmap)
            )
            plt.axis('off')
        plt.savefig(join(analysis_path, figname), bbox_inches='tight', pad_inches = 0)


if __name__=='__main__':

    data_path, results_path, analysis_path = generate_paths()

    # load datasets
    datasets = load_datasets(data_dir=data_path)

    # load results
    results = []
    for name in RESULTS_NAMES:
        file_path = join(results_path, f'{name}.pkl')
        results.append(pd.read_pickle(file_path))

    # combine and select results
    results = combine_results(*results)
    results = select_results(results, oversamplers_names=OVRS_NAMES, classifiers_names=CLFS_NAMES)

    # datasets description
    summarize_multiclass_datasets(datasets).to_csv(join(analysis_path, 'datasets_description.csv'), index=False)

    # Main results
    main_results = generate_main_results()
    for name, result in main_results:
        result['Metric'] = result['Metric'].map(METRICS_MAPPING)
        result.to_csv(join(analysis_path, f'{name}.csv'), index=False)

    # Statistical results
    statistical_results = generate_statistical_results()
    for name, result in statistical_results:
        result['Metric'] = result['Metric'].map(METRICS_MAPPING)
        result.to_csv(join(analysis_path, f'{name}.csv'), index=False)
