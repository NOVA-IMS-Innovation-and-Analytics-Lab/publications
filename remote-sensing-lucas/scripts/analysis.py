"""
Analyze the experimental results.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
#         João Fonseca <jpmrfonseca@gmail.com>
# License: MIT

import sys
import os
from os.path import (
    join,
    dirname,
    realpath,
    splitext,
    basename
)
import pickle
import numpy as np
import pandas as pd
from itertools import product

from rlearn.tools import (
    combine_results,
    calculate_mean_sem_perc_diff_scores,
    calculate_wide_optimal
)

RESULTS_PATH = realpath(join(dirname(__file__), '..', 'results'))
ANALYSIS_PATH = realpath(join(dirname(__file__), '..', 'analysis'))
DATA_PATH = realpath(join(dirname(__file__), '..', 'data', 'lucas.csv'))

OS_LABELS = {
    'NO OVERSAMPLING':'NONE',
    'RANDOM OVERSAMPLING':'ROS',
    'SMOTE': 'SMOTE',
    'BORDERLINE SMOTE':'B-SMOTE',
    'ADASYN': 'ADASYN',
    'G-SMOTE': 'G-SMOTE',
}
METRICS = {
    'mean_test_geometric_mean_macro_score':'G-mean',
    'mean_test_f1_macro':'F-score' ,
    'mean_test_accuracy':'Accuracy'
}
CLASSIFIERS = ['LR', 'KNN', 'DT', 'GBC', 'RF']
TARGET_LABELS = {'A':1, 'B':2, 'C':0, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7}
ANALYSIS_OUTPUT = [
    'mean_scores',
    'mean_ranking',
    'model_mean_ranking',
    'mean_perc_diff_scores',
    'dataset_descrip'
]

def format_dataframe(optimal):
    """Renames metrics, drops redundant classifiers, and sorts table"""
    optimal = optimal[
        optimal['Classifier']\
            .isin(CLASSIFIERS)
            ]
    optimal['Metric'] = optimal['Metric'].map(METRICS)
    optimal = optimal.set_index(['Classifier', 'Metric'])\
        .sort_index(ascending=[True, False])\
        .reindex([
            (clf, scorer)
            for clf, scorer
            in product(CLASSIFIERS,
                np.sort(list(METRICS.values())))
        ])
    return optimal


def generate_mean_table(mean_vals):
    """Calculates optimal results oversamplers and classifiers"""
    mean_vals = mean_vals\
            .rename(columns=OS_LABELS)\
            [OS_LABELS.values()]
    return mean_vals


def describe_dataset():
    """Generates dataframe with dataset description"""
    df = pd.read_csv(DATA_PATH)
    count_c = df[df['target']==TARGET_LABELS['C']].shape[0]
    count_h = df[df['target']==TARGET_LABELS['H']].shape[0]
    columns = ['Dataset', splitext(basename(DATA_PATH))[0].upper()]
    description = {
        'Features': df.shape[-1]-1,
        'Instances': df.shape[0],
        'Instances of class C': count_c,
        'Instances of class H': count_h,
        'IR of class H': count_c/count_h
    }
    desc = pd.DataFrame(data=description.items(), columns=columns)\
        .set_index('Dataset')
    desc.iloc[:-1] = desc.iloc[:-1].applymap('{:.0f}'.format)
    return desc

def mk_bold_round(row, maximum=True, decimals=2):
    row = row.astype(float)
    if maximum:
        val_ref = row.idxmax()
    else:
        val_ref = row.idxmin()
    row = row.map(('{:,.%sf}' % decimals).format )
    row[val_ref] = '\\textbf{%s}' % row[val_ref]
    return row

def percent_diff_table(combined, oversampler='G-SMOTE', comparison_ovs=['NO OVERSAMPLING','RANDOM OVERSAMPLING','SMOTE']):
    all_perc_diff = []
    for comp_ovs in comparison_ovs:
        mean_perc_diff_scores = format_dataframe(
            calculate_mean_sem_perc_diff_scores(combined, [comp_ovs, oversampler])[0]
        ).applymap('{:,.2f}'.format).rename(columns={'Difference':OS_LABELS[comp_ovs]})
        all_perc_diff.append(mean_perc_diff_scores)
    return pd.concat(all_perc_diff, axis=1)

def generate_main_results():
    """Generate the main results of the experiment."""
    # read pickled objects
    objs = os.listdir(RESULTS_PATH)
    results = [
        pickle.load(open(join(RESULTS_PATH, obj), 'rb'))
        for obj in objs
        if obj.split('.')[-1]=='pkl'
    ]

    # Create combined resuls dataframe
    combined = combine_results(*results)

    # Calculate results
    mean_vals = format_dataframe(calculate_wide_optimal(combined))
    _mean_table = generate_mean_table(mean_vals)
    _mean_ranking = _mean_table.rank(ascending=False, axis=1).astype(int)

    mean_scores = _mean_table.apply(lambda x: mk_bold_round(x, True), axis=1)
    mean_ranking = _mean_ranking.apply(lambda x: mk_bold_round(x, False, 0), axis=1)
    model_mean_ranking = _mean_ranking.groupby(['Metric']).mean()\
        .sort_index(ascending=False)\
        .apply(lambda x: mk_bold_round(x, False, 1), axis=1)
    mean_perc_diff_scores = percent_diff_table(
        combined,
        oversampler='G-SMOTE',
        comparison_ovs=['NO OVERSAMPLING','RANDOM OVERSAMPLING','SMOTE']
    )

    # Get dataset description
    dataset_descrip = describe_dataset()

    return ((k,v) for k,v in
        zip(ANALYSIS_OUTPUT,
            (mean_scores, mean_ranking, model_mean_ranking,
            mean_perc_diff_scores, dataset_descrip))
    )

if __name__ == '__main__':
    results = generate_main_results()
    for name, result in results:
        result.to_csv(join(ANALYSIS_PATH, f'{name}.csv'), index=True)
