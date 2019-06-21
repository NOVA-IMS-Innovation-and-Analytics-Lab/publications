"""
Format experimental results.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import pandas as pd

METRICS_NAMES_MAPPING = {'roc_auc': 'AUC', 'f1': 'F-SCORE', 'geometric_mean_score': 'G-MEAN'}


def generate_mean_std_tbl(experiment, name):
    """Generate table that combines mean and sem values."""
    mean_vals, std_vals = getattr(experiment, f'mean_{name}_'), getattr(experiment, f'sem_{name}_')
    index = mean_vals.iloc[:, :2]
    scores = mean_vals.iloc[:, 2:].applymap('{:,.2f}'.format) + r" $\pm$ "  + std_vals.iloc[:, 2:].applymap('{:,.2f}'.format)
    tbl = pd.concat([index, scores], axis=1)
    tbl['Metric'] = tbl['Metric'].apply(lambda metric: METRICS_NAMES_MAPPING[metric])
    return tbl


def generate_pvalues_tbl(experiment, name):
    """Format p-values."""
    tbl = getattr(experiment, f'{name}_test_').copy()
    for name in tbl.dtypes[tbl.dtypes == float].index:
        tbl[name] = tbl[name].apply(lambda pvalue: '%.1e' % pvalue)
    tbl['Metric'] = tbl['Metric'].apply(lambda metric: METRICS_NAMES_MAPPING[metric])
    return tbl
