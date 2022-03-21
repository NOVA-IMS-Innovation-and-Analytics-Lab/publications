"""
Functions to generate and format the manuscript.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
#         Joao Fonseca <jpmrfonseca@gmail.com>
# License: MIT


import pandas as pd


def generate_mean_std_tbl(mean_vals, std_vals):
    """Generate table that combines mean and sem values."""
    index = mean_vals.iloc[:, :2]
    scores = (
        mean_vals.iloc[:, 2:].applymap('{:,.2f}'.format)
        + r" $\pm$ "
        + std_vals.iloc[:, 2:].applymap('{:,.2f}'.format)
    )
    tbl = pd.concat([index, scores], axis=1)
    return tbl


def generate_pvalues_tbl(tbl):
    """Format p-values."""
    for name in tbl.dtypes[tbl.dtypes == float].index:
        tbl[name] = tbl[name].apply(lambda pvalue: '%.1e' % pvalue)
    return tbl


def sort_tbl(tbl, ds_order=None, ovrs_order=None, clfs_order=None, metrics_order=None):
    """Sort tables rows and columns."""
    cols = tbl.columns
    keys = ['Dataset', 'Oversampler', 'Classifier', 'Metric']
    for key, cat in zip(keys, (ds_order, ovrs_order, clfs_order, metrics_order)):
        if key in cols:
            tbl[key] = pd.Categorical(tbl[key], categories=cat)
    key_cols = [col for col in cols if col in keys]
    tbl.sort_values(key_cols, inplace=True)
    if ovrs_order is not None and set(ovrs_order).issubset(cols):
        tbl = tbl[key_cols + list(ovrs_order)]
    return tbl


def make_bold(row, maximum=True, num_decimals=2):
    """Make bold the lowest or highest value(s)."""
    row = round(row, num_decimals)
    val = row.max() if maximum else row.min()
    mask = row == val
    formatter = '{0:.%sf}' % num_decimals
    row = row.apply(lambda el: formatter.format(el))
    row[mask] = '\\textbf{%s}' % formatter.format(val)
    return row
