"""
Run the analysis of experimental results.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from argparse import ArgumentParser
from ast import literal_eval
from collections import Counter
from itertools import product
from math import ceil, floor
from os.path import dirname, join
from pickle import load
from re import match, sub

import pandas as pd
import statsmodels.api as sm
from bokeh.io import export_svgs
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.palettes import Category20, Spectral
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from scipy.stats import rankdata

from config import CONFIGURATIONS

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


def generate_cases_frequency_tbl(experiment, cases_func):
    """Generate a table of optimal hyperparameters for each dataset."""

    oversampler_name = experiment.oversamplers_names_[-1]
    parameters = list(experiment.oversamplers[-1][-1].keys())
    full_parameters = [f'{oversampler_name}__{param}' for param in parameters]
    
    results = experiment.results_.iloc[:, ::2].reset_index()
    results.columns = results.columns.get_level_values(0)
    results = results[results['Oversampler'] == oversampler_name]

    cols = results.columns.tolist()
    keys, metrics = cols[:3], cols[4:]
    
    optimal_values = results.groupby(cols[:3]).agg({metric: max for metric in metrics}).reset_index()
    optimal_params = pd.DataFrame()
    for metric in metrics:
        groups = keys + [metric]
        optimal_values_metric = pd.merge(optimal_values[groups], results[groups + ['params']], on=groups, how='left')
        optimal_values_metric = pd.merge(optimal_values_metric, experiment.datasets_summary_, left_on='Dataset', right_on='Dataset name', how='left')
        optimal_values_metric['Metric'] = metric.replace('mean_test_', '')
        optimal_values_metric[parameters] = pd.DataFrame(optimal_values_metric['params'].apply(lambda item: [literal_eval(item)[param] for param in full_parameters]).values.tolist())
        optimal_values_metric = optimal_values_metric.assign(Ratio=lambda df: df.Instances / df.Features).drop(columns=['Oversampler', 'params', 'Dataset name', 'Features', 'Instances', 'Minority instances', 'Majority instances'] + [metric])
        optimal_params = optimal_params.append(optimal_values_metric)
    optimal_params['Case'] = optimal_params.apply(cases_func, axis=1)
    cases_frequency = optimal_params.groupby('Case').size().reset_index().rename(columns={0: 'Frequency'}).sort_values('Frequency', ascending=False)
    cases_frequency.loc[-1] = ['Total experiments', cases_frequency.Frequency.sum()]

    return cases_frequency.reset_index(drop=True)


def generate_mean_perc_diff_scores_plot(experiment):
    """Generate a plot of mean percentace difference."""
    mean_perc_diff_scores = experiment.mean_perc_diff_scores_.copy()
    mean_perc_diff_scores['Metric'] = mean_perc_diff_scores['Metric'].apply(lambda metric: METRICS_NAMES_MAPPING[metric])
    classifiers_names = mean_perc_diff_scores['Classifier'].unique().tolist()
    metrics_names = mean_perc_diff_scores['Metric'].unique().tolist()
    combinations = list(product(classifiers_names, metrics_names))
    values = mean_perc_diff_scores['Difference'].tolist()
    source = ColumnDataSource(data=dict(x=combinations, counts=values))
    p = figure(x_range=FactorRange(*combinations), y_range=(floor(min(values)), ceil(max(values))), plot_height=650, plot_width=1500,
               title=f'{experiment.oversamplers_names_[-1]} and {experiment.oversamplers_names_[-2]} percentage difference', 
               toolbar_location=None, tools='')
    p.vbar(x='x', top='counts', width=0.9, source=source,
           fill_color=factor_cmap('x', palette=Spectral[3], factors=metrics_names, start=1, end=2))
    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = 1.0
    p.title.text_font_size = '20pt'
    p.xaxis.major_label_text_font_size = '11pt'
    p.xaxis.group_text_font_size = '14pt'
    p.output_backend = 'svg'
    return p


def parse_arguments():
    """Parse command-line arguments."""
    parser = ArgumentParser(f'Run an experimental analysis.')
    parser.add_argument('name', help='The name of the experiment.')
    parser.add_argument('experiment', nargs='?', default='.', help='The relative or absolute path to load the experiment object.')
    parser.add_argument('resources', nargs='?', default='.', help='The relative or absolute path to save the analysis results.')
    parsed_args = parser.parse_args() 
    experiment_path = join(dirname(__file__), parsed_args.experiment)
    resources_path = join(dirname(__file__), parsed_args.resources)
    return parsed_args.name, experiment_path, resources_path


if __name__ == '__main__':

    # Parse arguments
    name, experiment_path, resources_path = parse_arguments()

    # Get experiment's configuration
    configuration = CONFIGURATIONS[name]

    # Load experiment object
    with open(join(experiment_path, f'{name}.pkl'), 'rb') as file:
        experiment = load(file)
    
    # Generate tables
    experiment.datasets_summary_.to_csv(join(resources_path, 'datasets_summary.csv'), index=False)
    generate_mean_std_tbl(experiment, 'scores').to_csv(join(resources_path, 'scores.csv'), index=False)
    generate_mean_std_tbl(experiment, 'perc_diff_scores').to_csv(join(resources_path, 'perc_diff_scores.csv'), index=False)
    generate_mean_std_tbl(experiment, 'ranking').to_csv(join(resources_path, 'ranking.csv'), index=False)
    generate_pvalues_tbl(experiment, 'friedman').to_csv(join(resources_path, 'friedman_test.csv'), index=False)
    generate_pvalues_tbl(experiment, 'holms').to_csv(join(resources_path, 'holms_test.csv'), index=False)
    generate_cases_frequency_tbl(experiment, configuration['cases_func']).to_csv(join(resources_path, 'cases_frequency.csv'), index=False)
    
    # Generate plots
    export_svgs(generate_mean_perc_diff_scores_plot(experiment), filename=join(resources_path, 'mean_perc_diff_scores.svg'))
