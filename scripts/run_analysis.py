"""
Run the analysis of experimental results.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from argparse import ArgumentParser
from os.path import dirname, join
from pickle import load

import pandas as pd

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

