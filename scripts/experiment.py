#!usr/bin/env python

"""
Run the experimental procedure.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from os.path import join, dirname, exists, basename, abspath
from sqlite3 import connect
from argparse import ArgumentParser

import pandas as pd
from sklearnext.tools import BinaryExperiment

from config import CONFIGURATIONS


def load_datasets(path):
    """Load datasets from a sqlite database."""

    if not exists(path):
        raise FileNotFoundError(f'Database {basename(path)} was not found.')

    with connect(path) as connection:

        tbl_names = [name[0] for name in connection.execute("SELECT name FROM sqlite_master WHERE type='table';")]

        datasets = []

        for tbl_name in tbl_names:
            tbl = pd.read_sql(f'select * from "{tbl_name}"', connection)
            X, y = tbl.iloc[:, :-1], tbl.iloc[:, -1]
            datasets.append((tbl_name, (X, y)))
    
    return datasets


def parse_arguments():
    """Parse command-line arguments."""
    parser = ArgumentParser(f'Run an experiment.')
    parser.add_argument('name', help='The name of the experiment.')
    parser.add_argument('datasets_path', nargs='?', default='.', help='The relative or absolute path of the datasets.')
    parser.add_argument('experiment_path', nargs='?', default='.', help='The relative or absolute path to save the experiment object.')
    parser.add_argument('--calculate-results', default='n', help='Calculate results of the experiment.')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Number of jobs to run in parallel.')
    parser.add_argument('--verbose', type=int, default=0, help='Controls the verbosity level.')
    parsed_args = parser.parse_args() 
    return parsed_args.name, abspath(parsed_args.datasets_path), abspath(parsed_args.experiment_path), parsed_args.calculate_results, parsed_args.n_jobs, parsed_args.verbose


if __name__ == '__main__':

    # Parse arguments
    name, datasets_path, experiment_path, calculate_results, n_jobs, verbose = parse_arguments()

    # Get experiment's configuration
    configuration = CONFIGURATIONS[name]

    # Download datasets
    datasets = load_datasets(join(datasets_path, configuration['db_name']))
    
    # Run experiment and save object
    experiment = BinaryExperiment(name, datasets, configuration['oversamplers'], configuration['classifiers'], configuration['scoring'], configuration['n_splits'], configuration['n_runs'], configuration['random_state'])
    experiment.run(n_jobs=n_jobs, verbose=verbose)
    if calculate_results == 'y':
        experiment.calculate_results()
    experiment.dump(experiment_path)


