"""
Implement the command-line interface.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from argparse import ArgumentParser, RawTextHelpFormatter
from os.path import dirname, join, exists
from pickle import load
from sqlite3 import connect

import pandas as pd
from sklearnext.tools import BinaryExperiment

from . import DATA_PATH, EXPERIMENTS_PATH
from .data import ImbalancedBinaryClassDatasets, BinaryClassDatasets
from .experiment import CONFIG

DATABASES_MAPPING = {'imbalanced_binary_class': ImbalancedBinaryClassDatasets, 'binary_class': BinaryClassDatasets}


def load_datasets(db_name, datasets_names):
    """Load datasets from sqlite database."""

    path = join(dirname(__file__), DATA_PATH, f'{db_name}.db')
    if not exists(path):
        raise FileNotFoundError(f'Database {db_name} was not found.')

    with connect(path) as connection:

        if datasets_names == 'all':
            datasets_names = [name[0] for name in connection.execute("SELECT name FROM sqlite_master WHERE type='table';")]

        datasets = []

        for dataset_name in datasets_names:
            ds = pd.read_sql(f'select * from "{dataset_name}"', connection)
            X, y = ds.iloc[:, :-1], ds.iloc[:, -1]
            datasets.append((dataset_name, (X, y)))
    
    return datasets


def create_parser():
    """Parse command-line arguments."""

    # Define parameters
    databases_names = '\n'.join(DATABASES_MAPPING.keys())
    experiments_names = '\n'.join(CONFIG.keys())
    
    # Create parser and subparsers
    parser = ArgumentParser(description='Download databases and run experiments.')
    subparsers = parser.add_subparsers()
    subparsers.required = True
    subparsers.dest = 'subcommand'
    
    # Downloading subparser
    downloading_parser = subparsers.add_parser('downloading', help='Download data as sqlite database.', formatter_class=RawTextHelpFormatter)
    downloading_parser.add_argument('name', help=f'The name of the database. It should be one of the following:\n\n{databases_names}')

    # Add arguments
    experiment_parser = subparsers.add_parser('experiment', help='Run experiment from available experimental configurations.', formatter_class=RawTextHelpFormatter)
    experiment_parser.add_argument('name', help=f'The name of the experiment. It should be one of the following:\n\n{experiments_names}')
    experiment_parser.add_argument('--n-jobs', type=int, default=-1, help='Number of jobs to run in parallel. -1 means using all processors.')
    experiment_parser.add_argument('--verbose', type=int, default=0, help='Controls the verbosity: the higher, the more messages.')
    experiment_parser.add_argument('--compared-oversamplers', nargs=2, default=None, help='Pair of oversamplers to compare when percentage difference of performance is calculated.')
    experiment_parser.add_argument('--alpha', type=float, default=0.05, help='Significance level of the Friedman test.')
    experiment_parser.add_argument('--control-oversampler', default=None, help='Control oversampler of the Holms method.')

    return parser


def run():
    
    # Create parser and get arguments
    parser = create_parser()
    args = parser.parse_args()

    if args.subcommand == 'downloading':

        # Select database
        if args.name not in DATABASES_MAPPING.keys():
            raise ValueError(f'Database {args.name} not available to download. Select one from {list(DATABASES_MAPPING.keys())}.')
        datasets = DATABASES_MAPPING[args.name]()

        # Download and save database
        datasets.download().save(join(dirname(__file__), DATA_PATH), args.name)
    
    elif args.subcommand == 'experiment':
        
        # Get configuration
        if args.name not in CONFIG.keys():
            raise ValueError(f'Experiment {args.name} not available to run. Select one from {list(CONFIG.keys())}.')
        configuration = CONFIG[args.name]

        # Load datasets from database
        db_name, datasets_names = configuration.pop('db_name'), configuration.pop('datasets_names')
        datasets = load_datasets(db_name, datasets_names)
    
        # Run and save experiment
        experiment = BinaryExperiment(args.name, datasets, **configuration)
        experiment.run(args.n_jobs, args.verbose)
        experiment.calculate_results(args.compared_oversamplers, args.alpha, args.control_oversampler)
        experiment.dump(join(dirname(__file__), EXPERIMENTS_PATH))
