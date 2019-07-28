"""
Implement the command-line interface.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from pathlib import Path
from argparse import ArgumentParser, RawTextHelpFormatter
from os import listdir
from os.path import dirname, join, exists
from pickle import load
from sqlite3 import connect

import pandas as pd
from sklearnext.tools import (
    ImbalancedExperiment,
    combine_experiments,
    calculate_ranking,
    calculate_mean_sem_ranking,
    calculate_mean_sem_scores,
    calculate_mean_sem_perc_diff_scores,
    apply_friedman_test,
    apply_holms_test
)

from . import DATA_PATH, EXPERIMENTS_PATH
from .data import ImbalancedBinaryClassDatasets, BinaryClassDatasets
from .experiment import CONFIG

DATABASES_MAPPING = {'imbalanced_binary_class': ImbalancedBinaryClassDatasets, 'binary_class': BinaryClassDatasets}


def load_datasets(db_name, datasets_names):
    """Load datasets from sqlite database."""

    path = join(DATA_PATH, f'{db_name}.db')
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
    parser = ArgumentParser(description='Download databases, run and combine experiments.')
    subparsers = parser.add_subparsers()
    subparsers.required = True
    subparsers.dest = 'subcommand'
    
    # Downloading subparser
    downloading_parser = subparsers.add_parser('downloading', help='Download data as sqlite database.', formatter_class=RawTextHelpFormatter)
    downloading_parser.add_argument('name', help=f'The name of the database. It should be one of the following:\n\n{databases_names}')

    # Experiment subparser
    experiment_parser = subparsers.add_parser('experiment', help='Run experiment from available experimental configurations.', formatter_class=RawTextHelpFormatter)
    experiment_parser.add_argument('exp', help=f'The name of the experiment. It should be one of the following:\n\n{experiments_names}')
    experiment_parser.add_argument('ovs', help=f'The name of the oversampler(s).')
    experiment_parser.add_argument('--n-jobs', type=int, default=-1, help='Number of jobs to run in parallel. -1 means using all processors.')
    experiment_parser.add_argument('--verbose', type=int, default=0, help='Controls the verbosity: the higher, the more messages.')

    # Combine subparser
    combine_parser = subparsers.add_parser('combine', help='Combine multiple experiments.', formatter_class=RawTextHelpFormatter)
    combine_parser.add_argument('exp', help=f'The name of the experiment. It should be one of the following:\n\n{experiments_names}')

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
        if args.exp not in CONFIG.keys():
            raise ValueError(f'Experiment `{args.exp}`` not available to run. Select one from {list(CONFIG.keys())}.')
        if args.ovs not in CONFIG[args.exp].keys():
            raise ValueError(f'Oversampler `{args.ovs}` for experiment `{args.exp}` not available to run. Select one from {list(CONFIG[args.exp].keys())}.')
        configuration = CONFIG[args.exp][args.ovs]

        # Load datasets from database
        db_name, datasets_names = configuration.pop('db_name'), configuration.pop('datasets_names')
        datasets = load_datasets(db_name, datasets_names)
        
        # Create directory
        experiment_path = join(EXPERIMENTS_PATH, args.exp)
        Path(experiment_path).mkdir(exist_ok=True)

        # Run and save experiment
        experiment = ImbalancedExperiment(args.ovs, datasets, **configuration)
        experiment.run(args.n_jobs, args.verbose)
        experiment.dump(experiment_path)

    elif args.subcommand == 'combine':

        if args.exp not in CONFIG.keys():
            raise ValueError(f'Experiment `{args.exp}`` not available to combine sub-experiments. Select one from {list(CONFIG.keys())}.')

        # Find experiments
        experiment_path = join(EXPERIMENTS_PATH, args.exp)
        filenames = [filename for filename in listdir(experiment_path) if filename.endswith('.pkl')]
        
        # Load experiments
        experiments = []
        for filename in filenames:
            with open(join(experiment_path, filename), 'rb') as f:
                experiment = load(f)
            experiments.append(experiment)

        # Create and dump combined experiment
        combine_experiments('combined', *experiments).dump(experiment_path)
