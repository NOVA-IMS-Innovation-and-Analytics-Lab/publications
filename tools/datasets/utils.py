"""
Utilities for datasets.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
#         Joao Fonseca <jpmrfonseca@gmail.com>
# License: MIT

from os.path import join
from os import listdir
from sqlite3 import connect
import pandas as pd


def load_datasets_from_csv(data_dir):
    """Load datasets from csv files."""
    datasets = []
    for file_name in listdir(data_dir):
        if file_name.endswith('.csv'):
            file_path = join(data_dir, file_name)
            data = pd.read_csv(file_path)
            X, y = data.drop(columns='target'), data['target']
            datasets.append((file_path.replace('.csv', ''), (X, y)))
    return datasets


def load_datasets_from_db(path):
    """Load datasets from sqlite database."""
    datasets = []
    with connect(path) as connection:
        names = [
            name[0]
            for name in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            )
        ]
        for name in names:
            data = pd.read_sql(f'select * from "{name}"', connection)
            target_ind = str(data.shape[1] - 1)
            X, y = data.drop(columns=target_ind).values, data[target_ind].values
            datasets.append((name, (X, y)))
    return datasets
