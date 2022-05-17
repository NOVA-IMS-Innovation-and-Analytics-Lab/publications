"""
Base class for datasets.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
#         Joao Fonseca <jpmrfonseca@gmail.com>
# License: MIT

from os.path import join
from sqlite3 import connect

import click
import pandas as pd


class BaseDatasets:
    """Base class to download and save datasets."""

    def __init__(self, names='all'):
        self.names = names

    def download(self):
        """Download the datasets."""
        if self.names == 'all':
            func_names = [func_name for func_name in dir(self) if 'fetch_' in func_name]
        else:
            func_names = [
                f'fetch_{name}'.lower().replace(' ', '_') for name in self.names
            ]
        self.content_ = []
        with click.progressbar(func_names, label='Datasets') as bar:
            for func_name in bar:
                name = func_name.replace('fetch_', '')
                fetch_data = getattr(self, func_name)
                self.content_.append((name, fetch_data()))
        return self

    def save_to_db(self, db_path):
        """Save datasets to sqlite database."""
        with connect(join(db_path)) as connection:
            for name, (X, y) in self.content_:
                data = pd.concat(
                    [pd.DataFrame(X), pd.Series(y, name=X.shape[1])], axis=1
                )
                data.to_sql(name, connection, index=False, if_exists='replace')
