"""
Base class for datasets.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
#         Joao Fonseca <jpmrfonseca@gmail.com>
# License: MIT

from os.path import join
from sqlite3 import connect
from rich.progress import track
import pandas as pd


class Datasets:
    """Class to download and save datasets."""

    def __init__(self, names='all'):
        self.names = names

    @staticmethod
    def _modify_columns(data):
        """Rename and reorder columns of dataframe."""
        X, y = data.drop(columns='target'), data.target
        X.columns = range(len(X.columns))
        return pd.concat([X, y], axis=1)

    def download(self):
        """Download the datasets."""
        if self.names == 'all':
            func_names = [func_name for func_name in dir(self) if 'fetch_' in func_name]
        else:
            func_names = [
                f'fetch_{name}'.lower().replace(' ', '_') for name in self.names
            ]
        self.content_ = []
        for func_name in track(func_names, description='Datasets'):
            name = func_name.replace('fetch_', '').upper().replace('_', ' ')
            fetch_data = getattr(self, func_name)
            data = self._modify_columns(fetch_data())
            self.content_.append((name, data))
        return self

    def save_to_db(self, dir_name, db_name):
        """Save datasets to sqlite database."""
        with connect(join(dir_name, f'{db_name}.db')) as connection:
            for name, data in self.content_:
                data.to_sql(name, connection, index=False, if_exists='replace')
