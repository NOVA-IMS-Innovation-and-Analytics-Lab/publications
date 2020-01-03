"""
Extract the database.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
#         Joao Fonseca <jpmrfonseca@gmail.com>
# License: MIT

import sys
from os.path import join, dirname
from sqlite3 import connect
import pandas as pd

sys.path.append(join(dirname(__file__), '..', '..'))
from utils import RemoteSensingDatasets

DATA_PATH = join(dirname(__file__), '..', 'data', 'kmeans_smote_rs.db')


if __name__ == '__main__':

    # Download datasets
    data = RemoteSensingDatasets().download().datasets_
    data.append(('Coimbra', pd.read_csv('coimbra_sentinel.csv')))

    # Sample data
    # TODO

    # Save data to database
    with connect(DATA_PATH) as connection:
        for name, df in data:
            df.to_sql(name, connection, if_exists='replace')
