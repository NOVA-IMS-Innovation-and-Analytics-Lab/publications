"""
Extract the database.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import sys
from os.path import join, dirname
from sqlite3 import connect

sys.path.append(join(dirname(__file__), '..', '..'))
from utils import ImbalancedBinaryDatasets

DATA_PATH = join(dirname(__file__), '..', 'data', 'gsmote.db')


if __name__ == '__main__':

    # Download datasets
    data = ImbalancedBinaryDatasets().download().datasets_

    # Save data to database
    with connect(DATA_PATH) as connection:
        for name, df in data:
            df.to_sql(name, connection, if_exists='replace')