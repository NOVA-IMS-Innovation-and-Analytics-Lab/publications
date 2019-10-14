"""
Extract the database.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import sys
from os.path import join, dirname
from sqlite3 import connect

sys.path.append('../..')
from utils import ImbalancedBinaryClassDatasets

DATA_PATH = join(dirname(__file__), '..', 'data', 'gsmote.db')


if __name__ == '__main__':

    # TODO: Download the correct datasets of the paper

    # Download datasets
    cgan_data = ImbalancedBinaryClassDatasets().download().datasets_

    # Save data to database
    with connect(DATA_PATH) as connection:
        for name, data in cgan_data:
            data.to_sql(name, connection, if_exists='replace')