"""
Extract the database.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import sys
from os import pardir
from os.path import join, dirname
from sqlite3 import connect

from utils import BinaryDatasets

DATA_PATH = join(dirname(__file__), pardir, 'data')


if __name__ == '__main__':

    BinaryDatasets().download().save(DATA_PATH, 'small_data_oversampling')