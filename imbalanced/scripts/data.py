"""
Extract the database.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from os import pardir
from os.path import join, dirname
from sqlite3 import connect

from utils import ImbalancedBinaryDatasets

DATA_PATH = join(dirname(__file__), pardir, 'data')


if __name__ == '__main__':

    ImbalancedBinaryDatasets().download().save(DATA_PATH, 'imbalanced')