"""
Download the datasets.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from os.path import dirname, join

from tools.datasets import ImbalancedBinaryDatasets


def get_datasets_path():
    """Get the datasets path."""
    return join(dirname(__file__), 'imbalanced_binary.db')


def save_datasets(path):
    """Download and save the datasets."""
    ImbalancedBinaryDatasets().download().save_to_db(path)
