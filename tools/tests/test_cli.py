"""
Test the cli module.
"""

from os import remove
from os.path import dirname, join
from sqlite3 import connect

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from .. import DATA_PATH
from ..cli import load_datasets, create_parser, run


def test_load_datasets_raise_error():
    """Test raising a file not found error."""
    with pytest.raises(FileNotFoundError):
        load_datasets('test', None)


def test_load_datasets():
    """Test the loading of datasets."""
    
    # Generate data
    n_features1 = 10
    n_features2 = 5
    X1, y1 = make_classification(random_state=0, weights=[0.7, 0.3], n_features=n_features1)
    X2, y2 = make_classification(random_state=0, weights=[0.8, 0.2], n_features=n_features2)

    # Create database
    db_path = join(DATA_PATH, 'test.db')
    print(db_path)
    connection = connect(db_path)

    # Save table 1 to database
    ds1_expected = pd.DataFrame(np.column_stack([X1, y1]))
    ds1_expected.columns = pd.Index(np.arange(0, n_features1 + 1), dtype=str)
    X1_expected, y1_expected = ds1_expected.iloc[:, :-1], ds1_expected.iloc[:, -1]
    ds1_expected.to_sql('ds1', connection, index=False)
    
    # Save table 2 to database
    ds2_expected = pd.DataFrame(np.column_stack([X2, y2]))
    ds2_expected.columns = pd.Index(np.arange(0, n_features2 + 1), dtype=str)
    X2_expected, y2_expected = ds2_expected.iloc[:, :-1], ds2_expected.iloc[:, -1]
    ds2_expected.to_sql('ds2', connection, index=False)
    
    # Load datasets from database
    datasets = load_datasets('test', 'all')
    (name1, (X1, y1)), (name2, (X2, y2)) = datasets

    # Assertions
    assert name1 == 'ds1'
    assert name2 == 'ds2'
    pd.testing.assert_frame_equal(X1, X1_expected)
    np.testing.assert_array_equal(y1, y1_expected)
    pd.testing.assert_frame_equal(X2, X2_expected)
    np.testing.assert_array_equal(y2, y2_expected)

    # Delete databse
    remove(db_path)
