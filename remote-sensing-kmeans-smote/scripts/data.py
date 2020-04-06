"""
Extract the database.
"""
# Author: Georgios Douzas <gdouzas@icloud.com>
#         Joao Fonseca <jpmrfonseca@gmail.com>
# License: MIT

import sys
from sklearn.model_selection import train_test_split
from os.path import join, dirname

sys.path.append(join(dirname(__file__), '..', '..'))
from utils import RemoteSensingDatasets

data = RemoteSensingDatasets().download()
datasets = dict(data.datasets_)

for name, df in datasets.items():
    df = df[df.target != 0]
    datasets[name] = train_test_split(
        df, train_size=2000, stratify=df.target, random_state=0)[0]

data.datasets_ = list(datasets.items())
data.save(join(dirname(__file__), '..', 'data'), 'preprocessed_rs_datasets')
