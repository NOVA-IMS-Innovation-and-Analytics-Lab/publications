"""
Extract the database.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
#         Joao Fonseca <jpmrfonseca@gmail.com>
# License: MIT

import sys
from collections import Counter
from os import pardir
from os.path import join, dirname

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.append(join(dirname(__file__), '..', '..', '..'))
from utils import RemoteSensingDatasets

DATA_PATH = join(dirname(__file__), pardir, 'data')


if __name__ == '__main__':

    # Download datasets
    datasets = RemoteSensingDatasets().download()

    # Sample datasets
    min_n_samples, max_n_samples, fraction, rnd_seed = 20, 1000, 0.1, 5
    content = []
    for name, data in datasets.content_:
        data = data.sample(frac=fraction, random_state=rnd_seed)
        classes = [cl for cl,count in Counter(data.target).items() if count >= min_n_samples and count <= max_n_samples]
        data = data[data.target.isin(classes)].reset_index(drop=True)
        data = pd.concat([pd.DataFrame(MinMaxScaler().fit_transform(data.drop(columns='target'))), data.target], axis=1)
        content.append((name, data))

    # Save database
    datasets.content_ = content
    datasets.save(DATA_PATH, 'active_learning')
