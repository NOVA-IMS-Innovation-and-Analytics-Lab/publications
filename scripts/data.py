#!usr/bin/env python

"""
Downloads, transforms and simulates imbalanced data.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from os.path import join, dirname, abspath
from re import match, sub
from collections import Counter
from itertools import product
from urllib.parse import urljoin
from string import ascii_lowercase
from zipfile import ZipFile
from io import BytesIO, StringIO
from sqlite3 import connect
from argparse import ArgumentParser

from tqdm import tqdm
import requests
import numpy as np
import pandas as pd
from sklearn.utils import check_X_y
from sklearn.datasets import make_classification
from imblearn.datasets import make_imbalance

UCI_ML_DBS = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'
KEEL = 'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/'
OPENML_URL = 'https://www.openml.org/data/get_csv/3625/dataset_194_eucalyptus.arff'
GITHUB_URL = 'https://raw.githubusercontent.com/IMS-ML-Lab/publications/master/assets/data/pima.csv'
MULTIPLICATION_FACTORS = [1, 2, 3]
RANDOM_STATE = 0


def _calculate_ratio(multiplication_factor, y):
    """Calculate ratio based on IRs multiplication factor."""
    ratio = Counter(y).copy()
    ratio[1] = int(ratio[1] / multiplication_factor)
    return ratio


def _make_imbalance(data, multiplication_factor):
    """Undersample the minority class."""
    X_columns = [col for col in data.columns if col != 'target']
    X, y = check_X_y(data.loc[:, X_columns], data.target)
    if multiplication_factor > 1.0:
        sampling_strategy = _calculate_ratio(multiplication_factor, y)
        X, y = make_imbalance(X, y, sampling_strategy=sampling_strategy, random_state=RANDOM_STATE)
    data = pd.DataFrame(np.column_stack((X, y)))
    data.iloc[:, -1] = data.iloc[:, -1].astype(int)
    return data


def _modifiy_columns(data):
    """Rename and reorder columns of dataframe."""
    X, y = data.drop(columns='target'), data.target
    X.columns = range(len(X.columns))
    return pd.concat([X, y], axis=1)


def fetch_breast_tissue():
    """Download and transform the Breast Tissue Data Set.
    The minority class is identified as the `car` and `fad`
    labels and the majority class as the rest of the labels.

    http://archive.ics.uci.edu/ml/datasets/breast+tissue
    """
    url = urljoin(UCI_ML_DBS, '00192/BreastTissue.xls')
    data = pd.read_excel(url, sheet_name='Data')
    data = data.drop(columns='Case #').rename(columns={'Class': 'target'})
    data['target'] = data['target'].isin(['car', 'fad']).astype(int)
    return data


def fetch_ecoli():
    """Download and transform the Ecoli Data Set.
    The minority class is identified as the `pp` label
    and the majority class as the rest of the labels.

    https://archive.ics.uci.edu/ml/datasets/ecoli
    """
    url = urljoin(UCI_ML_DBS, 'ecoli/ecoli.data')
    data = pd.read_csv(url, header=None, delim_whitespace=True)
    data = data.drop(columns=0).rename(columns={8: 'target'})
    data['target'] = data['target'].isin(['pp']).astype(int)
    return data


def fetch_eucalyptus():
    """Download and transform the Eucalyptus Data Set.
    The minority class is identified as the `best` label
    and the majority class as the rest of the labels.

    https://www.openml.org/d/188
    """
    data = pd.read_csv(OPENML_URL)
    data = data.iloc[:, -9:].rename(columns={'Utility': 'target'})
    data = data[data != '?'].dropna()
    data['target'] = data['target'].isin(['best']).astype(int)
    return data


def fetch_glass():
    """Download and transform the Glass Identification Data Set.
    The minority class is identified as the `1` label
    and the majority class as the rest of the labels.

    https://archive.ics.uci.edu/ml/datasets/glass+identification
    """
    url = urljoin(UCI_ML_DBS, 'glass/glass.data')
    data = pd.read_csv(url, header=None)
    data = data.drop(columns=0).rename(columns={10: 'target'})
    data['target'] = data['target'].isin([1]).astype(int)
    return data


def fetch_haberman():
    """Download and transform the Haberman's Survival Data Set.
    The minority class is identified as the `1` label
    and the majority class as the `0` label.

    https://archive.ics.uci.edu/ml/datasets/Haberman's+Survival
    """
    url = urljoin(UCI_ML_DBS, 'haberman/haberman.data')
    data = pd.read_csv(url, header=None)
    data.rename(columns={3: 'target'}, inplace=True)
    data['target'] = data['target'].isin([2]).astype(int)
    return data


def fetch_heart():
    """Download and transform the Heart Data Set.
    The minority class is identified as the `2` label
    and the majority class as the `1` label.

    http://archive.ics.uci.edu/ml/datasets/statlog+(heart)
    """
    url = urljoin(UCI_ML_DBS, 'statlog/heart/heart.dat')
    data = pd.read_csv(url, header=None, delim_whitespace=True)
    data.rename(columns={13: 'target'}, inplace=True)
    data['target'] = data['target'].isin([2]).astype(int)
    return data


def fetch_iris():
    """Download and transform the Iris Data Set.
    The minority class is identified as the `1` label
    and the majority class as the rest of the labels.

    https://archive.ics.uci.edu/ml/datasets/iris
    """
    url = urljoin(UCI_ML_DBS, 'iris/bezdekIris.data')
    data = pd.read_csv(url, header=None)
    data.rename(columns={4: 'target'}, inplace=True)
    data['target'] = data['target'].isin(['Iris-setosa']).astype(int)
    return data


def fetch_libras():
    """Download and transform the Libras Movement Data Set.
    The minority class is identified as the `1` label
    and the majority class as the rest of the labels.

    https://archive.ics.uci.edu/ml/datasets/Libras+Movement
    """
    url = urljoin(UCI_ML_DBS, 'libras/movement_libras.data')
    data = pd.read_csv(url, header=None)
    data.rename(columns={90: 'target'}, inplace=True)
    data['target'] = data['target'].isin([1]).astype(int)
    return data


def fetch_liver():
    """Download and transform the Liver Disorders Data Set.
    The minority class is identified as the `1` label
    and the majority class as the '2' label.

    https://archive.ics.uci.edu/ml/datasets/liver+disorders
    """
    url = urljoin(UCI_ML_DBS, 'liver-disorders/bupa.data')
    data = pd.read_csv(url, header=None)
    data.rename(columns={6: 'target'}, inplace=True)
    data['target'] = data['target'].isin([1]).astype(int)
    return data


def fetch_pima():
    """Download and transform the Pima Indians Diabetes Data Set.
    The minority class is identified as the `1` label
    and the majority class as the '0' label.

    https://www.kaggle.com/uciml/pima-indians-diabetes-database
    """
    data = pd.read_csv(GITHUB_URL)
    data.rename(columns={'8': 'target'}, inplace=True)
    return data


def fetch_segmentation():
    """Download and transform the Image Segmentation Data Set.
    The minority class is identified as the `1` label
    and the majority class as the rest of the labels.

    https://archive.ics.uci.edu/ml/datasets/Statlog+%28Image+Segmentation%29
    """
    url = urljoin(UCI_ML_DBS, 'statlog/segment/segment.dat')
    data = pd.read_csv(url, header=None, delim_whitespace=True)
    data = data.drop(columns=[2, 3, 4]).rename(columns={19: 'target'})
    data['target'] = data['target'].isin([1]).astype(int)
    return data


def fetch_vehicle():
    """Download and transform the Vehicle Silhouettes Data Set.
    The minority class is identified as the `1` label
    and the majority class as the rest of the labels.

    https://archive.ics.uci.edu/ml/datasets/Statlog+(Vehicle+Silhouettes)
    """
    data = pd.DataFrame()
    for letter in ascii_lowercase[0:9]:
        url = urljoin(UCI_ML_DBS, 'statlog/vehicle/xa%s.dat') % letter
        partial_data = pd.read_csv(url, header=None, delim_whitespace=True)
        partial_data = partial_data.rename(columns={18: 'target'})
        partial_data['target'] = partial_data['target'].isin(['van']).astype(int)
        data = data.append(partial_data)
    return data


def fetch_wine():
    """Download and transform the Wine Data Set.
    The minority class is identified as the `2` label
    and the majority class as the rest of the labels.

    https://archive.ics.uci.edu/ml/datasets/wine
    """
    url = urljoin(UCI_ML_DBS, 'wine/wine.data')
    data = pd.read_csv(url, header=None)
    data = data.rename(columns={0: 'target'})
    data['target'] = data['target'].isin([2]).astype(int)
    return data


def fetch_new_thyroid_1():
    """Download and transform the Thyroid 1 Disease Data Set.
    The minority class is identified as the `positive`
    label and the majority class as the `negative` label.

    http://sci2s.ugr.es/keel/dataset.php?cod=145
    """
    url = urljoin(join(KEEL, 'imb_IRlowerThan9/'), 'new-thyroid1.zip')
    zipped_data = requests.get(url).content
    unzipped_data = ZipFile(BytesIO(zipped_data)).read('new-thyroid1.dat').decode('utf-8')
    data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None, sep=', ', engine='python')
    data = data.rename(columns={5: 'target'})
    data['target'] = data['target'].isin(['positive']).astype(int)
    return data


def fetch_new_thyroid_2():
    """Download and transform the Thyroid 2 Disease Data Set.
    The minority class is identified as the `positive`
    label and the majority class as the `negative` label.

    http://sci2s.ugr.es/keel/dataset.php?cod=146
    """
    url = urljoin(join(KEEL, 'imb_IRlowerThan9/'), 'new-thyroid2.zip')
    zipped_data = requests.get(url).content
    unzipped_data = ZipFile(BytesIO(zipped_data)).read('newthyroid2.dat').decode('utf-8')
    data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None, sep=', ', engine='python')
    data = data.rename(columns={5: 'target'})
    data['target'] = data['target'].isin(['positive']).astype(int)
    return data


def fetch_cleveland():
    """Download and transform the Heart Disease Cleveland Data Set.
    The minority class is identified as the `positive` label and
    the majority class as the `negative` label.

    http://sci2s.ugr.es/keel/dataset.php?cod=980
    """
    url = urljoin(join(KEEL, 'imb_IRhigherThan9p2/'), 'cleveland-0_vs_4.zip')
    zipped_data = requests.get(url).content
    unzipped_data = ZipFile(BytesIO(zipped_data)).read('cleveland-0_vs_4.dat').decode('utf-8')
    data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None)
    data = data.rename(columns={13: 'target'})
    data['target'] = data['target'].isin(['positive']).astype(int)
    return data


def fetch_dermatology():
    """Download and transform the Dermatology Data Set.
    The minority class is identified as the `positive` label and
    the majority class as the `negative` label.

    http://sci2s.ugr.es/keel/dataset.php?cod=1330
    """
    url = urljoin(join(KEEL, 'imb_IRhigherThan9p3/'), 'dermatology-6.zip')
    zipped_data = requests.get(url).content
    unzipped_data = ZipFile(BytesIO(zipped_data)).read('dermatology-6.dat').decode('utf-8')
    data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None)
    data = data.rename(columns={34: 'target'})
    data['target'] = data['target'].isin(['positive']).astype(int)
    return data


def fetch_led():
    """Download and transform the LED Display Domain Data Set.
    The minority class is identified as the `positive` label and
    the majority class as the `negative` label.

    http://sci2s.ugr.es/keel/dataset.php?cod=998
    """
    url = urljoin(join(KEEL, 'imb_IRhigherThan9p2/'), 'led7digit-0-2-4-5-6-7-8-9_vs_1.zip')
    zipped_data = requests.get(url).content
    unzipped_data = ZipFile(BytesIO(zipped_data)).read('led7digit-0-2-4-5-6-7-8-9_vs_1.dat').decode('utf-8')
    data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None)
    data = data.rename(columns={7: 'target'})
    data['target'] = data['target'].isin(['positive']).astype(int)
    return data


def fetch_page_blocks_0():
    """Download and transform the Page Blocks 0 Data Set.
    The minority class is identified as the `positive` label and
    the majority class as the `negative` label.

    http://sci2s.ugr.es/keel/dataset.php?cod=147
    """
    url = urljoin(join(KEEL, 'imb_IRlowerThan9/'), 'page-blocks0.zip')
    zipped_data = requests.get(url).content
    unzipped_data = ZipFile(BytesIO(zipped_data)).read('page-blocks0.dat').decode('utf-8')
    data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None)
    data = data.rename(columns={10: 'target'})
    data['target'] = data['target'].isin([' positive']).astype(int)
    return data


def fetch_page_blocks_1_3():
    """Download and transform the Page Blocks 1-3 Data Set.
    The minority class is identified as the `positive` label and
    the majority class as the `negative` label.

    http://sci2s.ugr.es/keel/dataset.php?cod=124
    """
    url = urljoin(join(KEEL, 'imb_IRhigherThan9p1/'), 'page-blocks-1-3_vs_4.zip')
    zipped_data = requests.get(url).content
    unzipped_data = ZipFile(BytesIO(zipped_data)).read('page-blocks-1-3_vs_4.dat').decode('utf-8')
    data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None)
    data = data.rename(columns={10: 'target'})
    data['target'] = data['target'].isin(['positive']).astype(int)
    return data


def fetch_vowel():
    """Download and transform the Vowel Recognition Data Set.
    The minority class is identified as the `positive` label and
    the majority class as the `negative` label.

    http://sci2s.ugr.es/keel/dataset.php?cod=127
    """
    url = urljoin(join(KEEL, 'imb_IRhigherThan9p1/'), 'vowel0.zip')
    zipped_data = requests.get(url).content
    unzipped_data = ZipFile(BytesIO(zipped_data)).read('vowel0.dat').decode('utf-8')
    data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None)
    data = data.rename(columns={13: 'target'})
    data['target'] = data['target'].isin([' positive']).astype(int)
    return data


def fetch_yeast_1():
    """Download and transform the Yeast 1 Data Set.
    The minority class is identified as the `positive` label and
    the majority class as the `negative` label.

    http://sci2s.ugr.es/keel/dataset.php?cod=153
    """
    url = urljoin(join(KEEL, 'imb_IRlowerThan9/'), 'yeast1.zip')
    zipped_data = requests.get(url).content
    unzipped_data = ZipFile(BytesIO(zipped_data)).read('yeast1.dat').decode('utf-8')
    data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None)
    data = data.rename(columns={8: 'target'})
    data['target'] = data['target'].isin([' positive']).astype(int)
    return data


def fetch_yeast_3():
    """Download and transform the Yeast 3 Data Set.
    The minority class is identified as the `positive` label and
    the majority class as the `negative` label.

    http://sci2s.ugr.es/keel/dataset.php?cod=154
    """
    url = urljoin(join(KEEL, 'imb_IRlowerThan9/'), 'yeast3.zip')
    zipped_data = requests.get(url).content
    unzipped_data = ZipFile(BytesIO(zipped_data)).read('yeast3.dat').decode('utf-8')
    data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None)
    data = data.rename(columns={8: 'target'})
    data['target'] = data['target'].isin([' positive']).astype(int)
    return data


def fetch_yeast_4():
    """Download and transform the Yeast 4 Data Set.
    The minority class is identified as the `positive` label and
    the majority class as the `negative` label.

    http://sci2s.ugr.es/keel/dataset.php?cod=133
    """
    url = urljoin(join(KEEL, 'imb_IRhigherThan9p1/'), 'yeast4.zip')
    zipped_data = requests.get(url).content
    unzipped_data = ZipFile(BytesIO(zipped_data)).read('yeast4.dat').decode('utf-8')
    data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None)
    data = data.rename(columns={8: 'target'})
    data['target'] = data['target'].isin([' positive']).astype(int)
    return data


def fetch_yeast_5():
    """Download and transform the Yeast 5 Data Set.
    The minority class is identified as the `positive` label and
    the majority class as the `negative` label.

    http://sci2s.ugr.es/keel/dataset.php?cod=134
    """
    url = urljoin(join(KEEL, 'imb_IRhigherThan9p1/'), 'yeast5.zip')
    zipped_data = requests.get(url).content
    unzipped_data = ZipFile(BytesIO(zipped_data)).read('yeast5.dat').decode('utf-8')
    data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None)
    data = data.rename(columns={8: 'target'})
    data['target'] = data['target'].isin([' positive']).astype(int)
    return data


def fetch_yeast_6():
    """Download and transform the Yeast 6 Data Set.
    The minority class is identified as the `positive` label and
    the majority class as the `negative` label.

    http://sci2s.ugr.es/keel/dataset.php?cod=135
    """
    url = urljoin(join(KEEL, 'imb_IRhigherThan9p1/'), 'yeast6.zip')
    zipped_data = requests.get(url).content
    unzipped_data = ZipFile(BytesIO(zipped_data)).read('yeast6.dat').decode('utf-8')
    data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None)
    data = data.rename(columns={8: 'target'})
    data['target'] = data['target'].isin([' positive']).astype(int)
    return data


def fetch_mandelon_1():
    """Simulate a variation of the MANDELON Data Set."""
    X, y = make_classification(n_samples=4000, n_features=20, weights=[0.97, 0.03], random_state=RANDOM_STATE)
    data = pd.DataFrame(np.column_stack([X, y]))
    data = data.rename(columns={20: 'target'})
    data.target = data.target.astype(int)
    return data


def fetch_mandelon_2():
    """Simulate a variation of the MANDELON Data Set."""
    X, y = make_classification(n_samples=3000, n_features=200, weights=[0.97, 0.03], random_state=RANDOM_STATE)
    data = pd.DataFrame(np.column_stack([X, y]))
    data = data.rename(columns={200: 'target'})
    data.target = data.target.astype(int)
    return data


def download(fetch_functions):
    """Download datasets."""
    datasets = []
    for func_name, fetch_data in tqdm(fetch_functions.items(), desc='Datasets'):
        name = sub('fetch_', '', func_name).upper().replace('_', ' ')
        data = _modifiy_columns(fetch_data())
        datasets.append((name, data))
    return datasets


def save(path, datasets):
    """Save datasets."""
    with connect(join(path, 'imbalanced_data.db')) as connection:
        for (name, data), factor in list(product(datasets, MULTIPLICATION_FACTORS)):
            tbl_name = f'{name} ({factor})' if factor > 1.0 else name
            ratio = _calculate_ratio(factor, data.target)
            if ratio[1] >= 15:
                data = _make_imbalance(data, factor)
                data.to_sql(tbl_name, connection, index=False)


def parse_path():
    """Parse path from command-line arguments."""
    parser = ArgumentParser('Download and save datasets.')
    parser.add_argument('path', nargs='?', default='.', help='The relative or absolute path to save the datasets.')
    return abspath(parser.parse_args().path)


if __name__ == '__main__':

    # Parse path
    path = parse_path()

    # Download datasets
    datasets = download({key: value for key, value in locals().items() if match('fetch', key)})

    # Save datasets
    save(path, datasets)
