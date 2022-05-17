"""
Create a database of binary class datasets.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
#         Joao Fonseca <jpmrfonseca@gmail.com>
# License: MIT

from re import sub
from urllib.parse import urljoin
from zipfile import ZipFile
from io import BytesIO, StringIO

import requests
import pandas as pd

from .base import BaseDatasets

UCI_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'
FETCH_URLS = {
    'banknote_authentication': urljoin(
        UCI_URL, '00267/data_banknote_authentication.txt'
    ),
    'arcene': urljoin(UCI_URL, 'arcene/'),
    'audit': urljoin(UCI_URL, '00475/audit_data.zip'),
    'spambase': urljoin(UCI_URL, 'spambase/spambase.data'),
    'parkinsons': urljoin(UCI_URL, 'parkinsons/parkinsons.data'),
    'ionosphere': urljoin(UCI_URL, 'ionosphere/ionosphere.data'),
    'breast_cancer': urljoin(UCI_URL, 'breast-cancer-wisconsin/wdbc.data'),
}


class BinaryDatasets(BaseDatasets):
    """Class to download, transform and save binary class datasets."""

    def fetch_banknote_authentication(self):
        """Download and transform the Banknote Authentication Data Set.

        https://archive.ics.uci.edu/ml/datasets/banknote+authentication
        """
        data = pd.read_csv(FETCH_URLS['banknote_authentication'], header=None)
        data.rename(columns={4: 'target'}, inplace=True)
        return data

    def fetch_arcene(self):
        """Download and transform the Arcene Data Set.

        https://archive.ics.uci.edu/ml/datasets/Arcene
        """
        url = FETCH_URLS['arcene']
        data, labels = [], []
        for data_type in ('train', 'valid'):
            data.append(
                pd.read_csv(
                    urljoin(url, f'ARCENE/arcene_{data_type}.data'),
                    header=None,
                    sep=' ',
                ).drop(columns=list(range(1998, 10001)))
            )
            labels.append(
                pd.read_csv(
                    urljoin(
                        url,
                        ('ARCENE/' if data_type == 'train' else '')
                        + f'arcene_{data_type}.labels',
                    ),
                    header=None,
                ).rename(columns={0: 'target'})
            )
        data = pd.concat(data, ignore_index=True)
        labels = pd.concat(labels, ignore_index=True)
        data = pd.concat([data, labels], axis=1)
        data['target'] = data['target'].isin([1]).astype(int)
        return data

    def fetch_audit(self):
        """Download and transform the Audit Data Set.

        https://archive.ics.uci.edu/ml/datasets/Audit+Data
        """
        zipped_data = requests.get(FETCH_URLS['audit']).content
        unzipped_data = (
            ZipFile(BytesIO(zipped_data))
            .read('audit_data/audit_risk.csv')
            .decode('utf-8')
        )
        data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), engine='python')
        data = (
            data.drop(columns=['LOCATION_ID'])
            .rename(columns={'Risk': 'target'})
            .dropna()
        )
        return data

    def fetch_spambase(self):
        """Download and transform the Spambase Data Set.

        https://archive.ics.uci.edu/ml/datasets/Spambase
        """
        data = pd.read_csv(FETCH_URLS['spambase'], header=None)
        data.rename(columns={57: 'target'}, inplace=True)
        return data

    def fetch_parkinsons(self):
        """Download and transform the Parkinsons Data Set.

        https://archive.ics.uci.edu/ml/datasets/parkinsons
        """
        data = pd.read_csv(FETCH_URLS['parkinsons'])
        data = pd.concat(
            [
                data.drop(columns=['name', 'status']),
                data[['status']].rename(columns={'status': 'target'}),
            ],
            axis=1,
        )
        data['target'] = data['target'].isin([0]).astype(int)
        return data

    def fetch_ionosphere(self):
        """Download and transform the Ionosphere Data Set.

        https://archive.ics.uci.edu/ml/datasets/ionosphere
        """
        data = pd.read_csv(FETCH_URLS['ionosphere'], header=None)
        data = data.drop(columns=[0, 1]).rename(columns={34: 'target'})
        data['target'] = data['target'].isin(['b']).astype(int)
        return data

    def fetch_breast_cancer(self):
        """Download and transform the Breast Cancer Wisconsin Data Set.

        https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
        """
        data = pd.read_csv(FETCH_URLS['breast_cancer'], header=None)
        data = pd.concat(
            [data.drop(columns=[0, 1]), data[[1]].rename(columns={1: 'target'})], axis=1
        )
        data['target'] = data['target'].isin(['M']).astype(int)
        return data
