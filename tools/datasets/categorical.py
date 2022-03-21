"""
Create a database of datasets that contain categorical features.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
#         Joao Fonseca <jpmrfonseca@gmail.com>
# License: MIT

from os.path import join
from urllib.parse import urljoin
from sqlite3 import connect

from rich.progress import track
import numpy as np
import pandas as pd

from .base import Datasets

UCI_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'
FETCH_URLS = {
    'adult': urljoin(UCI_URL, 'adult/adult.data'),
    'abalone': urljoin(UCI_URL, 'abalone/abalone.data'),
    'acute': urljoin(UCI_URL, 'acute/diagnosis.data'),
    'annealing': urljoin(UCI_URL, 'annealing/anneal.data'),
    'census': urljoin(UCI_URL, 'census-income-mld/census-income.data.gz'),
    'contraceptive': urljoin(UCI_URL, 'cmc/cmc.data'),
    'covertype': urljoin(UCI_URL, 'covtype/covtype.data.gz'),
    'credit_approval': urljoin(UCI_URL, 'credit-screening/crx.data'),
    'dermatology': urljoin(UCI_URL, 'dermatology/dermatology.data'),
    'echocardiogram': urljoin(UCI_URL, 'echocardiogram/echocardiogram.data'),
    'flags': urljoin(UCI_URL, 'flags/flag.data'),
    'heart_disease': [
        urljoin(UCI_URL, 'heart-disease/processed.cleveland.data'),
        urljoin(UCI_URL, 'heart-disease/processed.hungarian.data'),
        urljoin(UCI_URL, 'heart-disease/processed.switzerland.data'),
        urljoin(UCI_URL, 'heart-disease/processed.va.data'),
    ],
    'hepatitis': urljoin(UCI_URL, 'hepatitis/hepatitis.data'),
    'german_credit': urljoin(UCI_URL, 'statlog/german/german.data'),
    'thyroid': urljoin(UCI_URL, 'thyroid-disease/thyroid0387.data'),
}


class CategoricalDatasets(Datasets):
    """Class to download, transform and save datasets with both continuous
    and categorical features."""

    @staticmethod
    def _modify_columns(data, categorical_features):
        """Rename and reorder columns of dataframe."""
        X, y = data.drop(columns='target'), data.target
        X.columns = range(len(X.columns))
        return pd.concat([X, y], axis=1), categorical_features

    def download(self):
        """Download the datasets."""
        if self.names == 'all':
            func_names = [func_name for func_name in dir(self) if 'fetch_' in func_name]
        else:
            func_names = [
                f'fetch_{name}'.lower().replace(' ', '_') for name in self.names
            ]
        self.content_ = []
        for func_name in track(func_names, description='Datasets'):
            name = func_name.replace('fetch_', '').upper().replace('_', ' ')
            fetch_data = getattr(self, func_name)
            data, categorical_features = self._modify_columns(*fetch_data())
            self.content_.append((name, data, categorical_features))
        return self

    def save(self, path, db_name):
        """Save datasets."""
        with connect(join(path, f'{db_name}.db')) as connection:
            for name, data in self.content_:
                data.to_sql(name, connection, index=False, if_exists='replace')

    def fetch_adult(self):
        """Download and transform the Adult Data Set.

        https://archive.ics.uci.edu/ml/datasets/Adult
        """
        data = pd.read_csv(FETCH_URLS['adult'], header=None, na_values=' ?').dropna()
        data.rename(columns={data.columns[-1]: 'target'}, inplace=True)
        categorical_features = [1, 3, 5, 6, 7, 8, 9, 13]
        return data, categorical_features

    def fetch_abalone(self):
        """Download and transform the Abalone Data Set.

        https://archive.ics.uci.edu/ml/datasets/Abalone
        """
        data = pd.read_csv(FETCH_URLS['abalone'], header=None)
        data.rename(columns={data.columns[-1]: 'target'}, inplace=True)
        categorical_features = [0]
        return data, categorical_features

    def fetch_acute(self):
        """Download and transform the Acute Inflammations Data Set.

        https://archive.ics.uci.edu/ml/datasets/Acute+Inflammations
        """
        data = pd.read_csv(
            FETCH_URLS['acute'], header=None, sep='\t', decimal=',', encoding='UTF-16'
        )
        data['target'] = data[6].str[0] + data[7].str[0]
        data.drop(columns=[6, 7], inplace=True)
        categorical_features = list(range(1, 6))
        return data, categorical_features

    def fetch_annealing(self):
        """Download and transform the Annealing Data Set.

        https://archive.ics.uci.edu/ml/datasets/Annealing
        """
        data = pd.read_csv(FETCH_URLS['annealing'], header=None, na_values='?')

        # some features are dropped; they have too many missing values
        missing_feats = (data.isnull().sum(0) / data.shape[0]) < 0.1
        data = data.iloc[:, missing_feats.values]
        data[2].fillna(data[2].mode().squeeze(), inplace=True)

        data = data.T.reset_index(drop=True).T
        data.rename(columns={data.columns[-1]: 'target'}, inplace=True)

        categorical_features = [0, 1, 5, 9]
        return data, categorical_features

    def fetch_census(self):
        """Download and transform the Census-Income (KDD) Data Set.

        https://archive.ics.uci.edu/ml/datasets/Census-Income+%28KDD%29
        """
        data = pd.read_csv(FETCH_URLS['census'], header=None)

        categorical_features = (
            list(range(1, 5))
            + list(range(6, 16))
            + list(range(19, 29))
            + list(range(30, 38))
            + [39]
        )

        # some features are dropped; they have too many missing values
        cols_ids = [1, 6, 9, 13, 14, 20, 21, 29, 31, 37]
        categorical_features = np.argwhere(
            np.delete(
                data.rename(columns={k: f'nom_{k}' for k in categorical_features})
                .columns.astype('str')
                .str.startswith('nom_'),
                cols_ids,
            )
        ).squeeze()
        data = data.drop(columns=cols_ids).T.reset_index(drop=True).T
        # some rows are dropped; they have rare missing values
        data = data.iloc[
            data.applymap(lambda x: x != ' Not in universe').all(1).values, :
        ]

        data.rename(columns={data.columns[-1]: 'target'}, inplace=True)
        return data, categorical_features

    def fetch_contraceptive(self):
        """Download and transform the Contraceptive Method Choice Data Set.

        https://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice
        """
        data = pd.read_csv(FETCH_URLS['contraceptive'], header=None)
        data.rename(columns={data.columns[-1]: 'target'}, inplace=True)
        categorical_features = [4, 5, 6, 8]
        return data, categorical_features

    def fetch_covertype(self):
        """Download and transform the Covertype Data Set.

        https://archive.ics.uci.edu/ml/datasets/Covertype
        """
        data = pd.read_csv(FETCH_URLS['covertype'], header=None)
        data.rename(columns={data.columns[-1]: 'target'}, inplace=True)
        wilderness_area = pd.Series(
            np.argmax(data.iloc[:, 10:14].values, axis=1), name=10
        )
        soil_type = pd.Series(np.argmax(data.iloc[:, 14:54].values, axis=1), name=11)
        data = (
            data.drop(columns=list(range(10, 54)))
            .join(wilderness_area)
            .join(soil_type)[list(range(0, 12)) + ['target']]
        )
        categorical_features = [10, 11]
        return data, categorical_features

    def fetch_credit_approval(self):
        """Download and transform the Credit Approval Data Set.

        https://archive.ics.uci.edu/ml/datasets/Credit+Approval
        """
        data = pd.read_csv(
            FETCH_URLS['credit_approval'], header=None, na_values='?'
        ).dropna()
        data.rename(columns={data.columns[-1]: 'target'}, inplace=True)
        categorical_features = [0, 3, 4, 5, 6, 8, 9, 11, 12]
        return data, categorical_features

    def fetch_dermatology(self):
        """Download and transform the Dermatology Data Set.

        https://archive.ics.uci.edu/ml/datasets/Dermatology
        """
        data = pd.read_csv(
            FETCH_URLS['dermatology'], header=None, na_values='?'
        ).dropna()
        data.rename(columns={data.columns[-1]: 'target'}, inplace=True)
        categorical_features = list(range(data.shape[0]))
        categorical_features.remove(33)
        return data, categorical_features

    def fetch_echocardiogram(self):
        """Download and transform the Echocardiogram Data Set.

        https://archive.ics.uci.edu/ml/datasets/Echocardiogram
        """
        data = pd.read_csv(
            FETCH_URLS['echocardiogram'],
            header=None,
            error_bad_lines=False,
            warn_bad_lines=False,
            na_values='?',
        )
        data.drop(columns=[10, 11], inplace=True)
        data.dropna(inplace=True)
        data.rename(columns={data.columns[-1]: 'target'}, inplace=True)
        categorical_features = [1, 3]
        return data, categorical_features

    def fetch_flags(self):
        """Download and transform the Flags Data Set.

        https://archive.ics.uci.edu/ml/datasets/Flags
        """
        data = pd.read_csv(FETCH_URLS['flags'], header=None)
        target = data[6].rename('target')
        data = data.drop(columns=[0, 6]).T.reset_index(drop=True).T.join(target)
        categorical_features = [
            0,
            1,
            4,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
        ]
        return data, categorical_features

    def fetch_heart_disease(self):
        """Download and transform the Heart Disease Data Set.

        https://archive.ics.uci.edu/ml/datasets/Heart+Disease
        """
        data = (
            pd.concat(
                [
                    pd.read_csv(url, header=None, na_values='?')
                    for url in FETCH_URLS['heart_disease']
                ],
                ignore_index=True,
            )
            .drop(columns=[10, 11, 12])
            .dropna()
        )
        data.rename(columns={data.columns[-1]: 'target'}, inplace=True)
        categorical_features = [1, 2, 5, 6, 8]
        return data, categorical_features

    def fetch_hepatitis(self):
        """Download and transform the Hepatitis Data Set.

        https://archive.ics.uci.edu/ml/datasets/Hepatitis
        """
        data = (
            pd.read_csv(FETCH_URLS['hepatitis'], header=None, na_values='?')
            .drop(columns=[15, 18])
            .dropna()
        )
        target = data[0].rename('target')
        data = data.drop(columns=[0]).T.reset_index(drop=True).T.join(target)
        categorical_features = list(range(1, 13)) + [16]
        return data, categorical_features

    def fetch_german_credit(self):
        """Download and transform the German Credit Data Set.

        https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29
        """
        data = pd.read_csv(FETCH_URLS['german_credit'], header=None, sep=' ')
        data.rename(columns={data.columns[-1]: 'target'}, inplace=True)
        categorical_features = (
            np.argwhere(data.iloc[0, :-1].apply(lambda x: str(x)[0] == 'A').values)
            .squeeze()
            .tolist()
        )
        return data, categorical_features

    def fetch_heart(self):
        """Download and transform the Heart Data Set.

        http://archive.ics.uci.edu/ml/datasets/statlog+(heart)
        """
        data = pd.read_csv(FETCH_URLS['heart'], header=None, delim_whitespace=True)
        data.rename(columns={data.columns[-1]: 'target'}, inplace=True)
        categorical_features = [1, 2, 5, 6, 8, 10, 12]
        return data, categorical_features

    def fetch_thyroid(self):
        """Download and transform the Thyroid Disease Data Set.
        Label 0 corresponds to no disease found.
        Label 1 corresponds to one or multiple diseases found.

        https://archive.ics.uci.edu/ml/datasets/Thyroid+Disease
        """
        data = (
            pd.read_csv(FETCH_URLS['thyroid'], header=None, na_values='?')
            .drop(columns=27)
            .dropna()
            .T.reset_index(drop=True)
            .T
        )
        data.rename(columns={data.columns[-1]: 'target'}, inplace=True)
        data['target'] = (
            data['target'].apply(lambda x: x.split('[')[0]) != '-'
        ).astype(int)
        categorical_features = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            18,
            20,
            22,
            24,
            26,
            27,
        ]
        return data, categorical_features
