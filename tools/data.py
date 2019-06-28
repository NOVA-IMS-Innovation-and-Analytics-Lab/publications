"""
Download, transform and simulate various datasets.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from os.path import join
from os import remove
from re import sub
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

UCI_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'
KEEL_URL = 'http://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/'
FETCH_URLS = {
    'breast_tissue': urljoin(UCI_URL, '00192/BreastTissue.xls'),
    'ecoli': urljoin(UCI_URL, 'ecoli/ecoli.data'),
    'eucalyptus': 'https://www.openml.org/data/get_csv/3625/dataset_194_eucalyptus.arff',
    'glass': urljoin(UCI_URL, 'glass/glass.data'),
    'haberman': urljoin(UCI_URL, 'haberman/haberman.data'),
    'heart': urljoin(UCI_URL, 'statlog/heart/heart.dat'),
    'iris': urljoin(UCI_URL, 'iris/bezdekIris.data'),
    'libras': urljoin(UCI_URL, 'libras/movement_libras.data'),
    'liver': urljoin(UCI_URL, 'liver-disorders/bupa.data'),
    'pima': 'https://raw.githubusercontent.com/IMS-ML-Lab/publications/master/assets/data/various.db',
    'segmentation': urljoin(UCI_URL, 'statlog/segment/segment.dat'),
    'vehicle': urljoin(UCI_URL, 'statlog/vehicle/'),
    'wine': urljoin(UCI_URL, 'wine/wine.data'),
    'new_thyroid_1': urljoin(urljoin(KEEL_URL, 'imb_IRlowerThan9/'), 'new-thyroid1.zip'),
    'new_thyroid_2': urljoin(urljoin(KEEL_URL, 'imb_IRlowerThan9/'), 'new-thyroid2.zip'),
    'cleveland': urljoin(urljoin(KEEL_URL, 'imb_IRhigherThan9p2/'), 'cleveland-0_vs_4.zip'),
    'dermatology': urljoin(urljoin(KEEL_URL, 'imb_IRhigherThan9p3/'), 'dermatology-6.zip'),
    'led': urljoin(urljoin(KEEL_URL, 'imb_IRhigherThan9p2/'), 'led7digit-0-2-4-5-6-7-8-9_vs_1.zip'),
    'page_blocks_0': urljoin(urljoin(KEEL_URL, 'imb_IRlowerThan9/'), 'page-blocks0.zip'),
    'page_blocks_1_3': urljoin(urljoin(KEEL_URL, 'imb_IRhigherThan9p1/'), 'page-blocks-1-3_vs_4.zip'),
    'vowel': urljoin(urljoin(KEEL_URL, 'imb_IRhigherThan9p1/'), 'vowel0.zip'),
    'yeast_1': urljoin(urljoin(KEEL_URL, 'imb_IRlowerThan9/'), 'yeast1.zip'),
    'yeast_3': urljoin(urljoin(KEEL_URL, 'imb_IRlowerThan9/'), 'yeast3.zip'),
    'yeast_4': urljoin(urljoin(KEEL_URL, 'imb_IRhigherThan9p1/'), 'yeast4.zip'),
    'yeast_5': urljoin(urljoin(KEEL_URL, 'imb_IRhigherThan9p1/'), 'yeast4.zip'),
    'yeast_6': urljoin(urljoin(KEEL_URL, 'imb_IRhigherThan9p1/'), 'yeast5.zip'),
    'banknote_authentication': urljoin(UCI_URL, '00267/data_banknote_authentication.txt'),
    'arcene': urljoin(UCI_URL, 'arcene/'),
    'audit': urljoin(UCI_URL, '00475/audit_data.zip'),
    'spambase': urljoin(UCI_URL, 'spambase/spambase.data')
}
MULTIPLICATION_FACTORS = [2, 3]
RANDOM_STATE = 0


class Datasets:
    """Class to download and save datasets."""

    @staticmethod
    def _modify_columns(data):
        """Rename and reorder columns of dataframe."""
        X, y = data.drop(columns='target'), data.target
        X.columns = range(len(X.columns))
        return pd.concat([X, y], axis=1)
    
    def download(self):
        """Download the datasets."""
        self.datasets_ = []
        func_names = [func_name for func_name in dir(self) if 'fetch_' in func_name]
        for func_name in tqdm(func_names, desc='Datasets'):
            name = sub('fetch_', '', func_name).upper().replace('_', ' ')
            fetch_data = getattr(self, func_name)
            data = self._modify_columns(fetch_data())
            self.datasets_.append((name, data))
        return self
    
    def save(self, path, db_name):
        """Save datasets."""
        with connect(join(path, f'{db_name}.db')) as connection:
            for name, data in self.datasets_:
                data.to_sql(name, connection, index=False, if_exists='replace')


class ImbalancedBinaryClassDatasets(Datasets):
    """Class to download, transform and save binary class imbalanced datasets."""

    @staticmethod
    def _calculate_ratio(multiplication_factor, y):
        """Calculate ratio based on IRs multiplication factor."""
        ratio = Counter(y).copy()
        ratio[1] = int(ratio[1] / multiplication_factor)
        return ratio

    def _make_imbalance(self, data, multiplication_factor):
        """Undersample the minority class."""
        X_columns = [col for col in data.columns if col != 'target']
        X, y = check_X_y(data.loc[:, X_columns], data.target)
        if multiplication_factor > 1.0:
            sampling_strategy = self._calculate_ratio(multiplication_factor, y)
            X, y = make_imbalance(X, y, sampling_strategy=sampling_strategy, random_state=self.RANDOM_STATE)
        data = pd.DataFrame(np.column_stack((X, y)))
        data.iloc[:, -1] = data.iloc[:, -1].astype(int)
        return data
    
    def download(self):
        """Download the datasets and append undersampled versions of them."""
        super(ImbalancedBinaryClassDatasets, self).download()
        undersampled_datasets = []
        for (name, data), factor in list(product(self.datasets_, self.MULTIPLICATION_FACTORS)):
            ratio = self._calculate_ratio(factor, data.target)
            if ratio[1] >= 15:
                data = self._make_imbalance(data, factor)
                undersampled_datasets.append((f'{name} ({factor})', data))
        self.datasets_ += undersampled_datasets
        return self
                
    def fetch_breast_tissue(self):
        """Download and transform the Breast Tissue Data Set.
        The minority class is identified as the `car` and `fad`
        labels and the majority class as the rest of the labels.

        http://archive.ics.uci.edu/ml/datasets/breast+tissue
        """
        data = pd.read_excel(FETCH_URLS['breast_tissue'], sheet_name='Data')
        data = data.drop(columns='Case #').rename(columns={'Class': 'target'})
        data['target'] = data['target'].isin(['car', 'fad']).astype(int)
        return data

    def fetch_ecoli(self):
        """Download and transform the Ecoli Data Set.
        The minority class is identified as the `pp` label
        and the majority class as the rest of the labels.

        https://archive.ics.uci.edu/ml/datasets/ecoli
        """
        data = pd.read_csv(FETCH_URLS['ecoli'], header=None, delim_whitespace=True)
        data = data.drop(columns=0).rename(columns={8: 'target'})
        data['target'] = data['target'].isin(['pp']).astype(int)
        return data

    def fetch_eucalyptus(self):
        """Download and transform the Eucalyptus Data Set.
        The minority class is identified as the `best` label
        and the majority class as the rest of the labels.

        https://www.openml.org/d/188
        """
        data = pd.read_csv(FETCH_URLS['eucalyptus'])
        data = data.iloc[:, -9:].rename(columns={'Utility': 'target'})
        data = data[data != '?'].dropna()
        data['target'] = data['target'].isin(['best']).astype(int)
        return data

    def fetch_glass(self):
        """Download and transform the Glass Identification Data Set.
        The minority class is identified as the `1` label
        and the majority class as the rest of the labels.

        https://archive.ics.uci.edu/ml/datasets/glass+identification
        """
        data = pd.read_csv(FETCH_URLS['glass'], header=None)
        data = data.drop(columns=0).rename(columns={10: 'target'})
        data['target'] = data['target'].isin([1]).astype(int)
        return data

    def fetch_haberman(self):
        """Download and transform the Haberman's Survival Data Set.
        The minority class is identified as the `1` label
        and the majority class as the `0` label.

        https://archive.ics.uci.edu/ml/datasets/Haberman's+Survival
        """
        data = pd.read_csv(FETCH_URLS['haberman'], header=None)
        data.rename(columns={3: 'target'}, inplace=True)
        data['target'] = data['target'].isin([2]).astype(int)
        return data

    def fetch_heart(self):
        """Download and transform the Heart Data Set.
        The minority class is identified as the `2` label
        and the majority class as the `1` label.

        http://archive.ics.uci.edu/ml/datasets/statlog+(heart)
        """
        data = pd.read_csv(FETCH_URLS['heart'], header=None, delim_whitespace=True)
        data.rename(columns={13: 'target'}, inplace=True)
        data['target'] = data['target'].isin([2]).astype(int)
        return data

    def fetch_iris(self):
        """Download and transform the Iris Data Set.
        The minority class is identified as the `1` label
        and the majority class as the rest of the labels.

        https://archive.ics.uci.edu/ml/datasets/iris
        """
        data = pd.read_csv(FETCH_URLS['iris'], header=None)
        data.rename(columns={4: 'target'}, inplace=True)
        data['target'] = data['target'].isin(['Iris-setosa']).astype(int)
        return data

    def fetch_libras(self):
        """Download and transform the Libras Movement Data Set.
        The minority class is identified as the `1` label
        and the majority class as the rest of the labels.

        https://archive.ics.uci.edu/ml/datasets/Libras+Movement
        """
        data = pd.read_csv(FETCH_URLS['libras'], header=None)
        data.rename(columns={90: 'target'}, inplace=True)
        data['target'] = data['target'].isin([1]).astype(int)
        return data

    def fetch_liver(self):
        """Download and transform the Liver Disorders Data Set.
        The minority class is identified as the `1` label
        and the majority class as the '2' label.

        https://archive.ics.uci.edu/ml/datasets/liver+disorders
        """
        data = pd.read_csv(FETCH_URLS['liver'], header=None)
        data.rename(columns={6: 'target'}, inplace=True)
        data['target'] = data['target'].isin([1]).astype(int)
        return data

    def fetch_pima(self):
        """Download and transform the Pima Indians Diabetes Data Set.
        The minority class is identified as the `1` label
        and the majority class as the '0' label.

        https://www.kaggle.com/uciml/pima-indians-diabetes-database
        """
        database = requests.get(FETCH_URLS['pima']).content
        with open('temp.db', 'wb') as file:
            file.write(database)
        with connect('temp.db') as con:
            data = pd.read_sql('select * from pima', con)
        data.rename(columns={'8': 'target'}, inplace=True)
        remove('temp.db')
        return data

    def fetch_segmentation(self):
        """Download and transform the Image Segmentation Data Set.
        The minority class is identified as the `1` label
        and the majority class as the rest of the labels.

        https://archive.ics.uci.edu/ml/datasets/Statlog+%28Image+Segmentation%29
        """
        data = pd.read_csv(FETCH_URLS['segmentation'], header=None, delim_whitespace=True)
        data = data.drop(columns=[2, 3, 4]).rename(columns={19: 'target'})
        data['target'] = data['target'].isin([1]).astype(int)
        return data

    def fetch_vehicle(self):
        """Download and transform the Vehicle Silhouettes Data Set.
        The minority class is identified as the `1` label
        and the majority class as the rest of the labels.

        https://archive.ics.uci.edu/ml/datasets/Statlog+(Vehicle+Silhouettes)
        """
        data = pd.DataFrame()
        for letter in ascii_lowercase[0:9]:
            partial_data = pd.read_csv(urljoin(FETCH_URLS['vehicle'], 'xa%s.dat'% letter), header=None, delim_whitespace=True)
            partial_data = partial_data.rename(columns={18: 'target'})
            partial_data['target'] = partial_data['target'].isin(['van']).astype(int)
            data = data.append(partial_data)
        return data

    def fetch_wine(self):
        """Download and transform the Wine Data Set.
        The minority class is identified as the `2` label
        and the majority class as the rest of the labels.

        https://archive.ics.uci.edu/ml/datasets/wine
        """
        data = pd.read_csv(FETCH_URLS['wine'], header=None)
        data.rename(columns={0: 'target'}, inplace=True)
        data['target'] = data['target'].isin([2]).astype(int)
        return data

    def fetch_new_thyroid_1(self):
        """Download and transform the Thyroid 1 Disease Data Set.
        The minority class is identified as the `positive`
        label and the majority class as the `negative` label.

        http://sci2s.ugr.es/keel/dataset.php?cod=145
        """
        zipped_data = requests.get(FETCH_URLS['new_thyroid_1']).content
        unzipped_data = ZipFile(BytesIO(zipped_data)).read('new-thyroid1.dat').decode('utf-8')
        data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None, sep=', ', engine='python')
        data.rename(columns={5: 'target'}, inplace=True)
        data['target'] = data['target'].isin(['positive']).astype(int)
        return data

    def fetch_new_thyroid_2(self):
        """Download and transform the Thyroid 2 Disease Data Set.
        The minority class is identified as the `positive`
        label and the majority class as the `negative` label.

        http://sci2s.ugr.es/keel/dataset.php?cod=146
        """
        zipped_data = requests.get(FETCH_URLS['new_thyroid_2']).content
        unzipped_data = ZipFile(BytesIO(zipped_data)).read('newthyroid2.dat').decode('utf-8')
        data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None, sep=', ', engine='python')
        data.rename(columns={5: 'target'}, inplace=True)
        data['target'] = data['target'].isin(['positive']).astype(int)
        return data

    def fetch_cleveland(self):
        """Download and transform the Heart Disease Cleveland Data Set.
        The minority class is identified as the `positive` label and
        the majority class as the `negative` label.

        http://sci2s.ugr.es/keel/dataset.php?cod=980
        """
        zipped_data = requests.get(FETCH_URLS['cleveland']).content
        unzipped_data = ZipFile(BytesIO(zipped_data)).read('cleveland-0_vs_4.dat').decode('utf-8')
        data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None)
        data.rename(columns={13: 'target'}, inplace=True)
        data['target'] = data['target'].isin(['positive']).astype(int)
        return data

    def fetch_dermatology(self):
        """Download and transform the Dermatology Data Set.
        The minority class is identified as the `positive` label and
        the majority class as the `negative` label.

        http://sci2s.ugr.es/keel/dataset.php?cod=1330
        """
        zipped_data = requests.get(FETCH_URLS['dermatology']).content
        unzipped_data = ZipFile(BytesIO(zipped_data)).read('dermatology-6.dat').decode('utf-8')
        data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None)
        data.rename(columns={34: 'target'}, inplace=True)
        data['target'] = data['target'].isin(['positive']).astype(int)
        return data

    def fetch_led(self):
        """Download and transform the LED Display Domain Data Set.
        The minority class is identified as the `positive` label and
        the majority class as the `negative` label.

        http://sci2s.ugr.es/keel/dataset.php?cod=998
        """
        zipped_data = requests.get(FETCH_URLS['led']).content
        unzipped_data = ZipFile(BytesIO(zipped_data)).read('led7digit-0-2-4-5-6-7-8-9_vs_1.dat').decode('utf-8')
        data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None)
        data.rename(columns={7: 'target'}, inplace=True)
        data['target'] = data['target'].isin(['positive']).astype(int)
        return data

    def fetch_page_blocks_0(self):
        """Download and transform the Page Blocks 0 Data Set.
        The minority class is identified as the `positive` label and
        the majority class as the `negative` label.

        http://sci2s.ugr.es/keel/dataset.php?cod=147
        """
        zipped_data = requests.get(FETCH_URLS['page_blocks_0']).content
        unzipped_data = ZipFile(BytesIO(zipped_data)).read('page-blocks0.dat').decode('utf-8')
        data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None)
        data.rename(columns={10: 'target'}, inplace=True)
        data['target'] = data['target'].isin([' positive']).astype(int)
        return data

    def fetch_page_blocks_1_3(self):
        """Download and transform the Page Blocks 1-3 Data Set.
        The minority class is identified as the `positive` label and
        the majority class as the `negative` label.

        http://sci2s.ugr.es/keel/dataset.php?cod=124
        """
        zipped_data = requests.get(FETCH_URLS['page_blocks_1_3']).content
        unzipped_data = ZipFile(BytesIO(zipped_data)).read('page-blocks-1-3_vs_4.dat').decode('utf-8')
        data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None)
        data.rename(columns={10: 'target'}, inplace=True)
        data['target'] = data['target'].isin(['positive']).astype(int)
        return data

    def fetch_vowel(self):
        """Download and transform the Vowel Recognition Data Set.
        The minority class is identified as the `positive` label and
        the majority class as the `negative` label.

        http://sci2s.ugr.es/keel/dataset.php?cod=127
        """
        zipped_data = requests.get(FETCH_URLS['vowel']).content
        unzipped_data = ZipFile(BytesIO(zipped_data)).read('vowel0.dat').decode('utf-8')
        data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None)
        data.rename(columns={13: 'target'}, inplace=True)
        data['target'] = data['target'].isin([' positive']).astype(int)
        return data

    def fetch_yeast_1(self):
        """Download and transform the Yeast 1 Data Set.
        The minority class is identified as the `positive` label and
        the majority class as the `negative` label.

        http://sci2s.ugr.es/keel/dataset.php?cod=153
        """
        zipped_data = requests.get(FETCH_URLS['yeast_1']).content
        unzipped_data = ZipFile(BytesIO(zipped_data)).read('yeast1.dat').decode('utf-8')
        data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None)
        data.rename(columns={8: 'target'}, inplace=True)
        data['target'] = data['target'].isin([' positive']).astype(int)
        return data

    def fetch_yeast_3(self):
        """Download and transform the Yeast 3 Data Set.
        The minority class is identified as the `positive` label and
        the majority class as the `negative` label.

        http://sci2s.ugr.es/keel/dataset.php?cod=154
        """
        zipped_data = requests.get(FETCH_URLS['yeast_3']).content
        unzipped_data = ZipFile(BytesIO(zipped_data)).read('yeast3.dat').decode('utf-8')
        data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None)
        data.rename(columns={8: 'target'}, inplace=True)
        data['target'] = data['target'].isin([' positive']).astype(int)
        return data

    def fetch_yeast_4(self):
        """Download and transform the Yeast 4 Data Set.
        The minority class is identified as the `positive` label and
        the majority class as the `negative` label.

        http://sci2s.ugr.es/keel/dataset.php?cod=133
        """
        zipped_data = requests.get(FETCH_URLS['yeast_4']).content
        unzipped_data = ZipFile(BytesIO(zipped_data)).read('yeast4.dat').decode('utf-8')
        data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None)
        data.rename(columns={8: 'target'}, inplace=True)
        data['target'] = data['target'].isin([' positive']).astype(int)
        return data

    def fetch_yeast_5(self):
        """Download and transform the Yeast 5 Data Set.
        The minority class is identified as the `positive` label and
        the majority class as the `negative` label.

        http://sci2s.ugr.es/keel/dataset.php?cod=134
        """
        zipped_data = requests.get(FETCH_URLS['yeast_5']).content
        unzipped_data = ZipFile(BytesIO(zipped_data)).read('yeast5.dat').decode('utf-8')
        data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None)
        data.rename(columns={8: 'target'}, inplace=True)
        data['target'] = data['target'].isin([' positive']).astype(int)
        return data

    def fetch_yeast_6(self):
        """Download and transform the Yeast 6 Data Set.
        The minority class is identified as the `positive` label and
        the majority class as the `negative` label.

        http://sci2s.ugr.es/keel/dataset.php?cod=135
        """
        zipped_data = requests.get(FETCH_URLS['yeast_6']).content
        unzipped_data = ZipFile(BytesIO(zipped_data)).read('yeast6.dat').decode('utf-8')
        data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None)
        data.rename(columns={8: 'target'}, inplace=True)
        data['target'] = data['target'].isin([' positive']).astype(int)
        return data

    def fetch_mandelon_1(self):
        """Simulate a variation of the MANDELON Data Set."""
        X, y = make_classification(n_samples=4000, n_features=20, weights=[0.97, 0.03], random_state=self.RANDOM_STATE)
        data = pd.DataFrame(np.column_stack([X, y]))
        data.rename(columns={20: 'target'}, inplace=True)
        data.target = data.target.astype(int)
        return data

    def fetch_mandelon_2(self):
        """Simulate a variation of the MANDELON Data Set."""
        X, y = make_classification(n_samples=3000, n_features=200, weights=[0.97, 0.03], random_state=self.RANDOM_STATE)
        data = pd.DataFrame(np.column_stack([X, y]))
        data.rename(columns={200: 'target'}, inplace=True)
        data.target = data.target.astype(int)
        return data


class BinaryClassDatasets(Datasets):
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
            data.append(pd.read_csv(urljoin(url, f'ARCENE/arcene_{data_type}.data'), header=None, sep=' ').drop(columns=list(range(1999, 10001))))
            labels.append(pd.read_csv(urljoin(url, ('ARCENE/' if data_type == 'train' else '') + f'arcene_{data_type}.labels'), header=None).rename(columns={0:'target'}))
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
        unzipped_data = ZipFile(BytesIO(zipped_data)).read('audit_data/audit_risk.csv').decode('utf-8')
        data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), engine='python')
        data = data.rename(columns={'Risk': 'target'}).dropna()
        return data

    def fetch_spambase(self):
        """Download and transform the Spambase Data Set.

        https://archive.ics.uci.edu/ml/datasets/Spambase
        """
        data = pd.read_csv(FETCH_URLS['spambase'], header=None)
        data.rename(columns={57: 'target'}, inplace=True)
        return data

