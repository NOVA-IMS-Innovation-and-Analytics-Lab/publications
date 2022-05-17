"""
Create a database of remote sensing datasets.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
#         Joao Fonseca <jpmrfonseca@gmail.com>
# License: MIT

from urllib.parse import urljoin
from scipy.io import loadmat
import io
import requests

import numpy as np
import pandas as pd

from .base import BaseDatasets

GIC_URL = 'http://www.ehu.eus/ccwintco/uploads/'
FETCH_URLS = {
    'indian_pines': [
        urljoin(GIC_URL, '2/22/Indian_pines.mat'),
        urljoin(GIC_URL, 'c/c4/Indian_pines_gt.mat'),
    ],
    'salinas': [
        urljoin(GIC_URL, 'f/f1/Salinas.mat'),
        urljoin(GIC_URL, 'f/fa/Salinas_gt.mat'),
    ],
    'salinas_a': [
        urljoin(GIC_URL, 'd/df/SalinasA.mat'),
        urljoin(GIC_URL, 'a/aa/SalinasA_gt.mat'),
    ],
    'pavia_centre': [
        urljoin(GIC_URL, 'e/e3/Pavia.mat'),
        urljoin(GIC_URL, '5/53/Pavia_gt.mat'),
    ],
    'pavia_university': [
        urljoin(GIC_URL, 'e/ee/PaviaU.mat'),
        urljoin(GIC_URL, '5/50/PaviaU_gt.mat'),
    ],
    'kennedy_space_center': [
        urljoin(GIC_URL, '2/26/KSC.mat'),
        urljoin(GIC_URL, 'a/a6/KSC_gt.mat'),
    ],
    'botswana': [
        urljoin(GIC_URL, '7/72/Botswana.mat'),
        urljoin(GIC_URL, '5/58/Botswana_gt.mat'),
    ],
}


def img_array_to_pandas(X, y):
    """Converts an image numpy array (with ground truth) to a pandas dataframe"""
    shp = X.shape
    columns = [i for i in range(shp[-1])] + ['target']
    dat = np.concatenate(
        [np.moveaxis(X, -1, 0), np.moveaxis(y, -1, 0)], axis=0
    ).reshape((len(columns), shp[0] * shp[1]))
    return pd.DataFrame(data=dat.T, columns=columns)


class RemoteSensingDatasets(BaseDatasets):
    """Class to download, transform and save remote sensing datasets."""

    def __init__(self, names='all', return_coords=False):
        self.names = names
        self.return_coords = return_coords

    def _load_gic_dataset(self, dataset_name):
        for url in FETCH_URLS[dataset_name]:
            r = requests.get(url, stream=True)
            content = loadmat(io.BytesIO(r.content))
            arr = np.array(list(content.values())[-1])
            arr = np.expand_dims(arr, -1) if arr.ndim == 2 else arr
            if self.return_coords and arr.shape[-1] != 1:
                indices = np.moveaxis(np.indices(arr.shape[:-1]), 0, -1)
                arr = np.insert(arr, [0, 0], indices, -1)
            yield arr

    def fetch_indian_pines(self):
        """Download and transform the Indian Pines Data Set. Label "0" means the pixel is
        not labelled. It is therefore dropped.

        http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Indian_Pines
        """
        df = img_array_to_pandas(*self._load_gic_dataset('indian_pines'))
        return df[df.target != 0]

    def fetch_salinas(self):
        """Download and transform the Salinas Data Set. Label "0" means the pixel is
        not labelled. It is therefore dropped.

        http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Salinas_scene
        """
        df = img_array_to_pandas(*self._load_gic_dataset('salinas'))
        return df[df.target != 0]

    def fetch_salinas_a(self):
        """Download and transform the Salinas-A Data Set. Label "0" means the pixel is
        not labelled. It is therefore dropped.

        http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Salinas-A_scene
        """
        df = img_array_to_pandas(*self._load_gic_dataset('salinas_a'))
        return df[df.target != 0]

    def fetch_pavia_centre(self):
        """Download and transform the Pavia Centre Data Set. Label "0" means the pixel is
        not labelled. It is therefore dropped.

        http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_Centre_scene
        """
        df = img_array_to_pandas(*self._load_gic_dataset('pavia_centre'))
        return df[df.target != 0]

    def fetch_pavia_university(self):
        """Download and transform the Pavia University Data Set. Label "0" means the pixel is
        not labelled. It is therefore dropped.

        http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_University_scene
        """
        df = img_array_to_pandas(*self._load_gic_dataset('pavia_university'))
        return df[df.target != 0]

    def fetch_kennedy_space_center(self):
        """Download and transform the Kennedy Space Center Data Set. Label "0" means the pixel is
        not labelled. It is therefore dropped.

        http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Kennedy_Space_Center_.28KSC.29
        """
        df = img_array_to_pandas(*self._load_gic_dataset('kennedy_space_center'))
        return df[df.target != 0]

    def fetch_botswana(self):
        """Download and transform the Botswana Data Set. Label "0" means the pixel is
        not labelled. It is therefore dropped.

        http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Botswana
        """
        df = img_array_to_pandas(*self._load_gic_dataset('botswana'))
        return df[df.target != 0]
