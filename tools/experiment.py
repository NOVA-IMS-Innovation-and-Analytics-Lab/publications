"""
Configure and run the experimental procedure.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from os.path import join, exists
from sqlite3 import connect

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from imblearn.pipeline import Pipeline
from sklearnext.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN, GeometricSMOTE, DensityDistributor
from sklearnext.cluster import KMeans, SOM

from . import DATA_PATH


def load_datasets(db_name, datasets_names):
    """Load datasets from sqlite database."""

    path = join(DATA_PATH, f'{db_name}.db')
    if not exists(path):
        raise FileNotFoundError(f'Database {db_name} was not found.')

    with connect(path) as connection:

        if datasets_names == 'all':
            datasets_names = [name[0] for name in connection.execute("SELECT name FROM sqlite_master WHERE type='table';")]

        datasets = []

        for dataset_name in datasets_names:
            ds = pd.read_sql(f'select * from "{dataset_name}"', connection)
            X, y = ds.iloc[:, :-1], ds.iloc[:, -1]
            datasets.append((dataset_name, (X, y)))
    
    return datasets


def generate_configuration(db_name, datasets_names='all', classifiers_names='all', oversamplers_names='all', 
                           scoring='imbalanced', n_splits=5, n_runs=3, random_state=0):
    """Generate configuration dictionary for an experiment."""

    # Define classifiers and oversamplers
    classifiers = [
    ('LR', LogisticRegression(solver='lbfgs', max_iter=1e4, multi_class='auto'), {}),
    ('KNN', KNeighborsClassifier(), {'n_neighbors': [3, 5]}),
    ('DT', DecisionTreeClassifier(), {'max_depth': [3, 6]}),
    ('GBC', GradientBoostingClassifier(), {'max_depth': [3, 6], 'n_estimators': [50, 100]})
    ]
    oversamplers = [
        ('NO OVERSAMPLING', None),
        ('RANDOM OVERSAMPLING', RandomOverSampler()),
        ('SMOTE', SMOTE(), {'k_neighbors': [3, 5]}),
        ('BORDERLINE SMOTE', BorderlineSMOTE(), {'k_neighbors': [3, 5]}),
        ('ADASYN', ADASYN(), {'n_neighbors': [2, 3]}),
        ('G-SMOTE', GeometricSMOTE(), {
            'k_neighbors': [3, 5], 
            'selection_strategy': ['combined', 'minority', 'majority'], 
            'truncation_factor': [-1.0, -0.5, .0, 0.25, 0.5, 0.75, 1.0], 
            'deformation_factor': [.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
            }
        ),
        ('K-MEANS RANDOM OVERSAMPLING', RandomOverSampler(clusterer=KMeans(), distributor=DensityDistributor()), {
            'k_neighbors': [3, 5],
            'clusterer__n_clusters': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'distributor__distances_exponent': [0, 1, 2, 5],
            'distributor__filtering_threshold': [0.0, 0.5, 1.0, 2.0]
            }
        ),
        ('K-MEANS SMOTE', SMOTE(clusterer=KMeans(), distributor=DensityDistributor()), {
            'k_neighbors': [3, 5],
            'clusterer__n_clusters': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'distributor__distances_exponent': [0, 1, 2, 5],
            'distributor__filtering_threshold': [0.0, 0.5, 1.0, 2.0]
            }
        ),
        ('K-MEANS BORDERLINE SMOTE', BorderlineSMOTE(clusterer=KMeans(), distributor=DensityDistributor()), {
            'k_neighbors': [3, 5],
            'clusterer__n_clusters': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'distributor__distances_exponent': [0, 1, 2, 5],
            'distributor__filtering_threshold': [0.0, 0.5, 1.0, 2.0]
            }
        ),
        ('K-MEANS G-SMOTE', GeometricSMOTE(clusterer=KMeans(), distributor=DensityDistributor()), {
            'k_neighbors': [3, 5],
            'selection_strategy': ['combined', 'minority', 'majority'],
            'truncation_factor': [-1.0, -0.5, .0, 0.25, 0.5, 0.75, 1.0], 
            'deformation_factor': [.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
            'clusterer__n_clusters': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'distributor__distances_exponent': [0, 1, 2, 5],
            'distributor__filtering_threshold': [0.0, 0.5, 1.0, 2.0]
            }
        ),
        ('SOMO', SMOTE(clusterer=SOM(), distributor=DensityDistributor()), {
            'k_neighbors': [3, 5],
            'clusterer__n_clusters': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'distributor__distances_exponent': [0, 1, 2, 5],
            'distributor__filtering_threshold': [0.0, 0.5, 1.0, 2.0],
            'distributor__distribution_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]
            }
        ),
        ('G-SOMO', GeometricSMOTE(clusterer=SOM(), distributor=DensityDistributor()), {
            'k_neighbors': [3, 5],
            'selection_strategy': ['combined', 'minority', 'majority'],
            'truncation_factor': [-1.0, -0.5, .0, 0.25, 0.5, 0.75, 1.0], 
            'deformation_factor': [.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
            'clusterer__n_clusters': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'distributor__distances_exponent': [0, 1, 2, 5],
            'distributor__filtering_threshold': [0.0, 0.5, 1.0, 2.0],
            'distributor__distribution_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]
            }
        )
    ]
    if scoring == 'imbalanced':
        scoring = ['roc_auc', 'f1', 'geometric_mean_score']
    n_splits = 5
    n_runs = 3
    random_state = 0

    # Select datasets
    datasets = load_datasets(db_name, datasets_names)

    # Select classifiers and oversamplers
    if classifiers == 'scaled':
        classifiers = [(
            name, Pipeline([ ('scaler', MinMaxScaler()), ('clf', clf) ]), 
            {f'clf__{param}':val for param, val in param_grid.items()}) for name, clf, param_grid in classifiers]
    elif classifiers_names != 'all':
        classifiers = [(name, clf) for name, clf in classifiers if name in classifiers_names]
    
    if oversamplers_names != 'all':
        if oversamplers_names == 'basic':
            oversamplers_names = ('NO OVERSAMPLING', 'RANDOM OVERSAMPLING', 'SMOTE', 'BORDERLINE SMOTE', 'ADASYN', 'G-SMOTE')
        oversamplers = [(name, oversampler) for name, oversampler in oversamplers if name in oversamplers_names]
    
    return dict(datasets=datasets, classifiers=classifiers, oversamplers=oversamplers, scoring=scoring, n_splits=n_splits, n_runs=n_runs, random_state=random_state)


CONFIG = {
    'no_oversampling_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_names=['NO OVERSAMPLING']),
    'random_oversampling_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_names=['RANDOM OVERSAMPLING']),
    'smote_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_names=['SMOTE']),
    'borderline_smote_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_names=['BORDERLINE SMOTE']),
    'adasyn_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_names=['ADASYN']),
    'gsmote_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_names=['G-SMOTE']),
    'kmeans_ros_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_names=['NO OVERSAMPLING']),
    'kmeans_smote_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_names=['NO OVERSAMPLING']),
    'kmeans_borderline_smote_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_names=['NO OVERSAMPLING']),
    'kmeans_gsmote_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_names=['NO OVERSAMPLING']),
    'somo_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_names=['NO OVERSAMPLING']),
    'gsomo_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_names=['NO OVERSAMPLING']),
    'lucas': generate_configuration('remote_sensing', datasets_names=['lucas'], classifiers_names=['KNN' , 'DT', 'GBC'], oversamplers_names='basic', scoring=['f1_macro'], n_splits=3),
    'insurance': generate_configuration('various', datasets_names=['insurance'], classifiers_names='scaled', oversamplers_names='basic'),
    'small_data': generate_configuration('binary_class', oversamplers_names='basic', scoring=['accuracy']),
}
