#!usr/bin/env python

"""
Configure and run the experimental procedure.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from os.path import join, dirname, exists, basename, abspath
from sqlite3 import connect
from argparse import ArgumentParser

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from imblearn.pipeline import Pipeline
from sklearnext.tools import BinaryExperiment
from sklearnext.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN, GeometricSMOTE, DensityDistributor
from sklearnext.cluster import KMeans, SOM

CLASSIFIERS = [
    ('LR', LogisticRegression(solver='lbfgs', max_iter=1e4, multi_class='auto'), {}),
    ('KNN', KNeighborsClassifier(), {'n_neighbors': [3, 5]}),
    ('DT', DecisionTreeClassifier(), {'max_depth': [3, 6]}),
    ('GBC', GradientBoostingClassifier(), {'max_depth': [3, 6], 'n_estimators': [50, 100]})
]
OVERSAMPLERS = [
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
CONFIG1 = {
    'datasets': 'all',
    'classifiers': CLASSIFIERS,
    'scoring': ['roc_auc', 'f1', 'geometric_mean_score'],
    'n_splits': 5,
    'n_runs': 3,
    'random_state': 0
}
CONFIG2 = {
    'datasets': ['lucas'],
    'classifiers': CLASSIFIERS[1:], 
    'oversamplers': OVERSAMPLERS[0:6],
    'scoring': ['f1_macro'],
    'n_splits': 3,
    'n_runs': 3,
    'random_state': 0
}
CONFIG3 = {
    'datasets': ['insurance'],
    'classifiers': [
        (
            clf_name, 
            Pipeline([ ('scaler', MinMaxScaler()), ('clf', clf) ]), 
            {f'clf__{param}':val for param, val in param_grid.items()}
        ) for clf_name, clf, param_grid in CLASSIFIERS
    ],
    'oversamplers': OVERSAMPLERS[0:6],
    'scoring': ['roc_auc', 'f1', 'geometric_mean_score'],
    'n_splits': 5,
    'n_runs': 3,
    'random_state': 0
},
CONFIG4 = {
    'datasets': 'all',
    'classifiers': CLASSIFIERS,
    'scoring': ['roc_auc', 'f1', 'geometric_mean_score'],
    'n_splits': 5,
    'n_runs': 3,
    'random_state': 0
}
CONFIGURATIONS = {
    ('imbalanced_binary_class', 'no_oversampling'): dict(IMBALANCED_BINARY_CLASS, **{'oversamplers': [OVERSAMPLERS[0]]}),
    ('imbalanced_binary_class', 'random_oversampling'): dict(IMBALANCED_BINARY_CLASS, **{'oversamplers': [OVERSAMPLERS[1]]}),
    ('imbalanced_binary_class', 'smote'): dict(IMBALANCED_BINARY_CLASS, **{'oversamplers': [OVERSAMPLERS[2]]}),
    ('imbalanced_binary_class', 'borderline_smote'): dict(IMBALANCED_BINARY_CLASS, **{'oversamplers': [OVERSAMPLERS[3]]}),
    ('imbalanced_binary_class', 'adasyn'): dict(IMBALANCED_BINARY_CLASS, **{'oversamplers': [OVERSAMPLERS[4]]}),
    ('imbalanced_binary_class', 'gsmote'): dict(IMBALANCED_BINARY_CLASS, **{'oversamplers': [OVERSAMPLERS[5]]}),
    ('imbalanced_binary_class', 'kmeans_ros'): dict(IMBALANCED_BINARY_CLASS, **{'oversamplers': [OVERSAMPLERS[6]]}),
    ('imbalanced_binary_class', 'kmeans_smote'): dict(IMBALANCED_BINARY_CLASS, **{'oversamplers': [OVERSAMPLERS[7]]}),
    ('imbalanced_binary_class', 'kmeans_borderline_smote'): dict(IMBALANCED_BINARY_CLASS, **{'oversamplers': [OVERSAMPLERS[8]]}),
    ('imbalanced_binary_class', 'kmeans_gsmote'): dict(IMBALANCED_BINARY_CLASS, **{'oversamplers': [OVERSAMPLERS[9]]}),
    ('imbalanced_binary_class', 'somo'): dict(IMBALANCED_BINARY_CLASS, **{'oversamplers': [OVERSAMPLERS[10]]}),
    ('imbalanced_binary_class', 'gsomo'): dict(IMBALANCED_BINARY_CLASS, **{'oversamplers': [OVERSAMPLERS[11]]}),
    ('remote_sensing', 'lucas'): LUCAS,
    ('various', 'insurance'): INSURANCE,
    ('binary_class', 'small_data'): SMALL_DATA
}


def load_datasets(path, tbls_names):
    """Load datasets from a sqlite database."""

    if not exists(path):
        raise FileNotFoundError(f'Database {basename(path)} was not found.')

    with connect(path) as connection:

        if tbls_names == 'all':
            tbls_names = [name[0] for name in connection.execute("SELECT name FROM sqlite_master WHERE type='table';")]

        datasets = []

        for tbl_name in tbls_names:
            tbl = pd.read_sql(f'select * from "{tbl_name}"', connection)
            X, y = tbl.iloc[:, :-1], tbl.iloc[:, -1]
            datasets.append((tbl_name, (X, y)))
    
    return datasets


def check_calculate_results(calculate_results):
    """Validate input for calculate results argument"""
    if calculate_results.upper() in ('Y', 'YES', 'TRUE'):
        return True
    elif calculate_results.upper() in ('N', 'NO', 'FALSE'):
        return False
    else:
        raise ValueError('Argument `calculate_results` is not correct.')


def generate_configuration(experiment_name, db_name, datasets_names='all', classifiers='all', oversamplers='basic', 
                           scoring='binary', n_splits=5, n_runs=5, random_state=0):
    """Generate configration for experiment."""
    configuration = {(experiment_name, db_name): {}}


def parse_arguments():
    """Parse command-line arguments."""
    parser = ArgumentParser(f'Run an experiment.')
    parser.add_argument('experiment', help='The name of the experiment.')
    return parser.parse_args()


if __name__ == '__main__':

    # Parse arguments
    name, datasets_path, experiment_path, calculate_results, n_jobs, verbose = parse_arguments()

    # Get experiment's configuration
    configuration = CONFIGURATIONS[name]

    # Download datasets
    db_name, tbls_names = configuration['datasets']
    datasets = load_datasets(join(datasets_path, db_name), tbls_names)
    
    # Run experiment and save object
    experiment = BinaryExperiment(name, datasets, configuration['oversamplers'], configuration['classifiers'], configuration['scoring'], 
                                  configuration['n_splits'], configuration['n_runs'], configuration['random_state'])
    experiment.run(n_jobs=n_jobs, verbose=verbose)
    if check_calculate_results(calculate_results):
        experiment.calculate_results()
    experiment.dump(experiment_path)
