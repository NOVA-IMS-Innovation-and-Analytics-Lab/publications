"""
Define configurations of experiments.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from imblearn.pipeline import Pipeline
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
IMBALANCED_CONFIGURATION = {
    'db_name': 'imbalanced_data.db',
    'classifiers': CLASSIFIERS,
    'scoring': ['roc_auc', 'f1', 'geometric_mean_score'],
    'n_splits': 5,
    'n_runs': 3,
    'random_state': 0
}
REMOTE_SENSING_CONFIGURATION = {
    'db_name': 'remote_sensing_data.db',
    'classifiers': CLASSIFIERS[1:], 
    'oversamplers': OVERSAMPLERS[0:6],
    'scoring': ['f1_macro'],
    'n_splits': 3,
    'n_runs': 3,
    'random_state': 0
}
INSURANCE_CONFIGURATION = {
    'db_name': 'insurance_data.db',
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
}
CONFIGURATIONS = {
    'no_oversampling': dict(IMBALANCED_CONFIGURATION, **{'oversamplers': [OVERSAMPLERS[0]]}),
    'random_oversampling': dict(IMBALANCED_CONFIGURATION, **{'oversamplers': [OVERSAMPLERS[1]]}),
    'smote': dict(IMBALANCED_CONFIGURATION, **{'oversamplers': [OVERSAMPLERS[2]]}),
    'borderline_smote': dict(IMBALANCED_CONFIGURATION, **{'oversamplers': [OVERSAMPLERS[3]]}),
    'adasyn': dict(IMBALANCED_CONFIGURATION, **{'oversamplers': [OVERSAMPLERS[4]]}),
    'gsmote': dict(IMBALANCED_CONFIGURATION, **{'oversamplers': [OVERSAMPLERS[5]]}),
    'kmeans_ros': dict(IMBALANCED_CONFIGURATION, **{'oversamplers': [OVERSAMPLERS[6]]}),
    'kmeans_smote': dict(IMBALANCED_CONFIGURATION, **{'oversamplers': [OVERSAMPLERS[7]]}),
    'kmeans_borderline_smote': dict(IMBALANCED_CONFIGURATION, **{'oversamplers': [OVERSAMPLERS[8]]}),
    'kmeans_gsmote': dict(IMBALANCED_CONFIGURATION, **{'oversamplers': [OVERSAMPLERS[9]]}),
    'somo': dict(IMBALANCED_CONFIGURATION, **{'oversamplers': [OVERSAMPLERS[10]]}),
    'gsomo': dict(IMBALANCED_CONFIGURATION, **{'oversamplers': [OVERSAMPLERS[11]]}),
    'remote_sensing': REMOTE_SENSING_CONFIGURATION,
    'insurance': INSURANCE_CONFIGURATION
}






