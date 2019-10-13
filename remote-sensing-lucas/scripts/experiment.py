"""
Configure and run the experimental procedure.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import Counter, OrderedDict

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.cluster import KMeans
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling.base import BaseOverSampler
from gsmote import GeometricSMOTE
from clover.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from clover.distribution import DensityDistributor
from somlearn import SOM


CLASSIFIERS_BASIC = [
    ('CONSTANT CLASSIFIER', DummyClassifier(strategy='constant', constant=0), {}),
    ('RANDOM CLASSIFIER', DummyClassifier(strategy='stratified'), {}),
    ('LR', LogisticRegression(solver='liblinear', multi_class='auto'), {}),
    ('KNN', KNeighborsClassifier(), {'n_neighbors': [3, 5]}),
    ('DT', DecisionTreeClassifier(), {'max_depth': [3, 6]}),
    ('GBC', GradientBoostingClassifier(), {'max_depth': [3, 6], 'n_estimators': [50, 100]}),
    ('RF', RandomForestClassifier(), {'max_depth': [None, 3, 6], 'n_estimators': [10, 50, 100]})
]
OVERSAMPLERS_BASIC = [
    ('NO OVERSAMPLING', None, {}),
    ('RANDOM OVERSAMPLING', RandomOverSampler(), {}),
    ('SMOTE', SMOTE(), {'k_neighbors': [3, 5]}),
    ('BORDERLINE SMOTE', BorderlineSMOTE(), {'k_neighbors': [3, 5]}),
    ('ADASYN', ADASYN(), {'n_neighbors': [2, 3]}),
    ('G-SMOTE', GeometricSMOTE(), {
        'k_neighbors': [3, 5],
        'selection_strategy': ['combined', 'minority', 'majority'],
        'truncation_factor': [-1.0, -0.5, .0, 0.25, 0.5, 0.75, 1.0],
        'deformation_factor': [.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
        }
    )
]
OVERSAMPLERS_CLUSTERING = [
    ('K-MEANS RANDOM OVERSAMPLING', RandomOverSampler(clusterer=KMeans(), distributor=DensityDistributor()), {
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
    ('SOMO', SMOTE(clusterer=SOM(), distributor=DensityDistributor()), {
        'k_neighbors': [3, 5],
        'clusterer__n_clusters': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'distributor__distances_exponent': [0, 1, 2, 5],
        'distributor__filtering_threshold': [0.0, 0.5, 1.0, 2.0],
        'distributor__distribution_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]
        }
    )
]
CLASSIFIERS_MAPPING = {
    'basic': CLASSIFIERS_BASIC

}
OVERSAMPLERS_MAPPING = {
    'basic': OVERSAMPLERS_BASIC,
    'clustering': OVERSAMPLERS_CLUSTERING,
}


def generate_configuration(db_name, datasets_names='all',
                           classifiers_category='basic', classifiers_names=None,
                           oversamplers_category='basic', oversamplers_names=None,
                           scoring=['roc_auc', 'f1', 'geometric_mean_score'], n_splits=5,
                           n_runs=3, random_state=0):
    """Generate configuration dictionary for an experiment."""
    classifiers = check_estimators(classifiers_category, classifiers_names, CLASSIFIERS_MAPPING)
    oversamplers = check_estimators(oversamplers_category, oversamplers_names, OVERSAMPLERS_MAPPING)
    return dict(db_name=db_name, datasets_names=datasets_names, classifiers=classifiers, oversamplers=oversamplers, scoring=scoring, n_splits=n_splits, n_runs=n_runs, random_state=random_state)


CONFIG = {
    'imbalanced': {
        'no_oversampling': generate_configuration('imbalanced_binary_class', classifiers_names=['LR', 'KNN' , 'DT', 'GBC'], oversamplers_names=['NO OVERSAMPLING']),
        'random_oversampling': generate_configuration('imbalanced_binary_class', classifiers_names=['LR', 'KNN' , 'DT', 'GBC'], oversamplers_names=['RANDOM OVERSAMPLING']),
        'smote': generate_configuration('imbalanced_binary_class', classifiers_names=['LR', 'KNN' , 'DT', 'GBC'], oversamplers_names=['SMOTE']),
        'borderline_smote': generate_configuration('imbalanced_binary_class', classifiers_names=['LR', 'KNN' , 'DT', 'GBC'], oversamplers_names=['BORDERLINE SMOTE']),
        'gsmote': generate_configuration('imbalanced_binary_class', classifiers_names=['LR', 'KNN' , 'DT', 'GBC'], oversamplers_names=['G-SMOTE']),
        'kmeans_ros': generate_configuration('imbalanced_binary_class', classifiers_names=['LR', 'KNN' , 'DT', 'GBC'], oversamplers_category='clustering', oversamplers_names=['K-MEANS RANDOM OVERSAMPLING']),
        'kmeans_smote': generate_configuration('imbalanced_binary_class', classifiers_names=['LR', 'KNN' , 'DT', 'GBC'], oversamplers_category='clustering', oversamplers_names=['K-MEANS SMOTE']),
        'kmeans_borderline_smote': generate_configuration('imbalanced_binary_class', classifiers_names=['LR', 'KNN' , 'DT', 'GBC'], oversamplers_category='clustering', oversamplers_names=['K-MEANS BORDERLINE SMOTE']),
        'somo': generate_configuration('imbalanced_binary_class', classifiers_names=['LR', 'KNN' , 'DT', 'GBC'], oversamplers_category='clustering', oversamplers_names=['SOMO']),
    },
    'lucas': {
        'no_oversampling': generate_configuration('remote_sensing',  datasets_names=['lucas'], oversamplers_names=['NO OVERSAMPLING'], scoring=['geometric_mean_macro_score', 'f1_macro', 'accuracy'], n_splits=3),
        'random_oversampling': generate_configuration('remote_sensing',  datasets_names=['lucas'], oversamplers_names=['RANDOM OVERSAMPLING'], scoring=['geometric_mean_macro_score', 'f1_macro', 'accuracy'], n_splits=3),
        'smote': generate_configuration('remote_sensing',  datasets_names=['lucas'], oversamplers_names=['SMOTE'], scoring=['geometric_mean_macro_score', 'f1_macro', 'accuracy'], n_splits=3),
        'borderline_smote': generate_configuration('remote_sensing',  datasets_names=['lucas'], oversamplers_names=['BORDERLINE SMOTE'], scoring=['geometric_mean_macro_score', 'f1_macro', 'accuracy'], n_splits=3),
        'adasyn': generate_configuration('remote_sensing',  datasets_names=['lucas'], oversamplers_names=['ADASYN'], scoring=['geometric_mean_macro_score', 'f1_macro', 'accuracy'], n_splits=3),
        'gsmote': generate_configuration('remote_sensing',  datasets_names=['lucas'], oversamplers_names=['G-SMOTE'], scoring=['geometric_mean_macro_score', 'f1_macro', 'accuracy'], n_splits=3)
    },
    'insurance': {
        'no_oversampling': generate_configuration('various', datasets_names=['insurance'], classifiers_names=['LR', 'KNN' , 'DT', 'GBC'], oversamplers_names=['NO OVERSAMPLING']),
        'random_oversampling': generate_configuration('various', datasets_names=['insurance'], classifiers_names=['LR', 'KNN' , 'DT', 'GBC'], oversamplers_names=['RANDOM OVERSAMPLING']),
        'smote': generate_configuration('various', datasets_names=['insurance'], classifiers_names=['LR', 'KNN' , 'DT', 'GBC'], oversamplers_names=['SMOTE']),
        'borderline_smote': generate_configuration('various', datasets_names=['insurance'], classifiers_names=['LR', 'KNN' , 'DT', 'GBC'], oversamplers_names=['BORDERLINE SMOTE']),
        'adasyn': generate_configuration('various', datasets_names=['insurance'], classifiers_names=['LR', 'KNN' , 'DT', 'GBC'], oversamplers_names=['ADASYN']),
        'gsmote': generate_configuration('various', datasets_names=['insurance'], classifiers_names=['LR', 'KNN' , 'DT', 'GBC'], oversamplers_names=['G-SMOTE'])
    }
}
