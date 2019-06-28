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
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearnext.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN, GeometricSMOTE, DensityDistributor
from sklearnext.cluster import KMeans, SOM
from sklearnext.over_sampling.base import BaseClusterOverSampler


class UnderOverSampler(BaseClusterOverSampler):
    """A class that applies random undersampling and oversampling."""

    def __init__(self,
                 random_state=None,
                 oversampler=None,
                 factor=3):
        super(UnderOverSampler, self).__init__(sampling_strategy='auto', clusterer=None, distributor=None)
        self.random_state = random_state
        self.oversampler = oversampler
        self.factor = factor
    
    def fit(self, X, y):
        self._deprecate_ratio()
        X, y, _ = self._check_X_y(X, y)
        counts = Counter(y)

        # Overwrite default sampling strategy
        self.sampling_strategy_ = OrderedDict([(list(counts.keys())[0], 0)])

        # Create undersampler, oversampler and pipeline
        self.undersampler_ = RandomUnderSampler(random_state=self.random_state, sampling_strategy={k:int(v / self.factor) for k,v in counts.items()})
        self.oversampler_ = clone(self.oversampler).set_params(random_state=self.random_state, sampling_strategy=dict(counts))
        self.pipeline_ = make_pipeline(self.undersampler_, self.oversampler_)
        
        return self

    def _basic_sample(self, X, y):
        X_resampled, y_resampled = self.pipeline_.fit_resample(X, y)
        return X_resampled, y_resampled


def check_estimators(category, names, mapping):
    """Check estimators."""
    if category not in mapping:
        raise ValueError(f'Parameter `category` should be one of {list(mapping.keys())}.')
    all_names, *_ = zip(*mapping[category])
    if names is None:
        estimators = mapping[category]
    elif set(names).issubset(all_names):
        estimators = [(name, est, param_grid) for name, est, param_grid in mapping[category] if name in names]
    else:
        raise ValueError(f'Parameter names should be a subset of {all_names}.')
    return estimators


CLASSIFIERS_BASIC = [
    ('LR', LogisticRegression(solver='lbfgs', max_iter=1e4, multi_class='auto'), {}),
    ('KNN', KNeighborsClassifier(), {'n_neighbors': [3, 5]}),
    ('DT', DecisionTreeClassifier(), {'max_depth': [3, 6]}),
    ('GBC', GradientBoostingClassifier(), {'max_depth': [3, 6], 'n_estimators': [50, 100]})
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
OVERSAMPLERS_UNDERSAMPLED = [
    ('BENCHMARK METHOD', None, {}),
    ('NO OVERSAMPLING', RandomUnderSampler(), {}),
    ('RANDOM OVERSAMPLING', UnderOverSampler(oversampler=RandomOverSampler()), {}),
    ('SMOTE', UnderOverSampler(oversampler=SMOTE()), {'oversampler__k_neighbors': [3, 5]}),
    ('BORDERLINE SMOTE', UnderOverSampler(oversampler=BorderlineSMOTE()), {'oversampler__k_neighbors': [3, 5]}),
    ('ADASYN', UnderOverSampler(oversampler=ADASYN()), {'oversampler__n_neighbors': [2, 3]}),
    ('G-SMOTE', UnderOverSampler(oversampler=GeometricSMOTE()), {
        'oversampler__k_neighbors': [3, 5], 
        'oversampler__selection_strategy': ['combined', 'minority', 'majority'], 
        'oversampler__truncation_factor': [-1.0, -0.5, .0, 0.25, 0.5, 0.75, 1.0], 
        'oversampler__deformation_factor': [.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
        }
    )
]
CLASSIFIERS_MAPPING = {
    'basic': CLASSIFIERS_BASIC
}
OVERSAMPLERS_MAPPING = {
    'basic': OVERSAMPLERS_BASIC,
    'clustering': OVERSAMPLERS_CLUSTERING,
    'undersampled': OVERSAMPLERS_UNDERSAMPLED
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
    
    # All imbalanced datasets
    'no_oversampling_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_names=['NO OVERSAMPLING']),
    'random_oversampling_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_names=['RANDOM OVERSAMPLING']),
    'smote_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_names=['SMOTE']),
    'borderline_smote_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_names=['BORDERLINE SMOTE']),
    'adasyn_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_names=['ADASYN']),
    'gsmote_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_names=['G-SMOTE']),
    'kmeans_ros_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_category='clustering', oversamplers_names=['K-MEANS RANDOM OVERSAMPLING']),
    'kmeans_smote_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_category='clustering', oversamplers_names=['K-MEANS SMOTE']),
    'kmeans_borderline_smote_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_category='clustering', oversamplers_names=['K-MEANS BORDERLINE SMOTE']),
    'kmeans_gsmote_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_category='clustering', oversamplers_names=['K-MEANS G-SMOTE']),
    'somo_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_category='clustering', oversamplers_names=['SOMO']),
    'gsomo_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_category='clustering', oversamplers_names=['G-SOMO']),
    
    # Remote sensing lucas data
    'no_oversampling_lucas': generate_configuration('remote_sensing',  datasets_names=['lucas'], classifiers_names=['KNN' , 'DT', 'GBC'], oversamplers_names=['NO OVERSAMPLING'], scoring=['f1_macro'], n_splits=3),
    'random_oversampling_lucas': generate_configuration('remote_sensing',  datasets_names=['lucas'], classifiers_names=['KNN' , 'DT', 'GBC'], oversamplers_names=['RANDOM OVERSAMPLING'], scoring=['f1_macro'], n_splits=3),
    'smote_lucas': generate_configuration('remote_sensing',  datasets_names=['lucas'], classifiers_names=['KNN' , 'DT', 'GBC'], oversamplers_names=['SMOTE'], scoring=['f1_macro'], n_splits=3),
    'borderline_smote_lucas': generate_configuration('remote_sensing',  datasets_names=['lucas'], classifiers_names=['KNN' , 'DT', 'GBC'], oversamplers_names=['BORDERLINE SMOTE'], scoring=['f1_macro'], n_splits=3),
    'adasyn_lucas': generate_configuration('remote_sensing',  datasets_names=['lucas'], classifiers_names=['KNN' , 'DT', 'GBC'], oversamplers_names=['ADASYN'], scoring=['f1_macro'], n_splits=3),
    'gsmote_lucas': generate_configuration('remote_sensing',  datasets_names=['lucas'], classifiers_names=['KNN' , 'DT', 'GBC'], oversamplers_names=['G-SMOTE'], scoring=['f1_macro'], n_splits=3),
    
    # Insurance data
    'no_oversampling_insurance': generate_configuration('various', datasets_names=['insurance'], oversamplers_names=['NO OVERSAMPLING']),
    'random_oversampling_insurance': generate_configuration('various', datasets_names=['insurance'], oversamplers_names=['RANDOM OVERSAMPLING']),
    'smote_insurance': generate_configuration('various', datasets_names=['insurance'], oversamplers_names=['SMOTE']),
    'borderline_smote_insurance': generate_configuration('various', datasets_names=['insurance'], oversamplers_names=['BORDERLINE SMOTE']),
    'adasyn_insurance': generate_configuration('various', datasets_names=['insurance'], oversamplers_names=['ADASYN']),
    'gsmote_insurance': generate_configuration('various', datasets_names=['insurance'], oversamplers_names=['G-SMOTE']),
    
    # Small data oversampling
    'benchmark_method_small_data': generate_configuration('binary_class', oversamplers_category='undersampled', oversamplers_names=['BENCHMARK METHOD'], scoring=['accuracy']),
    'no_oversampling_small_data': generate_configuration('binary_class', oversamplers_category='undersampled', oversamplers_names=['NO OVERSAMPLING'], scoring=['accuracy']),
    'random_oversampling_small_data': generate_configuration('binary_class', oversamplers_category='undersampled', oversamplers_names=['RANDOM OVERSAMPLING'], scoring=['accuracy']),
    'smote_small_data': generate_configuration('binary_class', oversamplers_category='undersampled', oversamplers_names=['SMOTE'], scoring=['accuracy']),
    'borderline_smote_method_small_data': generate_configuration('binary_class', oversamplers_category='undersampled', oversamplers_names=['BORDERLINE SMOTE'], scoring=['accuracy']),
    'adasyn_small_data': generate_configuration('binary_class', oversamplers_category='undersampled', oversamplers_names=['ADASYN'], scoring=['accuracy']),
    'gsmote_method_small_data': generate_configuration('binary_class', oversamplers_category='undersampled', oversamplers_names=['G-SMOTE'], scoring=['accuracy'])

}
