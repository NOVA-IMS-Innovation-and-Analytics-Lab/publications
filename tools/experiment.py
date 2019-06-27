"""
Configure and run the experimental procedure.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import Counter

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearnext.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN, GeometricSMOTE, DensityDistributor
from sklearnext.cluster import KMeans, SOM
from sklearnext.over_sampling.base import BaseClusterOverSampler


class UnderOverSampler(BaseClusterOverSampler):
    """A class that applies random undersampling and oversampling."""

    def __init__(self,
                 sampling_strategy='auto',
                 clusterer=None,
                 distributor=None,
                 random_state=None,
                 oversampler=None,
                 factor=3):
        super(UnderOverSampler, self).__init__(sampling_strategy=sampling_strategy, clusterer=clusterer, distributor=distributor)
        self.random_state = random_state
        self.oversampler = oversampler
        self.factor = factor

    @staticmethod
    def _generate_sampling_strategy(y, factor):
        """"Generate a dictionary of the sampling strategy.""" 
        sampling_strategy = {k:int(v / factor) for k,v in Counter(y).items()}
        return sampling_strategy

    def _basic_sample(self, X, y):
        oversampler = clone(self.oversampler)
        pipeline = make_pipeline(
            RandomUnderSampler(random_state=self.random_state, sampling_strategy=self._generate_sampling_strategy(y, self.factor)), 
            oversampler.set_params(random_state=self.random_state, sampling_strategy=self._generate_sampling_strategy(y, 1 / self.factor))
        )
        X_resampled, y_resampled = pipeline.fit_resample(X, y)
        return X_resampled, y_resampled


def append_transformer(transformer, oversamplers):
    """Append transformer to oversamplers."""
    names, ovs, param_grids = zip(*oversamplers)
    ovs_names = [ov.__class__.__name__.lower() if ov is not None else None for ov in ovs]
    param_grids = [{f'{ovs_name}__{param}': values for param, values in param_grid.items()} for ovs_name, param_grid in zip(ovs_names, param_grids)]
    oversamplers = [(name, (make_pipeline(transformer, ov) if ov is not None else ov), param_grid) for name, ov, param_grid in zip(names, ovs, param_grids)]
    return oversamplers


def set_sampling_strategy(value, oversamplers):
    """Set sampling strategy to oversamplers."""
    oversamplers = [(name, ov.set_params(sampling_strategy=value) if ov is not None else None, param_grid) for name, ov, param_grid in oversamplers]
    return oversamplers


def select_pipelines(pipelines, names):
    """Filter pipelines given a list of names"""
    return [(name, pipeline, param_grid) for name, pipeline, param_grid in pipelines if name in names]


def generate_classifiers(classifiers_names):
    """Generate classifiers."""
    classifiers = [
        ('LR', LogisticRegression(solver='lbfgs', max_iter=1e4, multi_class='auto'), {}),
        ('KNN', KNeighborsClassifier(), {'n_neighbors': [3, 5]}),
        ('DT', DecisionTreeClassifier(), {'max_depth': [3, 6]}),
        ('GBC', GradientBoostingClassifier(), {'max_depth': [3, 6], 'n_estimators': [50, 100]})
    ]
    if classifiers_names != 'all':
        classifiers = select_pipelines(classifiers, classifiers_names)

    return classifiers
    

def generate_oversamplers(oversamplers_names):
    "Generate oversamplers."
    oversamplers = [
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
    if oversamplers_names in ('basic', 'scaled', 'undersampled'):
        oversamplers = select_pipelines(oversamplers, ('NO OVERSAMPLING', 'RANDOM OVERSAMPLING', 'SMOTE', 'BORDERLINE SMOTE', 'ADASYN', 'G-SMOTE'))
    if oversamplers_names == 'scaled':
        oversamplers = append_transformer(MinMaxScaler(), oversamplers)
    elif oversamplers_names == 'undersampled':
        oversamplers = set_sampling_strategy(lambda y: generate_sampling_strategy(y, 1/3), oversamplers)
        oversamplers = append_transformer(RandomUnderSampler(sampling_strategy=lambda y: generate_sampling_strategy(y, 3)), oversamplers)

    return oversamplers


def generate_configuration(db_name, datasets_names='all', classifiers_names='all', oversamplers_names='all', 
                           scoring='imbalanced', n_splits=5, n_runs=3, random_state=0):
    """Generate configuration dictionary for an experiment."""
    if scoring == 'imbalanced':
        scoring = ['roc_auc', 'f1', 'geometric_mean_score']
    n_splits = 5
    n_runs = 3
    random_state = 0
    classifiers = generate_classifiers(classifiers_names)
    oversamplers = generate_oversamplers(oversamplers_names)
    return dict(db_name=db_name, datasets_names=datasets_names, classifiers=classifiers, oversamplers=oversamplers, scoring=scoring, n_splits=n_splits, n_runs=n_runs, random_state=random_state)


CONFIG = {
    'no_oversampling_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_names=['NO OVERSAMPLING']),
    'random_oversampling_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_names=['RANDOM OVERSAMPLING']),
    'smote_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_names=['SMOTE']),
    'borderline_smote_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_names=['BORDERLINE SMOTE']),
    'adasyn_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_names=['ADASYN']),
    'gsmote_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_names=['G-SMOTE']),
    'kmeans_ros_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_names=['K-MEANS RANDOM OVERSAMPLING']),
    'kmeans_smote_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_names=['K-MEANS SMOTE']),
    'kmeans_borderline_smote_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_names=['K-MEANS BORDERLINE SMOTE']),
    'kmeans_gsmote_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_names=['K-MEANS G-SMOTE']),
    'somo_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_names=['SOMO']),
    'gsomo_imbalanced': generate_configuration('imbalanced_binary_class', oversamplers_names=['G-SOMO']),
    'lucas': generate_configuration('remote_sensing', datasets_names=['lucas'], classifiers_names=['KNN' , 'DT', 'GBC'], oversamplers_names='basic', scoring=['f1_macro'], n_splits=3),
    'random_oversampling_insurance': generate_configuration('various', datasets_names=['insurance'], oversamplers_names='scaled'),
    'small_data_oversampling': generate_configuration('binary_class', oversamplers_names='undersampled', scoring=['accuracy'])
}
