"""
Define configurations of experiments.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearnext.over_sampling import RandomOverSampler, SMOTE, GeometricSMOTE, DensityDistributor
from sklearnext.cluster import KMeans, SOM, AgglomerativeClustering, Birch, SpectralClustering


CONFIGURATIONS = {
    'GSMOTE': {
        'db_name': 'imbalanced_data.db',
        'oversamplers': [
            ('NO OVERSAMPLING', None),
            ('RANDOM OVERSAMPLING', RandomOverSampler()),
            ('SMOTE', SMOTE(), {'k_neighbors': [3, 5]}),
            ('G-SMOTE', GeometricSMOTE(), {'selection_strategy': ['combined', 'minority', 'majority'], 'k_neighbors': [3, 5], 'truncation_factor': [-1.0, -0.5, .0, 0.25, 0.5, 0.75, 1.0], 'deformation_factor': [.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]})
        ],
        'classifiers': [
            ('LR', LogisticRegression(solver='lbfgs', max_iter=1e4)),
            ('KNN', KNeighborsClassifier(), {'n_neighbors': [3, 5]}),
            ('DT', DecisionTreeClassifier(), {'max_depth': [3, 6]}),
            ('GBC', GradientBoostingClassifier(), {'max_depth': [3, 6], 'n_estimators': [50, 100]})
        ],
        'scoring': ['roc_auc', 'f1', 'geometric_mean_score'],
        'n_splits': 5,
        'n_runs': 3,
        'random_state': 0
    },
    'Clustering-SMOTE': {
        'db_name': 'imbalanced_data.db',
        'oversamplers': [
            ('NO OVERSAMPLING', None),
            ('SMOTE', SMOTE(), {'k_neighbors': [3, 4, 5]}),
            ('K-MEANS SMOTE', SMOTE(clusterer=KMeans(), distributor=DensityDistributor()), {
                'k_neighbors': [3, 4, 5],
                'clusterer__n_clusters': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'distributor__distances_exponent': [0, 1, 2, 5],
                'distributor__filtering_threshold': [0.0, 0.5, 1.0, 2.0]
                }
            ),
            ('SOMO', SMOTE(clusterer=SOM(), distributor=DensityDistributor()), {
                'k_neighbors': [3, 4, 5],
                'clusterer__n_clusters': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'distributor__distances_exponent': [0, 1, 2, 5],
                'distributor__filtering_threshold': [0.0, 0.5, 1.0, 2.0],
                'distributor__distribution_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]
                }
            ),
            ('AGGLOMERATIVE SMOTE', SMOTE(clusterer=AgglomerativeClustering(), distributor=DensityDistributor()), {
                'k_neighbors': [3, 4, 5],
                'clusterer__n_clusters': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'distributor__distances_exponent': [0, 1, 2, 5],
                'distributor__filtering_threshold': [0.0, 0.5, 1.0, 2.0]
                }
            ),
            ('BIRCH SMOTE', SMOTE(clusterer=Birch(), distributor=DensityDistributor()), {
                'k_neighbors': [3, 4, 5],
                'clusterer__n_clusters': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'distributor__distances_exponent': [0, 1, 2, 5],
                'distributor__filtering_threshold': [0.0, 0.5, 1.0, 2.0]
                }
            ),
            ('SPECTRAL SMOTE', SMOTE(clusterer=SpectralClustering(), distributor=DensityDistributor()), {
                'k_neighbors': [3, 4, 5],
                'clusterer__n_clusters': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'distributor__distances_exponent': [0, 1, 2, 5],
                'distributor__filtering_threshold': [0.0, 0.5, 1.0, 2.0]
                }
            )
        ],
        'classifiers': [
            ('LR', LogisticRegression()),
            ('KNN', KNeighborsClassifier(), {'n_neighbors': [3, 5]}),
            ('DT', DecisionTreeClassifier(), {'max_depth': [3, 6]}),
            ('GBC', GradientBoostingClassifier(), {'max_depth': [3, 6], 'n_estimators': [50, 100]})
        ],
        'scoring': ['roc_auc', 'f1', 'geometric_mean_score'],
        'n_splits': 5,
        'n_runs': 3,
        'random_state': 0
    },
    'GSOMO': {
        'db_name': 'imbalanced_data.db',
        'oversamplers': [
            ('NO OVERSAMPLING', None),
            ('RANDOM OVERSAMPLING', RandomOverSampler(random_state=0)),
            ('SMOTE', SMOTE(random_state=1), {'k_neighbors': [3, 5]}),
            ('G-SOMO', GeometricSMOTE(clusterer=SOM(), distributor=DensityDistributor(distances_exponent=2, filtering_threshold=1.0), random_state=3), {
                'k_neighbors': [3, 5],
                'truncation_factor': [-1.0, 0.0, 0.25, 1.0],
                'deformation_factor': [0.0, 0.5, 1.0],
                'clusterer__n_clusters': [0.2, 0.5],
                'distributor__distribution_ratio': [0.75, 1.0]
                }
            ) 
        ],
        'classifiers': [
            ('LR', LogisticRegression()),
            ('KNN', KNeighborsClassifier(), {'n_neighbors': [3, 5]}),
            ('DT', DecisionTreeClassifier(random_state=3), {'max_depth': [3, 6]}),
            ('GBC', GradientBoostingClassifier(random_state=4), {'max_depth': [3, 6]})
        ],
        'scoring': ['roc_auc', 'f1', 'geometric_mean_score'],
        'n_splits': 5,
        'n_runs': 3,
        'random_state': 0
    },
    'KMeans-ROS': {
        'db_name': 'imbalanced_data.db',
        'oversamplers': [
            ('NO OVERSAMPLING', None), 
            ('RANDOM OVERSAMPLING', RandomOverSampler()),
            ('K-MEANS RANDOM OVERSAMPLING', RandomOverSampler(clusterer=KMeans(n_init=1), distributor=DensityDistributor()), {
                'clusterer__n_clusters': [0.0, 0.25, 0.5, 0.75, 1.0],
                'distributor__distances_exponent': [0, 1, 2],
                'distributor__filtering_threshold': [0.5, 1.0]
                }
            )
        ],
        'classifiers': [
            ('LR', LogisticRegression()),
            ('KNN', KNeighborsClassifier(), {'n_neighbors': [3, 5]}),
            ('DT', DecisionTreeClassifier(), {'max_depth': [3, 6]}),
            ('GBC', GradientBoostingClassifier(), {'max_depth': [3, 6], 'n_estimators': [50, 100]})
        ],
        'scoring': ['roc_auc', 'f1', 'geometric_mean_score'],
        'n_splits': 5,
        'n_runs': 3,
        'random_state': 0
    },
    'KMeans-SMOTE': {
        'db_name': 'imbalanced_data.db',
        'oversamplers': [
            ('NO OVERSAMPLING', None),
            ('SMOTE', SMOTE(), {'k_neighbors': [3, 5]}),
            ('K-MEANS SMOTE', SMOTE(clusterer=KMeans(n_init=1), distributor=DensityDistributor()), {
                'k_neighbors': [3, 5],
                'clusterer__n_clusters': [0.0, 0.25, 0.5, 0.75, 1.0],
                'distributor__distances_exponent': [0, 1, 2],
                'distributor__filtering_threshold': [0.5, 1.0]
                }
            ) 
        ],
        'classifiers': [
            ('LR', LogisticRegression()),
            ('KNN', KNeighborsClassifier(), {'n_neighbors': [3, 5]}),
            ('DT', DecisionTreeClassifier(), {'max_depth': [3, 6]}),
            ('GBC', GradientBoostingClassifier(), {'max_depth': [3, 6], 'n_estimators': [50, 100]})
        ],
        'scoring': ['roc_auc', 'f1', 'geometric_mean_score'],
        'n_splits': 5,
        'n_runs': 3,
        'random_state': 0
    },
    'KMeans-BorderlineSMOTE': {
        'db_name': 'imbalanced_data.db',
        'oversamplers': [
            ('NO OVERSAMPLING', None),
            ('BORDERLINE-SMOTE', SMOTE(kind='borderline1'), {'k_neighbors': [3, 5]}),
            ('K-MEANS BORDERLINE-SMOTE', SMOTE(clusterer=KMeans(n_init=1), distributor=DensityDistributor(), kind='borderline1'), {
                'k_neighbors': [3, 5],
                'clusterer__n_clusters': [0.0, 0.25, 0.5, 0.75, 1.0],
                'distributor__distances_exponent': [0, 1, 2],
                'distributor__filtering_threshold': [0.5, 1.0]
                }
            ) 
        ],
        'classifiers': [
            ('LR', LogisticRegression()),
            ('KNN', KNeighborsClassifier(), {'n_neighbors': [3, 5]}),
            ('DT', DecisionTreeClassifier(), {'max_depth': [3, 6]}),
            ('GBC', GradientBoostingClassifier(), {'max_depth': [3, 6], 'n_estimators': [50, 100]})
        ],
        'scoring': ['roc_auc', 'f1', 'geometric_mean_score'],
        'n_splits': 5,
        'n_runs': 3,
        'random_state': 0
    },
    'KMeans-GSMOTE': {
        'db_name': 'imbalanced_data.db',
        'oversamplers': [
            ('NO OVERSAMPLING', None),
            ('G-SMOTE', GeometricSMOTE(), {
                'k_neighbors': [3, 5],
                'truncation_factor': [-1.0, 0.0, 1.0],
                'deformation_factor': [0.0, 0.5, 1.0]
                }
            ),
            ('K-MEANS G-SMOTE', GeometricSMOTE(clusterer=KMeans(n_init=1), distributor=DensityDistributor()), {
                'k_neighbors': [3, 5],
                'truncation_factor': [-1.0, 0.0, 1.0],
                'deformation_factor': [0.0, 0.5, 1.0],
                'clusterer__n_clusters': [0.0, 0.25, 0.5, 0.75, 1.0],
                'distributor__distances_exponent': [0, 1, 2],
                'distributor__filtering_threshold': [0.5, 1.0]
                }
            )
        ],
        'classifiers': [
            ('LR', LogisticRegression()),
            ('KNN', KNeighborsClassifier(), {'n_neighbors': [3, 5]}),
            ('DT', DecisionTreeClassifier(), {'max_depth': [3, 6]}),
            ('GBC', GradientBoostingClassifier(), {'max_depth': [3, 6], 'n_estimators': [50, 100]})
        ],
        'scoring': ['roc_auc', 'f1', 'geometric_mean_score'],
        'n_splits': 5,
        'n_runs': 3,
        'random_state': 0
    }
}






