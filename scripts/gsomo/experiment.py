"""
Run the experimental procedure of the
G-SOMO paper.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

# Imports
from os.path import join, dirname

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearnext.over_sampling import RandomOverSampler, SMOTE, GeometricSMOTE, DensityDistributor
from sklearnext.cluster import SOM
from sklearnext.tools import evaluate_binary_imbalanced_experiments, read_csv_dir, summarize_binary_datasets


# Paths
datasets_path = join(dirname(__file__), '..', '..', 'data', 'binary-numerical-imbalanced')
results_path = join(dirname(__file__), '..', '..', 'data', 'results', 'gsomo')

# Oversamplers and classifiers
oversamplers = [
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
]
classifiers = [
    ('LR', LogisticRegression()),
    ('KNN', KNeighborsClassifier(), {'n_neighbors': [3, 5]}),
    ('DT', DecisionTreeClassifier(random_state=3), {'max_depth': [3, 6]}),
    ('GBC', GradientBoostingClassifier(random_state=4), {'max_depth': [3, 6]})
]

# Load datasets
imbalanced_datasets = read_csv_dir(datasets_path)

# Summarize datasets
imbalanced_datasets_summary = summarize_binary_datasets(imbalanced_datasets)

# Run main experiment
results = evaluate_binary_imbalanced_experiments(datasets=imbalanced_datasets,
                                                 oversamplers=oversamplers,
                                                 classifiers=classifiers,
                                                 scoring=['roc_auc', 'f1', 'geometric_mean_score'],
                                                 n_splits=3,
                                                 n_runs=2,
                                                 random_state=5,
                                                 scheduler='multiprocessing')

# Save various datasets 
imbalanced_datasets_summary.to_csv(join(results_path, 'imbalanced_datasets_summary.csv'), index=False)
for name, result in results.items():
    result.to_csv(join(results_path, '%s.csv' % name), index=(name == 'aggregated'))
