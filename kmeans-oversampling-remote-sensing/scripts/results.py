"""
Generate the main experimental results.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
#         Joao Fonseca <jpmrfonseca@gmail.com>
# License: MIT

import sys
from os.path import join, dirname

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import make_scorer, SCORERS
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from clover.over_sampling import (
    SMOTE as ClusterSMOTE,
    BorderlineSMOTE as ClusterBorderlineSMOTE,
    SVMSMOTE as ClusterSVMSMOTE,
    ADASYN as ClusterADASYN,
    RandomOverSampler as ClusterRandomOverSampler
)
from sklearn.cluster import KMeans
from rlearn.tools import ImbalancedExperiment

sys.path.append(join(dirname(__file__), '..', '..'))
from utils import load_datasets, generate_paths, RemoteSensingDatasets

CONFIG = {
    'oversamplers': [
        ('NONE', None, {}),
        ('ROS', RandomOverSampler(), {}),
        ('SMOTE', SMOTE(), {'k_neighbors': [3, 5]}),
        ('B-SMOTE', BorderlineSMOTE(), {'k_neighbors': [3, 5]}),
        ('ADASYN', ADASYN(), {'n_neighbors': [2, 3]}),
        ('KMEANS-SMOTE', ClusterSMOTE(), {
                'clusterer': [
                    KMeans(n_clusters=10), KMeans(n_clusters=30),
                    KMeans(n_clusters=50), KMeans(n_clusters=70),
                    KMeans(n_clusters=100)
                    ], 'k_neighbors': [3, 5]}),
        ('KMEANS-B-SMOTE', ClusterBorderlineSMOTE(), {
                'clusterer': [
                    KMeans(n_clusters=10), KMeans(n_clusters=30),
                    KMeans(n_clusters=50), KMeans(n_clusters=70),
                    KMeans(n_clusters=100)
                    ], 'k_neighbors': [3, 5]}),
        ('KMEANS-SVM-SMOTE', ClusterSVMSMOTE(), {
                'clusterer': [
                    KMeans(n_clusters=10), KMeans(n_clusters=30),
                    KMeans(n_clusters=50), KMeans(n_clusters=70),
                    KMeans(n_clusters=100)
                    ]}),
        ('KMEANS-ADASYN', ClusterADASYN(), {
                'clusterer': [
                    KMeans(n_clusters=10), KMeans(n_clusters=30),
                    KMeans(n_clusters=50), KMeans(n_clusters=70),
                    KMeans(n_clusters=100)
                    ], 'n_neighbors': [2, 3]}),
        ('KMEANS-ROS', ClusterRandomOverSampler(), {
                'clusterer': [
                    KMeans(n_clusters=10), KMeans(n_clusters=30),
                    KMeans(n_clusters=50), KMeans(n_clusters=70),
                    KMeans(n_clusters=100)
                    ]})
        ],
    'classifiers': [
        ('CONSTANT CLASSIFIER', DummyClassifier(strategy='constant', constant=0), {}),
        ('LR', LogisticRegression(solver='liblinear', multi_class='auto'), {}),
        ('KNN', KNeighborsClassifier(), {'n_neighbors': [3, 5]}),
        ('DT', DecisionTreeClassifier(), {'max_depth': [3, 6]}),
        ('GBC', GradientBoostingClassifier(), {'max_depth': [3, 6], 'n_estimators': [50, 100]}),
        ('RF', RandomForestClassifier(), {'max_depth': [None, 3, 6], 'n_estimators': [10, 50, 100]})
    ],
    'scoring': ['accuracy', 'f1_macro', 'geometric_mean_score_macro'],
    'n_splits': 5,
    'n_runs': 3,
    'rnd_seed': 0,
    'n_jobs': -1
}


if __name__ == '__main__':

    # Extract paths
    data_path, results_path, _ = generate_paths()

    # Load datasets
    datasets = load_datasets(data_path=data_path, data_type='db')

    # Extract oversamplers
    oversamplers = CONFIG['oversamplers']

    # Generate oversamplers
    for oversampler in oversamplers:

        # Define and fit experiment
        experiment = ImbalancedExperiment(
            oversamplers=[oversampler],
            classifiers=CONFIG['classifiers'],
            scoring=CONFIG['scoring'],
            n_splits=CONFIG['n_splits'],
            n_runs=CONFIG['n_runs'],
            random_state=CONFIG['rnd_seed'],
            n_jobs=CONFIG['n_jobs']
        ).fit(datasets)

        # Save results
        file_name = f'{oversampler[0].replace("-", "").lower()}.pkl'
        experiment.results_.to_pickle(join(results_path, file_name))
