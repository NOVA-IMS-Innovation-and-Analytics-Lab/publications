"""
Generate the experimental results and the manuscript.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import sys
from os.path import join, dirname

from sklearn.metrics import SCORERS, make_scorer
from sklearn.linear_model import LogisticRegression
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import RandomOverSampler, SMOTE
from gsmote import GeometricSMOTE
from rlearn.tools import ImbalancedExperiment

from tools.datasets import load_db_datasets_from_db, ImbalancedBinaryDatasets


SCORERS['geometric_mean_score'] = make_scorer(geometric_mean_score)
OVERSAMPLERS = [
    ('NONE', None, {}),
    ('ROS', RandomOverSampler(), {}),
    ('SMOTE', SMOTE(), {}),
    ('G-SMOTE', GeometricSMOTE(), {}),
]
CLASSIFIERS = [('LR', LogisticRegression(solver='liblinear', multi_class='auto'), {})]
SCORING = ['accuracy', 'f1', 'geometric_mean_score']
PATH = join(dirname(__file__), 'artifacts')


if __name__ == '__main__':

    if sys.argv[1] == 'experiment':
        datasets = load_db_datasets_from_db(join(PATH, 'imbalanced.db'))
        if not datasets:
            raise FileNotFoundError(
                'Database not found. Download the database using the `datasets` subcommand.'
            )
        experiment = ImbalancedExperiment(
            oversamplers=OVERSAMPLERS,
            classifiers=CLASSIFIERS,
            scoring=SCORING,
            n_splits=int(sys.argv[2]),
            n_runs=int(sys.argv[3]),
            random_state=int(sys.argv[4]),
        )
        experiment.fit(datasets)
    elif sys.argv[1] == 'datasets':
        datasets = ImbalancedBinaryDatasets()
        datasets.download().save(path=PATH, db_name='imbalanced')
