"""
Configuration of the experiment.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sklearn.metrics import SCORERS, make_scorer
from sklearn.linear_model import LogisticRegression
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import RandomOverSampler, SMOTE
from gsmote import GeometricSMOTE


SCORERS['geometric_mean_score'] = make_scorer(geometric_mean_score)
OVERSAMPLERS = [
    ('NONE', None, {}),
    ('ROS', RandomOverSampler(), {}),
    ('SMOTE', SMOTE(), {}),
    (
        'G-SMOTE',
        GeometricSMOTE(),
        {
            'truncation_factor': [-1.0, -0.75, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0],
            'deformation_factor': [0.0, 0.25, 0.5, 0.75, 1.0],
        },
    ),
]
CLASSIFIERS = [('LR', LogisticRegression(solver='liblinear', multi_class='auto'), {})]
SCORING = ['accuracy', 'f1', 'geometric_mean_score']
