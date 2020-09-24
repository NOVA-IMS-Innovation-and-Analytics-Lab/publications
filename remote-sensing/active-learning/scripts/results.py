"""
Generate the main experimental results.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
#         Joao Fonseca <jpmrfonseca@gmail.com>
# License: MIT

import sys
from os.path import join, dirname

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import (
    SMOTE,
    RandomOverSampler,
    BorderlineSMOTE
)
from clover.over_sampling import ClusterOverSampler
from gsmote import GeometricSMOTE
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.model_selection import StratifiedKFold, train_test_split
from rlearn.model_selection import ModelSearchCV

sys.path.append(join(dirname(__file__), '..', '..', '..'))
from utils import load_datasets, generate_paths, RemoteSensingDatasets


# Imports related to AL
from sklearn.metrics import SCORERS, make_scorer
from imblearn.metrics import geometric_mean_score
from itertools import product
from rlearn.utils import check_random_states
from imblearn.pipeline import Pipeline

def geometric_mean_score_macro(y_true, y_pred):
    """Geometric mean score with macro average."""
    return geometric_mean_score(y_true, y_pred, average='macro')

SCORERS['geometric_mean_score_macro'] = make_scorer(geometric_mean_score_macro)

def check_pipelines(objects_list, random_state, n_runs):
    """Extract estimators and parameters grids."""

    # Create random states
    random_states = check_random_states(random_state, n_runs)

    pipelines = []
    param_grid = []
    for comb in product(*objects_list):
        name  = '|'.join([i[0] for i in comb])
        comb = [(nm,ob,grd) for nm,ob,grd in comb if ob is not None] # name, object, grid

        pipelines.append((name, Pipeline([(nm,ob) for nm,ob,_ in comb])))

        grids = {'est_name': [name]}
        for obj_name, obj, sub_grid in comb:
            if 'random_state' in obj.get_params().keys():
                grids[f'{name}__{obj_name}__random_state'] = random_states
            for param, values in sub_grid.items():
                grids[f'{name}__{obj_name}__{param}'] = values
        param_grid.append(grids)

    return pipelines, param_grid

def check_pipelines_wrapper(
    objects_list,
    wrapper,
    random_state,
    n_runs,
):
    wrapper_label = wrapper[0]
    wrapper_obj = wrapper[1]
    wrapper_grid = wrapper[2]

    estimators, param_grids = check_pipelines(
        objects_list, random_state, n_runs
    )

    wrapped_estimators = [
        (f'{wrapper_label}|{name}', clone(wrapper_obj).set_params(**{'classifier':pipeline})) for name, pipeline in estimators
    ]

    wrapped_param_grids = [
        {
            'est_name': [f'{wrapper_label}|{d["est_name"][0]}'],
            **{k.replace(d["est_name"][0], f'{wrapper_label}|{d["est_name"][0]}__classifier'):v
                for k, v in d.items() if k!='est_name'},
            **{f'{wrapper_label}|{d["est_name"][0]}__{k}':v for k, v in wrapper_grid.items()}
        } for d in param_grids
    ]

    return estimators + wrapped_estimators, param_grids + wrapped_param_grids


def _entropy_selection(probabs, leftover_ids, increment):
    e = (-probabs * np.log2(probabs)).sum(axis=1)
    new_ids = leftover_ids[np.argsort(e)[::-1][:increment]]
    return new_ids

def _margin_sampling_selection(probabs, leftover_ids, increment):
    """
    Selecting samples as a smallest difference of probability values
    between the first and second most likely classes
    """
    probs_sorted = np.sort(probabs, axis=1)[:, ::-1]
    values = probs_sorted[:, 0] - probs_sorted[:, 1]
    new_ids = leftover_ids[np.argsort(values)[:increment]]
    return new_ids

def _random_selection(leftover_ids, increment, rng):
    """
    Random sample selection. Random State object is required.
    """
    new_ids = rng.choice(leftover_ids, increment, replace=False)
    return new_ids

def data_selection(probabs, leftover_ids, increment, rng, selection_strategy):
    if selection_strategy == 'entropy':
        return _entropy_selection(probabs, leftover_ids, increment)
    elif selection_strategy == 'margin sampling':
        return _margin_sampling_selection(probabs, leftover_ids, increment)
    elif selection_strategy == 'random':
        return _random_selection(leftover_ids, increment, rng)
    else:
        msg = f"Selection strategy {selection_strategy} is \
        not implemented. Possible values are \
        ['entropy', 'margin sampling', 'random']."
        raise ValueError(f'')

class ALWrapper(ClassifierMixin, BaseEstimator):
    def __init__(
            self,
            classifier=RandomForestClassifier(),
            max_iter=1000,
            #split_method=None,
            selection_strategy='entropy',
            n_initial=100,
            increment=50,
            save_classifiers=False,
            auto_load=True,
            test_size=.1,
            evaluation_metric=None,
            random_state=None
        ):
        self.classifier = classifier
        self.max_iter = max_iter
        #self.split_method = split_method
        self.selection_strategy = selection_strategy
        self.n_initial = n_initial
        self.increment = increment
        self.random_state = random_state

        # For finding the optimal classifier purposes
        self.auto_load = auto_load
        self.test_size = test_size
        self.save_classifiers = save_classifiers
        self.evaluation_metric = evaluation_metric


    def fit(self, X, y):

        if self.evaluation_metric is None:
            self.evaluation_metric_ = SCORERS['accuracy']
        elif type(self.evaluation_metric) == str:
            self.evaluation_metric_ = SCORERS[self.evaluation_metric]
        else:
            self.evaluation_metric_ = self.evaluation_metric

        if self.save_classifiers:
            self.classifiers_ = []

        if self.auto_load:
            self.classifier_ = None
            self._top_score = 0
            X, X_test, y, y_test = train_test_split(
                X, y,
                test_size = self.test_size,
                random_state = self.random_state,
                stratify = y
            )

        self.increment_ = self.increment

        iter_n = 0
        rng = np.random.RandomState(self.random_state)
        selection = np.zeros(shape=(X.shape[0])).astype(bool)

        while iter_n<self.max_iter:

            classifier = clone(self.classifier)

            # add new samples to dataset
            leftover_ids = np.argwhere(~selection).squeeze()
            ids = (
                data_selection(
                    probabs, leftover_ids, self.increment_, rng, self.selection_strategy
                )
                if iter_n != 0
                else
                rng.choice(leftover_ids, self.n_initial, replace=False)
            )
            selection[ids] = True

            # train classifier and get probabilities
            try:
                classifier.fit(X[selection], y[selection])
            except:
               print('error found.')
               return X, y, selection

            # save classifier
            if self.save_classifiers:
                self.classifiers_.append((selection.sum(),classifier))

            # Replace top classifier
            if self.auto_load:
                score = self.evaluation_metric_(
                    classifier,
                    X_test,
                    y_test
                )
                if score > self._top_score:
                    self._top_score = score
                    self.classifier_ = classifier


            # keep track of iter_n
            if self.max_iter is not None:
                iter_n+=1

            # stop if all examples have been included
            if selection.all():
                break
            elif selection.sum()+self.increment_ > y.shape[0]:
                self.increment_ = y.shape[0] - selection.sum()

            probabs = classifier.predict_proba(X[~selection])

            # some selection strategies don't deal well with 0. values
            probabs = np.where(probabs==0., 1e-10, probabs)

        return self

    def load_best_classifier(self, X, y):
        scores = []
        for _, classifier in self.classifiers_:
            scores.append(
                self.evaluation_metric_(classifier, X, y)
            )

        self.classifier_ = self.classifiers_[np.argmax(scores)][-1]
        return self

    def predict(self, X):
        return self.classifier_.predict(X)



CONFIG = {
    'oversamplers': [
        #('NONE', None, {}),
        ('G-SMOTE', ClusterOverSampler(GeometricSMOTE(), n_jobs=1), {})
        #    'oversampler__k_neighbors': [3, 5],
        #    'oversampler__selection_strategy': ['combined', 'minority', 'majority'],
        #    'oversampler__truncation_factor': [-1.0, .0, 1.0],
        #    'oversampler__deformation_factor': [.0, 0.5, 1.0]
        #    })
    ],
    'classifiers': [
        ('LR', LogisticRegression(
            multi_class='multinomial', solver='sag', penalty='none', max_iter=1e4), {}),
        ('KNN', KNeighborsClassifier(), {'n_neighbors': [3]}),#, 5, 8]}),
        ('RF', RandomForestClassifier(), {'max_depth':
        [None, 3, 6], 'n_estimators': [50]})#, 100, 200]})
    ],
    'wrapper': ('AL', ALWrapper(n_initial=100, increment=20, max_iter=3000, random_state=0), {
        'evaluation_metric': ['accuracy'],#, 'f1_macro', 'geometric_mean_score_macro'],
        'selection_strategy': ['random']#, 'entropy', 'margin sampling']
    }),
    'scoring': ['accuracy', 'f1_macro', 'geometric_mean_score_macro'],
    'n_splits': 4,
    'n_runs': 3,
    'rnd_seed': 0,
    'n_jobs': -1,
    'verbose': 1
}


if __name__ == '__main__':

    # Extract paths
    data_dir, results_dir, _ = generate_paths()

    # Load datasets
    datasets = load_datasets(data_dir=data_dir)

    # Extract pipelines and parameter grids
    estimators, param_grids = check_pipelines_wrapper(
        [CONFIG['oversamplers'], CONFIG['classifiers']],
        CONFIG['wrapper'],
        CONFIG['rnd_seed'],
        CONFIG['n_runs'],
    )

    for name, (X, y) in datasets:
        # Define and fit experiment
        experiment = ModelSearchCV(
            estimators,
            param_grids,
            scoring = CONFIG['scoring'],
            n_jobs = CONFIG['n_jobs'],
            cv = StratifiedKFold(
                n_splits=CONFIG['n_splits'],
                shuffle=True,
                random_state=CONFIG['rnd_seed']
            ),
            verbose = CONFIG['verbose'],
            return_train_score = False,
            refit = False
        ).fit(X, y)

        # Save results
        file_name = f'{name.replace(" ", "_").lower()}.pkl'
        pd.DataFrame(experiment.cv_results_).to_pickle(join(results_dir, file_name))
