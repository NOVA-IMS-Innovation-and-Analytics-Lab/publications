from os.path import dirname, join
from pickle import load

import ast
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearnext.tools import (
    combine_experiments,
    calculate_mean_sem_scores,
    calculate_mean_sem_perc_diff_scores,
    calculate_mean_sem_ranking,
)
from tools import EXPERIMENTS_PATH
from tools.format import METRICS_NAMES_MAPPING

EXPERIMENTS_NAMES = [
    'no_oversampling',
    'random_oversampling',
    'smote',
    'borderline_smote',
    'adasyn',
    'gsmote'
]
RESOURCES_PATH = join(dirname(__file__), 'resources')
MAIN_RESULTS_NAMES = [
    'mean_scores',
    'mean_perc_diff_scores',
    'mean_ranking',
    'comparative_confusion_matrix',
    'conf_matrix_scores'
]
OVERSAMPLERS_NAMES_MAPPER = {
    'NO OVERSAMPLING': 'NO OS',
    'RANDOM OVERSAMPLING': 'RAND OS',
    'BORDERLINE SMOTE': 'B-SMOTE',
}

def generate_mean_tbl(mean_vals, std_vals):
    """Generate table that combines mean and sem values."""
    index = mean_vals.iloc[:, :2]
    scores = mean_vals.iloc[:, 2:].applymap('{:,.2f}'.format)
    tbl = pd.concat([index, scores], axis=1).rename(columns=OVERSAMPLERS_NAMES_MAPPER)
    tbl['Metric'] = tbl['Metric'].apply(lambda metric: METRICS_NAMES_MAPPING[metric])
    return tbl


def _make_confusion_matrix(y_true, y_pred, label_mapper):
    """Generates a confusion matrix. Returns 1) Pandas Dataframe with confusion
    matrix, Producer's accuracy and User's accuracy and 2) Pandas Dataframe with
    Overall accuracy, F-Score, and G-mean score."""
    labels = list(label_mapper.values())
    labels.sort()
    cm = confusion_matrix(y_true.map(label_mapper), pd.Series(y_pred).map(label_mapper), labels=labels).T
    total_h = np.sum(cm, 0)
    total_v = np.sum(cm, 1)
    total   = sum(total_h)
    tp      = cm.diagonal()
    fp      = total_v-tp
    fn      = total_h-tp
    tn      = total-(fn+tp+fp)
    spec    = tn/(fp+tn)
    ua      = tp/(total_v+1e-100)
    pa      = tp/(total_h+1e-100)
    oa      = sum(tp)/total
    fscore  = 2*((np.mean(ua)*np.mean(pa))/(np.mean(ua)+np.mean(pa)))
    gmean   = np.sqrt(np.mean(pa)*np.mean(spec))
    core_cm = pd.DataFrame(index=labels, columns=labels, data=cm)\
                .append([
                    pd.Series(data=total_h, index=labels, name='Total'),
                    pd.Series(data=pa, index=labels, name='PA').map('{:,.2f}'.format)
                    ])
    core_cm['Total'] = np.append(total_v, [total, ''])
    core_cm['UA']    = np.append(ua, [np.nan,np.nan])
    core_cm['UA']    = core_cm['UA'].map('{:,.2f}'.format).replace('nan', '')
    core_cm.columns = pd.MultiIndex.from_product([['True Condition'],core_cm.columns])
    core_cm.index = pd.MultiIndex.from_product([['Predicted Condition'],core_cm.index])
    scores = pd.DataFrame(data= [oa, fscore, gmean], index=['ACCURACY', 'F-SCORE MACRO', 'G-MEAN MACRO'], columns=['Score'])
    return core_cm, scores


def generate_comparative_conf_matrix(experiment, mean_vals, *args):
    """Generates a comparative confusion matrix with the model with best
    accuracy score using no oversampling (1st row) and model with best g-mean
    score using g-smote (2nd row). Returns 1) Comparative confusion matrix and
    2) Comparative table with scores for each model"""

    X, y = experiment.datasets[0][1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    label_mapper = {2: 'B', 0: 'C', 3: 'D', 4: 'E', 5: 'F', 1: 'A', 6: 'G', 7: 'H'}
    _metric_prefix = 'mean_test_'
    matrices = []
    scores = []
    for metric, _y in zip(['accuracy', 'geometric_mean_macro_score'], ['NO OVERSAMPLING', 'G-SMOTE']):
        _x = mean_vals[mean_vals['Metric']==metric].iloc[:,2:].idxmax()[_y]
        score = mean_vals.loc[_x,_y]
        results = experiment.results_[_metric_prefix+metric]
        _model = results[results['mean']==score].reset_index()
        _pre_name = f'{_model.loc[0,"Oversampler"]}|{_model.loc[0,"Classifier"]}'
        _params = ast.literal_eval(_model.loc[0,'params'])
        for m in experiment.estimators_:
            if m[0].startswith(_pre_name):
                model = m[-1]
                for key, value in _params.items():
                    clf, par = key.split('__')
                    setattr(model.named_steps[clf], par, value)
                break
        y_pred = model.fit(X_train, y_train).predict(X_test)
        df_cm, df_score = _make_confusion_matrix(y_test, y_pred, label_mapper)
        df_score['Score'] = df_score['Score'].map('{:,.2f}'.format)
        matrices.append(df_cm)
        scores.append(df_score.rename(columns={'Score':_pre_name}))


    conf_matrix = matrices[0].append(matrices[-1].rename(index=dict([[x,x+'1'] for x in df_cm.index.levels[-1]])))\
                    .sort_index().rename(index=dict([[x+'1',''] for x in df_cm.index.levels[-1]]))\
                    .iloc[:-1,:]
    model_scores = pd.concat(scores, axis=1)
    return conf_matrix, model_scores


def generate_main_results():
    """Generate the main results of the experiment."""

    # Load experiments object
    experiments = []
    for name in EXPERIMENTS_NAMES:
        file_path = join(EXPERIMENTS_PATH, 'lucas', f'{name}.pkl')
        with open(file_path, 'rb') as file:
            experiments.append(load(file))

    # Create combined experiment
    experiment = combine_experiments('combined', *experiments)

    # Exclude constant and random classifiers
    mask_clf1 = (experiment.results_.reset_index().Classifier != 'CONSTANT CLASSIFIER')
    mask_clf2 = (experiment.results_.reset_index().Classifier != 'RANDOM CLASSIFIER')
    mask = (mask_clf1 & mask_clf2).values
    experiment.results_ = experiment.results_[mask]
    experiment.classifiers = experiment.classifiers[2:]
    experiment.classifiers_names_ = experiment.classifiers_names_[2:]

    # Recreate optimal and wide optimal results
    experiment._calculate_optimal_results()._calculate_wide_optimal_results()

    # Calculate results
    mean_scores = generate_mean_tbl(*calculate_mean_sem_scores(experiment))
    mean_perc_diff_scores = generate_mean_tbl(*calculate_mean_sem_perc_diff_scores(experiment, ['SMOTE', 'G-SMOTE']))
    mean_ranking = generate_mean_tbl(*calculate_mean_sem_ranking(experiment))
    conf_matrix, model_scores = generate_comparative_conf_matrix(experiment, *calculate_mean_sem_scores(experiment))

    return mean_scores, mean_perc_diff_scores, mean_ranking, conf_matrix, model_scores


if __name__ == '__main__':

    main_results = generate_main_results()
    for name, result in zip(MAIN_RESULTS_NAMES, main_results):
        if name in ['comparative_confusion_matrix','conf_matrix_scores']:
            keep_index=True
        else:
            keep_index=False
        result.to_csv(join(RESOURCES_PATH, f'{name}.csv'), index=keep_index)
