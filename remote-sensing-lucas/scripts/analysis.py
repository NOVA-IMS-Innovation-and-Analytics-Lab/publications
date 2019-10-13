from os.path import dirname, join
from pickle import load
from copy import deepcopy

import ast
import pandas as pd
import numpy as np
from sklearn.metrics import (
    make_scorer,
    confusion_matrix,
    accuracy_score,
    f1_score
)
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    StratifiedKFold
)
from sklearnext.tools import (
    combine_experiments,
    calculate_mean_sem_scores,
    calculate_mean_sem_perc_diff_scores,
    calculate_mean_sem_ranking,
)
from sklearnext.utils import check_random_states, check_oversamplers_classifiers
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
    'LR_comparative_confusion_matrix',
    'KNN_comparative_confusion_matrix',
    'DT_comparative_confusion_matrix',
    'GBC_comparative_confusion_matrix',
    'RF_comparative_confusion_matrix',
    'LR_conf_matrix_scores',
    'KNN_conf_matrix_scores',
    'DT_conf_matrix_scores',
    'GBC_conf_matrix_scores',
    'RF_conf_matrix_scores',
    'LR_accuracy_model',
    'KNN_accuracy_model',
    'DT_accuracy_model',
    'GBC_accuracy_model',
    'RF_accuracy_model',
    'LR_gmean_model',
    'KNN_gmean_model',
    'DT_gmean_model',
    'GBC_gmean_model',
    'RF_gmean_model',
]
OVERSAMPLERS_NAMES_MAPPER = {
    'NO OVERSAMPLING': 'NO OS',
    'RANDOM OVERSAMPLING': 'RAND OS',
    'BORDERLINE SMOTE': 'B-SMOTE',
}

def generate_mean_tbl(mean_vals, std_vals, type=float):
    """Generate table that combines mean and sem values."""
    index = mean_vals.iloc[:, :2]
    if type == float:
        scores = mean_vals.iloc[:, 2:].applymap('{:,.2f}'.format)
    elif type == int:
        scores = mean_vals.iloc[:, 2:].applymap('{:,.0f}'.format)
    else:
        raise TypeError('Only types str and float are accepted')

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
                    pd.Series(data=pa, index=labels, name='PA')
                    ])
    core_cm['UA']    = np.append(ua, [np.nan,np.nan])
    core_cm['UA']    = core_cm['UA'].replace('nan', np.nan) # replace with .2f
    core_cm['Total'] = np.append(total_v, [total, np.nan])
    scores = pd.DataFrame(data= [oa, fscore, gmean], index=['ACCURACY', 'F-SCORE MACRO', 'G-MEAN MACRO'], columns=['Score'])
    return core_cm, scores


def generate_comparative_conf_matrix(experiment, classifier, mean_vals, *args):
    """Generates a comparative confusion matrix with the model with best
    accuracy score using no oversampling (1st row) and model with best g-mean
    score using g-smote (2nd row). Returns 1) Comparative confusion matrix and
    2) Comparative table with scores for each model

    Params:
        - experiment: Pickle object generated in the experiment phase
        - classifier: Name of the classifier to which generate the confusion
                      matrix. Pass None if the overall best models are desired
                      instead. Otherwise, pass either one of 'LR', 'KNN', 'DT',
                      'GBC' or 'RF'
        - mean_vals:  Mean scores of the experiments
    """
    # set up labels and data
    X, y = experiment.datasets[0][1]
    label_mapper = {2: 'B', 0: 'C', 3: 'D', 4: 'E', 5: 'F', 1: 'A', 6: 'G', 7: 'H'}
    _metric_prefix = 'mean_test_'

    # setup k-fold cross validation, setup scorers and get weights for the mean scores
    cv = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
    scorers = {
        'accuracy': make_scorer(accuracy_score),
        'f_score_mean_macro_score': make_scorer(f1_score, average='macro'),
        'geometric_mean_macro_score': make_scorer(geometric_mean_score, average='macro')
    }
    mean_weights = []
    for _, test_  in cv.split(X,y):
         mean_weights.append(test_.shape[0])

    # query results to only include the classifier of interest
    q_mean_vals = mean_vals[mean_vals['Classifier']==classifier]

    # get random states
    pre_random_states = np.array([[est['est_name'][0].split('_')[0], est['random_state'][0]] for est in experiment.param_grids_])

    matrices = []
    scores = []
    for metric, _y in zip(['accuracy', 'geometric_mean_macro_score'], ['NO OVERSAMPLING', 'G-SMOTE']):
        target_score = q_mean_vals.set_index('Metric').loc[metric, _y]
        pre_name = f'{_y}|{classifier}'

        results = experiment.results_[_metric_prefix+metric].reset_index()
        results = results[(results['Classifier'] == classifier)&(results['Oversampler'] == _y)]
        meta_model = results[results['mean']==target_score].reset_index(drop=True)
        pre_params = ast.literal_eval(meta_model.loc[0,'params'])

        # get list of random states
        random_states = list(pre_random_states[pre_random_states[:,0]==pre_name][:,1].astype(int))

        for m in experiment.estimators_:
            if m[0].startswith(pre_name):
                pre_model = m[-1]
                break

        _scores = {'ACCURACY': [], 'G-MEAN MACRO': []}
        estimators = {}
        for random_state in range(3):

            params = deepcopy(pre_params)

            if f'{_y}__random_state' in pre_model.get_params().keys():
                params[f'{_y}__random_state'] = random_states[random_state]
            if f'{classifier}__random_state' in pre_model.get_params().keys():
                params[f'{classifier}__random_state'] = random_states[random_state]

            model = deepcopy(pre_model).set_params(**params)

            model_cv = cross_validate(
                estimator=model, X=X, y=y,
                cv=cv, scoring=scorers,
                return_estimator=True, n_jobs=1,
            )
            estimators[f'state{random_state}']=list(model_cv['estimator'])
            _scores['ACCURACY']+=[np.sum((model_cv['test_accuracy']*mean_weights))/np.sum(mean_weights)]
            _scores['G-MEAN MACRO']+=[np.sum((model_cv['test_geometric_mean_macro_score']*mean_weights))/np.sum(mean_weights)]


        pre_scores = {'state0':[], 'state1':[], 'state2':[]}
        pre_cms = {'state0':[], 'state1':[], 'state2':[]}
        for state in pre_scores.keys():
            for model, (train_index, test_index) in zip(estimators[state], cv.split(X,y)):
                X_test = X.loc[test_index]
                y_test = y.loc[test_index]
                y_pred = model.predict(X_test)
                single_df_cm, single_df_score = _make_confusion_matrix(y_test, y_pred, label_mapper)
                pre_cms[state].append(single_df_cm)
                pre_scores[state].append(single_df_score)

        dfs = ()
        for result in [pre_cms, pre_scores]:
            # get weighted mean for scores and confusiong matrix on each state
            averaged_result_per_state = []
            for df_list in result.values():
                df = pd.concat(df_list)
                df = df.astype(float).groupby(df.index).agg(lambda x: np.average(x, weights=mean_weights, axis=0))
                averaged_result_per_state.append(df)
            # get the final average confusion matrix and scores, across all random states
            df = pd.concat(averaged_result_per_state)
            df = df.astype(float).groupby(df.index).mean()
            #for col in df.columns:
            #    df[col] = df[col]#.map('{:,.2f}'.format).replace('nan', '')
            dfs+=(df,)

        df_cm, df_score = dfs

        # DEBUGGING analysis.py
        mapper__ = {'accuracy': 'ACCURACY', 'geometric_mean_macro_score': 'G-MEAN MACRO'}
        print(f"""
        ############################
        DEBUGGING analysis.py
        ############################
        {pre_name}
        ############################
        SCORE SUMMARY:
        ########## {metric} ##########
        Diff: {target_score - df_score['Score'].loc[mapper__[metric]]}
        """)
        
        matrices.append(df_cm)
        scores.append(df_score.rename(columns={'Score':pre_name}))

    conf_matrix = matrices[0].append(matrices[-1].rename(index=dict([[x,x+'1'] for x in df_cm.index])))\
                    .sort_index().rename(index=dict([[x+'1',''] for x in df_cm.index]))\
                    .reset_index()\
                    .rename(columns={'index': 'Labels'})\
                    .iloc[:-1,:]

    model_scores = pd.concat(scores, axis=1).reset_index().rename(columns={'index': 'Metrics'})

    accuracy_model = matrices[0].sort_index().reset_index().rename(columns={'index':'Labels'})
    gmean_model    = matrices[1].sort_index().reset_index().rename(columns={'index':'Labels'})

    return conf_matrix, model_scores, accuracy_model, gmean_model


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
    mean_ranking = generate_mean_tbl(type=int, *calculate_mean_sem_ranking(experiment))

    t_conf_matrix = ()
    t_model_scores = ()
    t_acc_model = ()
    t_gm_model = ()
    for model_name in ['LR', 'KNN', 'DT', 'GBC', 'RF']:
        conf_matrix, model_scores, acc_model, gm_model = generate_comparative_conf_matrix(experiment, model_name, *calculate_mean_sem_scores(experiment))
        t_conf_matrix+= (conf_matrix,)
        t_model_scores+= (model_scores,)
        t_acc_model+= (acc_model,)
        t_gm_model+= (gm_model,)

    all_res = (mean_scores, mean_perc_diff_scores, mean_ranking)+t_conf_matrix+t_model_scores+t_acc_model+t_gm_model
    return all_res


if __name__ == '__main__':

    main_results = generate_main_results()
    for name, result in zip(MAIN_RESULTS_NAMES, main_results):
        result.to_csv(join(RESOURCES_PATH, f'{name}.csv'), index=False)
