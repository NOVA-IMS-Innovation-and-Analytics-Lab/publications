"""
Runs the experimental procedure of the
Geometric SMOTE journal paper.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

# Imports
from os.path import join
from collections import Counter
from re import match, sub
from ast import literal_eval
from scipy.stats import rankdata
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearnext.tools import evaluate_binary_imbalanced_experiments, read_csv_dir, summarize_binary_datasets
from sklearnext.over_sampling import GeometricSMOTE
from bokeh.io import export_svgs
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.palettes import Category20, Spectral

# Configuration
results_path = join('..', 'gsmote-journal', 'resources')

# Oversamplers and classifiers
oversamplers = [
    ('NO OVERSAMPLING', None),
    ('RANDOM OVERSAMPLING', RandomOverSampler()),
    ('SMOTE', SMOTE(), {'k_neighbors': [3, 5]}),
    ('G-SMOTE', GeometricSMOTE(), {
        'selection_strategy':['combined', 'minority', 'majority'],
        'k_neighbors':[3, 5],
        'truncation_factor': [-1.0, -0.5, .0, 0.25, 0.5, 0.75, 1.0],
        'deformation_factor': [.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    })
]
classifiers = [
    ('LR', LogisticRegression()),
    ('KNN', KNeighborsClassifier(), {'n_neighbors':[3, 5]}),
    ('DT', DecisionTreeClassifier(), {'max_depth': [3, 6]}),
    ('GBC', GradientBoostingClassifier(), {'max_depth':[3, 6], 'n_estimators':[50, 100]})
]

# Parameters
classifiers_names, _ = zip(*classifiers)
oversamplers_names, _ = zip(*oversamplers)
_, metrics_names = zip(*metric_map)

# Load datasets
imbalanced_datasets = read_csv_dir('data/binary-numerical-imbalanced')


if __name__ == "__main__":

    # Summarize datasets
    imbalanced_datasets_summary = summarize_binary_datasets(imbalanced_datasets)

    # Run main experiment
    results = evaluate_binary_imbalanced_experiments(datasets=imbalanced_datasets,
                                                     oversamplers=oversamplers,
                                                     classifiers=classifiers,
                                                     scoring=['roc_auc', 'f1', 'geometric_mean_score'],
                                                     n_splits=5,
                                                     n_runs=3,
                                                     random_state=0,
                                                     scheduler='multiprocessing')

    # Extract results
    wide_optimal = results['wide_optimal']
    mean_ranking = results['mean_ranking']
    friedman_test = results['friedman_test']
    friedman_test['p-value'] = friedman_test['p-value'].apply(lambda p_value: '%.1e' % p_value)

    # Rename metrics
    metric_map = [('ROC AUC', 'AUC'), ('F1', 'F-SCORE'), ('GEOMETRIC MEAN SCORE', 'G-MEAN')]
    for result in [wide_optimal, mean_ranking, friedman_test]:
        result['Metric'] = result['Metric'].apply(lambda row: dict(metric_map)[row])

    # Sort results by IR
    wide_optimal['Dataset'] = pd.Categorical(wide_optimal['Dataset'], imbalanced_datasets_summary['Dataset name'])
    wide_optimal.sort_values(['Dataset', 'Classifier', 'Metric'], inplace=True)

    # Save datasets
    imbalanced_datasets_summary.to_csv(join(results_path, 'imbalanced_datasets_summary.csv'), index=False)
    for name, result in results.items():
        result.to_csv(join(results_path, '%s.csv' % name), index=(name == 'aggregated'))

    # Calculate mean score of oversamplers by classifier and metric
    mean_score_clf_metric = wide_optimal.groupby(['Classifier', 'Metric']).mean().reset_index()

    # Calculate percentage difference in mean scores between SMOTE and G-SMOTE by classifier and metric
    perc_diff_mean_scores = pd.DataFrame((100 * (mean_score_clf_metric['G-SMOTE'] - mean_score_clf_metric['SMOTE']) /
                                          mean_score_clf_metric['NO OVERSAMPLING']), columns=['Difference %'])
    perc_diff_mean_scores = pd.concat([mean_score_clf_metric.iloc[:, :2], perc_diff_mean_scores], axis=1)

    # Mean scores
    x = [(clf, smpl) for clf in classifiers_names for smpl in oversamplers_names]
    for name in metrics_names:
        data = {'classifiers_names': classifiers_names}
        data.update(dict(mean_score_clf_metric.loc[mean_score_clf_metric.Metric == name, oversamplers_names].items()))
        counts = sum(zip(data['NO OVERSAMPLING'], data['RANDOM OVERSAMPLING'], data['SMOTE'], data['G-SMOTE']), ())
        source = ColumnDataSource(data=dict(x=x, counts=counts))

        p = figure(x_range=FactorRange(*x), y_range=(0.0, 1.0001), plot_height=650, plot_width=1500,
                   title='Mean %s' % name, toolbar_location=None, tools='')
        p.vbar(x='x', top='counts', width=0.9, source=source,
               fill_color=factor_cmap('x', palette=Category20[4], factors=oversamplers_names, start=1, end=2))
        p.y_range.start = 0
        p.x_range.range_padding = 0.1
        p.xaxis.major_label_orientation = 1.2
        p.title.text_font_size = '20pt'
        p.xaxis.major_label_text_font_size = '11pt'
        p.xaxis.group_text_font_size = '14pt'
        p.output_backend = 'svg'
        export_svgs(p, filename='%s.svg' % name)

    # G-SMOTE and SMOTE percentage difference
    data = {'classifiers_names': classifiers_names}
    for name in metrics_names:
        data[name] = perc_diff_mean_scores.loc[perc_diff_mean_scores.Metric == name, 'Difference %']
    x = [(clf, name) for clf in classifiers_names for name in metrics_names]
    counts = sum(zip(data['AUC'], data['F-SCORE'], data['G-MEAN']), ())
    source = ColumnDataSource(data=dict(x=x, counts=counts))
    p = figure(x_range=FactorRange(*x), y_range=(0.0, 8.001), plot_height=650, plot_width=1500,
               title='G-SMOTE and SMOTE percentage difference', toolbar_location=None, tools='')
    p.vbar(x='x', top='counts', width=0.9, source=source,
           fill_color=factor_cmap('x', palette=Spectral[3], factors=metrics_names, start=1, end=2))
    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = 1.0
    p.title.text_font_size = '20pt'
    p.xaxis.major_label_text_font_size = '11pt'
    p.xaxis.group_text_font_size = '14pt'
    p.output_backend = 'svg'
    export_svgs(p, filename='percentage_difference.svg')

    # Hyperparameters analysis
    aggregated = results['aggregated'] #aggregated = pd.read_csv('aggregated.csv', header =[0,1], index_col=[0,1,2,3])
    aggregated = aggregated.iloc[:, ::2].reset_index()
    aggregated.columns = aggregated.columns.get_level_values(0)
    cols = aggregated.columns.tolist()
    agg_measures = {score: max for score in cols[4:]}
    optimal_params = aggregated.groupby(cols[:3]).agg(agg_measures).reset_index()
    params = pd.Series()
    for metric in cols[4:]:
        group_cols = cols[:3] + [metric]
        optimal_params_group = pd.merge(optimal_params[group_cols], aggregated[group_cols + ['params']], on=group_cols, how='left')
        optimal_params_group = optimal_params_group.loc[optimal_params_group.Oversampler == 'G-SMOTE', 'params']
        params = params.append(optimal_params_group)
    params = params.apply(literal_eval).reset_index(drop=True)
    hyperparameters = ['selection_strategy', 'truncation_factor', 'deformation_factor', 'k_neighbors']
    params = params.apply(lambda grid: {sub('G-SMOTE__', '', param): value for param, value in grid.items() if match('G-SMOTE', param)})
    params = params.apply(lambda grid: [grid[param] for param in hyperparameters])
    params = pd.DataFrame(params.values.tolist(), columns=hyperparameters)
    param_values_map = {
        'selection_strategy': ['Minority', 'Majority', 'Combined'],
        'truncation_factor': ['-1.0', '-0.5', '0.0', '0.25', '0.5', '0.75', '1.0'],
        'deformation_factor': ['0.0', '0.2', '0.4', '0.5', '0.6', '0.8', '1.0']
    }
    characteristic_cases = pd.DataFrame(
        {
            'case': ['SMOTE', 'Majority selection SMOTE', 'Combined selection SMOTE', 'Inverse SMOTE',
                     'Hyper-sphere SMOTE', 'Half hyper-sphere SMOTE'],
            'selection_strategy': ['minority', 'majority', 'combined', 'minority', 'minority', 'minority'],
            'truncation_factor': [1.0, 1.0, 1.0, -1.0, 0.0, 1.0],
            'deformation_factor': [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        }
    )
    params_ranking = params.groupby(params.columns[:-1].tolist()).size().reset_index()
    params_ranking['rank'] = rankdata(params_ranking[0], 'min')
    cases_rank = pd.merge(characteristic_cases, params_ranking).loc[:, ['case', 'rank']]


    for param_name in params.columns[:-1]:
        param_count = Counter(params[param_name])
        x = param_values_map[param_name]
        p = figure(x_range=x, y_range=(0, 100.01), plot_height=250,
                   title=sub('_', ' ', param_name).capitalize(),
                   toolbar_location=None, tools="")
        p.vbar(x=x,
               top=[100 * param_count[value.lower() if param_name == 'selection_strategy' else float(value)] / len(params) for value in x],
               width=0.9)
        p.y_range.start = 0
        p.x_range.range_padding = 0.1
        p.xaxis.major_label_orientation = 1.0
        p.title.text_font_size = '12pt'
        p.xaxis.major_label_text_font_size = '11pt'
        p.xaxis.group_text_font_size = '14pt'
        p.output_backend = 'svg'
        export_svgs(p, filename='%s.svg' % param_name)

    x = cases_rank['case'].values
    p = figure(x_range=x, plot_height=550, title='Rank', toolbar_location=None, tools="")
    p.vbar(x=x, top=cases_rank['rank'], width=0.9)
    max_rank = round(cases_rank['rank'].max(), -1)
    ticker = list(range(0, max_rank + 10, 10))
    p.yaxis.ticker = ticker
    p.yaxis.major_label_overrides = {tick: str(max_rank - tick) for tick in ticker}
    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = 1.0
    p.title.text_font_size = '12pt'
    p.xaxis.major_label_text_font_size = '11pt'
    p.xaxis.group_text_font_size = '14pt'
    p.output_backend = 'svg'
    export_svgs(p, filename='cases_rank.svg')