"""
Analyse the experiment's results of the
Geometric SMOTE journal paper.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from os.path import join, dirname
from collections import Counter
from re import match, sub
from ast import literal_eval

from scipy.stats import rankdata
import pandas as pd
from bokeh.io import export_svgs
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.palettes import Category20, Spectral

RESULTS_PATH = join(dirname(__file__), '..', '..', 'data', 'results', 'gsmote-journal')


def rename_metrics(df):
    metrics_names_mapping = {'roc_auc': 'AUC', 'f1': 'F-SCORE', 'geometric_mean_score': 'G-MEAN'}
    df['Metric'] = df['Metric'].apply(lambda metric: metrics_names_mapping[metric])
    return df

def calculate_mean_score(wide_optimal):
    
    # Sort classifier and metric
    categories = {}
    for column in ['Classifier', 'Metric']:
        categories.update({column: wide_optimal[column].unique()})
        wide_optimal[column] = pd.Categorical(wide_optimal[column], categories=categories[column])

    # Rename metrics
    wide_optimal = rename_metrics(wide_optimal)

    # Calculate mean score
    mean_score_clf_metric = wide_optimal.groupby(['Classifier', 'Metric']).mean().reset_index()

    return mean_score_clf_metric


def calculate_perc_diff_mean_scores(mean_score_clf_metric):
    perc_diff_mean_scores = pd.DataFrame((100 * (mean_score_clf_metric['G-SMOTE'] - mean_score_clf_metric['SMOTE']) /
                                          mean_score_clf_metric['NO OVERSAMPLING']), columns=['Difference %'])
    perc_diff_mean_scores = pd.concat([mean_score_clf_metric.iloc[:, :2], perc_diff_mean_scores], axis=1)
    return perc_diff_mean_scores


def plot_perc_diff_mean_scores(perc_diff_mean_scores):
    classifiers_names = perc_diff_mean_scores['Classifier'].unique().tolist()
    metrics_names = perc_diff_mean_scores['Metric'].unique().tolist()
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
    export_svgs(p, filename=join(RESULTS_PATH, 'percentage_difference.svg'))


def plot_hyperparameters_analysis(aggregated):
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
        export_svgs(p, filename=join(RESULTS_PATH, '%s.svg' % param_name))

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
    export_svgs(p, filename=join(RESULTS_PATH, 'cases_rank.svg'))


if __name__ == '__main__':

    # Load datasets
    datasets = {}
    csv_filenames = ['imbalanced_datasets_summary', 'aggregated', 'wide_optimal', 'mean_ranking', 'friedman_test', 'adjusted_pvalues']
    for name in csv_filenames: 
        datasets[name] = pd.read_csv(
            join(RESULTS_PATH, '%s.csv' % name), 
            header='infer' if name != 'aggregated' else [0, 1],
            index_col=None if name != 'aggregated' else [0, 1, 2, 3]
        )

    # Extract results
    mean_score_clf_metric = calculate_mean_score(datasets['wide_optimal'])
    perc_diff_mean_scores = calculate_perc_diff_mean_scores(mean_score_clf_metric)
    for name in csv_filenames[3:]:
        datasets[name] = rename_metrics(datasets[name])
    
    # Save tables and plots
    mean_score_clf_metric.to_csv(join(RESULTS_PATH, 'mean_score_clf_metric_table.csv'), index=False)
    plot_perc_diff_mean_scores(perc_diff_mean_scores)
    datasets['mean_ranking'].to_csv(join(RESULTS_PATH, 'mean_ranking_table.csv'), index=False)
    datasets['friedman_test'].to_csv(join(RESULTS_PATH, 'friedman_test_table.csv'), index=False)
    datasets['adjusted_pvalues'].to_csv(join(RESULTS_PATH, 'adjusted_pvalues_table.csv'), index=False)
    plot_hyperparameters_analysis(datasets['aggregated'])