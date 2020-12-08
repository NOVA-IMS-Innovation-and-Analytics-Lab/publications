"""
Analyze the experimental results.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
#         João Fonseca <jpmrfonseca@gmail.com>
# License: MIT

import sys
from os.path import join, dirname
from collections import OrderedDict
from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from rlearn.tools import (
    combine_results,
    select_results,
    summarize_datasets,
    calculate_wide_optimal,
    calculate_mean_sem_scores,
    calculate_mean_sem_perc_diff_scores,
    calculate_mean_sem_ranking,
    apply_friedman_test,
    apply_holms_test
)
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
sys.path.append(join(dirname(__file__),'..', '..', '..'))
from utils import (
    load_datasets,
    generate_paths,
    generate_pvalues_tbl,
    generate_mean_std_tbl,
    sort_tbl,
    make_bold,
    RemoteSensingDatasets
)

RESULTS_NAMES = ('none', 'ros', 'smote', 'bsmote', 'ksmote')
OVRS_NAMES = ('NONE', 'ROS', 'SMOTE', 'B-SMOTE', 'K-SMOTE')
CLFS_NAMES = ('LR', 'KNN', 'RF')
METRICS_MAPPING = dict([('accuracy', 'Accuracy'), ('f1_macro', 'F-score'), ('geometric_mean_score_macro', 'G-mean')])
DATASETS_NAMES = ('indian_pines', 'salinas', 'salinas_a', 'pavia_centre', 'pavia_university', 'kennedy_space_center', 'botswana')


def _make_bold_stat_signif(value, sig_level=.05):
    """Make bold the lowest or highest value(s)."""

    val = '{%.1e}' % value
    val = (
        '\\textbf{%s}' % val
        if value <= sig_level
        else val
    )
    return val

def generate_pvalues_tbl_bold(tbl, sig_level=.05):
    """Format p-values."""
    for name in tbl.dtypes[tbl.dtypes == float].index:
        tbl[name] = tbl[name].apply(
            lambda pvalue: _make_bold_stat_signif(pvalue, sig_level)
        )
    return tbl


def _make_bold(row, maximum=True, num_decimals=2):
    """Make bold the lowest or highest value(s)."""
    row = round(row, num_decimals)
    val = row.max() if maximum else row.min()
    mask = (row == val)
    formatter = '{0:.%sf}' % num_decimals
    row = row.apply(lambda el: formatter.format(el))
    row[mask] = '\\textbf{%s' % formatter.format(val)
    return row, mask


def calculate_max_improvement(results, oversamplers=None):

    # Calculate wide optimal results
    res = calculate_wide_optimal(results)
    ovrs_names = res.columns[3:]

    # Extract oversamplers
    control, test = (
        oversamplers if oversamplers is not None else ovrs_names[-2:]
    )

    # Calculate percentage difference
    res['Difference'] = (
        (res[test] - res[control]) / res[control]
    ) * 100

    return res\
        .groupby(['Classifier', 'Metric'])[['Difference']]\
        .max()\
        .reset_index()


def generate_mean_std_tbl_bold(mean_vals, std_vals, maximum=True, decimals=2):
    """Generate table that combines mean and sem values."""
    index = mean_vals.iloc[:, :2]
    mean_bold = mean_vals.iloc[:, 2:].apply(lambda row: _make_bold(row, maximum, decimals)[0], axis=1)
    mask = mean_vals.iloc[:, 2:].apply(lambda row: _make_bold(row, maximum, decimals)[1], axis=1).values
    std_bold = np.round(std_vals.iloc[:, 2:], decimals).astype(str)
    std_bold = np.where(mask, std_bold+'}', std_bold)
    scores = mean_bold + r" $\pm$ "  + std_bold

    tbl = pd.concat([index, scores], axis=1)
    return tbl

def generate_main_results():
    """Generate the main results of the experiment."""

    wide_optimal = sort_tbl(
        calculate_wide_optimal(results),
        ovrs_order=OVRS_NAMES, clfs_order=CLFS_NAMES
    )\
        .set_index(['Dataset', 'Classifier', 'Metric'])\
        .apply(lambda row: make_bold(row, num_decimals=3), axis=1)\
        .reset_index()
    wide_optimal['Dataset'] = wide_optimal['Dataset'].apply(
        lambda x: x.title()
        if len(x.split(' ')) == 1
        else ''.join([w[0] for w in x.split(' ')])
    )

    mean_sem_scores = sort_tbl(
        generate_mean_std_tbl_bold(*calculate_mean_sem_scores(results), maximum=True, decimals=3),
        ovrs_order=OVRS_NAMES, clfs_order=CLFS_NAMES
    )
    mean_sem_perc_diff_scores = sort_tbl(
        generate_mean_std_tbl(*calculate_mean_sem_perc_diff_scores(results, ['SMOTE', 'K-SMOTE'])),
        ovrs_order=OVRS_NAMES, clfs_order=CLFS_NAMES
    )
    mean_sem_ranking = sort_tbl(
        generate_mean_std_tbl_bold(*calculate_mean_sem_ranking(results), maximum=False),
        ovrs_order=OVRS_NAMES, clfs_order=CLFS_NAMES
    )
    main_results_names = ('wide_optimal', 'mean_sem_scores', 'mean_sem_perc_diff_scores', 'mean_sem_ranking')

    return zip(main_results_names, (wide_optimal, mean_sem_scores, mean_sem_perc_diff_scores, mean_sem_ranking))

def generate_statistical_results():
    """Generate the statistical results of the experiment."""

    friedman_test = sort_tbl(generate_pvalues_tbl(apply_friedman_test(results)), ovrs_order=OVRS_NAMES, clfs_order=CLFS_NAMES)
    holms_test = sort_tbl(generate_pvalues_tbl_bold(apply_holms_test(results, control_oversampler='K-SMOTE')), ovrs_order=OVRS_NAMES[:-1], clfs_order=CLFS_NAMES)
    statistical_results_names = ('friedman_test', 'holms_test')
    statistical_results = zip(statistical_results_names, (friedman_test, holms_test))

    return statistical_results


def load_plt_sns_configs(font_size=8):
    """
    Load LaTeX style configurations for Matplotlib/Seaborn
    Visualizations.
    """
    sns.set_style('whitegrid')
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": (10/8)*font_size,
        "font.size": (10/8)*font_size,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        # Subplots size/shape
        "figure.subplot.left":.098,
        "figure.subplot.right":.938,
        "figure.subplot.bottom":.12,
        "figure.subplot.top":.944,
        "figure.subplot.wspace":.071,
        "figure.subplot.hspace":.2
    }
    plt.rcParams.update(tex_fonts)


def make_mean_rank_bar_chart():
    """Generates bar chart."""

    load_plt_sns_configs()

    ranks = calculate_mean_sem_ranking(results)[0]
    ranks['Metric'] = ranks['Metric'].apply(lambda x: METRICS_MAPPING[x])
    fig, axes = plt.subplots(
        ranks.Classifier.unique().shape[0],
        ranks.Metric.unique().shape[0],
        figsize=(5,6)
    )
    lranks = ranks.set_index(['Classifier', 'Metric'])
    for (row, clf), (col, metric) in product(
            enumerate(ranks.Classifier.unique()),
            enumerate(ranks.Metric.unique())
    ):
        dat = len(OVRS_NAMES) - lranks.loc[(clf,metric)].loc[list(OVRS_NAMES[::-1])]
        axes[row, col].bar(
            dat.index,
            dat.values,
            color=['indianred']+['steelblue' for i in range(len(OVRS_NAMES)-1)]
        )
        plt.sca(axes[row, col])
        plt.yticks(range(len(OVRS_NAMES)), [None]+list(range(1, len(OVRS_NAMES)))[::-1])
        plt.xticks(rotation=90)
        if row == 0:
            plt.title(metric)
        if col == 0:
            plt.ylabel(f'{clf}')
        if row != len(ranks.Classifier.unique())-1:
            plt.xticks(range(len(OVRS_NAMES)), [])
        if col != 0:
            plt.yticks(range(len(OVRS_NAMES)), [])
        sns.despine(left=True)
        plt.grid(b=None, axis='x')

    fig.savefig(join(analysis_path, 'mean_rankings_bar_chart.pdf'),
            format='pdf', bbox_inches='tight')
    plt.close()


def make_score_heatmaps():
    """Generates score heatmaps (mean and max oercentage differences)."""

    # Visualization Formatting
    sns.set(rc={'figure.figsize':(3,2)})
    load_plt_sns_configs()

    # Get data
    fnames = [
        'mean_score_improvement_heatmap',
        'max_score_improvement_heatmap'
    ]
    perc_diffs = [
        calculate_mean_sem_perc_diff_scores(results, ['SMOTE', 'K-SMOTE'])[0],
        calculate_max_improvement(results, ['SMOTE', 'K-SMOTE'])
    ]

    # Plot and save results
    for fname, dat in zip(fnames, perc_diffs):
        dat['Metric'] = dat['Metric'].apply(lambda x: METRICS_MAPPING[x])
        dat['Difference'] = dat['Difference'].apply(lambda x: np.round(x, 2))
        sns.heatmap(
            dat.pivot('Classifier', 'Metric', 'Difference'),
            annot=True,
            center=0,
            cmap='RdBu_r'
        )
        plt.savefig(join(analysis_path, f'{fname}.pdf'),
                format='pdf',bbox_inches='tight')
        plt.close()


def summarize_multiclass_datasets(datasets):
    summarized = summarize_datasets(datasets)\
        .rename(columns={'Dataset name':'Dataset', 'Imbalance Ratio': 'IR'})\
        .set_index('Dataset')\
        .join(pd.Series(dict([(name, dat[-1].unique().size) for name, dat in datasets]), name='Classes'))\
        .reset_index()\
        .rename(columns={
            'Minority instances': 'Min. Instances',
            'Majority instances': 'Maj. Instances'
        })
    summarized.loc[:, 'Dataset'] = summarized.loc[:, 'Dataset']\
        .apply(lambda x: x.title())
    return summarized


def plot_lulc_images():
    arrays_x = []
    arrays_y = []
    for dat_name in DATASETS_NAMES:
        X, y = RemoteSensingDatasets()._load_gic_dataset(dat_name)
        arrays_x.append(X[:,:,100])
        arrays_y.append(np.squeeze(y))

    for X, y, figname in zip(arrays_x, arrays_y, DATASETS_NAMES):
        plt.figure(
            figsize=(20,10),
            dpi=320
        )
        if figname == 'kennedy_space_center':
            X = np.clip(X, 0, 350)
        for i, (a, cmap) in enumerate(zip([X, y],['gist_gray','terrain'])):
            plt.subplot(2, 1, i+1)
            plt.imshow(
                a, cmap=plt.get_cmap(cmap)
            )
            plt.axis('off')
        plt.savefig(join(analysis_path, figname), bbox_inches='tight', pad_inches = 0)

def make_resampling_example():
    def create_dataset(n_samples=100, weights=(.05, .95), n_classes=2,
                       class_sep=0.5, n_clusters=1):
        return make_classification(n_samples=n_samples, n_features=2,
                                   n_informative=2, n_redundant=0, n_repeated=0,
                                   n_classes=n_classes,
                                   n_clusters_per_class=n_clusters,
                                   weights=list(weights),
                                   class_sep=class_sep, random_state=21)

    def plot_decision_function(X, y, clf, ax, label=None, cmap="cividis"):

        plot_step = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.4)
        ax.scatter(X[:, 0], X[:, 1], s=15, alpha=0.8, c=y, cmap=cmap, edgecolor='k')

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.sca(ax)
        plt.xlabel(label)
        sns.despine(left=True, right=True, bottom=True, top=True)
        plt.grid(b=False, axis='both')

    load_plt_sns_configs(14)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    X, y = create_dataset(n_samples=700, weights=(0.1, 0.9))
    query = np.array([(X[:,0]<3),(X[:,1]>-3)]).all(0)
    X, y = X[query], y[query]

    clf = LinearSVC(random_state=0)
    clf.fit(X, y)
    plot_decision_function(X, y, clf, ax[0], '(a)')

    clf = make_pipeline(SMOTE(random_state=0), LinearSVC(random_state=0))
    clf.fit(X, y)
    plot_decision_function(X, y, clf, ax[1], '(b)')
    fig.tight_layout()

    plt.savefig(
        join(analysis_path, 'resampling_decision_function.pdf'),
        bbox_inches='tight',
        pad_inches = 0
    )

if __name__=='__main__':

    data_path, results_path, analysis_path = generate_paths()

    # load datasets
    datasets = load_datasets(data_dir=data_path)

    # load results
    results = []
    for name in RESULTS_NAMES:
        file_path = join(results_path, f'{name}.pkl')
        results.append(pd.read_pickle(file_path))

    # combine and select results
    results = combine_results(*results)
    results = select_results(results, oversamplers_names=OVRS_NAMES, classifiers_names=CLFS_NAMES)

    # datasets description
    summarize_multiclass_datasets(datasets).to_csv(join(analysis_path, 'datasets_description.csv'), index=False)

    # Main results
    main_results = generate_main_results()
    for name, result in main_results:
        result['Metric'] = result['Metric'].map(METRICS_MAPPING)
        result.to_csv(join(analysis_path, f'{name}.csv'), index=False)

    # Visualizations
    make_mean_rank_bar_chart()
    make_score_heatmaps()
    make_resampling_example()
    #plot_lulc_images() # Generating this viz can be time consuming

    # Statistical results
    statistical_results = generate_statistical_results()
    for name, result in statistical_results:
        result['Metric'] = result['Metric'].map(METRICS_MAPPING)
        result.to_csv(join(analysis_path, f'{name}.csv'), index=False)
