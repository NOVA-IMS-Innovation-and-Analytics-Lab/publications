from os.path import dirname, join
from pickle import load

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import make_scorer, SCORERS
from imblearn.metrics import geometric_mean_score
from rlearn.tools import (
    ImbalancedExperiment,
    combine_experiments,
    calculate_mean_sem_scores,
    calculate_mean_sem_perc_diff_scores,
    calculate_mean_sem_ranking,
    apply_friedman_test,
    apply_holms_test
)

from tools import EXPERIMENTS_PATH
from tools.format import generate_mean_std_tbl, generate_pvalues_tbl


UNDERSAMPLING_RATIO = [50, 75, 90, 95]
EXPERIMENTS_NAMES = [
    "no_oversampling",
    "random_oversampling",
    "smote",
    "borderline_smote",
    "gsmote",
    "benchmark_method",
]
RESOURCES_PATH = join(dirname(__file__), "resources")
STATISTICAL_RESULTS_NAMES = [
    'friedman_test',
    'holms_test'
]
SCORERS['geometric_mean_score'] = make_scorer(geometric_mean_score)
METRICS_NAMES_MAPPING = {
    "roc_auc": "AUC",
    "f1": "F-SCORE",
    "geometric_mean_score": "G-MEAN",
    "geometric_mean_macro_score": "G-MEAN MACRO",
    "f1_macro": "F-SCORE MACRO",
    "accuracy": "ACCURACY",
}
MEAN_SEM_SCORES = pd.DataFrame()
MEAN_RANKING = pd.DataFrame()
MEAN_PERC_DIFF = pd.DataFrame()


def generate_experiment(ratio):
    """Generate an experiment including all oversamplers."""

    # Load experiments object
    experiments = []
    for name in EXPERIMENTS_NAMES:
        file_path = join(EXPERIMENTS_PATH, f"small_data_{ratio}", f"{name}.pkl")
        with open(file_path, "rb") as file:
            experiments.append(load(file))

    # Create combined experiment
    experiment = combine_experiments("combined", *experiments)

    # Exclude constant classifier
    mask_clf = (experiment.results_.reset_index().Classifier != "CONSTANT CLASSIFIER")
    mask = mask_clf.values
    experiment.results_ = experiment.results_[mask]

    # Modify attributes
    experiment.classifiers = experiment.classifiers[1:]
    experiment.classifiers_names_ = experiment.classifiers_names_[1:]

    return experiment


def generate_mean_sem_scores(ratio):
    """Generate mean scores inkl. standard deviation values."""

    # Generate experiment
    experiment = generate_experiment(ratio)

    # Recreate optimal and wide optimal results
    experiment._calculate_optimal_results()._calculate_wide_optimal_results()

    # Calculate results
    mean_sem_scores = generate_mean_std_tbl(*calculate_mean_sem_scores(experiment))

    return mean_sem_scores


def generate_mean_sem_perc_diff_scores(ratio):
    """Generate percentage difference between G-SMOTE and SMOTE."""

    # Generate experiment
    experiment = generate_experiment(ratio)

    # Recreate optimal and wide optimal results
    experiment._calculate_optimal_results()._calculate_wide_optimal_results()

    # Calculate results
    mean_sem_perc_diff_scores = generate_mean_std_tbl(
        *calculate_mean_sem_perc_diff_scores(experiment, ["SMOTE", "G-SMOTE"])
    )

    return mean_sem_perc_diff_scores


def generate_mean_tbl(mean_vals, std_vals):
    """Generate mean table that takes out std values for mean ranking graph."""

    index = mean_vals.iloc[:, :2]
    scores = mean_vals.iloc[:, 2:].applymap("{:,.2f}".format)
    tbl = pd.concat([index, scores], axis=1)
    tbl["Metric"] = tbl["Metric"].apply(lambda metric: METRICS_NAMES_MAPPING[metric])

    return tbl


def generate_mean_ranking(ratio):
    """Generate mean rankings without std values."""

    # Generate experiment
    experiment = generate_experiment(ratio)

    # Recreate optimal and wide optimal results
    experiment._calculate_optimal_results()._calculate_wide_optimal_results()

    # Calculate results
    mean_sem_ranking = generate_mean_tbl(*calculate_mean_sem_ranking(experiment))

    return mean_sem_ranking


def generate_plot(MEAN_RANKING):
    """Generate mean ranking plot."""

    # Create tables for generation of subplots

    # Logistic Regression
    LR_accuracy = MEAN_RANKING[
        (MEAN_RANKING["Metric"] == "ACCURACY") & (MEAN_RANKING["Classifier"] == "LR")
    ]
    LR_accuracy = LR_accuracy[
        [
            "Ratio",
            "G-SMOTE",
            "SMOTE",
            "BORDERLINE SMOTE",
            "RANDOM OVERSAMPLING",
            "NO OVERSAMPLING",
            "BENCHMARK METHOD",
        ]
    ]
    LR_accuracy.set_index("Ratio", drop=True, inplace=True, verify_integrity=True)

    # K-Nearest Neighbors
    KNN_accuracy = MEAN_RANKING[
        (MEAN_RANKING["Metric"] == "ACCURACY") & (MEAN_RANKING["Classifier"] == "KNN")
    ]
    KNN_accuracy = KNN_accuracy[
        [
            "Ratio",
            "G-SMOTE",
            "SMOTE",
            "BORDERLINE SMOTE",
            "RANDOM OVERSAMPLING",
            "NO OVERSAMPLING",
            "BENCHMARK METHOD",
        ]
    ]
    KNN_accuracy.set_index("Ratio", drop=True, inplace=True, verify_integrity=True)

    # Decision Tree
    DT_accuracy = MEAN_RANKING[
        (MEAN_RANKING["Metric"] == "ACCURACY") & (MEAN_RANKING["Classifier"] == "DT")
    ]
    DT_accuracy = DT_accuracy[
        [
            "Ratio",
            "G-SMOTE",
            "SMOTE",
            "BORDERLINE SMOTE",
            "RANDOM OVERSAMPLING",
            "NO OVERSAMPLING",
            "BENCHMARK METHOD",
        ]
    ]
    DT_accuracy.set_index("Ratio", drop=True, inplace=True, verify_integrity=True)

    # Gradient Boosting
    GBC_accuracy = MEAN_RANKING[
        (MEAN_RANKING["Metric"] == "ACCURACY") & (MEAN_RANKING["Classifier"] == "GBC")
    ]
    GBC_accuracy = GBC_accuracy[
        [
            "Ratio",
            "G-SMOTE",
            "SMOTE",
            "BORDERLINE SMOTE",
            "RANDOM OVERSAMPLING",
            "NO OVERSAMPLING",
            "BENCHMARK METHOD",
        ]
    ]
    GBC_accuracy.set_index("Ratio", drop=True, inplace=True, verify_integrity=True)

    # Create plot
    sns.set()
    sns.set_style("darkgrid")
    blacknwhite = sns.color_palette(
        ["#696969", "#808080", "#A9A9A9", "#C0C0C0", "#D3D3D3", "#000000"]
    )
    sns.set_palette(blacknwhite)

    fig, axes = plt.subplots(figsize=(15, 10), ncols=2, nrows=2)
    fig.subplots_adjust(hspace=0.4)
    fig.suptitle("Mean ranking per classifier (Accuracy)")

    sns.lineplot(data=LR_accuracy, markers=True, dashes=True, ax=axes[0][0])
    sns.lineplot(data=KNN_accuracy, markers=True, dashes=True, ax=axes[0][1])
    sns.lineplot(data=DT_accuracy, markers=True, dashes=True, ax=axes[1][0])
    sns.lineplot(data=GBC_accuracy, markers=True, dashes=True, ax=axes[1][1])

    # Logisitic Regression subplot
    axes[0][0].set_title("Logistic Regression")
    axes[0][0].set_ylabel("Mean Ranking")
    axes[0][0].set_xlabel("Undersampling Ratio (%)")
    axes[0][0].set_ylim(6, 0)
    axes[0][0].legend(
        [
            "G-SMOTE",
            "SMOTE",
            "Borderline SMOTE",
            "Random Oversampling",
            "No Oversampling",
            "Benchmark",
        ],
        loc="lower left",
    )

    # K-Nearest Neighbors subplot
    axes[0][1].set_title("K-Nearest Neighbors")
    axes[0][1].set_ylabel("Mean Ranking")
    axes[0][1].set_xlabel("Undersampling Ratio (%)")
    axes[0][1].set_ylim(6, 0)
    axes[0][1].legend(
        [
            "G-SMOTE",
            "SMOTE",
            "Borderline SMOTE",
            "Random Oversampling",
            "No Oversampling",
            "Benchmark",
        ],
        loc="lower left",
    )

    # Decision Tree subplot
    axes[1][0].set_title("Decision Tree")
    axes[1][0].set_ylabel("Mean Ranking")
    axes[1][0].set_xlabel("Undersampling Ratio (%)")
    axes[1][0].set_ylim(6, 0)
    axes[1][0].legend(
        [
            "G-SMOTE",
            "SMOTE",
            "Borderline SMOTE",
            "Random Oversampling",
            "No Oversampling",
            "Benchmark",
        ],
        loc="lower left",
    )

    # Gradient Boosting subplot
    axes[1][1].set_title("Gradient Boosting")
    axes[1][1].set_ylabel("Mean Ranking")
    axes[1][1].set_xlabel("Undersampling Ratio (%)")
    axes[1][1].set_ylim(6, 0)
    axes[1][1].legend(
        [
            "G-SMOTE",
            "SMOTE",
            "Borderline SMOTE",
            "Random Oversampling",
            "No Oversampling",
            "Benchmark",
        ],
        loc="lower left",
    )

    # Print plot
    plt.savefig(join(RESOURCES_PATH, "mean_ranking_per_classifier_accuracy.png"))
    plt.close()


def generate_statistical_experiment(ratio):
    """Generate an experiment including all oversamplers."""

    # Load experiments object
    experiments = []
    for name in EXPERIMENTS_NAMES:
        file_path = join(EXPERIMENTS_PATH, f'small_data_{ratio}', f'{name}.pkl')
        with open(file_path, 'rb') as file:
            experiments.append(load(file))

    # Create combined experiment
    experiment = combine_experiments('combined', *experiments)

    # Exclude constant classifier and benchmark method
    mask_clf = (experiment.results_.reset_index().Classifier != 'CONSTANT CLASSIFIER')
    mask_ovr = (experiment.results_.reset_index().Oversampler != 'BENCHMARK METHOD')
    mask = (mask_clf & mask_ovr).values
    experiment.results_ = experiment.results_[mask]

    # Modify attributes
    experiment.classifiers = experiment.classifiers[1:]
    experiment.classifiers_names_ = experiment.classifiers_names_[1:]
    experiment.oversamplers = experiment.oversamplers[:-1]
    experiment.oversamplers_names_ = experiment.oversamplers_names_[:-1]

    return experiment


def generate_statistical_results():
    """Generate the statistical results of the experiment."""

    # Combine experiments objects
    experiment_results, datasets = [], []
    for ratio in UNDERSAMPLING_RATIO:

        # Generate experiment with all oversamplers
        experiment = generate_statistical_experiment(ratio)

        # Populate datasets
        datasets += [(f'{name}({ratio})', (X, y)) for name, (X, y) in experiment.datasets]

        # Extract results
        results = experiment.results_.reset_index()
        results['Dataset'] = results['Dataset'].apply(lambda name: f'{name}({ratio})')
        results.set_index(['Dataset', 'Oversampler', 'Classifier', 'params'], inplace=True)
        results.columns = experiment.results_.columns
        experiment_results.append(results)

    experiment_results = pd.concat(experiment_results)

    # Create combined experiment
    combined_experiment = ImbalancedExperiment(
        'experiment',
        datasets,
        experiment.oversamplers,
        experiment.classifiers,
        experiment.scoring,
        experiment.n_splits,
        experiment.n_runs,
        experiment.random_state,
    )
    combined_experiment._initialize(-1, 0)
    combined_experiment.results_ = experiment_results

    # Calculate optimal and wide optimal results
    combined_experiment._calculate_optimal_results()._calculate_wide_optimal_results()

    # Calculate results
    friedman_test = generate_pvalues_tbl(apply_friedman_test(combined_experiment))
    holms_test = generate_pvalues_tbl(apply_holms_test(combined_experiment, control_oversampler='G-SMOTE'))

    return friedman_test, holms_test


if __name__ == "__main__":

    # Statistical results
    statistical_results = generate_statistical_results()
    for name, result in zip(STATISTICAL_RESULTS_NAMES, statistical_results):
        result.to_csv(join(RESOURCES_PATH, f'{name}.csv'), index=False)

    # Generate mean_sem_scores.csv
    for ratio in UNDERSAMPLING_RATIO:
        mean_sem_scores = generate_mean_sem_scores(ratio)
        df_scores = pd.DataFrame(mean_sem_scores)
        df_scores["Ratio"] = ratio
        df_scores = df_scores[
            [
                "Ratio",
                "Classifier",
                "Metric",
                "NO OVERSAMPLING",
                "RANDOM OVERSAMPLING",
                "SMOTE",
                "BORDERLINE SMOTE",
                "G-SMOTE",
                "BENCHMARK METHOD",
            ]
        ]
        df_scores.columns = [
            "Ratio",
            "Classifier",
            "Metric",
            "NONE",
            "RANDOM",
            "SMOTE",
            "B-SMOTE",
            "G-SMOTE",
            "BENCHMARK",
        ]
        MEAN_SEM_SCORES = MEAN_SEM_SCORES.append(df_scores, ignore_index=True)

    MEAN_SEM_SCORES.to_csv(join(RESOURCES_PATH, "mean_sem_scores.csv"), index=False)

    # Generate mean_sem_perc_diff.csv
    for ratio in UNDERSAMPLING_RATIO:
        mean_perc_diff = generate_mean_sem_perc_diff_scores(ratio)
        df_perc_diff = pd.DataFrame(mean_perc_diff)
        df_perc_diff["Ratio"] = ratio
        df_perc_diff = df_perc_diff[["Ratio", "Classifier", "Metric", "Difference"]]
        MEAN_PERC_DIFF = MEAN_PERC_DIFF.append(df_perc_diff, ignore_index=True)

    MEAN_PERC_DIFF.to_csv(
        join(RESOURCES_PATH, "mean_sem_perc_diff_scores.csv"), index=False
    )

    # Generate mean_ranking.csv
    for ratio in UNDERSAMPLING_RATIO:
        mean_ranking = generate_mean_ranking(ratio)
        df_ranking = pd.DataFrame(mean_ranking)
        df_ranking["Ratio"] = ratio
        df_ranking = df_ranking[
            [
                "Ratio",
                "Classifier",
                "Metric",
                "NO OVERSAMPLING",
                "RANDOM OVERSAMPLING",
                "SMOTE",
                "BORDERLINE SMOTE",
                "G-SMOTE",
                "BENCHMARK METHOD",
            ]
        ]
        MEAN_RANKING = MEAN_RANKING.append(df_ranking, ignore_index=True)

    # Format numbers
    MEAN_RANKING["Ratio"] = MEAN_RANKING["Ratio"].astype(int)
    MEAN_RANKING["NO OVERSAMPLING"] = MEAN_RANKING["NO OVERSAMPLING"].astype(float)
    MEAN_RANKING["RANDOM OVERSAMPLING"] = MEAN_RANKING["RANDOM OVERSAMPLING"].astype(float)
    MEAN_RANKING["SMOTE"] = MEAN_RANKING["SMOTE"].astype(float)
    MEAN_RANKING["BORDERLINE SMOTE"] = MEAN_RANKING["BORDERLINE SMOTE"].astype(float)
    MEAN_RANKING["G-SMOTE"] = MEAN_RANKING["G-SMOTE"].astype(float)
    MEAN_RANKING["BENCHMARK METHOD"] = MEAN_RANKING["BENCHMARK METHOD"].astype(float)

    # Save mean ranking scores to csv
    MEAN_RANKING.to_csv(join(RESOURCES_PATH, "mean_ranking.csv"), index=False)

    # Generate mean ranking plot
    generate_plot(MEAN_RANKING)
