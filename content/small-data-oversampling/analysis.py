from os.path import dirname, join
from pickle import load

import pandas as pd
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
    'no_oversampling', 
    'random_oversampling',
    'smote',
    'borderline_smote',
    'gsmote',
    'benchmark_method'
]
RESOURCES_PATH = join(dirname(__file__), 'resources')
MAIN_RESULTS_NAMES = [
    'mean_sem_scores', 
    'mean_sem_perc_diff_scores', 
    'mean_sem_ranking'
]
STATISTICAL_RESULTS_NAMES = [
    'friedman_test',
    'holms_test'
]
SCORERS['geometric_mean_score'] = make_scorer(geometric_mean_score)


def generate_experiment(ratio):
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
    

def generate_main_results(ratio):
    """Generate the main results of the experiment."""

    # Generate experiment
    experiment = generate_experiment(ratio)

    # Recreate optimal and wide optimal results
    experiment._calculate_optimal_results()._calculate_wide_optimal_results()

    # Calculate results
    mean_sem_scores = generate_mean_std_tbl(*calculate_mean_sem_scores(experiment))
    mean_sem_perc_diff_scores = generate_mean_std_tbl(*calculate_mean_sem_perc_diff_scores(experiment, ['SMOTE', 'G-SMOTE']))
    mean_sem_ranking = generate_mean_std_tbl(*calculate_mean_sem_ranking(experiment))
    
    return mean_sem_scores, mean_sem_perc_diff_scores, mean_sem_ranking


def generate_statistical_results():
    """Generate the statistical results of the experiment."""

    # Combine experiments objects
    experiment_results, datasets = [], []
    for ratio in UNDERSAMPLING_RATIO:

        # Generate experiment with all oversamplers
        experiment = generate_experiment(ratio)

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


if __name__ == '__main__':
    
    # Main results
    for ratio in UNDERSAMPLING_RATIO:
        main_results = generate_main_results(ratio)
        for name, result in zip(MAIN_RESULTS_NAMES, main_results):
            result.to_csv(join(RESOURCES_PATH, f'{name}_{ratio}.csv'), index=False)

    # Statistical results
    statistical_results = generate_statistical_results()
    for name, result in zip(STATISTICAL_RESULTS_NAMES, statistical_results):
        result.to_csv(join(RESOURCES_PATH, f'{name}.csv'), index=False)

