from os.path import dirname, join
from pickle import load

import pandas as pd
from sklearn.metrics import make_scorer, SCORERS
from imblearn.metrics import geometric_mean_score
from rlearn.tools import (
    filter_experiment,
    combine_experiments,
    calculate_mean_sem_scores,
    calculate_mean_sem_perc_diff_scores,
    calculate_mean_sem_ranking,
    apply_friedman_test,
    apply_holms_test
)

from tools import EXPERIMENTS_PATH
from tools.format import generate_mean_std_tbl, generate_pvalues_tbl


UNDERSAMPLING_RATIOS = [50, 75, 90, 95]
EXPERIMENTS_NAMES = [
    'no_oversampling', 
    'random_oversampling',
    'smote',
    'borderline_smote',
    'gsmote'
]

STATISTICAL_RESULTS_NAMES = [
    'friedman_test',
    'holms_test'
]
RESOURCES_PATH = join(dirname(__file__), 'resources')
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
    experiment = combine_experiments(experiments)

    # Filter experiment
    classifiers_names = [name for name in experiment.classifiers_names_ if name != 'CONSTANT CLASSIFIER']
    experiment = filter_experiment(experiment, classifiers_names=classifiers_names)

    return experiment
    

def generate_main_results():
    """Generate the main results of the experiment."""

    main_results = {}
    for ratio in UNDERSAMPLING_RATIOS:

        # Generate experiment   
        experiment = generate_experiment(ratio)

        # Calculate results
        mean_sem_scores = generate_mean_std_tbl(*calculate_mean_sem_scores(experiment))
        mean_sem_perc_diff_scores = generate_mean_std_tbl(*calculate_mean_sem_perc_diff_scores(experiment, ['NO OVERSAMPLING', 'G-SMOTE']))
        mean_sem_ranking = generate_mean_std_tbl(*calculate_mean_sem_ranking(experiment))

        # Populate main results
        main_results_names = ('mean_sem_scores', 'mean_sem_perc_diff_scores', 'mean_sem_ranking')
        main_results[ratio] = zip(main_results_names, (mean_sem_scores, mean_sem_perc_diff_scores, mean_sem_ranking))
    
    return main_results


def generate_statistical_results():
    """Generate the statistical results of the experiment."""

    # Combine experiments objects
    experiments = []
    for ratio in UNDERSAMPLING_RATIOS:

        # Generate experiment with all oversamplers
        experiment = generate_experiment(ratio)

        # Extract results
        results = experiment.results_.reset_index()
        results['Dataset'] = results['Dataset'].apply(lambda name: f'{name}({ratio})')
        results.set_index(['Dataset', 'Oversampler', 'Classifier', 'params'], inplace=True)
        results.columns = experiment.results_.columns
        experiment.results_ = results
        experiments.append(experiment)

    # Generate final experiment
    experiment = combine_experiments(experiments)

    # Calculate statistical results
    friedman_test = generate_pvalues_tbl(apply_friedman_test(experiment))
    holms_test = generate_pvalues_tbl(apply_holms_test(experiment, control_oversampler='G-SMOTE'))
    statistical_results = zip(STATISTICAL_RESULTS_NAMES, (friedman_test, holms_test))

    return statistical_results


if __name__ == '__main__':
    
    # Main results
    main_results = generate_main_results()
    for ratio, results in main_results.items():
        for name, result in results:
            result.to_csv(join(RESOURCES_PATH, f'{name}_{ratio}.csv'), index=False)

    # Statistical results
    statistical_results = generate_statistical_results()
    for name, result in statistical_results:
        result.to_csv(join(RESOURCES_PATH, f'{name}.csv'), index=False)

