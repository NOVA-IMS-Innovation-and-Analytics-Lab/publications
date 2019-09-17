from os.path import dirname, join
from pickle import load

from tools import EXPERIMENTS_PATH
from tools.format import generate_mean_std_tbl, generate_pvalues_tbl
from sklearnext.tools import (
    combine_experiments,
    calculate_mean_sem_scores,
    calculate_mean_sem_perc_diff_scores,
    calculate_mean_sem_ranking,
    apply_friedman_test,
    apply_holms_test
)

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
    'mean_sem_ranking', 
    'friedman_test', 
    'holms_test'
]


def generate_main_results(ratio):
    """Generate the main results of the experiment."""

    # Load experiments object
    experiments = []
    for name in EXPERIMENTS_NAMES:
        file_path = join(EXPERIMENTS_PATH, f'small_data_{ratio}', f'{name}.pkl')
        with open(file_path, 'rb') as file:
            experiments.append(load(file))
    
    # Create combined experiment
    experiment = combine_experiments('combined', *experiments)

    # Exclude constant classifier
    mask_clf = (experiment.results_.reset_index().Classifier != 'CONSTANT CLASSIFIER')
    mask_ovr = (experiment.results_.reset_index().Oversampler != 'BENCHMARK METHOD')
    mask = (mask_clf & mask_ovr).values
    experiment.results_ = experiment.results_[mask]
    experiment.classifiers = experiment.classifiers[1:]
    experiment.classifiers_names_ = experiment.classifiers_names_[1:]
    experiment.oversamplers = experiment.oversamplers[:-1]
    experiment.oversamplers_names_ = experiment.oversamplers_names_[:-1]

    # Recreate optimal and wide optimal results
    experiment._calculate_optimal_results()._calculate_wide_optimal_results()

    # Calculate results
    mean_sem_scores = generate_mean_std_tbl(*calculate_mean_sem_scores(experiment))
    mean_sem_perc_diff_scores = generate_mean_std_tbl(*calculate_mean_sem_perc_diff_scores(experiment, ['SMOTE', 'G-SMOTE']))
    mean_sem_ranking = generate_mean_std_tbl(*calculate_mean_sem_ranking(experiment))
    friedman_test = generate_pvalues_tbl(apply_friedman_test(experiment))
    holms_test = generate_pvalues_tbl(apply_holms_test(experiment, control_oversampler='G-SMOTE'))
    
    return mean_sem_scores, mean_sem_perc_diff_scores, mean_sem_ranking, friedman_test, holms_test


if __name__ == '__main__':
    
    for ratio in UNDERSAMPLING_RATIO:
        main_results = generate_main_results(ratio)
        for name, result in zip(MAIN_RESULTS_NAMES, main_results):
            result.to_csv(join(RESOURCES_PATH, f'{name}_{ratio}.csv'), index=False)
