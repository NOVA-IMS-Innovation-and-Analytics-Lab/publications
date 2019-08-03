import pickle

from os.path import dirname, join
from pickle import load

from tools import EXPERIMENTS_PATH
from tools.format import generate_mean_std_tbl, generate_pvalues_tbl
from sklearnext.tools import (
    combine_experiments,
    calculate_mean_sem_scores,
    calculate_mean_sem_perc_diff_scores,
    calculate_mean_sem_ranking,
)

data = pickle.load(open(join(EXPERIMENTS_PATH, f'lucas', f'combined.pkl'), 'rb')) 

RESOURCES_PATH = join(dirname(__file__), 'resources')

EVALUATION = 'drop_accuracy'
MAIN_RESULTS_NAMES = names = [
    'mean_sem_scores', 
    'mean_sem_perc_diff_scores', 
    'mean_sem_ranking',  
]

def generate_main_results(evaluation):
    """Generate the main results of the experiment."""

    # Load experiments object
    experiments = [data]
    
    # Create combined experiment
    experiment = combine_experiments('combined', *experiments)

    # Exclude constant and random classifiers
    mask_clf1 = (experiment.results_.reset_index().Classifier != 'CONSTANT CLASSIFIER')
    mask_clf2 = (experiment.results_.reset_index().Classifier != 'RANDOM CLASSIFIER')
    mask = (mask_clf1 & mask_clf2).values
    experiment.results_ = experiment.results_[mask]
    experiment.classifiers = experiment.classifiers[2:]
    experiment.classifiers_names_ = experiment.classifiers_names_[2:]
    
    # Drop accuracy metric
    experiment.results_ = experiment.results_.drop('mean_test_accuracy', axis=1, level=0)
    experiment.scoring_cols_ = experiment.scoring_cols_[:2]
    
    # Recreate optimal and wide optimal results
    experiment._calculate_optimal_results()._calculate_wide_optimal_results()

    # Calculate results
    mean_sem_scores = generate_mean_std_tbl(*calculate_mean_sem_scores(experiment))
    mean_sem_perc_diff_scores = generate_mean_std_tbl(*calculate_mean_sem_perc_diff_scores(experiment, ['SMOTE', 'G-SMOTE']))
    mean_sem_ranking = generate_mean_std_tbl(*calculate_mean_sem_ranking(experiment))
        
    return mean_sem_scores, mean_sem_perc_diff_scores, mean_sem_ranking


if __name__ == '__main__':
    
    main_results = generate_main_results(EVALUATION)
    for name, result in zip(MAIN_RESULTS_NAMES, main_results):
        result.to_csv(join(RESOURCES_PATH, f'{name}.csv'), index=False)