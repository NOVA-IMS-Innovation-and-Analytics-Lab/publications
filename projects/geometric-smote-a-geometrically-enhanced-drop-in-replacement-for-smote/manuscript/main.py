"""
Generate the manuscript.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import subprocess
from os.path import join, dirname, pardir, exists

from rlearn.reporting import (
    summarize_imbalanced_binary_datasets,
)
from tools.datasets import load_datasets_from_db


def generate_manuscript(experiment):
    """Analyze the experiment and generate the manuscript."""

    # Datasets summary
    datasets = load_datasets_from_db(
        join(dirname(__file__), pardir, 'datasets', 'imbalanced_binary.db')
    )
    datasets_summary = summarize_imbalanced_binary_datasets(datasets)
    columns_mapping = {
        col: ' '.join(col.split('_')).title() for col in datasets_summary.columns
    }
    datasets_summary = datasets_summary.rename(columns=columns_mapping)
    datasets_summary['Dataset Name'] = datasets_summary['Dataset Name'].apply(
        lambda name: ' '.join(name.split('_')).upper()
    )
    datasets_summary.to_csv(join(dirname(__file__), 'datasets_summary.csv'))

    # TODO
    # Produce results from experiment object

    # Create manuscript
    manuscript_path = join(dirname(__file__), 'manuscript.tex')
    if exists(manuscript_path):
        subprocess.run(['pdf2latex', join(dirname(__file__), 'manuscript.tex')])
