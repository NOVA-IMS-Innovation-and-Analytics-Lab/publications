"""
Analyze the experiment.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from os.path import join, dirname

import mlflow

PATH = join(dirname(__file__), 'artifacts')


def run_analysis():

    with mlflow.start_run():

        pass


if __name__ == '__main__':

    pass
