"""
CLI for running and analyzing experiments.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from os.path import join, dirname, pardir, isdir
from os import listdir

import click
import mlflow

ROOT_PATH = join(dirname(__file__), pardir)


def list_projects():
    projects_path = join(ROOT_PATH, 'projects')
    projects = []
    for name in listdir(projects_path):
        project_path = join(projects_path, name)
        if isdir(project_path) and 'MLproject' in listdir(project_path):
            projects.append(name)
    return projects


@click.command(help='Run experiment of MLflow project.')
@click.argument('project', type=click.Choice(list_projects()))
@click.option(
    '--n-splits',
    type=click.INT,
    default=5,
    help='The number of cross validation splits.',
)
@click.option(
    '--n-runs', type=click.INT, default=2, help='The number of experiment runs.'
)
@click.option('--rnd-seed', type=click.INT, default=1, help='The random seed.')
def experiment(project, n_splits, n_runs, rnd_seed):
    mlflow.set_tracking_uri(f'file://{join(ROOT_PATH, "mlruns")}')
    mlflow.run(
        uri=join(ROOT_PATH, 'projects', project),
        entry_point='main',
        experiment_name=project,
        parameters={'n_splits': n_splits, 'n_runs': n_runs, 'rnd_seed': rnd_seed},
    )


@click.command(help='Run analysis on the latest experiment of MLflow project.')
@click.argument('project', type=click.Choice(list_projects()))
def analysis(project):
    mlflow.set_tracking_uri(f'file://{join(ROOT_PATH, "mlruns")}')
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(project)
    if experiment is None:
        click.echo('No experiment was found to analyze it.')
        return
    run_infos = client.list_run_infos(
        experiment.experiment_id, order_by=['attribute.end_time DESC']
    )
    mlflow.run(
        uri=join(ROOT_PATH, 'projects', project),
        entry_point='analysis',
        experiment_name=project,
        run_id=run_infos[0].run_id,
    )
