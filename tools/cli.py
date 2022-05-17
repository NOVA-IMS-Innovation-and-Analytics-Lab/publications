"""
CLI for running and analyzing experiments.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from importlib import import_module
from os.path import join, dirname, pardir, isdir, exists
from os import listdir, makedirs

import click
import mlflow

ROOT_PATH = join(dirname(__file__), pardir)
TRACKING_URI = f'file://{join(ROOT_PATH, "mlruns")}'
DATASETS_CONTENT = """\"""
Download the datasets.
\"""
{}


def get_datasets_path():
    \"""Get the datasets path.\"""
    pass


def save_datasets(path):
    \"""Download and save the datasets.\"""
    pass
"""
EXPERIMENT_CONTENT = """\"""
Run the experiment.
\"""
{}


def run_experiment():
    \"""Run the experiment and log the experiment model.\"""
    pass


if __name__ == '__main__':

    run_experiment()
"""
MANUSCRIPT_CONTENT = """\"""
Generate the manuscript.
\"""
{}


def generate_manuscript(experiment):
    \"""Analyze the experiment and generate the manuscript.\"""
    pass
"""


def list_projects():
    projects_path = join(ROOT_PATH, 'projects')
    projects = []
    for name in listdir(projects_path):
        project_path = join(projects_path, name)
        if isdir(project_path) and 'MLproject' in listdir(project_path):
            projects.append(name)
    return projects


@click.command(help='Create a new MLflow project.')
@click.argument('project', type=str, required=True)
@click.argument('author-name', type=str, required=True)
@click.argument('author-email', type=str, required=True)
def create(project, author_name, author_email):
    contact = f"""\n# Author: {author_name} <{author_email}>\n# License: MIT"""
    content = {
        'datasets': DATASETS_CONTENT.format(contact),
        'experiment': EXPERIMENT_CONTENT.format(contact),
        'manuscript': MANUSCRIPT_CONTENT.format(contact),
    }
    project_path = join(ROOT_PATH, 'projects', project)
    if exists(project_path):
        click.confirm('Project already exists. Overwrite files?', abort=True)
    for name in ('datasets', 'experiment', 'manuscript'):
        path = join(project_path, name)
        makedirs(path, exist_ok=True)
        with open(join(path, 'main.py'), 'w') as file:
            file.write(content[name])
        if name == 'manuscript':
            open(join(path, 'manuscript.tex'), 'a').close()
    with open(join(project_path, 'MLproject'), 'w') as file:
        file.write("entry_points:\n  main:\n    command: \"python experiment/main.py\"")


@click.command(help='Download and save the data of MLflow project.')
@click.argument('project', type=click.Choice(list_projects()))
def datasets(project):
    module = import_module(f'projects.{project}.datasets.main')
    module.save_datasets(module.get_datasets_path())


@click.command(help='Run an experiment of MLflow project.')
@click.argument('project', type=click.Choice(list_projects()))
def experiment(project):
    module = import_module(f'projects.{project}.datasets.main')
    datasets_path = module.get_datasets_path()
    if datasets_path is None or not exists(datasets_path):
        click.echo(
            f'The datasets of the experiment were not found in {datasets_path}. '
            'You should first download them with the `datasets` command.'
        )
        return
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.run(
        uri=join(ROOT_PATH, 'projects', project),
        experiment_name=project,
        env_manager='local',
    )


@click.command(help='Create the manuscript of MLflow project.')
@click.argument('project', type=click.Choice(list_projects()))
@click.option('--run-id', '-r', type=str, help='The run ID of the experiment.')
def manuscript(project, run_id):
    client = mlflow.tracking.MlflowClient(tracking_uri=TRACKING_URI)
    experiment = client.get_experiment_by_name(project)
    if experiment is None:
        click.echo(
            'No experiment was found to create the manuscript. '
            'You should first start the experiment with the `experiment` command.'
        )
        return
    run_ids = [
        info.run_id
        for info in client.list_run_infos(
            experiment.experiment_id, order_by=['attribute.end_time DESC']
        )
    ]
    if run_id is None:
        run_id = run_ids[0]
    if run_id not in run_ids:
        click.echo(
            f'Run ID should be one of the following: {", ".join(run_ids)}. '
            'Instead {run_id} was given.'
        )
    model_uri = join(client.get_run(run_id).info.artifact_uri, 'experiment_model')
    experiment_obj = mlflow.sklearn.load_model(model_uri)
    module = import_module(f'projects.{project}.manuscript.main')
    module.generate_manuscript(experiment_obj)
