"""
CLI for running and analyzing experiments.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from importlib import import_module
from os.path import join, dirname, pardir, isdir, exists
from os import listdir
from subprocess import Popen, DEVNULL

import click
import mlflow

ROOT_PATH = join(dirname(__file__), pardir)
TRACKING_URI = f'file://{join(ROOT_PATH, "mlruns")}'


def list_projects():
    projects_path = join(ROOT_PATH, 'projects')
    projects = {}
    ind = 0
    for name in listdir(projects_path):
        project_path = join(projects_path, name)
        if isdir(project_path) and 'MLproject' in listdir(project_path):
            projects[str(ind)] = name
            ind += 1
    return projects


def import_main(project, msg):
    try:
        return import_module(f'projects.{project}.main')
    except ModuleNotFoundError:
        click.echo(
            click.style(
                f'Project \'{project}\' does not have a \'main\' module. {msg}.',
                fg='red',
            )
        )


def get_task(module, project, func_name, msg):
    try:
        return getattr(module, func_name)
    except AttributeError:
        click.echo(
            click.style(
                f'Project \'{project}\' does not have a \'{func_name}\' function '
                'in the \'main\' module. {msg}.',
                fg='red',
            )
        )


@click.command(help='List all projects')
def list():
    for ind, project in list_projects().items():
        click.echo(click.style(f'({ind}) {project}', fg='green'))


@click.command(help='Fetch the data of MLflow project.')
@click.option(
    '--project-number',
    '-p',
    type=click.Choice(list_projects().keys()),
    help='The project number to select.',
    required=True,
)
def data(project_number):
    msg = 'No data are fetched'
    project = list_projects()[project_number]
    data_path = join(ROOT_PATH, 'projects', project, 'data')
    module = import_main(project, msg)
    if module is not None:
        task = get_task(module, project, 'fetch_data', msg)
        if task is not None:
            task()
            click.echo(
                click.style(
                    f'The data were downloaded and saved to the path \'{data_path}\'.',
                    fg='green',
                )
            )


@click.command(help='Run an experiment of MLflow project.')
@click.option(
    '--project-number',
    '-p',
    type=click.Choice(list_projects().keys()),
    help='The project number to select.',
    required=True,
)
def experiment(project_number):
    msg = 'No experiment is run'
    project = list_projects()[project_number]
    module = import_main(project, msg)
    if module is not None:
        task = get_task(module, project, 'run_experiment', msg)
        if task is not None:
            mlflow.set_tracking_uri(TRACKING_URI)
            mlflow.run(
                uri=join(ROOT_PATH, 'projects', project),
                experiment_name=project,
                env_manager='local',
            )
            click.echo(
                click.style(
                    'The experiment was run and the model was saved.', fg='green'
                )
            )


@click.command(help='Create the manuscript of MLflow project.')
@click.option(
    '--project_number',
    '-p',
    type=click.Choice(list_projects().keys()),
    help='The project number to select.',
    required=True
)
@click.option('--run-id', '-r', type=str, help='The run ID of the experiment.', required=True)
def manuscript(project_number, run_id):
    msg = 'No manuscript\'s artifacts were generated'
    client = mlflow.tracking.MlflowClient(tracking_uri=TRACKING_URI)
    project = list_projects()[project_number]
    experiment = client.get_experiment_by_name(project)
    if experiment is None:
        click.echo(
            click.style(
                'No experiment was found to create the manuscript artifacts.', fg='red'
            )
        )
    else:
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
        if exists(model_uri):
            experiment_obj = mlflow.sklearn.load_model(model_uri)
            module = import_main(project, msg)
            if module is not None:
                task = get_task(module, project, 'generate_manuscript_artifacts', msg)
                if task is not None:
                    module.generate_manuscript_artifacts(experiment_obj)
    manuscript_path = join(
        ROOT_PATH, 'projects', project, 'manuscript', 'manuscript.tex'
    )
    Popen(
        f'pdflatex {manuscript_path}',
        shell=True,
        cwd=dirname(manuscript_path),
        stdout=DEVNULL,
    )
    click.echo(
        click.style(
            f'The manuscript pdf was generated and saved to the path {manuscript_path}.',
            fg='green',
        )
    )
