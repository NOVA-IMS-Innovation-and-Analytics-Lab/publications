"""improving-imbalanced-learning-through-a-heuristic-oversampling-method-based-on-k-means-and-smote file for ensuring the package is executable
as `improving-imbalanced-learning-through-a-heuristic-oversampling-method-based-on-k-means-and-smote` and `python -m improving_imbalanced_learning_through_a_heuristic_oversampling_method_based_on_k_means_and_smote`
"""
import importlib
from pathlib import Path

from kedro.framework.cli.utils import KedroCliError, load_entry_points
from kedro.framework.project import configure_project


def _find_run_command(package_name):
    try:
        project_cli = importlib.import_module(f"{package_name}.cli")
        # fail gracefully if cli.py does not exist
    except ModuleNotFoundError as exc:
        if f"{package_name}.cli" not in str(exc):
            raise
        plugins = load_entry_points("project")
        run = _find_run_command_in_plugins(plugins) if plugins else None
        if run:
            # use run command from installed plugin if it exists
            return run
        # use run command from the framework project
        from kedro.framework.cli.project import run

        return run
    # fail badly if cli.py exists, but has no `cli` in it
    if not hasattr(project_cli, "cli"):
        raise KedroCliError(f"Cannot load commands from {package_name}.cli")
    return project_cli.run


def _find_run_command_in_plugins(plugins):
    for group in plugins:
        if "run" in group.commands:
            return group.commands["run"]


def main(*args, **kwargs):
    package_name = Path(__file__).parent.name
    configure_project(package_name)
    run = _find_run_command(package_name)
    run(*args, **kwargs)


if __name__ == "__main__":
    main()
