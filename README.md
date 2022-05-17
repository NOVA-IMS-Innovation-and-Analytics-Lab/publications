# Publications

The repository contains the published work of the **NOVA IMS Innovation and Analytics Lab**. The `projects` directory contains each of the publications as an MLFlow project. Installation 
of the projects provides a CLI and allows to download the data and run the experiments.

## Installation

Initially, clone the project:

```
git clone https://github.com/NOVA-IMS-Innovation-and-Analytics-Lab/publications.git
```

### Main installation


From the root of the project you can use [pip](https://pip.pypa.io/en/stable):

```
pip install .
```

Alternatively you can use [PDM](https://pdm.fming.dev):

```
pdm install
```

The main installation allows to use the CLI. Some of the commands require additional dependencies.

### Optional dependencies

You can install the optional dependencies using the `-G` flag:

```
pdm install -G [GROUP]
```

You may find the various groups of optional dependencies in the `pyproject.toml` file.

### Optional dependencies

You can install the development dependencies using the `-G` and `-d` flags:

```
pdm install -G -d [GROUP]
```

You may find the various groups of development dependencies in the `pyproject.toml` file.

## Command Line Interface

There four commands available to manage the project:

- `create`
- `datasets`
- `experiment`
- `manuscript`

You can use `--help` flag to get help about their usage and options.
