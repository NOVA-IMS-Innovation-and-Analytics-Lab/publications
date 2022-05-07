# Publications

The repository contains the published work of the **NOVA IMS Innovation and Analytics Lab**. The `projects` directory contains each of the publications as an MLFlow project. Installation 
of the projects provides a CLI and allows to download the data and run the experiments.

## Installation

Initially, clone the project:

```
git clone https://github.com/NOVA-IMS-Innovation-and-Analytics-Lab/publications.git
```

Then from the root of the project you can use [pip](https://pip.pypa.io/en/stable):

```
pip install .
```

Alternatively you can use [PDM](https://pdm.fming.dev):

```
pdm install
```

## Command Line Interface

You can use `experiment --help` to get the available projects and their options. To run an experiment for a paricular project:

```
experiment [NAME]
```

To run an analysis on the latest experiment of a project:

```
analysis [NAME]
```
