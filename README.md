# Overview

The repository contains the published work of the NOVA IMS Innovation and Analytics Lab.

## Publications

A list of published papers and work in progress.

### Published work

- [Geometric SMOTE for regression](https://www.sciencedirect.com/science/article/abs/pii/S095741742101678X)
- [Improving the quality of predictive models in small data GSDOT: A new algorithm for generating synthetic data](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0265626)
- [G-SOMO: An oversampling approach based on self-organized maps and geometric SMOTE](https://www.sciencedirect.com/science/article/abs/pii/S095741742100662X)
- [Increasing the effectiveness of active learning: Introducing artificial data generation in active learning for land use/land cover classification](https://www.mdpi.com/2072-4292/13/13/2619)
- [Improving imbalanced land cover classification with k-means smote: Detecting and oversampling distinctive minority spectral signatures](https://www.mdpi.com/2078-2489/12/7/266)
- [Geometric SMOTE a geometrically enhanced drop-in replacement for SMOTE](https://www.sciencedirect.com/science/article/abs/pii/S0020025519305353)
- [Imbalanced learning in land cover classification: Improving minority classes’ prediction accuracy using the geometric SMOTE algorithm](https://www.mdpi.com/2072-4292/11/24/3040)
- [Effective data generation for imbalanced learning using conditional generative adversarial networks](https://www.sciencedirect.com/science/article/abs/pii/S0957417417306346)
- [Improving imbalanced learning through a heuristic oversampling method based on k-means and SMOTE](https://www.sciencedirect.com/science/article/abs/pii/S0020025518304997)
- [Self-Organizing Map Oversampling (SOMO) for imbalanced data set learning](https://www.sciencedirect.com/science/article/abs/pii/S0957417417302324)

### Work in progress

- Imbalanced text classification using Geometric SMOTE oversampling algorithm
- cluster-over-sampling: A Python package for clustering-based oversampling
- geometric-smote: A package for flexible and efficient oversampling
- Intraday trading via Deep Reinforcement Learning and technical indicators
- Genetic Programming for Offline Reinforcement Learning

## Projects

Each publication corresponds to a [Kedro](https://kedro.org/) project. Take a look at the [Kedro
documentation](https://docs.kedro.org) to get started.

### How to install dependencies

Declare any dependencies in `requirements.txt` for `pip` installation.

To install them, run:

```
pip install -r requirements.txt
```

### How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

### How to test your Kedro project

Have a look at the files `src/tests/test_run.py` and `src/tests/pipelines/test_data_science.py` for instructions on how to write your tests. Run the tests as follows:

```
pytest
```

To configure the coverage threshold, look at the `.coveragerc` file.
