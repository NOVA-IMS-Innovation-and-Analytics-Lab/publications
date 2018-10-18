============
Installation
============

The provided scripts require Python 3 and `Scikit-Learn-Extensions <https://github.com/georgedouzas/scikit-learn-extensions>`_.

====================
Imbalanced data sets
====================

The following command downloads and transforms various imbalanced datasets and
saves them as csv files.

.. code-block::

  $ python download_imbalanced_data.py --path <path>

===========
Experiments
===========

The following commands execute the experimental procedure of the various papers.

Geometric SMOTE: A geometrically enhanced drop-in replacement for SMOTE
=======================================================================

From the `gsmote-journal <https://github.com/IMS-ML-Lab/publications/tree/master/scripts/gsmote-journal>`_ directory run the following commands:

Run experiment
--------------

.. code-block::

  $ python experiment.py

Analyse results
---------------

.. code-block::

  $ python analysis.py

KMeans and oversampling
=======================

From the `kmeans-oversampling <https://github.com/IMS-ML-Lab/publications/tree/master/scripts/kmeans-oversampling>`_ directory run the following commands:

Run experiment
--------------

.. code-block::

  $ python random_oversampler_experiment.py
  $ python smote_experiment.py
  $ python bordeline_smote_experiment.py
  $ python gsmote_experiment.py
  $ python combined_experiment.py

Clustering and SMOTE
====================

From the `clustering-smote <https://github.com/IMS-ML-Lab/publications/tree/master/scripts/clustering-smote>`_ directory run the following commands:

Run experiment
--------------

.. code-block::

  $ python experiment.py

G-SOMO
======

From the `gsomo <https://github.com/IMS-ML-Lab/publications/tree/master/scripts/gsomo>`_ directory run the following commands:

Run experiment
--------------

.. code-block::

  $ python experiment.py



