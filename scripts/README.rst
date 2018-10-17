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

  $ python download_imbalanced_data.py --path ../data/binary-numerical-imbalanced

===========
Experiments
===========

The following commands execute the experimental procedure of the various papers.

Geometric SMOTE: A geometrically enhanced drop-in replacement for SMOTE
=======================================================================

Run experiment
--------------

.. code-block::

  $ python gsmote_journal_experiment.py

Analyse results
---------------

.. code-block::

  $ python gsmote_journal_analysis.py



