=======
Scripts
=======

Installation
============

The provided scripts require Python 3 and `Scikit-Learn-Extensions <https://github.com/georgedouzas/scikit-learn-extensions>`_.

Data sets
=========

The following command downloads and transforms various imbalanced datasets and
saves them as an SQLite database.

.. code-block::

  $ python data.py [path]

Experiments
===========

The following commands execute the experimental procedure of the various papers.

Run experimental procedure:

.. code-block::

  $ python experiment.py name

Analyze the results of experiment:

.. code-block::

  $ python analysis.py name

Both commands include various options. Add the ``-h`` flag for help. 
