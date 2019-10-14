============
Installation
============

Activate a Python 3 virtual environment and run the command::

    pip install -r requirements.txt

====
Data
====

To download and save the experimental datasets, run the command::

    python data.py

The data are saved in the *data* directory as a sqlite3 database.

=======
Results
=======

To run the experiments and get the experimental results, run the command::

    python results.py

The results for each oversampler are saved in the *results* directory as pickled
pandas dataframes.

========
Analysis
========

To analyze the experimental results, run the command::

    python analysis.py

The outcome is saved in the *results* directory in various formats.
