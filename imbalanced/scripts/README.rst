============
Installation
============

 The installation of various dependencies is described below.

Data::

    pip install -r requirements.txt

*cgan* experiment::

    pip install -r requirements.cgan.txt

*gsmote* experiment::

    pip install -r requirements.gsmote.txt

*cluster-over-sampling* and *kmeans-smote* experiments::

    pip install -r requirements.cluster.txt

*somo* experiment::

    pip install -r requirements.somo.txt
    pip install -r requirements.cluster.txt

*gsomo* experiment::

    pip install -r requirements.gsmote.txt
    pip install -r requirements.somo.txt
    pip install -r requirements.cluster.txt

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

    python results.py [experiment]

The results for each oversampler are saved in the *results* directory as pickled
pandas dataframes.

========
Analysis
========

To analyze the experimental results, run the command::

    python analysis.py [experiment]

The outcome is saved in the *results* directory in various formats.
