=================
Project structure
=================

Every research project contains the scripts, data, results, analysis
and content directories.

scripts
=======

It is the entry point of the project. To install the required dependencies,
first activate a Python 3 virtual environment and then run the command::

    pip install -r requirements.txt

In order to generate the content of the publication in a reproducible format,
various scripts are provided.

data.py
#######

Download and save the experimental datasets::

    python data.py

results.py
##########

Run the experiments and get the experimental results::

    python results.py

analysis.py
###########

Analyze the experimental results::

    python analysis.py

data
====

It contains the experimental data as a sqlite3 database. They are download and
saved, using the ``data.py`` script.

results
=======

It contains the experimental results as pickled pandas dataframes. They are
generated, using the ``results.py`` script.

analysis
========

It contains the analysis of experimental results in various formats. They are
generated, using the ``analysis.py`` script.

content
=======

It contains the LaTex source files of the project.

