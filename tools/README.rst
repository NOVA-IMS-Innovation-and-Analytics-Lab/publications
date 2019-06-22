=====
Tools
=====

Installation
============

Activate a Python virtual environment and from the current directory run the following command:

.. code-block::

  $ pip install .

The ``run`` command is then installed and includes various options.

Usage
=====

The basic usage of the ``run`` command is described below: 

Download database
#################

The following command downloads various datasets and
saves them as an SQLite database:

.. code-block::

  $ run downloading name

The argument ``name`` corresponds to the name of the database. For more information: 

.. code-block::

  $ run downloading --help

Run experiment
##############

The following command executes an experimental procedure and saves the outcome:

Run experimental procedure:

.. code-block::

  $ run experiment name

The argument ``name`` corresponds to the name of the experiment. For more information: 

.. code-block::

  $ run experiment --help
