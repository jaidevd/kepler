Kepler
======

.. image:: https://travis-ci.com/jaidevd/kepler.svg?branch=master
    :target: https://travis-ci.com/jaidevd/kepler

Kepler is a monitoring system for machine learning experiments. It allows users to perform efficient bookkeeping, logging and auditing on ML models and experiments. It rests on the idea that machine learning is cheap, but not cheap enough for repetitive mistakes.

Kepler empowers developers and data scientists to apply the DRY principle in machine learning practice by:

1. Applying sanity checks to models and continuously monitoring model activity

   It is easy to spin up a model that works (a model "works" or "learns" when it satisfactorily optimizes some metric), but unless carefully examined, it is quite likely that the model is acting against itself. It is very easy to throw together model components that are inherently contradictory or less than optimal. This sub-optimal design can manifest itself in many ways - from something as simple as not shuffling training samples to as complex as having a deep network with an internally incompatible set of layers. Kepler uses a set of `checks <doc/checks.rst>`_ to search for such inconsistencies or bad practices.

2. Enabling efficient bookkeeping with a searchable interface

   Kepler installs a sqlite DB which stores almost everything done within the Kepler instance. Kepler organizes every model under a "project". The definition and metadata associated with each model under a project is saved under that projects. Multiple projects may share a model. Each training / validation / testing action on a model is interpreted as an "experiment", and the results of all such experimens are stored in the DB. The projects, models and experiments are all searchable - allowing for better code reuse and more efficient grid search.


Installation
------------

To install Kepler, download or clone this repository and run:

.. code-block:: bash

   $ pip install -e .

After the installation, run the initialization script as follows:

.. code-block:: bash

   $ kepler init
   Welcome to Kepler!

This means that the Kepler database has been successfully installed on your system.


Usage
-----

The main entry point into Kepler is the ``kepler.ModelInspector`` class. It is a context manager which wraps a model during training or evaluation.

.. code-block:: python

   >>> from kepler import ModelInspector
   >>> from kepler.sample_models import mnist_shallow  # A 3 layer keras NN intended for MNIST
   >>> from sklearn.datasets import load_digits
   >>> digits = load_digits()
   >>> X, y = digits['data'], digits['target']
   >>> with ModelInspector(model=mnist_shallow()) as mi:
   ...      mi.fit(X, y)
