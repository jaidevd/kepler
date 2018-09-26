#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
"""
Kepler.
"""

from datetime import datetime
from functools import wraps
import json
import warnings

import numpy as np
from keras.models import Model
from keras.utils.layer_utils import count_params
from sklearn.base import BaseEstimator
from traitlets import HasTraits, Enum, Union, Unicode, Instance, Tuple, Dict
from h5py import File as H5File

from kepler.custom_traits import KerasModelWeights

warnings.simplefilter('always', category=UserWarning)


class Project(HasTraits):
    """An ML project."""

    problem_type = Enum('classification', 'regression')


class ModelInspector(HasTraits):

    model = Union([Instance(Model), Instance(BaseEstimator)])

    commit = Unicode()

    weights_path = KerasModelWeights()

    model_definition = Unicode()

    @property
    def model_config(self):
        with H5File(self.weights_path, 'r') as fout:
            config = fout.attrs.get('model_config')
        config = json.loads(config.decode())
        return config['config']

    @property
    def n_params(self):
        self.model._check_trainable_weights_consistency()
        tw = getattr(self.model, '_collected_trainable_weights',
                     self.model.trainable_weights)
        return count_params(tw)

    def check_fit_for_overfitting(self, fit_method):
        """Checks if the fit method might be overfitting."""
        @wraps(fit_method)
        def overfit_wrapper(X, y, *args, **kwargs):
            # Is this a standalone `model` or same as `self.model`?
            nrows = X.shape[0]
            if nrows < self.n_params:
                warnings.warn('You might overfit!')
            return fit_method(X, y, *args, **kwargs)
        return overfit_wrapper

    def __enter__(self):
        self.oldfit = self.model.fit
        self.model.fit = self.check_fit_for_overfitting(self.model.fit)
        return self.model

    def __exit__(self, _type, value, traceback):
        self.model.fit = self.oldfit


class Experiment(HasTraits):

    model = Instance(Model)

    train_x = Instance(np.ndarray)
    train_y = Instance(np.ndarray)
    validation_x = Instance(np.ndarray)
    validation_y = Instance(np.ndarray)

    start_ts = Instance(datetime)
    fit_args = Tuple()
    fit_kwargs = Dict()

    def __eq__(self, other):
        """Check if this experiment is _similar_ to `other`.

        Arguments:
            other {kepler.Experiment} -- Another experiment
        """

    def __enter__(self):
        self.start_ts = datetime.now()
        return self

    def __exit__(self):
        self.save()

    @property
    def history(self):
        return self.model.history

    @property
    def n_epochs(self):
        return self.fit_kwargs['epochs']

    def save(self):
        """Save the experiment in the db.
        """
        pass
