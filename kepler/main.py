#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
"""
Kepler.
"""

from functools import wraps
import warnings
from keras.models import Model
from keras.utils.layer_utils import count_params
from sklearn.base import BaseEstimator
from traitlets import HasTraits, Enum, Instance, Union, Unicode

warnings.simplefilter('always', category=UserWarning)


class Project(HasTraits):
    """An ML project."""

    problem_type = Enum('classification', 'regression')


class ModelInspector(HasTraits):

    model = Union([Instance(Model), Instance(BaseEstimator)])

    commit = Unicode()

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
