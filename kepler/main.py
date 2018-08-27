#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
"""
Kepler.
"""

from keras.models import Model
from sklearn.base import BaseEstimator
from traitlets import HasTraits, Enum, Instance, Union, Unicode


class Project(HasTraits):
    """An ML project."""

    problem_type = Enum('classification', 'regression')


class ModelInspector(HasTraits):

    model = Union([Instance(Model), Instance(BaseEstimator)])

    commit = Unicode()


if __name__ == '__main__':
    ModelInspector(commit='foo124')
