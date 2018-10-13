#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# JSM product code
#
# (C) Copyright 2018 Juxt SmartMandate Pvt Ltd
# All right reserved.
#
# This file is confidential and NOT open source.  Do not distribute.
#

"""
Module containing all checks.
"""
from abc import ABC
from functools import wraps
import warnings

import numpy as np
from scipy.stats import mode
from sklearn.decomposition import PCA

from kepler.utils import is_power2, is_1d, is_onehotencoded


def checker(fit_method, checks):
    @wraps(fit_method)
    def run(X, y, *args, **kwargs):
        for check in checks:
            check(X, y, *args, **kwargs)
        return fit_method(X, y, *args, **kwargs)
    return run


class BaseCheck(ABC):
    """Base class for all checks"""

    code = None
    msg = None

    @classmethod
    def __subclasshook__(cls, other):
        return bool(getattr(other, 'code', False))

    def check(self, X, y, *args, **kwargs):
        """Only supposed to return a boolean."""
        raise NotImplementedError

    def warn(self):
        """All logging logic goes here."""
        warnings.warn(self.code + ": " + self.msg)

    def __call__(self, X, y, *args, **kwargs):
        if not self.check(X, y, *args, **kwargs):
            self.warn()


# Minibatch Checks ############################################################

class BadMinibatchSize(BaseCheck):
    """BadMinibatchSize"""

    code = 'K101'
    msg = 'Batch size is not a power of 2.'

    def check(self, X, y, *args, **kwargs):
        if len(args) > 0:
            batch_size = args[0]
        else:
            batch_size = None
        if batch_size is None:
            batch_size = kwargs.pop('batch_size', 32)
        return is_power2(batch_size)


class MinibatchTooSmall(BaseCheck):
    """MinibatchTooSmall"""

    code = 'K1010'
    msg = 'Batch size too small.'

    def check(self, X, y, *args, **kwargs):
        n_samples = X.shape[0]
        if len(args) > 0:
            batch_size = args[0]
        else:
            batch_size = None
        if batch_size is None:
            batch_size = kwargs.get('batch_size', 32)
        return batch_size / n_samples > 0.0125


class MinibatchTooLarge(BaseCheck):
    """MinibatchTooLarge"""

    code = 'K1012'
    msg = 'Batch size too large.'

    def check(self, X, y, *args, **kwargs):
        n_samples = X.shape[0]
        if len(args) > 0:
            batch_size = args[0]
        else:
            batch_size = None
        if batch_size is None:
            batch_size = kwargs.get('batch_size', 32)
        return batch_size / n_samples < 0.33


class DataNotShuffled(BaseCheck):
    """DataNotShuffled.
    Not sure exactly how one would detect a random ordering of ints."""


class TrainDevNotStratified(BaseCheck):
    """TrainDevNotStratified"""

    code = 'K103'
    msg = 'Train and validation data may not be stratified.'

    def check(self, X, y, *args, **kwargs):
        validation_data = kwargs.get('validation_data')
        if validation_data is None:
            warnings.warn('{} not applicable.'.format(
                self.__class__.__name__))
            return True
        else:
            y_val = validation_data[1]
        if is_1d(y):
            y_train = y
        elif is_onehotencoded(y):
            y_train = y.sum(axis=1)
            y_val = y_val.sum(axis=1)
        trn_labels, trn_label_counts = np.unique(y_train, return_counts=True)
        val_labels, val_label_counts = np.unique(y_val, return_counts=True)
        stratified = True
        if trn_labels != val_labels:
            stratified = False
        elif (trn_label_counts / val_label_counts).var().round(3) > 1e-3:
            stratified = False
        return stratified


class TrainDevNormalizedSeparately(BaseCheck):
    """TrainDevNormalizedSeparately"""


# Model Weight Checks #########################################################

class WeightsTooSmall(BaseCheck):
    """WeightsTooSmall"""


class WeightsNotNormal(BaseCheck):
    """WeightsNotNormal"""


class UniformWeightInit(BaseCheck):
    """UniformWeightInit"""


class IncompatibleWeightInitializer(BaseCheck):
    """IncompatibleWeightInitializer"""


# Training Data Statistics ####################################################

class TrainingSamplesCorrelated(BaseCheck):
    """TrainingSamplesCorrelated"""

    code = 'K301'
    msg = 'Training samples are correlated. ' + \
          'There may be redundancy in the data.'

    def check(self, X, y, *args, **kwargs):
        pca = PCA()
        pca.fit(X)
        m, c = mode(np.cumsum(pca.explained_variance_ratio_))
        return not(m[0].round(3) == 1 and c[0] > 1)


class DuplicateTrainingSamples(BaseCheck):
    """DuplicateTrainingSamples"""
    # http://www.ryanhmckenna.com/2017/01/efficiently-remove-duplicate-rows-from.html
    code = 'K302'
    msg = 'There might be duplicate training samples.'

    def check(self, X, y, *args, **kwargs):
        if is_1d(y):
            y2 = y.ravel()
        elif is_onehotencoded(y):
            y2 = np.argmax(y, axis=1)
        A = np.c_[X, y2]
        sampler = np.random.rand(A.shape[1])
        y = A.dot(sampler)
        unique, ix = np.unique(y, return_index=True)
        return ix.shape[0] > A.shape[0]


class DataNotNormalized(BaseCheck):
    """DataNotNormalized"""

    code = 'K303'
    msg = 'Training data not normalized.'

    def check(self, X, y, *args, **kwargs):
        centered = np.all(X.mean(0).round(3) == 0)
        scaled = np.all(X.var(0).round(3) == 1)
        return centered and scaled


# Activations #################################################################

class SigmoidActivation(BaseCheck):
    """SigmoidActivation"""


class BadTanhEncoding(BaseCheck):
    """BadTanhEncoding"""


class BadLabelEncoding(BaseCheck):
    """BadLabelEncoding"""


# Training ####################################################################

class ParamsMoreThanTrainingSamples(BaseCheck):
    """ParamsMoreThanTrainingSamples"""


class VanishingGradients(BaseCheck):
    """VanishingGradients"""


class ExplodingGradients(BaseCheck):
    """ExplodingGradients"""


class BadBatchNormPosition(BaseCheck):
    """BadBatchNormPosition"""


class ModelOverfittingStarted(BaseCheck):
    """ModelOverfittingStarted"""


class NoisyWeightUpdates(BaseCheck):
    """NoisyWeightUpdates"""
