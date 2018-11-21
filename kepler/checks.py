#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
"""
Module containing all checks.
"""
from abc import ABC
from functools import wraps
import warnings

import numpy as np
from scipy.stats import mode
from sklearn.decomposition import PCA

from keras.activations import sigmoid
from kepler.utils import (is_power2, is_1d, is_onehotencoded,
                          is_initializer_uniform, runs_test)


def checker(fit_method, checks):
    @wraps(fit_method)
    def run(*args, **kwargs):
        for check in checks:
            check(*args, **kwargs)
        return fit_method(*args, **kwargs)
    return run

# Checks that happen at start of training #####################################


class BaseStartTrainingCheck(ABC):
    """Base class for all checks that happen at the beginning of training."""

    code = None
    msg = None
    __type = 'start-training'

    @classmethod
    def __subclasshook__(cls, other):
        if getattr(other, '__type', '') == cls.__type:
            return bool(getattr(other, 'code', False))
        return False

    def check(self, X, y, *args, **kwargs):
        """Only supposed to return a boolean."""
        raise NotImplementedError

    def warn(self):
        """All logging logic goes here."""
        warnings.warn(self.code + ": " + self.msg)

    def __call__(self, *args, **kwargs):
        if not self.check(*args, **kwargs):
            self.warn()


class BadMinibatchSize(BaseStartTrainingCheck):
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


class MinibatchTooSmall(BaseStartTrainingCheck):
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


class MinibatchTooLarge(BaseStartTrainingCheck):
    """MinibatchTooLarge"""

    code = 'K1011'
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


class DataNotShuffled(BaseStartTrainingCheck):
    """DataNotShuffled.
    Not sure exactly how one would detect a random ordering of ints.
    Maybe the Runs test."""

    code = 'K102'
    msg = 'Training data is not shuffled. This may slow training down.'

    def check(self, X, y, *args, **kwargs):
        if is_onehotencoded(y):
            y = np.argmax(y, axis=1)
        elif not is_1d(y):
            return True
        z = runs_test(y)
        return abs(z) < 1.96


class TrainDevNotStratified(BaseStartTrainingCheck):
    """TrainDevNotStratified"""

    code = 'K103'
    msg = 'Train and validation data may not be stratified.'

    def check(self, X, y, *args, **kwargs):
        validation_data = kwargs.get('validation_data')
        if validation_data is None:
            return True
        y_val = validation_data[1]
        if is_1d(y):
            y_train = y
        elif is_onehotencoded(y):
            y_train = y.argmax(axis=1)
            y_val = y_val.argmax(axis=1)
        trn_labels, trn_label_counts = np.unique(y_train, return_counts=True)
        val_labels, val_label_counts = np.unique(y_val, return_counts=True)
        stratified = True
        if trn_labels != val_labels:
            stratified = False
        elif (trn_label_counts / val_label_counts).var().round(3) > 1e-3:
            stratified = False
        return stratified


class TrainDevNormalizedSeparately(BaseStartTrainingCheck):
    """TrainDevNormalizedSeparately"""


class TrainingSamplesCorrelated(BaseStartTrainingCheck):
    """TrainingSamplesCorrelated"""

    code = 'K301'
    msg = 'Training samples are correlated. ' + \
          'There may be redundancy in the data.'

    def check(self, X, y, *args, **kwargs):
        pca = PCA()
        pca.fit(X)
        m, c = mode(np.cumsum(pca.explained_variance_ratio_))
        return not(m[0].round(3) == 1 and c[0] > 1)


class DuplicateTrainingSamples(BaseStartTrainingCheck):
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


class DataNotNormalized(BaseStartTrainingCheck):
    """DataNotNormalized"""

    code = 'K303'
    msg = 'Training data not normalized.'

    def check(self, X, y, *args, **kwargs):
        centered = np.all(X.mean(0).round(3) == 0)
        scaled = np.all(X.var(0).round(3) == 1)
        return centered and scaled


class IncompatibleWeightInitializer(BaseStartTrainingCheck):
    code = 'K204'
    """IncompatibleWeightInitializer.
    This means checking if input data is compatible with the chosen
    initialization. Look at the literature for different initliazations.
    For example: LeCun initialization works with normalized inputs and tanh
    activations."""


class BadTanhEncoding(BaseStartTrainingCheck):
    """BadTanhEncoding"""


class BadLabelEncoding(BaseStartTrainingCheck):
    """BadLabelEncoding"""


class ParamsMoreThanTrainingSamples(BaseStartTrainingCheck):
    """ParamsMoreThanTrainingSamples"""

# Checks that happen after a model has been defined ###########################


class BaseModelCheck(ABC):
    """BaseModelCheck"""

    code = None
    msg = None
    __type = 'model'

    @classmethod
    def __subclasshook__(cls, other):
        if getattr(other, '__type', '') == cls.__type:
            return bool(getattr(other, 'code', False))
        return False

    def check(self, model):
        raise NotImplementedError

    def warn(self):
        """All logging logic goes here."""
        warnings.warn(self.code + ": " + self.msg)

    def __call__(self, model):
        if not self.check(model):
            self.warn()


class BadBatchNormPosition(BaseModelCheck):
    """BadBatchNormPosition"""


class WeightsTooSmall(BaseModelCheck):
    """WeightsTooSmall"""


class WeightsNotNormal(BaseModelCheck):
    """WeightsNotNormal.
    scipy.stats.normaltest"""


class UniformWeightInit(BaseModelCheck):
    """UniformWeightInit.
    Use ks_2samp for this."""

    code = 'K203'
    msg = 'Some layers have uniform weight initialization. ' + \
        'This may slow down training.'

    def check(self, model):
        for l in model.layers:
            if is_initializer_uniform(l):
                return False
        return True


class SigmoidActivation(BaseModelCheck):
    """SigmoidActivation"""

    code = 'K401'
    msg = 'Sigmoid activation found in an intermediate layer.'

    def check(self, model):
        for l in model.layers[:-1]:
            if l.activation == sigmoid:
                return False
        return True

# Checks that happen during training ##########################################


class VanishingGradients(BaseStartTrainingCheck):
    """VanishingGradients"""


class ExplodingGradients(BaseStartTrainingCheck):
    """ExplodingGradients"""


class ModelOverfittingStarted(BaseStartTrainingCheck):
    """ModelOverfittingStarted"""


class NoisyWeightUpdates(BaseStartTrainingCheck):
    """NoisyWeightUpdates"""
