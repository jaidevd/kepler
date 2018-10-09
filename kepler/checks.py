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
from functools import wraps
import warnings

import numpy as np

from kepler.utils import is_power2, is_1d, is_onehotencoded


class BaseCheck(object):

    code = None

    def run(self):
        raise NotImplementedError


# Minibatch Checks ############################################################

class BadMinibatchSize(BaseCheck):
    """BadMinibatchSize"""

    def run(self, fit_method):
        @wraps(fit_method)
        def fit_method_wrapper(X, y, *args, **kwargs):
            if len(args) > 0:
                batch_size = args[0]
            else:
                batch_size = None
            if batch_size is None:
                batch_size = kwargs.get('batch_size', 32)
            if not is_power2(batch_size):
                warnings.warn('Batch size is not a power of 2.')
            return fit_method(X, y, *args, **kwargs)
        return fit_method_wrapper


class MinibatchTooSmall(BaseCheck):
    """MinibatchTooSmall"""

    def run(self, fit_method):
        @wraps(fit_method)
        def fit_method_wrapper(X, y, *args, **kwargs):
            n_samples = X.shape[0]
            if len(args) > 0:
                batch_size = args[0]
            else:
                batch_size = None
            if batch_size is None:
                batch_size = kwargs.get('batch_size', 32)
            if batch_size / n_samples <= 0.0125:
                warnings.warn('Batch size too small.')
            return fit_method(X, y, *args, **kwargs)
        return fit_method_wrapper


class MinibatchTooLarge(BaseCheck):
    """MinibatchTooLarge"""

    def run(self, fit_method):
        @wraps(fit_method)
        def fit_method_wrapper(X, y, *args, **kwargs):
            n_samples = X.shape[0]
            if len(args) > 0:
                batch_size = args[0]
            else:
                batch_size = None
            if batch_size is None:
                batch_size = kwargs.get('batch_size', 32)
            if batch_size / n_samples >= 0.33:
                warnings.warn('Batch size too large.')
            return fit_method(X, y, *args, **kwargs)
        return fit_method_wrapper


class DataNotShuffled(BaseCheck):
    """DataNotShuffled.
    Not sure exactly how one would detect a random ordering of ints."""


class TrainDevNotStratified(BaseCheck):
    """TrainDevNotStratified"""

    def run(self, fit_method):
        @wraps(fit_method)
        def fit_method_wrapper(X, y, *args, **kwargs):
            validation_data = kwargs.get('validation_data')
            if validation_data is None:
                warnings.warn('{} not applicable.'.format(
                    self.__class__.__name__))
                return fit_method(X, y, *args, **kwargs)
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
            if not stratified:
                warnings.warn('Train and validation data may not be stratified.')
            return fit_method(X, y, *args, **kwargs)


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


class DuplicateTrainingSamples(BaseCheck):
    """DuplicateTrainingSamples"""
    # http://www.ryanhmckenna.com/2017/01/efficiently-remove-duplicate-rows-from.html


class DataNotNormalized(BaseCheck):
    """DataNotNormalized"""


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
