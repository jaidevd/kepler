#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
"""
Unittests for Kepler checks.
"""

from unittest import TestCase, main
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.datasets import load_digits, make_classification
from kepler.main import ModelInspector
from kepler import checks as C
import pytest


class TestChecks(TestCase):

    # TODO: When runnin most tests, all DB transactions must be disabled.

    @classmethod
    def setUpClass(cls):
        cls.model = Sequential([
            Dense(32, input_shape=(64,)),
            Activation('sigmoid'),
            Dense(10),
            Activation('sigmoid')
        ])
        cls.model.compile(loss='categorical_crossentropy', optimizer=SGD())
        cls.digits = load_digits()
        cls.X = cls.digits['data']
        cls.y = to_categorical(cls.digits['target'])

    def _call_keras_method(self, check, method, *args, **kwargs):
        with ModelInspector(model=self.model, enable_model_search=False,
                            checks=[check]) as mp:
            mp.history = getattr(mp, method)(*args, **kwargs).history

    def assertWarns(self, check, func, *args, **kwargs):
        expected_msg = check.code + ": " + check.msg
        with pytest.warns(UserWarning, match=expected_msg):
            self._call_keras_method(check, func, *args, **kwargs)

    # Architecture level checks ###############################################

    def test_uniform_weight_init(self):
        msg = C.UniformWeightInit.code + ': ' + C.UniformWeightInit.msg
        with pytest.warns(UserWarning, match=msg):
            with ModelInspector(model=self.model, checks=[],
                                model_checks=[C.UniformWeightInit()],
                                enable_model_search=False) as mp:
                mp.history = mp.fit(self.X[:8], self.y[:8]).history

    def test_sigmoid_activation(self):
        msg = C.SigmoidActivation.code + ': ' + C.SigmoidActivation.msg
        with pytest.warns(UserWarning, match=msg):
            with ModelInspector(model=self.model, checks=[],
                                model_checks=[C.SigmoidActivation()],
                                enable_model_search=False) as mp:
                mp.history = mp.fit(self.X[:8], self.y[:8]).history

    # Miscellaneous checks ###################################################

    def test_bad_minibatch_size(self):
        self.assertWarns(C.BadMinibatchSize(), 'fit', self.X[:8], self.y[:8],
                         batch_size=3)

    def test_minibatch_too_small(self):
        self.assertWarns(C.MinibatchTooSmall(), 'fit', self.X[:100], self.y[:100],
                         batch_size=1)

    def test_minibatch_too_large(self):
        self.assertWarns(C.MinibatchTooLarge(), 'fit', self.X[:8], self.y[:8],
                         batch_size=4)

    def test_data_not_stratified(self):
        self.assertWarns(
            C.TrainDevNotStratified(), 'fit', self.X[:8], self.y[:8],
            validation_data=(self.X[8:11], self.y[8:11]), batch_size=8)

    def test_data_not_shuffled(self):
        y = np.random.choice(self.digits['target'], size=(128,))
        y = np.sort(y)
        y = to_categorical(y)
        self.assertWarns(C.DataNotShuffled(), 'fit', self.X[:128], y)

    def test_training_samples_correlated(self):
        X, _ = make_classification(128, n_features=64)
        self.assertWarns(
            C.TrainingSamplesCorrelated(), 'fit', X, self.y[:128],
            batch_size=128)

    def test_duplicate_training_samples(self):
        X = self.X[:8].copy()
        y = self.y[:8].copy()
        X[2] = X[1]
        y[2] = y[1]
        self.assertWarns(C.DuplicateTrainingSamples(), 'fit', X, y, batch_size=8)

    def test_data_normalized(self):
        self.assertWarns(C.DataNotNormalized(), 'fit', self.X[:8], self.y[:8],
                         batch_size=8)


if __name__ == '__main__':
    main()
