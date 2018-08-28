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
Example of how to use Kepler's `fit` wrappers and model inspectors.
"""

from sklearn.datasets import load_digits
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.models import Sequential
from keras import layers as L
from kepler import ModelInspector

digits = load_digits()
X = digits['data']
y = to_categorical(digits['target'])

model = Sequential([L.Dense(10, input_shape=(64,), activation='softmax')])
model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
              optimizer=SGD(lr=0.001))
mi = ModelInspector(model=model)
fitter = mi.check_fit_for_overfitting(model.fit)
fitter(X, y)
print('\n\n' + '-' * 80)
print('The following should raise a warning.')
print('-' * 80)
fitter(X[:10], y[:10])

# As a context manager
model = Sequential([L.Dense(10, input_shape=(64,), activation='softmax')])
model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
              optimizer=SGD(lr=0.001))
print('\n\n' + '-' * 80)
print('The following should raise a warning.')
print('-' * 80)
with ModelInspector(model=model) as clf:
    clf.fit(X[:10], y[:10])

print('\n\n' + '-' * 80)
print('The following should run w/o warnings.')
print('-' * 80)
model.fit(X[:10], y[:10])
