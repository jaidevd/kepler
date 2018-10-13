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

"""


from keras.models import Sequential
from keras.layers import Dense, Activation
from kepler.main import ModelInspector
from sklearn.datasets import load_digits
from keras.optimizers import SGD
from keras.utils import to_categorical

digits = load_digits()
X = digits['data']
y = to_categorical(digits['target'])

model = Sequential([
    Dense(32, input_shape=(64,)),
    Activation('sigmoid'),
    Dense(10),
    Activation('sigmoid')
])
model.compile(loss='categorical_crossentropy', optimizer=SGD())

with ModelInspector(model=model) as mp:
    mp.fit(X, y, batch_size=3)
