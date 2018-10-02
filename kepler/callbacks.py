#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
"""Wrappers around Keras callbacks."""

from keras.callbacks import History


class ExperimentLogger(History):
    pass