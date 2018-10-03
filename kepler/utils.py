#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
"""
Miscellaneous functions.
"""

import os
import os.path as op
import shutil
import sqlite3
from collections import Counter
from configparser import ConfigParser, NoOptionError, NoSectionError
from datetime import datetime

from keras import layers as L
from keras.engine.base_layer import Layer
from keras.engine.training import Model
from keras.utils.layer_utils import count_params as k_count_params
from scipy.io import mmread, mmwrite
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib
from sqlalchemy import create_engine
import yaml

from kepler.db_models import KeplerBase


def get_keras_layers():
    layers = []
    for attr in dir(L):
        try:
            if issubclass(getattr(L, attr), Layer):
                layers.append(attr)
        except TypeError:
            continue
    return layers


def load_config():
    config_dir = os.environ.get('KEPLER_HOME', False)
    if not config_dir:
        config_dir = op.expanduser('~/.kepler')
    path = op.join(config_dir, 'config.ini')
    config = ConfigParser()
    config.read(path)
    return config


def initdb(path, metadata=None):
    if op.isdir(path):
        path = op.join(path, 'kepler.db')
    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE metadata(created timestamp, location text);
        ''')
        cursor.execute('''
        INSERT INTO metadata(created, location) values (?, ?)''',
                       (datetime.now(), path))
        conn.commit()
    engine = get_engine(path)
    KeplerBase.metadata.create_all(engine)


def init_model_vectorizer(path=None):
    if path is None:
        config = load_config()
        path = config.get('models', 'vectorizer')
    path = op.expanduser(path)
    model_config_dir = op.dirname(path)
    if not op.isdir(model_config_dir):
        os.makedirs(model_config_dir)
    layers = get_keras_layers()
    layer_dv = DictVectorizer().fit([dict.fromkeys(layers, 0)])
    joblib.dump(layer_dv, path)


def get_model_vectorizer(path=None):
    if path is None:
        config = load_config()
        path = config.get('models', 'vectorizer')
    path = op.expanduser(path)
    return joblib.load(path)


def init_config(path):
    path = op.expanduser(path)
    if op.isdir(path):
        path = op.join(path, 'config.ini')
    sampleconfig = op.join(op.dirname(__file__), 'fixtures', 'sample.ini')
    shutil.copy(sampleconfig, path)


def get_engine(dbpath=None):
    """
    Get an SQLAlchemy engine configured from the ~/.kepler/config.ini file.

    """
    if not dbpath:
        config = load_config()
        try:
            dbpath = config.get('default', 'db')
        except (NoOptionError, NoSectionError):
            dbpath = op.join(os.environ.get('KEPLER_HOME', '~/.kepler'), 'kepler.db')
            dbpath = op.abspath(dbpath)
        dbpath = op.expanduser(dbpath)
    return create_engine('sqlite:///' + op.abspath(dbpath))


def is_power2(n):
    return n != 0 and ((n & (n - 1)) == 0)


def count_params(model):
    model._check_trainable_weights_consistency()
    tw = getattr(model, '_collected_trainable_weights',
                 model.trainable_weights)
    return k_count_params(tw)


def count_layers(model, trainable_only=True):
    if trainable_only:
        return sum([c.trainable for c in model.layers])
    return len(model.layers)


def count_layer_types(model):
    if isinstance(model, Model):
        layer_counts = Counter([c.__class__.__name__ for c in model.layers])
    else:
        layer_counts = Counter([c['class_name'] for c in model])
    return layer_counts


def layer_architecture(model):
    if isinstance(model, dict):
        layer_config = model['config']['layers']
    else:
        layer_config = model._updated_config()['config']['layers']
    for l in layer_config:
        l.pop('name', None)
        l.get('config', {}).pop('name', None)
    return layer_config


def model_representation(model, dv=None):
    """Get a sparse vector representation of a model.

    Arguments:
        model {keras.engine.training.Model or str} -- Path to a yaml
        file containing model descriptions or the actual model.

    """
    if isinstance(model, str):
        if op.isfile(model):
            with open(model, 'r') as fin:
                spec = yaml.load(fin)
        else:
            spec = yaml.load(model)
        model = layer_architecture(spec)
    layer_counts = count_layer_types(model)
    if not dv:
        dv = get_model_vectorizer()
    return dv.transform(layer_counts)


def load_model_arch_mat(path=None):
    if path is None:
        config = load_config()
        path = config.get('models', 'model_archs')
    path = op.expanduser(path)
    if not op.isfile(path):
        return
    return mmread(path)


def write_model_arch_mat(X, path=None):
    if path is None:
        config = load_config()
        path = config.get('models', 'model_archs')
    path = op.expanduser(path)
    mmwrite(path, X)
