#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
"""
Miscellaneous functions.
"""

import os
import os.path as op
import sqlite3
from collections import Counter
from configparser import ConfigParser, NoOptionError, NoSectionError, ExtendedInterpolation
from datetime import datetime

from keras import layers as L
from keras.engine.base_layer import Layer
from keras.engine.training import Model
from keras.models import Sequential
from keras.utils.layer_utils import count_params as k_count_params
import numpy as np
from scipy.io import mmread, mmwrite
from scipy.stats import ks_2samp
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib
from sqlalchemy import create_engine
import yaml

from kepler.db_models import KeplerBase, add_project


def find_configuration():
    """Find where the configuration might be located."""
    home = os.environ.get('KEPLER_HOME', False)
    if not home:
        home = op.join(os.getcwd(), '.kepler')
        if not op.isdir(home):
            home = op.expanduser('~/.kepler')
            if not op.isdir(home):
                raise RuntimeError('No Kepler configuration found!')
    return home


def get_keras_layers():
    """Get a list of all available layers in Keras."""
    layers = []
    for attr in dir(L):
        try:
            if issubclass(getattr(L, attr), Layer):
                layers.append(attr)
        except TypeError:
            continue
    return layers


def load_config(path=None):
    """Load Kepler config."""
    if path is None:
        config_dir = os.environ.get('KEPLER_HOME', False)
        if not config_dir:
            config_dir = op.expanduser('~/.kepler')
        path = op.join(config_dir, 'config.ini')
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(path)
    return config


def initdb(path):
    """Initialize the Kepler db.

    Parameters
    ----------

    path : str
        Destination path of the sqlite db.
    """
    if op.isdir(path):
        dbpath = op.join(path, 'kepler.db')
    else:
        dbpath = path
        path = op.dirname(path)
    with sqlite3.connect(dbpath) as conn:
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE metadata(created timestamp, location text);
        ''')
        cursor.execute('''
        INSERT INTO metadata(created, location) values (?, ?)''',
                       (datetime.now(), dbpath))
        conn.commit()
    engine = get_engine(dbpath)
    KeplerBase.metadata.create_all(engine)

    # add the default project
    add_project(engine, name='default', location=path, desc='Default project')


def init_model_vectorizer(path=None):
    """Create and save a sklearn DictVectorizer that can vectorize keras model
    architectures.

    Parameters
    ----------

    path : str, optional
        Path to which the serialized vectorizer will be dumped. If unspecified,
        this value defaults to the ('models', 'vectorizer') config option.
    """
    if path is None:
        config = load_config()
        path = config.get('models', 'vectorizer')
    path = op.expanduser(path)
    os.makedirs(op.dirname(path), exist_ok=True)
    layers = get_keras_layers()
    layer_dv = DictVectorizer().fit([dict.fromkeys(layers, 0)])
    joblib.dump(layer_dv, path)


def get_model_vectorizer(path=None):
    """Load the model vectorizer.

    Parameters
    ----------

    path : str, optional
        Location from which to load the model vectorizer. If unspecified,
        defaults to the ('models', 'vectorizer') config option.

    Returns
    -------
    sklearn.feature_extraction.dict_vectorizer.DictVectorizer
    """
    if path is None:
        config = load_config()
        path = config.get('models', 'vectorizer')
    path = op.expanduser(path)
    return joblib.load(path)


def init_config(path):
    """Write the initial Kepler configuration.

    Parameters
    ----------

    path : str
        Path to which to write the config.
    """
    path = op.expanduser(path)
    if op.isdir(path):
        cfgpath = op.join(path, 'config.ini')
    else:
        cfgpath = path
    home = op.dirname(cfgpath)
    sampleconfig = op.join(op.dirname(__file__), 'fixtures', 'sample.ini')
    config = ConfigParser(interpolation=ExtendedInterpolation())
    with open(sampleconfig, 'r') as fin:
        config.read_file(fin)
    config.set('default', 'home', home)
    with open(cfgpath, 'w') as fout:
        config.write(fout)


def get_engine(dbpath=None):
    """Get an SQLAlchemy engine configured from the ~/.kepler/config.ini file.

    Parameters
    ----------

    dbpath : str, optional
        Path from which to read the db. If unspecified, defaults to the
        ('default', 'db') config option.

    Returns
    -------
    sqlalchemy.engine.base.Engine
    """
    if not dbpath:
        config = load_config()
        try:
            dbpath = config.get('default', 'db')
        except (NoOptionError, NoSectionError):
            dbpath = op.join(os.environ.get('KEPLER_HOME', '~/.kepler'),
                             'kepler.db')
            dbpath = op.abspath(dbpath)
        dbpath = op.expanduser(dbpath)
    return create_engine('sqlite:///' + op.abspath(dbpath))


def is_power2(n):
    return n != 0 and ((n & (n - 1)) == 0)


def count_params(model):
    """Count the trainable parameters in a Keras model.

    Parameters
    ----------

    model : keras.engine.training.Model
        A Keras model.

    Returns
    -------
    int
        Number of trainable parameters in the model.
    """
    model._check_trainable_weights_consistency()
    tw = getattr(model, '_collected_trainable_weights',
                 model.trainable_weights)
    return k_count_params(tw)


def count_layers(model, trainable_only=True):
    """Count the number of layers in a Keras model.

    Parameters
    ----------

    model : keras.engine.training.Model
        A Keras model.
    trainable_only : bool, optional
        Whether to count only trainable layers.

    Returns
    -------
    int
        Number of layers in the model.
    """
    if trainable_only:
        return sum([c.trainable for c in model.layers])
    return len(model.layers)


def count_layer_types(model):
    """Count layers by layer type.

    Parameters
    ----------

    model : keras.engine.training.Model
        A Keras model.

    Returns
    -------
    collections.Counter
        Counter object containing number of layers in the model by layer type.
    """
    return Counter([c.__class__.__name__ for c in model.layers])


def layer_architecture(model):
    """Get a Keras model config as a dictionary.

    Strip all info like layer names that are not essential to the definition of
    the model.

    Parameters
    ----------

    model : dict or keras.engine.training.Model
        A Keras model or a dict representing the config of the model (result of
        `model.get_config()`)

    Returns
    -------
    dict
        The model configuration dict stripped of useless details.
    """
    if isinstance(model, dict):
        layer_config = model['config']['layers']
    else:
        layer_config = model._updated_config()['config']['layers']
    for l in layer_config:
        l.pop('name', None)
        l.get('config', {}).pop('name', None)
    return layer_config


def model_representation(model, dv):
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
    return dv.transform(layer_counts)


def load_model_arch_mat(path=None):
    """Load the model architecture matrix.

    Parameters
    ----------

    path : str, optional
        Path from which to read the matrix. If unspecified, defaults to the
        ('models', 'model_archs') config options.

    Returns
    -------
    sparse matrix
        A sparse matrix with each row a model and each column a layer type.
        The elements of the matrix represent how many layers of a given type
        are present in a model.
    """
    if path is None:
        config = load_config()
        path = config.get('models', 'model_archs')
    path = op.expanduser(path)
    if not op.isfile(path):
        return
    return mmread(path)


def write_model_arch_mat(X, path=None):
    """Save the model architecture matrix to disk.

    Parameters
    ----------

    X : sparse matrix
        A sparse matrix with each row a model and each column a layer type.
        The elements of the matrix represent how many layers of a given type
        are present in a model.
    path : str, optional
        Path to which to write the matrix. If unspecified, defaults to the
        ('models', 'model_archs') config options.
    """
    if path is None:
        config = load_config()
        path = config.get('models', 'model_archs')
    path = op.expanduser(path)
    mmwrite(path, X)


def binary_prompt(msg, default='y'):
    """A yes/no prompt.

    Parameters
    ----------

    msg : str
        The message to print at the prompt.
    default : str, optional
        Default result of the prompt.

    Returns
    -------
    bool
        Whether the prompt resulted in a 'yes'.
    """
    y = 'y yes'.split()
    n = 'n no'.split()
    all_choices = y + n + ['']
    if default.lower() in y:
        choices = '[Y/n]'
    elif default.lower() in n:
        choices = '[y/N]'
    output = input(' '.join((msg, choices, ': ')))
    while output.lower() not in all_choices:
        output = input(' '.join((msg, choices, ': ')))
    if not output:
        return default.lower() == 'y'
    return output.lower() in y


def is_1d(x):
    """If input is a column or row vector or an array.

    Parameters
    ----------

    x : array-like

    Returns
    -------
    bool
        Whether `x` is 1D.
    """
    return x.ndim == 1 or (x.ndim == 2 and 1 in x.shape)


def is_onehotencoded(x):
    """If input is a one-hot encoded representation of some set of values.

    Parameters
    ----------

    x : array-like

    Returns
    -------
    bool
        Whether `x` is a one-hot encoded / categorical representation.
    """
    if x.ndim != 2:
        return False
    fractional, integral = np.modf(x)
    if fractional.sum() != 0:
        return False
    if not np.array_equal(integral, integral.astype(bool)):
        return False
    return np.all(integral.sum(axis=1) == 1)


def is_uniform(x):
    marker = np.random.uniform(x.min(), x.max(), size=(100,))
    _, pval = ks_2samp(x.ravel(), marker)
    return pval > 1e-3


def is_initializer_uniform(layer):
    initializer = getattr(layer, 'kernel_initializer', False)
    if initializer:
        cname = initializer.__class__.__name__
        if cname not in ('RandomUniform', 'VarianceScaling'):
            return False
        if cname == 'RandomUniform':
            return True
        if initializer.distribution == 'uniform':
            return True
    return False


def runs_test(x):
    """Wald-Wolfowitz Runs Test for checking randomness.

    Adapted from
    https://www.itl.nist.gov/div898/handbook/eda/section3/eda35d.htm

    Parameters
    ----------
    x : array-like
        The array to be tested for randomness.

    Returns
    -------
    float
        The runs test statistic.
    """
    # check if `x` is already binary
    unx = np.unique(x)
    if unx.shape[0] == 2:
        low, high = unx
        seq = np.zeros(x.shape, dtype=bool)
        seq[x == high] = True
    else:
        med = np.median(x)
        seq = x[x != med] > med
    n_runs = np.diff(seq).sum() + 1
    n_pos = seq.sum()
    n_neg = seq.shape[0] - n_pos
    rhat = (2 * n_pos * n_neg) / (n_pos + n_neg) + 1
    sr2 = 2 * n_pos * n_neg * (2 * n_pos * n_neg - n_pos - n_neg) / \
        (n_pos + n_neg) ** 2 / (n_pos + n_neg - 1)
    sr = np.sqrt(sr2)
    return (n_runs - rhat) / sr


def name_keras_model(model):
    """Name the keras model.

    Parameters
    ----------
        model : keras.training.Model

    Returns
    -------
    str
        A made up name for the model

    Example
    -------

    """
    if isinstance(model, Sequential):
        mtype = 'Sequential'
    elif isinstance(model, Model):
        mtype = 'FunctionalModel'
    n_layers = count_layers(model)
    return f'{mtype}-{n_layers}-layers'
