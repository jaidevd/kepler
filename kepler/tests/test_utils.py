#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
"""
Tests for the kepler.utils module.
"""

from configparser import ConfigParser, ExtendedInterpolation
from datetime import datetime
import os.path as op
from shutil import rmtree
from sqlite3 import connect
from tempfile import NamedTemporaryFile, mkdtemp

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from kepler import utils
from kepler.tests import TestKepler
from sklearn.externals import joblib


class TestUtils(TestKepler):

    def _check_initdb(self, conn):
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table';")
        actual = [c[0] for c in cursor.fetchall()]
        self.assertSetEqual(set(actual), self.ideal_tables)

        # check that everything except metadat is empty
        for mname in 'models experiments history'.split():
            result = cursor.execute('SELECT COUNT(*) from ' + mname)
            self.assertListEqual(result.fetchall(), [(0,)])
        metadata = cursor.execute('SELECT * FROM metadata').fetchall()
        self.assertEqual(len(metadata), 1)
        metadata = metadata[0]
        created, location = metadata
        created = datetime.strptime(created, '%Y-%m-%d %H:%M:%S.%f')
        self.assertLessEqual(
            (datetime.now() - created).total_seconds(), 60)

    def test_initdb(self):
        # Check if initdb works.
        self.ideal_tables = {'metadata', 'models', 'experiments', 'history'}
        with NamedTemporaryFile() as ntf:
            utils.initdb(ntf.name)
            with connect(ntf.name) as conn:
                self._check_initdb(conn)
        # try with a directory now
        try:
            tempdir = mkdtemp()
            utils.initdb(tempdir)
            with connect(op.join(tempdir, 'kepler.db')) as conn:
                self._check_initdb(conn)
        finally:
            rmtree(tempdir)

    def test_model_vectorizer(self):
        # Test if the vectorizer functions work.
        ideal_layers = utils.get_keras_layers()
        with NamedTemporaryFile() as ntf:
            utils.init_model_vectorizer(ntf.name)
            vect = utils.get_model_vectorizer(ntf.name)
            self.assertSetEqual(set(vect.get_feature_names()),
                                set(ideal_layers))

    def test_init_config(self):
        """Test if config can be written to arbit locations."""
        ideal_configpath = op.join(op.dirname(__file__), '..', 'fixtures',
                                   'sample.ini')
        ideal_config = ConfigParser(interpolation=ExtendedInterpolation())
        with open(ideal_configpath, 'r') as fin:
            ideal_config.read_file(fin)
        actual_config = ConfigParser(interpolation=ExtendedInterpolation())
        with NamedTemporaryFile(mode='r+') as ntf:
            utils.init_config(ntf.name)
            ntf.seek(0)
            actual_config.read_file(ntf)
            ideal_config.set('default', 'home', op.dirname(ntf.name))
        self.assertListEqual(actual_config.sections(), ideal_config.sections())
        for secname, section in actual_config.items():
            self.assertDictEqual(dict(section), dict(ideal_config[secname]))

        # try with a directory
        try:
            tempdir = mkdtemp()
            utils.init_config(tempdir)
            ideal_config.set('default', 'home', tempdir)
            with open(op.join(tempdir, 'config.ini'), 'r') as fin:
                actual_config.read_file(fin)
            self.assertListEqual(actual_config.sections(), ideal_config.sections())
            for secname, section in actual_config.items():
                self.assertDictEqual(dict(section), dict(ideal_config[secname]))
        finally:
            rmtree(tempdir)

    def test_is_power2(self):
        for n in (0, 2, 16, 2048):
            self.assertTrue(utils.is_power2(2))
        self.assertFalse(utils.is_power2(3))
        self.assertFalse(utils.is_power2(-2))

    def test_count_params(self):
        model = Sequential([
            Dense(5, input_shape=(10,)),
            Dense(3)
        ])
        self.assertEqual(utils.count_params(model), 73)

    def test_count_layers(self):
        model = Sequential([
            Dense(5, input_shape=(10,)),
            Activation('sigmoid', trainable=False),
            Dense(3)
        ])
        self.assertEqual(utils.count_layers(model), 2)
        self.assertEqual(utils.count_layers(model, False), 3)

    def test_count_layer_types(self):
        model = Sequential([
            Dense(5, input_shape=(10,)),
            Activation('sigmoid', trainable=False),
            Dense(3)
        ])
        ideal = {'Dense': 2, 'Activation': 1}
        counts = utils.count_layer_types(model)
        for layer, cnt in counts.items():
            self.assertEqual(cnt, ideal.get(layer, 0))

    def test_layer_architecture(self):
        model = Sequential([
            Dense(5, input_shape=(10,)),
            Activation('sigmoid', trainable=False),
            Dense(3)
        ])
        arch = utils.layer_architecture(model)
        self.assertListEqual([c['class_name'] for c in arch],
                             ['Dense', 'Activation', 'Dense'])

    def test_model_reprsentation(self):
        model = Sequential([
            Dense(5, input_shape=(10,)),
            Activation('sigmoid', trainable=False),
            Dense(3)
        ])
        dv = joblib.load(self.config.get('models', 'vectorizer'))
        x = utils.model_representation(model, dv)
        self.assertEqual(x.getnnz(), 2)
        self.assertEqual(x.sum(), 3)
        self.assertEqual(x[0, dv.vocabulary_['Dense']], 2)
        self.assertEqual(x[0, dv.vocabulary_['Activation']], 1)

    def test_is_1d(self):
        x = np.random.rand(10,)
        self.assertTrue(utils.is_1d(x))
        self.assertTrue(utils.is_1d(x.reshape(1, -1)))
        self.assertTrue(utils.is_1d(x.reshape(-1, 1)))

    def test_is_onehotencoded(self):
        x = np.random.randint(0, 10, size=(100,))
        X = to_categorical(x)
        self.assertTrue(utils.is_onehotencoded(X))
        self.assertFalse(utils.is_onehotencoded(X * 2))
        self.assertFalse(utils.is_onehotencoded(X + 2))
        self.assertFalse(utils.is_onehotencoded(X - 1))


if __name__ == "__main__":
    from unittest import main
    main()
