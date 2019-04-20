from unittest import TestCase
from kepler.utils import init_config, initdb, init_model_vectorizer
from tempfile import mkdtemp
from shutil import rmtree
from configparser import ConfigParser, ExtendedInterpolation
import os.path as op


class TestKepler(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config_dir = mkdtemp()
        init_config(cls.config_dir)
        cls.config = ConfigParser(interpolation=ExtendedInterpolation())
        with open(op.join(cls.config_dir, 'config.ini'), 'r') as fin:
            cls.config.read_file(fin)
        initdb(cls.config.get('default', 'db'))
        init_model_vectorizer(cls.config.get('models', 'vectorizer'))

    @classmethod
    def tearDownClass(cls):
        rmtree(cls.config_dir)
