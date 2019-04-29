from unittest import TestCase
from kepler.utils import init_config, initdb, init_model_vectorizer
from tempfile import mkdtemp
from shutil import rmtree
from configparser import ConfigParser, ExtendedInterpolation
import os.path as op
from sqlalchemy import create_engine
import pandas as pd


class TestKepler(TestCase):

    TABLES = {
        'experiments': {
            'id', 'start_time', 'end_time', 'n_epochs', 'start_metrics', 'end_metrics', 'model'},
        'history': {'id', 'experiment', 'epoch', 'metrics'},
        'metadata': {'created', 'location'},
        'models': {
            'id', 'name', 'weights_path', 'created', 'model_config', 'n_layers', 'n_params',
            'keras_type', 'archmat_index'},
        'projectmodels': {'id', 'project_id', 'model_id'},
        'projects': {'name', 'created', 'location', 'desc'}
    }

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

    def check_config(self, path):
        """Check exhaustively if the configuration has been written to path."""
        self.assertTrue(op.isdir(path))
        parser = ConfigParser(interpolation=ExtendedInterpolation())
        with open(op.join(path, 'config.ini'), 'r') as fin:
            parser.read_file(fin)
        home = parser.get('default', 'home')
        dbpath = op.join(home, 'kepler.db')
        self.assertEqual(dbpath, parser.get('default', 'db'))
        self.check_init_db(dbpath)

    def check_init_db(self, dbpath):
        engine = create_engine(f'sqlite:///{dbpath}')
        self.assertSetEqual(set(engine.table_names()), set(self.TABLES.keys()))
        for tname, cols in self.TABLES.items():
            df = pd.read_sql_table(tname, engine)
            self.assertSetEqual(set(df.columns), cols)
            if tname not in 'metadata projects'.split():
                self.assertEqual(df.shape[0], 0)
            elif tname == 'metadata':
                self.assertEqual(df.shape[0], 1)
            else:
                self.assertSequenceEqual(df['name'].tolist(), ['default'])
                self.assertSequenceEqual(df['location'].tolist(), [f'{op.dirname(dbpath)}'])
