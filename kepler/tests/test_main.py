"""Test the kepler.main module."""

import os

from keras.utils import to_categorical
from sklearn.datasets import load_digits
from sqlalchemy.orm import sessionmaker

from kepler.tests import TestKepler
from kepler.sample_models import mnist_shallow
from kepler import ModelInspector
from kepler.utils import get_engine
import kepler.db_models as db


class TestMain(TestKepler):

    @classmethod
    def setUpClass(cls):
        super(TestMain, cls).setUpClass()
        cls.engine = get_engine(cls.config.get('default', 'db'))

    def setUp(self):
        self.session = sessionmaker(self.engine)()

    def tearDown(self):
        self.session.close()

    def test_add_project(self):
        """Test if adding a project works."""
        db.add_project(self.engine, 'test_project')
        res = self.session.query(db.Project).filter(db.Project.name == 'test_project').all()
        self.assertEqual(len(res), 1)
        res = res[0]
        self.assertEqual(res.name, 'test_project')
        self.assertEqual(res.location, os.getcwd())
        self.assertEqual(res.desc, '')

    def test_default_project(self):
        """Check if an arbitrary model appears in the default project."""
        model = mnist_shallow()
        digits = load_digits()
        X = digits['data']
        y = digits['target']
        y = to_categorical(y)
        with ModelInspector(model=model, config=self.config) as mi:
            mi.fit(X, y)
        res = self.session.query(db.ProjectModel).all()[0]
        self.assertEqual(res.project.name, 'default')
        self.assertEqual(res.model.name, 'Sequential-4-layers')
