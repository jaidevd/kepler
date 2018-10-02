#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
"""
Kepler.
"""

import json
from inspect import signature
import os
import os.path as op
import warnings
from uuid import uuid4
from datetime import datetime
from functools import wraps

import numpy as np  # noqa: F401
import pandas as pd
from h5py import File as H5File
from keras.models import Model
from sklearn.base import BaseEstimator
from sqlalchemy.orm import sessionmaker
from traitlets import (Dict, Enum, HasTraits, Instance, Tuple,  # noqa: F401
                       Unicode, Union, Integer)

from kepler.custom_traits import KerasModelWeights, KerasYamlSpec
from kepler.db_models import ModelDBModel, ExperimentDBModel, HistoryModel
from kepler.utils import count_params, get_engine, load_config

warnings.simplefilter('always', category=UserWarning)

engine = get_engine()


class Project(HasTraits):
    """An ML project."""

    problem_type = Enum('classification', 'regression')


class ModelInspector(HasTraits):

    model = Union([Instance(Model), Instance(BaseEstimator)])

    commit = Unicode()

    weights_path = KerasModelWeights()

    model_specs = KerasYamlSpec()

    keras_type = Unicode()

    def __init__(self, *args, **kwargs):
        """
        Overwritten from parent to include the created timestamp.
        """
        super(ModelInspector, self).__init__(*args, **kwargs)
        self.created = datetime.now()

    @property
    def keras_type(self):
        """Type of keras model.
        """
        return self.model.__class__.__name__

    @property
    def model_config(self):
        with H5File(self.weights_path, 'r') as fout:
            config = fout.attrs.get('model_config')
        config = json.loads(config.decode())
        return config['config']

    @property
    def n_params(self):
        return count_params(self.model)

    @property
    def n_layers(self):
        return sum([c.trainable for c in self.model.layers])

    def check_fit_for_overfitting(self, fit_method):
        """Checks if the fit method might be overfitting."""
        @wraps(fit_method)
        def overfit_wrapper(X, y, *args, **kwargs):
            # Is this a standalone `model` or same as `self.model`?
            nrows = X.shape[0]
            if nrows < self.n_params:
                warnings.warn('You might overfit!')
            return fit_method(X, y, *args, **kwargs)
        return overfit_wrapper

    def write_model_spec(self):
        if not self.model_specs:
            config = load_config()
            uid = uuid4()
            specs_dir = op.expanduser(config.get('models', 'spec_dir'))
            if not op.isdir(specs_dir):
                os.makedirs(specs_dir)
            outpath = op.join(specs_dir, str(uid) + '.yml')
            with open(outpath, 'w') as fout:
                fout.write(self.model.to_yaml())
            self.model_specs = outpath

    def __enter__(self):
        self.instance = ModelDBModel()
        self.session = sessionmaker(bind=engine)()
        self.session.add(self.instance)
        self.session.commit()
        self.oldfit = self.model.fit
        # Added just as an example
        # self.model.fit = self.check_fit_for_overfitting(self.model.fit)
        self.write_model_spec()
        return self

    def __exit__(self, _type, value, traceback):
        self.model.fit = self.oldfit
        self.save()

    def save(self):
        """
        Save the model details to the Kepler db.

        """
        attrs = [k.name for k in ModelDBModel.__table__.columns if not k.primary_key]
        for attr in attrs:
            setattr(self.instance, attr, getattr(self, attr))
        self.session.add(self.instance)
        self.session.commit()
        self.session.close()

    def get_experiment(self):
        return Experiment(model=self)


class Experiment(HasTraits):

    model = Instance(ModelInspector)

    # what's in an experiment?
    start_time = Instance(datetime)
    end_time = Instance(datetime)
    n_epochs = Integer()
    start_metrics = Unicode()
    end_metrics = Unicode()

    def save(self):
        session = self.model.session
        attrs = []
        for k in ExperimentDBModel.__table__.columns:
            if not (k.primary_key or k.foreign_keys):
                attrs.append(k.name)
        self.instance = ExperimentDBModel(model=self.model.instance.id)
        for attr in attrs:
            setattr(self.instance, attr, getattr(self, attr))
        session.add(self.instance)
        session.commit()

    def save_history(self):
        df = pd.DataFrame.from_dict(self.model.model.history.history)
        session = self.model.session
        for i, metric in enumerate(df.to_dict(orient='records')):
            inst = HistoryModel(experiment=self.instance.id, epoch=i + 1,
                                metrics=json.dumps(metric))
            session.add(inst)
        session.commit()

    def run(self, method, *args, **kwargs):
        self.start_time = datetime.now()
        if isinstance(method, str):
            method = getattr(self.model.model, method)
        sig = signature(method)
        self.n_epochs = kwargs.get(
            'epochs', sig.parameters['epochs'].default)
        h = method(*args, **kwargs)
        self.end_time = datetime.now()
        start_metrics = {}
        end_metrics = {}
        for k, v in h.history.items():
            start_metrics[k] = v[0]
            end_metrics[k] = v[-1]
        self.start_metrics = json.dumps(start_metrics)
        self.end_metrics = json.dumps(end_metrics)
        self.save()
        self.save_history()
