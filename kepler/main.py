#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
"""
Kepler.
"""

import json
import os
import os.path as op
import warnings
from datetime import datetime
from functools import wraps, partial
from inspect import signature
from uuid import uuid4

import numpy as np  # noqa: F401
import pandas as pd
from h5py import File as H5File
from scipy.sparse import vstack
from sklearn.base import BaseEstimator
from sklearn.metrics import pairwise_distances
from sqlalchemy.orm import sessionmaker
from traitlets import (Dict, Enum, HasTraits, Instance, Integer,  # noqa: F401
                       Tuple, Unicode, Union)

from kepler.custom_traits import KerasModelWeights, KerasYamlSpec
from kepler.db_models import ExperimentDBModel, HistoryModel, ModelDBModel
from kepler.utils import (count_params, get_engine, load_config,
                          load_model_arch_mat, model_representation,
                          write_model_arch_mat)
from keras.models import Model

warnings.simplefilter('always', category=UserWarning)

engine = get_engine()


class Project(HasTraits):
    """An ML project."""

    problem_type = Enum('classification', 'regression')


class ModelInspector(HasTraits):

    model = Union([Instance(Model), Instance(BaseEstimator)])

    commit = Unicode()

    weights_path = KerasModelWeights()

    model_summary = KerasYamlSpec()

    keras_type = Unicode()

    def __init__(self, *args, **kwargs):
        """
        Overwritten from parent to include the created timestamp.
        """
        super(ModelInspector, self).__init__(*args, **kwargs)
        # overwrite the model.save method to be able to save the model
        # weightspath here.
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

    def write_model_summary(self):
        if not self.model_summary:
            config = load_config()
            uid = uuid4()
            specs_dir = op.expanduser(config.get('models', 'spec_dir'))
            if not op.isdir(specs_dir):
                os.makedirs(specs_dir)
            outpath = op.join(specs_dir, str(uid) + '.txt')
            def _summary_writer(x, fh):
                fh.write(x + '\n')
            with open(outpath, 'w') as fout:
                self.model.summary(print_fn=partial(_summary_writer, fh=fout))
            self.model_specs = outpath
    
    def write_model_arch_vector(self, x=None):
        if not x:
            x = model_representation(self.model)
        X = load_model_arch_mat()
        if X is None:
            X = x
        else:
            X = vstack((X, x))
        write_model_arch_mat(X)

    def __enter__(self):
        self.instance = ModelDBModel()
        self.session = sessionmaker(bind=engine)()
        self.session.add(self.instance)
        self.session.commit()
        self.oldfit = self.model.fit
        # Added just as an example
        # self.model.fit = self.check_fit_for_overfitting(self.model.fit)
        self.write_model_summary()
        self.write_model_arch_vector()
        config = load_config()
        if config.get('models', 'enable_model_search'):
            self.search()
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
    
    def search(self, prompt=True):
        x = model_representation(self.model)
        X = load_model_arch_mat()
        d = pairwise_distances(x, X, metric='cosine').ravel()
        thresh = load_config().get('misc', 'model_sim_tol')
        d = d < float(thresh)
        if np.any(d):
            indices, = np.where(d)
            indices += 1
            if prompt:
                n_similar = d.sum()
                print('There are {} models similar to this one.'.format(n_similar))
                see_archs = input(' Would you like to see their summaries? [Y|n] ')
                if see_archs.lower() == 'y':
                    summary_preview_dir = op.join(os.getcwd(), 'model-summaries')
                    print('Enter location for summaries [{}]: '.format(summary_preview_dir))
                    user_choice = input('>>> ')
                    if user_choice:
                        summary_preview_dir = user_choice
                    if not op.isdir(summary_preview_dir):
                        os.makedirs(summary_preview_dir)
                    for summary_file in self.get_summaries(indices):
                        os.symlink(summary_file, op.join(summary_preview_dir, summary_file))
            continue_training = input('Continue training? [y|N]: ')
            if continue_training.lower() in ('', 'no', 'N'):
                import sys
                sys.exit()
            return indices
    
    def get_summaries(self, indices):
        q = self.session.query(self.instance.__class__)
        for inst in q.filter(self.instance.__class__.id.in_(indices)):
            yield inst.model_summary

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
