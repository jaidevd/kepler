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
from functools import partial
from inspect import isclass
from uuid import uuid4

import numpy as np  # noqa: F401
import pandas as pd
from h5py import File as H5File
from scipy.sparse import vstack
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import sessionmaker
from traitlets import (Dict, Enum, HasTraits, Instance, Integer,  # noqa: F401
                       Tuple, Unicode, Union)

from kepler.custom_traits import KerasModelWeights, File, KerasModelMethods
from kepler.db_models import ExperimentDBModel, HistoryModel, ModelDBModel
from kepler.utils import (count_params, get_engine, load_config,
                          load_model_arch_mat, model_representation,
                          write_model_arch_mat, binary_prompt)
from keras.models import Model
from kepler import checks as C

warnings.simplefilter('always', category=UserWarning)

engine = get_engine()


class ModelInspector(HasTraits):

    model = Union([Instance(Model), Instance(BaseEstimator)])

    commit = Unicode()

    weights_path = KerasModelWeights()

    model_summary = File()

    keras_type = Unicode()

    archmat_index = Integer()

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

    def write_model_summary(self):
        if not self.model_summary:
            config = load_config()
            uid = uuid4()
            specs_dir = op.expanduser(config.get('models', 'spec_dir'))
            if not op.isdir(specs_dir):
                os.makedirs(specs_dir)
            outpath = op.join(specs_dir, str(uid) + '.txt')
            def _summary_writer(x, fh):  # noqa: E306
                fh.write(x + '\n')
            with open(outpath, 'w') as fout:
                self.model.summary(print_fn=partial(_summary_writer, fh=fout))
            self.model_summary = outpath

    def write_model_arch_vector(self, x=None):
        if not x:
            x = model_representation(self.model)
        X = load_model_arch_mat()
        if X is None:
            X = x
        else:
            X = vstack((X, x))
        self.archmat_index = X.shape[0] - 1
        write_model_arch_mat(X)

    def __enter__(self):
        self.instance = ModelDBModel()
        self.session = sessionmaker(bind=engine)()
        self.session.add(self.instance)
        self.session.commit()
        self.write_model_summary()
        config = load_config()
        if config.get('models', 'enable_model_search'):
            x = model_representation(self.model)
            self.search(x)
        self.model_proxy = ModelProxy(self.model, self)
        self.model_proxy.setUp()
        return self.model_proxy

    def __exit__(self, _type, value, traceback):
        self.model_proxy.tearDown()
        self.write_model_arch_vector()
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

    def search(self, x=None, prompt=True):
        if x is None:
            x = model_representation(self.model)
        X = load_model_arch_mat()
        if X is None:  # nothing to search against
            return
        d = cosine_similarity(x, X).ravel()
        thresh = load_config().get('misc', 'model_sim_tol')
        d = d > float(thresh)
        if np.any(d):
            indices, = np.where(d)
            if prompt:
                n_similar = d.sum()
                print('There are {} models similar to this one.'.format(n_similar))
                see_archs = binary_prompt(
                    'Would you like to see their summaries?')
                if see_archs:
                    summary_preview_dir = op.join(os.getcwd(), 'model-summaries')
                    print('Enter location for summaries [{}]: '.format(summary_preview_dir))
                    user_choice = input('>>> ')
                    if user_choice:
                        summary_preview_dir = user_choice
                    if not op.isdir(summary_preview_dir):
                        os.makedirs(summary_preview_dir)
                    for summary_file in self.get_summaries(indices):
                        os.symlink(summary_file,
                                   op.join(summary_preview_dir,
                                           op.basename(summary_file)))
            continue_training = binary_prompt('Continue training?')
            if not continue_training:
                import sys
                sys.exit()
            return indices

    def get_summaries(self, indices):
        klass = self.instance.__class__
        q = self.session.query(klass)
        for inst in q.filter(klass.archmat_index.in_(map(lambda x: x.item(),
                                                         indices))):
            yield inst.model_summary


class ModelProxy(HasTraits):

    wrapped = KerasModelMethods(['fit', 'train_on_batch'])

    def __init__(self, model, caller=None):
        self.model = model
        self.caller = caller

    def register_checks(self):
        available_checks = []
        for attr in dir(C):
            obj = getattr(C, attr)
            if isclass(obj):
                if issubclass(obj, C.BaseCheck):
                    available_checks.append(obj())
        for func in self.wrapped:
            setattr(self.model, func, C.checker(getattr(self.model, func),
                                                available_checks))
            setattr(self, func, getattr(self.model, func))

    def setUp(self):
        self.orgfuncs = {func: getattr(self.model, func) for func in self.wrapped}
        self.register_checks()
        self.start_experiment()
        return self.model

    def tearDown(self):
        self.end_experiment()
        for funcname, orgfunc in self.orgfuncs.items():
            setattr(self.model, funcname, orgfunc)

    def start_experiment(self):
        self.experiment = Experiment(model=self.caller)
        self.experiment.start_time = datetime.now()

    def end_experiment(self):
        self.experiment.end_time = datetime.now()
        self.experiment.process_history()
        self.experiment.save()
        self.experiment.save_history()


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

    def process_history(self):
        h = self.model.model.history.history
        start_metrics = {}
        end_metrics = {}
        for k, v in h.items():
            start_metrics[k] = v[0]
            end_metrics[k] = v[-1]
        self.start_metrics = json.dumps(start_metrics)
        self.end_metrics = json.dumps(end_metrics)
        self.n_epochs = len(v)
