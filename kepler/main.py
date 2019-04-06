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
from uuid import uuid4

import numpy as np  # noqa: F401
import pandas as pd
from scipy.sparse import vstack
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError
import tensorflow as tf
from traitlets import (Dict, Enum, HasTraits, Instance, Integer,  # noqa: F401
                       Tuple, Unicode, Union, Bool, List, default)

from kepler.custom_traits import KerasModelWeights, File, KerasModelMethods
from kepler.db_models import ExperimentDBModel, HistoryModel, ModelDBModel
from kepler.utils import (count_params, get_engine, load_config,
                          load_model_arch_mat, model_representation,
                          write_model_arch_mat, binary_prompt)
from keras import backend as K
from keras.models import Model, model_from_yaml
from kepler import checks as C

warnings.simplefilter('always', category=UserWarning)

engine = get_engine()


class ModelInspector(HasTraits):
    """Main entry point to a Kepler session.

    This class orchestrates model logging, inspection, running experiments and
    saving results.
    """

    # the keras / sklearn model
    model = Union([Instance(Model), Instance(BaseEstimator)])

    # last git commit associated with the model
    commit = Unicode()

    # path to the saved weights of the model
    weights_path = KerasModelWeights()

    # Yaml config of the model, written to a file
    model_config = File()

    # Type of model, keras.engine.training.{Model, Sequential}, etc
    keras_type = Unicode()

    # Index of the archmat corresponding to this model.
    archmat_index = Integer()

    enable_model_search = Bool(True)

    checks = List()

    model_checks = List()

    @default('checks')
    def _default_checks(self):
        subcls = C.BaseStartTrainingCheck.__subclasses__()
        return [c() for c in subcls if c.enabled]

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
    def n_params(self):
        """Count number of trainable parameters."""
        return count_params(self.model)

    @property
    def n_layers(self):
        """Number of trainable layers."""
        return sum([c.trainable for c in self.model.layers])

    def write_model_config(self):
        """Write the model's config to a yaml file.

        The default location is ~/.kepler/models/specs, which is controlled
        from the ('models', 'spec_dir') config option."""
        if not self.model_config:
            config = load_config()
            uid = uuid4()
            specs_dir = op.expanduser(config.get('models', 'spec_dir'))
            if not op.isdir(specs_dir):
                os.makedirs(specs_dir)
            outpath = op.join(specs_dir, str(uid) + '.txt')
            with open(outpath, 'w') as fout:
                fout.write(self.model.to_yaml())
            self.model_config = outpath

    def write_model_arch_vector(self, x=None):
        """
        Write the vectorized representation of the model architecture to the
        archmat file.

        Parameters
        ----------
        x : sparse vector, optional
            The sparse vector representing a model. If not specified, it is
            calculated for the current model.
        """
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
        """Setup a Kepler session by:

        1. adding the current model to the db
        2. saving the model config
        3. searching for similar models
        4. creating a modelproxy for the user to work with
        """
        self.instance = ModelDBModel()
        self.session = sessionmaker(bind=engine)()
        self.session.add(self.instance)
        try:
            self.session.commit()
        except OperationalError:
            raise RuntimeError('Kepler may not have initialized properly.',
                               'Please run kepler init and try again.')
        self.write_model_config()
        config = load_config()
        if self.enable_model_search:
            if config.get('models', 'enable_model_search'):
                x = model_representation(self.model)
                self.search(x)
        self.run_model_checks()
        self.model_proxy = ModelProxy(self.model, self, self.checks)
        self.model_proxy.setUp()
        return self.model_proxy

    def __exit__(self, _type, value, traceback):
        """Teardown the Kepler session by:

        1. undoing the modelproxy
        2. writing the model architecture to the archmat
        3. saving the model metadata to the db.
        """
        self.model_proxy.tearDown()
        self.write_model_arch_vector()
        self.save()

    def save(self):
        """Save the model details to the Kepler db."""
        table_columns = ModelDBModel.__table__.columns
        attrs = [k.name for k in table_columns if not k.primary_key]
        for attr in attrs:
            setattr(self.instance, attr, getattr(self, attr))
        self.session.add(self.instance)
        self.session.commit()
        self.session.close()

    def search(self, x=None, prompt=True):
        """Search the archmat for similar models.

        Parameters
        ----------

        x : sparse vector, optional
            The sparse vector to search. If not specified, computed for the
            current model.
        prompt : bool, optional
            Whether to prompt the user if similar models are found.
        """
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
                print('There are {} models similar to this one.'.format(
                    n_similar))
                see_archs = binary_prompt(
                    'Would you like to see their graphs?')
                if see_archs:
                    tf_logdir = load_config().get('models', 'tensorflow_logdir')
                    print('Enter location for saving graphs [{}]: '.format(
                        tf_logdir))
                    user_choice = input('>>> ')
                    if user_choice:
                        tf_logdir = user_choice
                    tf_logdir = op.expanduser(tf_logdir)
                    with GraphWriter(logdir=tf_logdir) as gw:
                        gw.write_graphs(self.get_model_configs(indices))
                    print('Graphs written to ' + tf_logdir)
                    print('Please point Tensorboard to ' + tf_logdir)
            continue_training = binary_prompt('Continue training?')
            if not continue_training:
                import sys
                sys.exit()
            return indices

    def get_model_configs(self, indices):
        """Iterate over model config files.

        For models specified in `indices`, iterate over the corresponding
        `model_config` column values, which are paths to files containing the
        model summaries.

        Parameters
        ----------

        indices : sequence
            Sequence of DB indices over which to iterate and find the model
            summaries.

        Yields
        ------
        str
            path to a yaml file containing the config of a model
        """
        klass = self.instance.__class__
        q = self.session.query(klass)
        for inst in q.filter(klass.archmat_index.in_(map(lambda x: x.item(),
                                                         indices))):
            yield inst.model_config

    def run_model_checks(self):
        """Run all checks enabled at the model level."""
        if not self.model_checks:
            checks = [c for c in C.BaseModelCheck.__subclasses__() if c.enabled]
            for check in checks:
                check()(self.model)
        else:
            for check in self.model_checks:
                check(self.model)


class ModelProxy(HasTraits):
    """Wrapper around Keras models that accommodates hooks into the model API.

    All methods of the model that need to be wrapped are specified in the
    `wrapped` trait.
    """

    # Methods to hijack from the Keras model.
    wrapped = KerasModelMethods(['fit', 'train_on_batch'])

    def __init__(self, model, caller=None, checks=None, *args, **kwargs):
        super(ModelProxy, self).__init__(*args, **kwargs)
        self.model = model
        self.caller = caller
        if checks is not None:
            self.checks = checks

    def register_checks(self, checks=None):
        """Decorate the `wrapped` methods of the Keras model to register
        checks and add them to self, so this class can work as a veritable
        Keras model proxy."""
        if checks:
            self.checks = checks
        else:
            if not hasattr(self, 'checks'):
                self.checks = self.caller.checks
        for func in self.wrapped:
            setattr(self.model, func, C.checker(getattr(self.model, func),
                                                self.checks))
            setattr(self, func, getattr(self.model, func))

    def setUp(self):
        """Setup the modelproxy by:

        1. Copying `wrapped` model methods for re-assignment later.
        2. Decorate these methods to register checks.
        3. Start an `Experiment` session.
        """
        self.orgfuncs = {f: getattr(self.model, f) for f in self.wrapped}
        self.register_checks()
        self.start_experiment()
        return self.model

    def tearDown(self):
        """Teardown the model proxy by:

        1. Ending the `Experiment` session.
        2. Un-decorating the `wrapped` methods by adding the original methods
        back to the keras models.
        """
        self.end_experiment()
        for funcname, orgfunc in self.orgfuncs.items():
            setattr(self.model, funcname, orgfunc)

    def start_experiment(self):
        """Start an `Experiment` session."""
        self.experiment = Experiment(model=self.caller)
        self.experiment.start_time = datetime.now()

    def end_experiment(self):
        """End an `Experiment` session and save results to the db."""
        self.experiment.end_time = datetime.now()
        self.experiment.process_history()
        self.experiment.save()
        self.experiment.save_history()


class Experiment(HasTraits):

    # The ModelInspector instance that controls this instance.
    model = Instance(ModelInspector)

    start_time = Instance(datetime)
    end_time = Instance(datetime)

    # number of epochs for which the experiment ran
    n_epochs = Integer()

    # Keras metrics at the beginning and end of the experiment
    start_metrics = Unicode()
    end_metrics = Unicode()

    def save(self):
        """Save the experiment instance to the db."""
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
        """Save the experiment history to the db."""
        df = pd.DataFrame.from_dict(self.model.model.history.history)
        session = self.model.session
        for i, metric in enumerate(df.to_dict(orient='records')):
            inst = HistoryModel(experiment=self.instance.id, epoch=i + 1,
                                metrics=json.dumps(metric))
            session.add(inst)
        session.commit()

    def process_history(self, h=None):
        """Parse the history attribute of the fitted model to extract
        attributes for the db."""
        if not h:
            h = self.model.model.history.history
        start_metrics = {}
        end_metrics = {}
        for k, v in h.items():
            start_metrics[k] = v[0]
            end_metrics[k] = v[-1]
        self.start_metrics = json.dumps(start_metrics)
        self.end_metrics = json.dumps(end_metrics)
        self.n_epochs = len(v)


class GraphWriter(object):

    def __init__(self, logdir):
        self.logdir = logdir

    def __enter__(self):
        self.orgsession = K.get_session()
        return self

    def write_graphs(self, modelspecs):
        for spec in modelspecs:
            with tf.Session() as sess:
                K.set_session(sess)
                with open(spec, 'r') as mspec:
                    model = model_from_yaml(mspec.read())  # noqa: F841
                subdir = op.splitext(op.basename(spec))[0]
                with tf.summary.FileWriter(op.join(self.logdir, subdir),
                                           sess.graph) as fw:
                    fw.flush()

    def __exit__(self, a, b, c):
        K.set_session(self.orgsession)
