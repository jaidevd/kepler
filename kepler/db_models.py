#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
"""
Database models.
"""

from datetime import datetime
import os

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy import String, Integer, Column, ForeignKey, DateTime, Text


KeplerBase = declarative_base()


class ModelDBModel(KeplerBase):
    """
    Model for containing models. Yeah. Tautology.

    Parameters
    ----------
    KeplerBase : [type]
        [description]

    """
    __tablename__ = 'models'

    id = Column(Integer, primary_key=True)
    name = Column(Text)
    weights_path = Column(String)
    created = Column(DateTime)
    model_config = Column(String)
    n_layers = Column(Integer)
    n_params = Column(Integer)
    keras_type = Column(String)
    archmat_index = Column(Integer)

    def __repr__(self):
        return f'ID {self.id}: {self.name}'


class Project(KeplerBase):
    """Model for containing project to model mappings."""
    __tablename__ = 'projects'

    name = Column(Text, primary_key=True)
    created = Column(DateTime)
    location = Column(String)
    desc = Column(Text)

    def __repr__(self):
        return f'Project {self.name} located at {self.location}'


class ProjectModel(KeplerBase):
    """Table containing model-to-project mappings."""
    __tablename__ = 'projectmodels'

    id = Column(Integer, primary_key=True)

    project_id = Column(Text, ForeignKey('projects.name'))
    project = relationship('Project', backref='projectmodels')

    model_id = Column(Integer, ForeignKey('models.id'))
    model = relationship('ModelDBModel', backref='projectmodels')

    def __repr__(self):
        return f'ID {self.id}: Model {self.model} belonging to project {self.project}.'


class ExperimentDBModel(KeplerBase):
    """
    Model for logging experiments.

    Parameters
    ----------
    KeplerBase : [type]
        [description]

    """
    __tablename__ = 'experiments'

    id = Column(Integer, primary_key=True)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    n_epochs = Column(Integer)
    start_metrics = Column(String)
    end_metrics = Column(String)
    model = Column(Integer, ForeignKey('models.id'))

    def __repr__(self):
        return 'Some experiment ID: ' + str(self.id)


class HistoryModel(KeplerBase):
    """
    Model for maintaining histories.

    Parameters
    ----------
    KeplerBase : [type]
        [description]

    """
    __tablename__ = 'history'

    id = Column(Integer, primary_key=True)
    experiment = Column(Integer, ForeignKey('experiments.id'))
    epoch = Column(Integer)
    metrics = Column(String)

    def __repr__(self):
        return 'Some history log ID: ' + str(self.id)


def add_project(engine, name, location=None, desc=''):
    if location is None:
        location = os.getcwd()
    project = Project(name=name, created=datetime.now(), location=location, desc=desc)
    session = sessionmaker(bind=engine)()
    session.add(project)
    session.commit()
