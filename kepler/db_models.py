#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
"""
Database models.
"""

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import String, Integer, Column, ForeignKey, DateTime


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
    weights_path = Column(String)
    model_definition = Column(String)
    created = Column(DateTime)

    def __repr__(self):
        return 'Some model ID: ' + str(self.id)


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

    def __repr__(self):
        return 'Some history log ID: ' + str(self.id)