#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# JSM product code
#
# (C) Copyright 2018 Juxt SmartMandate Pvt Ltd
# All right reserved.
#
# This file is confidential and NOT open source.  Do not distribute.
#

"""

"""

from configparser import ConfigParser, NoOptionError, NoSectionError
from datetime import datetime
import os
import os.path as op
import shutil
import sqlite3
from sqlalchemy import create_engine

from kepler.db_models import KeplerBase


def load_config():
    config_dir = os.environ.get('KEPLER_HOME', False)
    if not config_dir:
        config_dir = op.expanduser('~/.kepler')
    path = op.join(config_dir, 'config.ini')
    config = ConfigParser()
    config.read(path)
    return config


def initdb(path, metadata=None):
    if op.isdir(path):
        path = op.join(path, 'kepler.db')
    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE metadata(created timestamp, location text);
        ''')
        cursor.execute('''
        INSERT INTO metadata(created, location) values (?, ?)''',
                       (datetime.now(), path))
        conn.commit()
    engine = get_engine(path)
    KeplerBase.metadata.create_all(engine)


def init_config(path):
    path = op.expanduser(path)
    if op.isdir(path):
        path = op.join(path, 'config.ini')
    sampleconfig = op.join(op.dirname(__file__), 'fixtures', 'sample.ini')
    shutil.copy(sampleconfig, path)


def get_engine(dbpath=None):
    """
    Get an SQLAlchemy engine configured from the ~/.kepler/config.ini file.
    
    """
    if not dbpath:
        config = load_config()
        try:
            dbpath = config.get('default', 'db')
        except (NoOptionError, NoSectionError):
            dbpath = op.join(os.environ.get('KEPLER_HOME', '~/.kepler'), 'kepler.db')
            dbpath = op.abspath(dbpath)
    return create_engine('sqlite:///' + op.abspath(dbpath))


def is_power2(n):
    return n != 0 and ((n & (n - 1)) == 0)