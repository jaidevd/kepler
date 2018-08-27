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

from ConfigParser import ConfigParser
from datetime import datetime
import os
import os.path as op
import shutil
import sqlite3


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
        path = op.join(path, 'db.sqlite3')
    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE metadata(created timestamp, location text);
        ''')
        cursor.execute('''
        INSERT INTO metadata(created, location) values (?, ?)''',
                       (datetime.now(), path))
        conn.commit()


def init_config(path):
    path = op.expanduser(path)
    if op.isdir(path):
        path = op.join(path, 'config.ini')
    sampleconfig = op.join(op.dirname(__file__), 'fixtures', 'sample.ini')
    shutil.copy(sampleconfig, path)
