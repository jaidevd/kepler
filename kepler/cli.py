#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

"""
CLI.
"""

import os
import os.path as op
import click
from sqlalchemy.exc import OperationalError
from kepler.utils import initdb, init_config, init_model_vectorizer, get_engine
from kepler import db_models as db


def get_project_name(path):
    default = op.basename(path)
    name = input(f'Name of the project? [{default}]: ')
    if not name:
        name = default
    return name


@click.command()
@click.option('--path', default=op.expanduser('~/.kepler'),
              help='Location of the Kepler configuration.')
def setup(path):
    path = op.expanduser(path)
    if not op.exists(path):
        os.mkdir(path)
    init_config(path)
    initdb(path)
    init_model_vectorizer(op.join(path, 'models', 'vectorizer.pkl'))
    click.echo('Welcome to Kepler!')


@click.group()
def add():
    pass


@click.command()
@click.option('-p', '--path', default='.', help='Location of the project.')
@click.option('-n', '--name', default='', help='Name of the project.')
@click.option('--desc', default='', help='Description of the project.')
def project(path, name, desc):
    path = op.abspath(path)
    if not op.isdir(path):
        click.echo(f'No such directory: {path}', err=True)
        return
    if not name:
        name = get_project_name(path)
    engine = get_engine()
    try:
        db.add_project(engine, name, path, desc)
    except OperationalError:
        click.echo('Please run kepler setup first.', err=True)


@click.group()
def main():
    pass


add.add_command(project)
main.add_command(setup)
main.add_command(add)
