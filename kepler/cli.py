#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

"""

"""

import os
import os.path as op
import click
from kepler.utils import initdb


@click.group()
def main():
    click.echo('Welcome to Kepler!')


@click.command()
@click.option('--path', default='~/.kepler')
def init(path):
    path = op.expanduser(path)
    if not op.exists(path):
        os.mkdir(path)
    initdb(path)


main.add_command(init)
