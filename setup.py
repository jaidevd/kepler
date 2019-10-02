#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

from setuptools import setup, find_packages
setup(name='kepler',
      version='0.0.2',
      py_modules=find_packages(),
      install_requires=[
        'Click',
        'h5py',
        'Keras',
        'nose',
        'numpy',
        'pandas',
        'pytest',
        'scikit-learn',
        'scipy',
        'SQLAlchemy',
        'tensorflow==1.12.2',
        'traitlets'
      ],
      entry_points='''
      [console_scripts]
      kepler=kepler.cli:main
      ''')
