#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#

from setuptools import setup, find_packages
setup(name='kepler',
      version='0.0.2',
      py_modules=find_packages(),
      install_requires=[
        'Click==7.0',
        'h5py==2.8.0',
        'Keras==2.2.4',
        'nose==1.3.7',
        'numpy==1.15.4',
        'pandas==0.23.4',
        'pytest==3.10.1',
        'scikit-learn==0.20.0',
        'scipy==1.1.0',
        'SQLAlchemy==1.2.14',
        'tensorflow==1.12.0',
        'traitlets==4.3.2'
      ],
      entry_points='''
      [console_scripts]
      kepler=kepler.cli:main
      ''')
