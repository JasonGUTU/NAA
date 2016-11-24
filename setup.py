# -*- coding: utf-8 -*-
"""
Author : JasonGUTU
Email  : hellojasongt@gmail.com
Python : anaconda3
Date   : 2016/11/24
"""

import os
from setuptools import setup
from setuptools import find_packages


setup(
    name = 'NAA',
    version = '0.0.1',
    author = 'JasonGUTU',
    author_email = 'hellojasongt@gmail.com',

    url = '',
    keywords = ('Global Optimization', 'Evolutionary computation', 'Natural Aggregation Algorithm', 'Collective Decision Making'),
    description = 'Natural Aggregation Algorithm: A New Powerful Global Optimizer over Continuous Spaces',
    license = 'MIT License',
    install_requires = ['numpy'],

    classifiers=['Development Status :: 3 - Alpha', 'Programming Language :: Python :: 3.5'],

    packages = find_packages(exclude=['docs', 'test']),
    platforms = 'any',
)
