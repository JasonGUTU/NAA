from setuptools import setup
from setuptools import find_packages

setup(
    name = 'NAA',
    version = '0.0.1',
    keywords = ('Global Optimization', 'Evolutionary computation', 'Natural Aggregation Algorithm'),
    description = 'Natural Aggregation Algorithm: A New Powerful Global Optimizer over Continuous Spaces',
    license = 'MIT License',
    install_requires = ['numpy', 'sympy', 'scipy'],

    author = 'Jason GT',
    author_email = 'hellojasongt@gmail.com',

    packages = find_packages(),
    platforms = 'any',
)
