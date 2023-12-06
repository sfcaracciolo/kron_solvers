from setuptools import setup
from Cython.Build import cythonize

setup(
    name='kron_solvers',
    ext_modules=cythonize("src/kron_solvers/solvers.pyx"),
)