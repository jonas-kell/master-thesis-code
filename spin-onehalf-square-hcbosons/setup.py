from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("*", ["*.py"], include_dirs=[numpy.get_include()]),
]
setup(
    name="Hello world app",
    ext_modules=cythonize(extensions, annotate=True),
)
