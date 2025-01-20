import os

from Cython.Build import cythonize
from setuptools import setup

this_dir = os.path.dirname(__file__)
pyx_file = os.path.join(this_dir, "cython_quadtree.pyx")

setup(
    name="cython_quadtree",
    ext_modules=cythonize(pyx_file, language_level=3),
    zip_safe=False,
)
