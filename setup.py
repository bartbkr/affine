"""
Setup for affine package
"""
import os
import sys

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

#Package requires cython
try:
    import cython
except ImportError:
    nonumpy_msg = ("# cython needed to finish setup.  run:\n\n" +
    "    $ pip install cython  # or easy_install cython\n")
    sys.exit(nonumpy_msg)
#Package requires numpy
try:
    import numpy
except ImportError:
    nonumpy_msg = ("# numpy needed to finish setup.  run:\n\n" +
    "    $ pip install numpy  # or easy_install numpy\n")
    sys.exit(nonumpy_msg)
#Package requires scipy
try:
    import scipy
except ImportError:
    noscipy_msg = ("# scipy needed to finish setup.  run:\n\n" +
    "    $ pip install scipy  # or easy_install scipy\n")
    sys.exit(noscipy_msg)
#Package requires pandas
try:
    import pandas
except ImportError:
    nopandas_msg = ("# pandas needed to finish setup.  run:\n\n" +
    "    $ pip install pandas  # or easy_install pandas\n")
    sys.exit(nopandas_msg)
#Package requires scipy
try:
    import statsmodels
except ImportError:
    nostatsmodels_msg = ("# statsmodels needed to finish setup.  run:\n\n" +
    "    $ pip install statsmodels  # or easy_install statsmodels\n")
    sys.exit(nostatsmodels_msg)
#Package requires tempita
try:
    try:
        from Cython import Tempita as tempita
    except ImportError:
        import tempita
except ImportError:
    notempita_msg = ("# tempita needed to finish setup.  run:\n\n" +
    "    $ pip install tempita  # or easy_install tempita\n")
    sys.exit(notempita_msg)

# DOC: Tempita create pyx files
code_cdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "affine")
for folder, subdirs, files in os.walk(code_cdir):
    for fil in files:
        if fil.endswith(".pyx.in"):
            with open(os.path.join(folder, fil), "r") as f:
                cython_file = tempita.sub(f.read())
            with open(os.path.join(folder, os.path.splitext(fil)[0]), "w") as f:
                f.write(cython_file)

extensions = [Extension('affine.model.Cython_extensions',
                       ['affine/extensions/Cython_extensions.pyx'])]

setup(
    name='affine',
    author='Barton Baker',
    version='0.3',
    packages=find_packages(),
    description='This package offers a solver class for affine ' \
                  + 'term structure models.',
    author_email="bartbkr@gmail.com",
    use_2to3=True,
    ext_modules=cythonize(extensions),
    platforms='any',
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
