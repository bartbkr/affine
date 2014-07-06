"""
Setup for affine package
"""
import os
import sys

from setuptools import setup, Extension

#Package requires numpy
try:
    import numpy
except ImportError:
    nonumpy_msg = ("# numpy needed to finish setup.  run:\n\n"
    "    $ pip install numpy  # or easy_install numpy\n")
    sys.exit(nonumpy_msg)
#Package requires scipy
try:
    import scipy
except ImportError:
    noscipy_msg = ("# scipy needed to finish setup.  run:\n\n"
    "    $ pip install scipy  # or easy_install scipy\n")
    sys.exit(noscipy_msg)
#Package requires pandas
try:
    import pandas
except ImportError:
    nopandas_msg = ("# pandas needed to finish setup.  run:\n\n"
    "    $ pip install pandas  # or easy_install pandas\n")
    sys.exit(nopandas_msg)
#Package requires scipy
try:
    import statsmodels
except ImportError:
    nostatsmodels_msg = ("# statsmodels needed to finish setup.  run:\n\n"
    "    $ pip install statsmodels  # or easy_install statsmodels\n")
    sys.exit(nostatsmodels_msg)

c_extension = Extension('affine.model._C_extensions',
                        depends=['affine/extensions/C_extensions.h'],
                        sources=['affine/extensions/C_extensions.c'],
                        include_dirs=[os.path.join(numpy.get_include(),
                                                   'numpy')],
                        #this flag needed to be added for current Python 3.4
                        #compilation
                        extra_compile_args =
                            ["-Wno-error=declaration-after-statement"])

setup(
    name='affine',
    author='Barton Baker',
    version='0.3',
    packages=['affine',
              'affine.model',
              'affine.constructors'],
    description='This package offers a solver class for affine ' \
                  + 'term structure models.',
    author_email="bartbkr@gmail.com",
    use_2to3=True,
    ext_modules=[c_extension],
    platforms='any'
)
