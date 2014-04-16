import os
import sys

from distutils.core import setup, Extension

#Package requires numpy
try:
    import numpy as np
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

c_extension = Extension('affine.model._C_extensions',
                        depends=['affine/extensions/C_extensions.h'],
                        sources=['affine/extensions/C_extensions.c'],
                        include_dirs=[os.path.join(np.get_include(), 'numpy')])

setup(
    name='py_affine',
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
