import os

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

c_extension = Extension('affine._C_extensions',
                        depends=['affine/c_extension/C_extensions.h'],
                        sources=['affine/c_extension/C_extensions.c'],
                        include_dirs=[os.path.join(np.get_include(), 'numpy')])

setup(name='py_affine',
      author='Barton Baker',
      version='0.3',
      packages=['affine',
                'affine.model'],
      include_dirs=[np.get_include()],
      description='This package offers a complete solver class for affine ' \
                    + 'models, specifically affine models of the term ' \
                    + 'structure',
     author_email="bartbkr@gmail.com",
     ext_modules=[c_extension],
     platforms='any')
