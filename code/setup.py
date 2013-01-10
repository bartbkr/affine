from distutils.core import setup, Extension

module1 = Extension('C_extension.c', ['affine/C_extension.c'])

#Package requires numpy
try:
    import numpy as np
except ImportError:
    nonumpy_msg = ("# numpy needed to finish setup.  run:\n\n"
    "    $ pip install numpy  # or easy_install numpy\n")
    sys.exit(nonumpy_msg)

setup(name = 'Barton Baker',
      version = '1.0',
      include_dirs = [np.get_include()],
      description = 'This package offers a complete solver class for affine ' \
                    + 'models, specifically affine models of the term ' \
                    + 'structure',
     author_email = "bartbkr@gmail.com",
     ext_modules = [module1])
