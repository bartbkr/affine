from distutils.core import setup, Extension

#Package requires numpy
try:
    import numpy as np
except ImportError:
    nonumpy_msg = ("# numpy needed to finish setup.  run:\n\n"
    "    $ pip install numpy  # or easy_install numpy\n")
    sys.exit(nonumpy_msg)

c_extentsion = Extension('_C_extensions', ['C_extensions.c'],
                    include_dirs=[np.get_include() + "/numpy"])

setup(name='py_affine',
      author='Barton Baker',
      version='1.0',
      packages=['affine']
      include_dirs=[np.get_include()],
      description='This package offers a complete solver class for affine ' \
                    + 'models, specifically affine models of the term ' \
                    + 'structure',
     author_email="bartbkr@gmail.com",
     ext_modules=[c_extentsion])
