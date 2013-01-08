from distutils.core import setup, Extension

module1 = Extension('C_extension.c', ['C_extension.c'])

setup(name = 'Barton Baker',
      version = '1.0',
      description = 'This package offers a complete solver class for affine ' \
                    + 'models, specifically affine models of the term ' \
                    + 'structure',
     author_email = "bartbkr@gmail.com",
     ext_modules = [module1])
