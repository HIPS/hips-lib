#!/usr/bin/env python

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(name='HIPS-Lib',
      version='0.1',
      description='Helpful package for Bayesian machine learning',
      author='Scott Linderman',
      author_email='slinderman@seas.harvard.edu',
      url='http://www.github.com/HIPS/hips-lib',
      # package_dir={'' : 'python'},
      packages=['hips',
                'hips.distributions',
                'hips.inference',
                'hips.movies',
                'hips.plotting'],
      ext_modules=cythonize('**/*.pyx'),
      include_dirs=[np.get_include(),],
     )

