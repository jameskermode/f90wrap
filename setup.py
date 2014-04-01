#!/usr/bin/env python
# HF XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# HF X
# HF X   f90wrap: F90 to Python interface generator with derived type support
# HF X
# HF X   Copyright James Kermode 2011
# HF X
# HF X   These portions of the source code are released under the GNU General
# HF X   Public License, version 2, http://www.gnu.org/copyleft/gpl.html
# HF X
# HF X   If you would like to license the source code under different terms,
# HF X   please contact James Kermode, james.kermode@gmail.com
# HF X
# HF X   When using this software, please cite the following reference:
# HF X
# HF X   http://www.jrkermode.co.uk/f90wrap
# HF X
# HF XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

import sys

major, minor = sys.version_info[0:2]
if (major, minor) < (2, 4):
    sys.stderr.write('Python 2.4 or later is needed to use this package\n')
    sys.exit(1)

try:
    import numpy
    if not tuple([int(x) for x in numpy.__version__.split('.')[0:3]]) >= (1, 2, 1):
        raise ImportError
except ImportError:
    sys.stderr.write('Numpy 1.2.1 (http://www.numpy.org) or later needed to use this package\n')
    sys.exit(1)

from numpy.distutils.core import setup, Extension

fortran_t = Extension('f90wrap.sizeof_fortran_t', ['f90wrap/sizeoffortran.f90'])

setup(name='f90wrap',
      packages=['f90wrap'],
      scripts=['scripts/f90doc', 'scripts/f90wrap'],
      version='0.0',
      description='Fortran to Python interface generator with derived type support',
      author='James Kermode',
      author_email='james.kermode@gmail.com',
      url='http://www.jrkermode.co.uk/f90wrap',
      ext_modules=[fortran_t])
