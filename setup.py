#!/usr/bin/env python

#  f90wrap: F90 to Python interface generator with derived type support
#
#  Copyright James Kermode 2011-2018
#
#  This file is part of f90wrap
#  For the latest version see github.com/jameskermode/f90wrap
#
#  f90wrap is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  f90wrap is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with f90wrap. If not, see <http://www.gnu.org/licenses/>.
# 
#  If you would like to license the source code under different terms,
#  please contact James Kermode, james.kermode@gmail.com

import os
import sys
from f90wrap import __version__
major, minor = sys.version_info[0:2]
if (major, minor) < (2, 4):
    sys.stderr.write('Python 2.4 or later is needed to use this package\n')
    sys.exit(1)

try:
    import setuptools
except ImportError:
    pass
from numpy.distutils.core import setup, Extension
from numpy.distutils.system_info import get_info

fortran_t = Extension('f90wrap.sizeof_fortran_t', ['f90wrap/sizeoffortran.f90'])

f2py_info = get_info('f2py')
arraydata_ext = Extension(name='f90wrap.arraydata',
                          sources=['f90wrap/arraydatamodule.c'] + f2py_info['sources'],
                          include_dirs=f2py_info['include_dirs'])

description = 'Fortran to Python interface generator with derived type support'
long_description = description
if os.path.exists('README.rst'):
    long_description = open('README.rst', 'r').read()
elif os.path.exists('README.md'):
    long_description = open('README.md', 'r').read()

setup(name='f90wrap',
      packages=['f90wrap'],
      scripts=['scripts/f90doc', 'scripts/f90wrap', 'scripts/f2py-f90wrap'],
      ext_modules=[fortran_t, arraydata_ext],
      version=__version__,
      description=description,
      long_description=long_description,
      author='James Kermode',
      author_email='james.kermode@gmail.com',
      url='https://github.com/jameskermode/f90wrap',
      download_url='https://github.com/jameskermode/f90wrap/archive/v%s.tar.gz' % __version__,
      install_requires=['numpy>=1.3.0'])
