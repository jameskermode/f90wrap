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

version = {}
with open('f90wrap/__init__.py') as fp:
    exec(fp.read(), version)
__version__ = version['__version__']

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




setup(
      ext_modules=[fortran_t, arraydata_ext],
      download_url=f'https://github.com/jameskermode/f90wrap/archive/refs/tags/v{__version__}.tar.gz',
)
