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

"""
f90wrap.runtime

Contains everything needed by f90wrap generated Python modules at runtime
"""

from f90wrap.fortrantype import (FortranDerivedType,
                                FortranDerivedTypeArray,
                                FortranModule)
from f90wrap.arraydata import get_array
from f90wrap.sizeof_fortran_t import sizeof_fortran_t as _sizeof_fortran_t

sizeof_fortran_t = _sizeof_fortran_t()
empty_handle = [0]*sizeof_fortran_t
empty_type = FortranDerivedType.from_handle(empty_handle)

_f90wrap_classes = {}

class register_class(object):
    def __init__(self, cls_name):
        self.cls_name = cls_name

    def __call__(self, cls):
        global _f90wrap_classes
        if self.cls_name in _f90wrap_classes:
            raise RuntimeError("Duplicate Fortran class name {0}".format(self.cls_name))
        _f90wrap_classes[self.cls_name] = cls
        return cls

def lookup_class(cls_name):
    global _f90wrap_classes
    return _f90wrap_classes[cls_name]
