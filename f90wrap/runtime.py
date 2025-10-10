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

import ctypes

import numpy as np

from f90wrap.fortrantype import (FortranDerivedType,
                                FortranDerivedTypeArray,
                                FortranModule)
from f90wrap.arraydata import get_array
from f90wrap.sizeof_fortran_t import sizeof_fortran_t as _sizeof_fortran_t

sizeof_fortran_t = _sizeof_fortran_t()
empty_handle = [0]*sizeof_fortran_t
empty_type = FortranDerivedType.from_handle(empty_handle)

_f90wrap_classes = {}

_DIRECT_C_CTYPES = {}


def _register_scalar_dtype(dtype_name: str, ctype) -> None:
    try:
        code = np.dtype(dtype_name).num
    except TypeError:
        return
    _DIRECT_C_CTYPES[code] = ctype


_register_scalar_dtype('bool_', ctypes.c_bool)
_register_scalar_dtype('int8', ctypes.c_int8)
_register_scalar_dtype('int16', ctypes.c_int16)
_register_scalar_dtype('int32', ctypes.c_int32)
_register_scalar_dtype('int64', ctypes.c_int64)
_register_scalar_dtype('float32', ctypes.c_float)
_register_scalar_dtype('float64', ctypes.c_double)
try:
    _register_scalar_dtype('longdouble', ctypes.c_longdouble)
except AttributeError:
    pass

_COMPLEX_HANDLERS = {}


def _register_complex_dtype(dtype_name: str, scalar_ctype, np_dtype_name: str) -> None:
    try:
        code = np.dtype(dtype_name).num
    except TypeError:
        return
    _COMPLEX_HANDLERS[code] = (scalar_ctype, np.dtype(np_dtype_name))


_register_complex_dtype('complex64', ctypes.c_float, 'complex64')
_register_complex_dtype('complex128', ctypes.c_double, 'complex128')
try:
    _register_complex_dtype('complex256', ctypes.c_longdouble, 'complex256')
except AttributeError:
    pass


def direct_c_array(dtype_code, shape, handle):
    """Construct a NumPy view over Fortran memory using Direct-C metadata."""

    if shape is None:
        raise ValueError("Shape metadata is required for Direct-C arrays")

    dims = tuple(int(dim) for dim in shape)
    total = 1
    for dim in dims:
        total *= dim

    if dtype_code in _COMPLEX_HANDLERS:
        scalar_ctype, np_dtype = _COMPLEX_HANDLERS[dtype_code]
        buffer = (scalar_ctype * (total * 2)).from_address(int(handle))
        complex_view = np.ctypeslib.as_array(buffer).view(np_dtype)
        return np.reshape(complex_view, dims, order='F')

    ctype = _DIRECT_C_CTYPES.get(dtype_code)
    if ctype is None:
        raise TypeError(f"Unsupported Direct-C dtype code {dtype_code}")

    buffer = (ctype * total).from_address(int(handle))
    array = np.ctypeslib.as_array(buffer)
    return np.reshape(array, dims, order='F')

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
