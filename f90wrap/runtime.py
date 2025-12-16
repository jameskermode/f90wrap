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
        dtype = np.dtype(dtype_name)
    except TypeError:
        return
    _DIRECT_C_CTYPES[dtype.num] = (ctype, dtype)


_register_scalar_dtype('bool_', ctypes.c_bool)
_register_scalar_dtype('int8', ctypes.c_int8)
_register_scalar_dtype('int16', ctypes.c_int16)
_register_scalar_dtype('int32', ctypes.c_int32)
_register_scalar_dtype('int64', ctypes.c_int64)
_register_scalar_dtype('uint8', ctypes.c_uint8)
_register_scalar_dtype('uint16', ctypes.c_uint16)
_register_scalar_dtype('uint32', ctypes.c_uint32)
_register_scalar_dtype('uint64', ctypes.c_uint64)
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
    """Construct a NumPy view over Fortran memory using Direct-C metadata.

    WARNING: This function creates a NumPy view over memory at the given address.
    The caller MUST ensure the handle points to valid, allocated Fortran memory.
    Invalid handles will cause segmentation faults or undefined behavior.

    Args:
        dtype_code: NumPy dtype code for the array elements
        shape: Tuple of array dimensions
        handle: Integer memory address from Fortran (must be valid)

    Returns:
        NumPy array view over the Fortran memory

    Raises:
        ValueError: If shape is None or handle is invalid
        TypeError: If handle is not an integer or dtype_code is unsupported
    """
    if shape is None:
        raise ValueError("Shape metadata is required for Direct-C arrays")

    # Validate and convert dimensions with overflow protection
    dims = []
    total = 1
    max_extent = 2**63 - 1
    for dim in shape:
        try:
            d = int(dim)
        except (TypeError, ValueError):
            raise TypeError(f"Invalid dimension value {dim!r}") from None
        if d < 0:
            raise ValueError("Negative dimensions are not allowed in Direct-C arrays")
        if total and d and total > max_extent // d:
            raise OverflowError("Direct-C array size is too large")
        dims.append(d)
        total *= d
    dims = tuple(dims)

    try:
        addr = int(handle)
    except (TypeError, ValueError):
        raise TypeError("Direct-C handle must be an integer address") from None
    if addr <= 0:
        raise ValueError(
            "Direct-C handle must be a positive memory address. "
            "Received invalid handle - this may indicate uninitialized Fortran memory."
        )

    if dtype_code in _COMPLEX_HANDLERS:
        scalar_ctype, np_dtype = _COMPLEX_HANDLERS[dtype_code]
        if total and total > max_extent // 2:
            raise OverflowError("Direct-C complex array size is too large")
        if total == 0:
            return np.empty(dims, dtype=np_dtype, order='F')
        buffer = (scalar_ctype * (total * 2)).from_address(addr)
        complex_view = np.ctypeslib.as_array(buffer).view(np_dtype)
        return np.reshape(complex_view, dims, order='F')

    entry = _DIRECT_C_CTYPES.get(dtype_code)
    if entry is None:
        raise TypeError(f"Unsupported Direct-C dtype code {dtype_code}")
    ctype, np_dtype = entry
    if total == 0:
        return np.empty(dims, dtype=np_dtype, order='F')

    buffer = (ctype * total).from_address(addr)
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
