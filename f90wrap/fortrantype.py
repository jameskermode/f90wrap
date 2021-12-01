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
import weakref


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class FortranModule(object):
    """
    Baseclass for Fortran modules

    Metaclass is set to Singleton, so only one instance of each subclass of
    FortranModule can be created.
    """

    _dt_array_initialisers = []

    __metaclass__ = Singleton

    def __init__(self):
        self._arrays = {}
        self._objs = {}

        # initialise any derived type arrays
        for init_array in self._dt_array_initialisers:
            init_array(self)


class FortranDerivedType(object):
    """
    Base class for Fortran derived types
    """

    _dt_array_initialisers = []

    def __init__(self):
        self._handle = None
        self._arrays = {}
        self._objs = {}
        self._alloc = True

        # initialise any derived type arrays
        for init_array in self._dt_array_initialisers:
            init_array(self)

    @classmethod
    def from_handle(cls, handle, alloc=False):
        self = cls.__new__(cls)
        FortranDerivedType.__init__(self)  # always call the base constructor only
        self._handle = handle
        self._alloc = alloc
        return self


class FortranDerivedTypeArray(object):
    def __init__(self, parent, getfunc, setfunc, lenfunc, doc, arraytype):
        self.parent = weakref.ref(parent)
        self.getfunc = getfunc
        self.setfunc = setfunc
        self.lenfunc = lenfunc
        self.doc = doc
        self.arraytype = arraytype

    def iterindices(self):
        return iter(range(len(self)))

    indices = property(iterindices)

    def items(self):
        for idx in self.indices:
            yield self[idx]

    def __iter__(self):
        return self.items()

    def __len__(self):
        parent = self.parent()
        if parent is None:
            raise RuntimeError("Array's parent has gone out of scope")
        return self.lenfunc(parent._handle)

    def __getitem__(self, i):
        parent = self.parent()
        if parent is None:
            raise RuntimeError("Array's parent has gone out of scope")

        # i += 1  # convert from 0-based (Python) to 1-based indices (Fortran)
        # YANN: as "i" is passed by reference, and would be incremented on each call ! This seems wrong to me
        #       so I propose to add the +1 on the function call instead, as following.
        element_handle = self.getfunc(parent._handle, i + 1)
        try:
            obj = parent._objs[tuple(element_handle)]
        except KeyError:
            obj = parent._objs[tuple(element_handle)] = self.arraytype.from_handle(element_handle)
        return obj

    def __setitem__(self, i, value):
        parent = self.parent()
        if parent is None:
            raise RuntimeError("Array's parent has gone out of scope")

        # i += 1 # convert from 0-based (Python) to 1-based indices (Fortran)
        # YANN: Same issue
        self.setfunc(parent._handle, i + 1, value._handle)
