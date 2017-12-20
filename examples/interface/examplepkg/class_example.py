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
Module class_example


Defined at ./example.f90 lines 2-73

"""
from __future__ import print_function, absolute_import, division
import _examplepkg
import f90wrap.runtime
import logging

_arrays = {}
_objs = {}

@f90wrap.runtime.register_class("Example")
class Example(f90wrap.runtime.FortranDerivedType):
    """
    Type(name=example)
    
    
    Defined at ./example.f90 lines 12-16
    
    """
    def __init__(self, handle=None):
        """
        self = Example()
        
        
        Defined at ./example.f90 lines 12-16
        
        
        Returns
        -------
        this : Example
        	Object to be constructed
        
        
        Automatically generated constructor for example
        """
        f90wrap.runtime.FortranDerivedType.__init__(self)
        self._handle = _examplepkg.f90wrap_example_initialise()
    
    def __del__(self):
        """
        Destructor for class Example
        
        
        Defined at ./example.f90 lines 12-16
        
        Parameters
        ----------
        this : Example
        	Object to be destructed
        
        
        Automatically generated destructor for example
        """
        if self._alloc:
            _examplepkg.f90wrap_example_finalise(this=self._handle)
    
    @property
    def first(self):
        """
        Element first ftype=integer  pytype=int
        
        
        Defined at ./example.f90 line 13
        
        """
        return _examplepkg.f90wrap_example__get__first(self._handle)
    
    @first.setter
    def first(self, first):
        _examplepkg.f90wrap_example__set__first(self._handle, first)
    
    @property
    def second(self):
        """
        Element second ftype=integer  pytype=int
        
        
        Defined at ./example.f90 line 14
        
        """
        return _examplepkg.f90wrap_example__get__second(self._handle)
    
    @second.setter
    def second(self, second):
        _examplepkg.f90wrap_example__set__second(self._handle, second)
    
    @property
    def third(self):
        """
        Element third ftype=integer  pytype=int
        
        
        Defined at ./example.f90 line 15
        
        """
        return _examplepkg.f90wrap_example__get__third(self._handle)
    
    @third.setter
    def third(self, third):
        _examplepkg.f90wrap_example__set__third(self._handle, third)
    
    def __str__(self):
        ret = ['<example>{\n']
        ret.append('    first : ')
        ret.append(repr(self.first))
        ret.append(',\n    second : ')
        ret.append(repr(self.second))
        ret.append(',\n    third : ')
        ret.append(repr(self.third))
        ret.append('}')
        return ''.join(ret)
    
    _dt_array_initialisers = []
    

def _return_example_first(first):
    """
    instance = _return_example_first(first)
    
    
    Defined at ./example.f90 lines 25-36
    
    Parameters
    ----------
    first : int
    
    Returns
    -------
    instance : Example
    
    """
    instance = _examplepkg.f90wrap_return_example_first(first=first)
    instance = f90wrap.runtime.lookup_class("Example").from_handle(instance)
    return instance

def _return_example_second(first, second):
    """
    instance = _return_example_second(first, second)
    
    
    Defined at ./example.f90 lines 40-53
    
    Parameters
    ----------
    first : int
    second : int
    
    Returns
    -------
    instance : Example
    
    """
    instance = _examplepkg.f90wrap_return_example_second(first=first, second=second)
    instance = f90wrap.runtime.lookup_class("Example").from_handle(instance)
    return instance

def _return_example_third(first, second, third):
    """
    instance = _return_example_third(first, second, third)
    
    
    Defined at ./example.f90 lines 57-72
    
    Parameters
    ----------
    first : int
    second : int
    third : int
    
    Returns
    -------
    instance : Example
    
    """
    instance = _examplepkg.f90wrap_return_example_third(first=first, second=second, \
        third=third)
    instance = f90wrap.runtime.lookup_class("Example").from_handle(instance)
    return instance

def return_example(*args, **kwargs):
    """
    return_example(*args, **kwargs)
    
    
    Defined at ./example.f90 lines 8-10
    
    Overloaded interface containing the following procedures:
      _return_example_first
      _return_example_second
      _return_example_third
    
    """
    for proc in [_return_example_first, _return_example_second, \
        _return_example_third]:
        try:
            return proc(*args, **kwargs)
        except TypeError:
            continue


_array_initialisers = []
_dt_array_initialisers = []

try:
    for func in _array_initialisers:
        func()
except ValueError:
    logging.debug('unallocated array(s) detected on import of module \
        "class_example".')

for func in _dt_array_initialisers:
    func()
