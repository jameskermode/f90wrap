"""
Module myclass_impl


Defined at myclass_impl.f90 lines 1-17

"""
from __future__ import print_function, absolute_import, division
import _itest
import f90wrap.runtime
import logging
import numpy

_arrays = {}
_objs = {}

@f90wrap.runtime.register_class("itest.myclass_impl_t")
class myclass_impl_t(f90wrap.runtime.FortranDerivedType):
    """
    Type(name=myclass_impl_t)
    
    
    Defined at myclass_impl.f90 lines 4-7
    
    """
    def __del__(self):
        """
        Destructor for class Myclass_Impl_T
        
        
        Defined at myclass_impl.f90 lines 15-17
        
        Parameters
        ----------
        self : Myclass_Impl_T
        
        """
        if self._alloc:
            _itest.f90wrap_myclass_impl__myclass_impl_destroy__binding__myclas021a(self=self._handle)
    
    def __init__(self, handle=None):
        """
        self = Myclass_Impl_T()
        
        
        Defined at myclass_impl.f90 lines 4-7
        
        
        Returns
        -------
        this : Myclass_Impl_T
        	Object to be constructed
        
        
        Automatically generated constructor for myclass_impl_t
        """
        f90wrap.runtime.FortranDerivedType.__init__(self)
        result = _itest.f90wrap_myclass_impl__myclass_impl_t_initialise()
        self._handle = result[0] if isinstance(result, tuple) else result
    
    def get_value(self):
        """
        value = get_value__binding__myclass_impl_t(self)
        
        
        Defined at myclass_impl.f90 lines 10-13
        
        Parameters
        ----------
        self : Myclass_Impl_T
        
        Returns
        -------
        value : float
        
        """
        value = \
            _itest.f90wrap_myclass_impl__get_value__binding__myclass_impl_t(self=self._handle)
        return value
    
    _dt_array_initialisers = []
    

def get_value_impl(self):
    """
    value = get_value_impl(self)
    
    
    Defined at myclass_impl.f90 lines 10-13
    
    Parameters
    ----------
    self : Myclass_Impl_T
    
    Returns
    -------
    value : float
    
    """
    value = _itest.f90wrap_myclass_impl__get_value_impl(self=self._handle)
    return value

def myclass_impl_destroy(self):
    """
    myclass_impl_destroy(self)
    
    
    Defined at myclass_impl.f90 lines 15-17
    
    Parameters
    ----------
    self : Myclass_Impl_T
    
    """
    _itest.f90wrap_myclass_impl__myclass_impl_destroy(self=self._handle)


_array_initialisers = []
_dt_array_initialisers = []

try:
    for func in _array_initialisers:
        func()
except ValueError:
    logging.debug('unallocated array(s) detected on import of module \
        "myclass_impl".')

for func in _dt_array_initialisers:
    func()
