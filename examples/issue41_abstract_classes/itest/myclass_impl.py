"""
Module myclass_impl


Defined at myclass_impl.f90 lines 1-16

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
    
    
    Defined at myclass_impl.f90 lines 3-6
    
    """
    def __del__(self):
        """
        Destructor for class Myclass_Impl_T
        
        
        Defined at myclass_impl.f90 lines 14-16
        
        Parameters
        ----------
        self : Myclass_Impl_T
        
        """
        if self._alloc:
            _itest.f90wrap_myclass_impl__myclass_impl_finalise__binding__mycla4a60(self=self._handle)
    
    def __del__(self):
        """
        Destructor for class Myclass_Impl_T
        
        
        Defined at myclass_impl.f90 lines 14-16
        
        Parameters
        ----------
        self : Myclass_Impl_T
        
        """
        if self._alloc:
            _itest.f90wrap_myclass_impl__myclass_impl_finalise(self=self._handle)
    
    def __init__(self, handle=None):
        """
        self = Myclass_Impl_T()
        
        
        Defined at myclass_impl.f90 lines 3-6
        
        
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
        
        
        Defined at myclass_impl.f90 lines 9-12
        
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
    
    
    Defined at myclass_impl.f90 lines 9-12
    
    Parameters
    ----------
    self : Myclass_Impl_T
    
    Returns
    -------
    value : float
    
    """
    value = _itest.f90wrap_myclass_impl__get_value_impl(self=self._handle)
    return value


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
